from pathlib import Path
import os
import json
import csv

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

TOP_K = 1   # paragraphs retrieved, passed to LLM, and reported in refs
EMBED_BATCH = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL = "BAAI/bge-m3"
MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def benchmark_lib(i):
    os.system(f"{PROGRESS_LIB} {i}")


def load_data(test_dir):
    path = Path(test_dir) / "test.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_valid_paragraphs(paragraphs):
    def is_valid(p):
        text = p["text"].strip()
        if not text:
            return False
        if set(text) <= set("_-=. \t"):
            return False
        return True
    return [p for p in paragraphs if is_valid(p)]


def encode_texts(model, texts, batch_size=64):
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def retrieve_top_k(query_emb, para_embs, valid_paras, k=3):
    if para_embs.shape[0] == 0:
        return []
    scores = F.cosine_similarity(query_emb.unsqueeze(0), para_embs, dim=1)
    k = min(k, len(valid_paras))
    top_indices = torch.topk(scores, k=k).indices.tolist()
    return [valid_paras[i]["para_id"] for i in top_indices]


def build_prompt(query, retrieved_texts):
    context = "\n".join(
        f"ย่อหน้า {i + 1}: {text}" for i, text in enumerate(retrieved_texts)
    )
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น\n"
        f"คำตอบ:"
    )


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]

    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    doc_para_data = {}
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        if valid:
            embs = encode_texts(embed_model, [p["text"] for p in valid], batch_size=EMBED_BATCH)
        else:
            embs = torch.zeros((0, embed_model.get_sentence_embedding_dimension()))
        doc_para_data[doc_id] = (valid, embs)

    # Encode all queries in one batch
    query_texts = [q["query"] for q in queries]
    query_embs = encode_texts(embed_model, query_texts, batch_size=EMBED_BATCH)

    del embed_model

    # Phase 1: Retrieval → build messages
    # items: (query_ID, ref_ids, messages_or_None, fallback_text)
    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)

        doc_id = query["doc_id"]
        q_text = query["query"]
        q_emb  = query_embs[i]

        valid_paras, para_embs = doc_para_data.get(doc_id, ([], torch.zeros((0, 1))))

        if valid_paras and para_embs.shape[0] > 0:
            ref_ids = retrieve_top_k(q_emb, para_embs, valid_paras, k=TOP_K)
        else:
            ref_ids = []

        para_text_map = {p["para_id"]: p["text"] for p in doc_index.get(doc_id, [])}
        retrieved_texts = [para_text_map[pid] for pid in ref_ids if para_text_map.get(pid, "").strip()]

        if retrieved_texts:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
                        "ตอบคำถามโดยสรุปจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเติม"
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(q_text, retrieved_texts),
                },
            ]
        else:
            messages = None

        items.append((query["ID"], ref_ids, messages, q_text))

    # Phase 2: Batch LLM generation
    llm = LLM(model=MODEL_NAME, max_model_len=4096, gpu_memory_utilization=0.90, dtype="half")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)

    prompts = []
    for _, it in enumerate(items):
        msgs = it[2] if it[2] is not None else [{"role": "user", "content": it[3]}]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    outputs = llm.generate(prompts, sampling)

    results = []
    for i, (it, out) in enumerate(zip(items, outputs)):
        summary = out.outputs[0].text.strip() or it[3]
        results.append({"ID": it[0], "abstractive": summary, "refs": ",".join(it[1])})

    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} rows to {out_path}")
    return len(queries)


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)  # must be last operation
