"""
Thai document summarization pipeline.

Retrieval is two-stage: dense (bge-m3) + BM25 build a candidate pool, a
cross-encoder (bge-reranker-v2-m3) re-scores it, the top-1 paragraph is
kept. Generation: Qwen3-32B-AWQ via vLLM (matches exp03, best train-set
composite 0.6256). Output: submission.csv with columns ID, abstractive, refs.
"""
from pathlib import Path
import os
import gc
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

TOP_K          = 1      # paragraphs retrieved, passed to LLM, and reported in refs
POOL_N         = 20     # top-N from each stage-1 retriever before rerank
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-AWQ")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยสรุปจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเติม"
)


def benchmark_lib(i):
    os.system(f"{PROGRESS_LIB} {i}")


def load_data(test_dir):
    with open(Path(test_dir) / "test.json", encoding="utf-8") as f:
        return json.load(f)


def filter_valid_paragraphs(paragraphs):
    def is_valid(p):
        text = p["text"].strip()
        if not text:
            return False
        if set(text) <= set("_-=. \t\n"):
            return False
        return True
    return [p for p in paragraphs if is_valid(p)]


def tokenize_th(text):
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def encode_texts(model, texts, batch_size=EMBED_BATCH):
    return model.encode(
        texts, batch_size=batch_size, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


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
    n = len(queries)
    print(f"{n} queries, {len(doc_index)} docs", flush=True)

    # ── Stage 1a: embed paragraphs + queries (CPU), build BM25 ────────────
    # CPU so CUDA isn't initialized before vLLM spawns its workers.
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    doc_para_data = {}   # doc_id -> (valid_paras, para_embs, bm25)
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        if valid:
            embs = encode_texts(embed_model, [p["text"] for p in valid])
            bm25 = BM25Okapi([tokenize_th(p["text"]) for p in valid])
        else:
            embs = torch.zeros((0, embed_model.get_sentence_embedding_dimension()))
            bm25 = None
        doc_para_data[doc_id] = (valid, embs, bm25)

    query_embs = encode_texts(embed_model, [q["query"] for q in queries])
    del embed_model
    print("Embeddings + BM25 done.", flush=True)

    # ── Stage 1b: build candidate pools, collect (query, paragraph) pairs ──
    pools = []
    pair_texts, pair_qidx = [], []
    for i, query in enumerate(queries):
        valid, para_embs, bm25 = doc_para_data.get(
            query["doc_id"], ([], torch.zeros((0, 1)), None))
        if not valid:
            pools.append([])
            continue
        sims = F.cosine_similarity(query_embs[i].unsqueeze(0), para_embs, dim=1)
        dense_idx = torch.topk(sims, k=min(POOL_N, len(valid))).indices.tolist()
        bm25_scores = np.asarray(bm25.get_scores(tokenize_th(query["query"])))
        bm25_idx = np.argsort(-bm25_scores)[:POOL_N].tolist()
        pool_idx = list(dict.fromkeys(dense_idx + bm25_idx))   # union, order-preserving
        pools.append([valid[j]["para_id"] for j in pool_idx])
        for j in pool_idx:
            pair_texts.append((query["query"], valid[j]["text"]))
            pair_qidx.append(i)

    # ── Stage 1c: cross-encoder rerank (GPU) ──────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
    pair_scores = reranker.predict(pair_texts, batch_size=64, show_progress_bar=False)
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Reranked {len(pair_texts)} pairs ({device}).", flush=True)

    scores_by_q = {}
    for qi, s in zip(pair_qidx, pair_scores):
        scores_by_q.setdefault(qi, []).append(float(s))

    # ── assemble retrieval results + LLM messages ─────────────────────────
    items = []   # (ID, ref_ids, messages_or_None, query_text)
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool_pids = pools[i]
        q_text = query["query"]
        if pool_pids:
            scores = scores_by_q.get(i, [])
            order = sorted(range(len(pool_pids)), key=lambda j: -scores[j])
            ref_ids = [pool_pids[j] for j in order[:TOP_K]]
        else:
            ref_ids = []

        para_text_map = {p["para_id"]: p["text"] for p in doc_index.get(query["doc_id"], [])}
        retrieved_texts = [para_text_map[pid] for pid in ref_ids
                           if para_text_map.get(pid, "").strip()]

        if retrieved_texts:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_prompt(q_text, retrieved_texts)},
            ]
        else:
            messages = None
        items.append((query["ID"], ref_ids, messages, q_text))

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=4096,
              gpu_memory_utilization=0.90, dtype="half", enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[2] if it[2] is not None else [{"role": "user", "content": it[3]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    outputs = llm.generate(prompts, sampling)

    results = []
    for it, out in zip(items, outputs):
        summary = out.outputs[0].text.strip() or it[3]
        results.append({"ID": it[0], "abstractive": summary, "refs": ",".join(it[1])})

    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} rows to {out_path}", flush=True)
    return n


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)  # must be last operation
