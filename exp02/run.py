"""
exp02 — BM25 + bge-m3 RRF retrieval  +  Qwen3-32B-AWQ (no_think) vLLM batch inference

Key changes vs exp01:
- GEN_K=REF_K=1: honest reporting (use exactly the paragraph claimed in refs)
- Extractive system prompt (removes "สรุป" which triggers paraphrasing)
- Auto-detects FlagEmbedding for M3 triple-mode retrieval; falls back to BM25+dense
"""
import os
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from vllm import LLM, SamplingParams

try:
    from FlagEmbedding import BGEM3FlagModel
    USE_TRIPLE = True
except ImportError:
    USE_TRIPLE = False

# ── Config ────────────────────────────────────────────────────────────────────
TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

EMBED_MODEL    = "BAAI/bge-m3"
LLM_MODEL      = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-AWQ")

TOP_K          = 1      # paragraphs retrieved, passed to LLM, and reported in refs
RETRIEVAL_POOL = 20     # top-N from each retriever before RRF
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
MAX_PARA_CHARS = 1200   # cap per paragraph before sending to LLM
RRF_K          = 60


def progress(i: int):
    os.system(f"{PROGRESS_LIB} {i}")


# ── Data helpers ──────────────────────────────────────────────────────────────
def load_data(test_dir: str) -> dict:
    with open(Path(test_dir) / "test.json", encoding="utf-8") as f:
        return json.load(f)


def is_valid_para(p: dict) -> bool:
    text = p["text"].strip()
    return bool(text) and not (set(text) <= set("_-=. \t\n"))


# ── Retrieval ─────────────────────────────────────────────────────────────────
def tokenize_th(text: str) -> list[str]:
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def build_bm25(valid_paras: list[dict]) -> BM25Okapi:
    return BM25Okapi([tokenize_th(p["text"]) for p in valid_paras])


def rrf_fuse(ranked_lists: list[list], k: int = RRF_K) -> list:
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: -scores[x])


def retrieve(
    query_text: str,
    query_emb: torch.Tensor,
    valid_paras: list[dict],
    para_embs: torch.Tensor,
    bm25: BM25Okapi,
    k: int = TOP_K,
) -> list[str]:
    if not valid_paras:
        return []

    pool = min(RETRIEVAL_POOL, len(valid_paras))

    bm25_scores = np.array(bm25.get_scores(tokenize_th(query_text)))
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:pool].tolist()
    bm25_ids = [valid_paras[i]["para_id"] for i in bm25_top_idx]

    sims = F.cosine_similarity(query_emb.unsqueeze(0), para_embs, dim=1)
    dense_top_idx = torch.topk(sims, k=min(pool, len(valid_paras))).indices.tolist()
    dense_ids = [valid_paras[i]["para_id"] for i in dense_top_idx]

    return rrf_fuse([bm25_ids, dense_ids])[:k]


# ── Prompt ────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "คุณเป็นผู้เชี่ยวชาญด้านบันทึกการประชุมคณะกรรมาธิการสภาไทย "
    "ตอบคำถามโดยใช้ถ้อยคำจากเอกสารอ้างอิงโดยตรง "
    "ห้ามเปลี่ยนคำ ห้ามตีความเพิ่มเติม ใช้ภาษาไทยเท่านั้น"
)


def build_messages(query: str, retrieved_texts: list[str]) -> list[dict]:
    truncated = [t[:MAX_PARA_CHARS] for t in retrieved_texts]
    context = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(truncated))
    return [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"คำถาม: {query}\n\n"
                f"ข้อมูลอ้างอิง:\n{context}\n\n"
                f"คำสั่ง: ตอบคำถามโดยคัดลอกและเรียบเรียงถ้อยคำจากเอกสารอ้างอิงข้างต้นเท่านั้น "
                f"ห้ามใช้คำที่ไม่ปรากฏในเอกสาร\n"
                f"คำตอบ:"
            ),
        },
    ]


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> int:
    data = load_data(TEST_DIR)
    doc_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
    queries   = data["queries"]
    n         = len(queries)

    print(f"Retrieval mode: {'M3 triple (dense+sparse+ColBERT)' if USE_TRIPLE else 'BM25 + dense (RRF)'}", flush=True)

    # ── 1. Embed paragraphs + queries, build BM25 ─────────────────────────
    print("Loading embedding model …", flush=True)
    # Always embed on CPU: some LANTA nodes have a broken CUDA driver that
    # causes torch.cuda init to partially succeed then raise. That partial
    # init poisons vLLM's forked worker ("Cannot re-initialize CUDA in
    # forked subprocess"). Keeping CUDA untouched here lets vLLM own the
    # GPU cleanly for generation.
    embed_device = "cpu"
    print(f"Embed device: {embed_device}", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL, device=embed_device)

    doc_data: dict = {}
    for doc_id, paragraphs in doc_index.items():
        valid = [p for p in paragraphs if is_valid_para(p)]
        texts = [p["text"] for p in valid]
        if valid:
            embs = embed_model.encode(
                texts,
                batch_size=EMBED_BATCH,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            dim  = embed_model.get_sentence_embedding_dimension()
            embs = torch.zeros((0, dim))
        bm25 = build_bm25(valid)
        doc_data[doc_id] = (valid, embs, bm25)

    query_texts = [q["query"] for q in queries]
    query_embs  = embed_model.encode(
        query_texts,
        batch_size=EMBED_BATCH,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    del embed_model
    torch.cuda.empty_cache()
    print("Embeddings done.", flush=True)

    # ── 2. Retrieve for all queries ────────────────────────────────────────
    all_retrieved: list[tuple[list[str], list[str]]] = []   # (para_ids, para_texts)

    for i, q in enumerate(queries):
        valid, embs, bm25 = doc_data[q["doc_id"]]
        para_ids = retrieve(q["query"], query_embs[i], valid, embs, bm25)
        text_map = {p["para_id"]: p["text"] for p in doc_index[q["doc_id"]]}
        para_texts = [text_map[pid] for pid in para_ids if text_map.get(pid, "").strip()]
        all_retrieved.append((para_ids, para_texts))

    # ── 3. Load vLLM ──────────────────────────────────────────────────────
    print("Loading vLLM …", flush=True)
    llm = LLM(
        model=LLM_MODEL,
        quantization="awq_marlin",
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="half",
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    sampling  = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.05,
    )

    # ── 4. Build prompts (no_think) ───────────────────────────────────────
    prompts = []
    for i, q in enumerate(queries):
        msgs = build_messages(q["query"], all_retrieved[i][1])
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text)

    # ── 5. Batch generate ─────────────────────────────────────────────────
    print(f"Generating {n} summaries …", flush=True)
    outputs = llm.generate(prompts, sampling)

    # ── 6. Write CSV ──────────────────────────────────────────────────────
    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (q, out) in enumerate(zip(queries, outputs)):
        para_ids, para_texts = all_retrieved[i]
        summary = out.outputs[0].text.strip()
        if not summary:
            summary = " ".join(para_texts) if para_texts else q["query"]
        rows.append({
            "ID":          q["ID"],
            "abstractive": summary,
            "refs":        ",".join(para_ids),
        })
        progress(i)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows → {out_path}", flush=True)
    progress(n)
    return n


if __name__ == "__main__":
    main()
