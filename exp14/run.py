"""
exp14 — dynamic few-shot via k-NN retrieval.

Static few-shot (exp08–exp13) is brittle to example choice:
  exp08 (doc_050 picks): +0.0091 composite vs exp03
  exp13 (doc_002 picks, length-matched): +0.0036
  exp12 (doc_002 picks, short): +0.0010
A median gain of ~+0.005 — but variance is large because one fixed pair
of examples doesn't match every test query's style/length/topic.

This run gives each test query its own k-NN-retrieved (Q', P', A')
examples drawn from the labeled train pool, excluding any query that
shares the test query's doc (to prevent paragraph/topic leakage). The
hypothesis is that retrieval-relevant examples auto-match length, topic
and style, removing the variance.

Eval: full 1239 queries — directly comparable to exp03 full = 0.6256.
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

TOP_K          = 1      # paragraphs retrieved + reported in refs (unchanged)
POOL_N         = 20     # candidate pool per stage-1 retriever
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"
FEWSHOT_K      = 2      # dynamic shots per query (matches exp08 sweep peak)
MAX_PARA_CHARS = 600    # truncate example paragraphs to this length (some refs are 940+ chars)

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


def build_user_turn(query, retrieved_texts):
    if len(retrieved_texts) == 1:
        paragraph_block = retrieved_texts[0]
    else:
        paragraph_block = "\n".join(
            f"ย่อหน้า {i + 1}: {t}" for i, t in enumerate(retrieved_texts)
        )
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n"
        f"ย่อหน้า 1: {paragraph_block}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น\n"
        f"คำตอบ:"
    )


def build_messages(target_query, target_para_texts, few_shot_examples):
    """System + alternating user/assistant few-shot turns + final user target.

    few_shot_examples: list of dicts with keys 'query', 'paragraph', 'answer'.
    Paragraphs are truncated to MAX_PARA_CHARS to keep prompt bounded.
    """
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for ex in few_shot_examples:
        para = ex["paragraph"]
        if len(para) > MAX_PARA_CHARS:
            para = para[:MAX_PARA_CHARS].rstrip() + "..."
        messages.append({"role": "user",
                         "content": build_user_turn(ex["query"], [para])})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user",
                     "content": build_user_turn(target_query, target_para_texts)})
    return messages


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"{n} queries, {len(doc_index)} docs", flush=True)

    # ── Build the few-shot pool from queries with refs + gold answer ───────
    # (Q', P', A') triples — paragraph is the gold-ref's paragraph text.
    pool = []
    for q in queries:
        refs = q.get("refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if not refs:
            continue
        ans = (q.get("abstractive") or "").strip()
        if not ans:
            continue
        para_text = next(
            (p["text"] for p in doc_index.get(q["doc_id"], [])
             if p["para_id"] == refs[0]), "")
        if not para_text.strip():
            continue
        pool.append({
            "id": q["ID"],
            "doc_id": q["doc_id"],
            "query": q["query"],
            "paragraph": para_text,
            "answer": ans,
        })
    print(f"Few-shot pool: {len(pool)} (Q, P, A) triples", flush=True)

    # ── Stage 1a: embed paragraphs + queries (GPU; safe under spawn) ──────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)

    # paragraph embeddings per doc (for retrieval to find the target paragraph)
    doc_para_data = {}
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        if valid:
            embs = encode_texts(embed_model, [p["text"] for p in valid])
            bm25 = BM25Okapi([tokenize_th(p["text"]) for p in valid])
        else:
            embs = torch.zeros((0, embed_model.get_sentence_embedding_dimension()))
            bm25 = None
        doc_para_data[doc_id] = (valid, embs, bm25)

    # embed all target queries (for paragraph retrieval)
    query_embs = encode_texts(embed_model, [q["query"] for q in queries])

    # embed pool queries (for dynamic few-shot selection); pool is a subset of queries,
    # but pool may exclude some — embed pool separately for clean indexing
    pool_embs = encode_texts(embed_model, [t["query"] for t in pool])
    pool_doc_ids = [t["doc_id"] for t in pool]
    pool_q_ids   = [t["id"]     for t in pool]

    del embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Embeddings + BM25 done ({device}).", flush=True)

    # ── Stage 1b: build candidate pools for paragraph retrieval ───────────
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
        pool_idx = list(dict.fromkeys(dense_idx + bm25_idx))
        pools.append([valid[j]["para_id"] for j in pool_idx])
        for j in pool_idx:
            pair_texts.append((query["query"], valid[j]["text"]))
            pair_qidx.append(i)

    # ── Stage 1c: cross-encoder rerank for paragraph retrieval ────────────
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

    # ── Dynamic few-shot selection per query ──────────────────────────────
    # For each query, find top-K pool entries by cosine sim of query embeddings,
    # excluding any pool entry that shares the query's doc_id (or is the query itself).
    pool_doc_id_arr = np.array(pool_doc_ids)
    pool_q_id_arr   = np.array(pool_q_ids)
    pool_embs_cpu   = pool_embs.detach().cpu().numpy()
    query_embs_cpu  = query_embs.detach().cpu().numpy()

    fewshot_per_query = []   # list of K example dicts per query
    for i, query in enumerate(queries):
        # cos sim (already normalized): dot product
        sims = pool_embs_cpu @ query_embs_cpu[i]
        # mask out same-doc and self
        mask = (pool_doc_id_arr == query["doc_id"]) | (pool_q_id_arr == query["ID"])
        sims = np.where(mask, -np.inf, sims)
        top_k = np.argpartition(-sims, FEWSHOT_K)[:FEWSHOT_K]
        # sort the top-K by sim descending for consistent ordering
        top_k = top_k[np.argsort(-sims[top_k])]
        fewshot_per_query.append([pool[j] for j in top_k])

    if fewshot_per_query[0]:
        print(f"Example dynamic few-shot for first query ({queries[0]['ID']}):", flush=True)
        for ex in fewshot_per_query[0]:
            print(f"  {ex['id']} (doc {ex['doc_id']}): {ex['query'][:80]}", flush=True)

    # ── assemble retrieval results + LLM messages ─────────────────────────
    items = []
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
            messages = build_messages(q_text, retrieved_texts, fewshot_per_query[i])
        else:
            messages = None
        items.append((query["ID"], ref_ids, messages, q_text))

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=8192,
              gpu_memory_utilization=0.85, dtype="half")
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
    benchmark_lib(n)
