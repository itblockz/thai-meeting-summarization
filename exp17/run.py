"""
exp17 — exp03 (rerank + Qwen3-32B-AWQ) + E5: decoupled context/refs via
LLM self-citation.

Single change vs exp03: instead of feeding the LLM the single top-1
reranked paragraph (TOP_K=1), feed the top-GEN_K reranked paragraphs as a
numbered list. The LLM answers the question AND cites which paragraph
numbers it used; the cited paragraphs become `refs` (adaptive K), the
answer becomes `abstractive`. Retrieval, reranker, Qwen3-32B-AWQ and
sampling are otherwise identical to exp03, so the score delta is
attributable to E5 alone.

E5 is the exp02 idea — exp02 scored 0.5833 (failed) because Qwen2.5-7B
could not follow the citation format. ua048's toey/exp01 confirmed E5
works once the model is Qwen3-32B-AWQ. exp17 uses exp02's two-line
citation format (`คำตอบ:` / `ย่อหน้าที่ใช้:`); the sibling exp18 uses
toey's inline `[อ้างอิง: X]` tag.

GEN_K=5 is the elbow of the rerank recall curve: eval_retrieval's E0
harness (result/e0_eval.json) measured rerank hit@5=0.912 vs hit@1=0.739
(+17.3 pts), while hit@10 adds only +3.6 — feeding more than 5 adds
distractors without meaningful recall.
"""
from pathlib import Path
import os
import gc
import re
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

POOL_N         = 20     # top-N from each stage-1 retriever before rerank
GEN_K          = 5      # reranked paragraphs fed to the LLM as numbered context
MAX_PARA_CHARS = 600    # per-paragraph cap in the prompt (keeps within max_model_len)
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"

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


def build_prompt(query, paras):
    context = "\n".join(
        f"[{i + 1}] {t[:MAX_PARA_CHARS]}" for i, t in enumerate(paras)
    )
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทยอย่างกระชับ ใช้ข้อมูลจากย่อหน้าที่ให้มาเท่านั้น "
        f"ห้ามแต่งเติม จากนั้นระบุเฉพาะหมายเลขย่อหน้าที่ใช้ตอบจริง ๆ\n\n"
        f"ตอบตามรูปแบบนี้:\n"
        f"คำตอบ: <คำตอบภาษาไทย>\n"
        f"ย่อหน้าที่ใช้: <หมายเลข คั่นด้วยจุลภาค>"
    )


def parse_output(text, n_paras):
    """Split LLM output into (answer, cited_indices). cited_indices: 1-based, <= n_paras."""
    answer = text.strip()
    cited = []
    m = re.search(r"ย่อหน้าที่ใช้\s*[:：]?\s*(.*)", text, re.DOTALL)
    if m:
        first_line = m.group(1).split("\n")[0]
        cited = [int(x) for x in re.findall(r"\d+", first_line)]
        answer = text[:m.start()].strip()
    am = re.search(r"คำตอบ\s*[:：]\s*(.*)", answer, re.DOTALL)
    if am:
        answer = am.group(1).strip()
    seen, ordered = set(), []
    for i in cited:
        if 1 <= i <= n_paras and i not in seen:
            seen.add(i)
            ordered.append(i)
    return answer, ordered


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp17: {n} queries, {len(doc_index)} docs", flush=True)

    # ── Stage 1a: embed (GPU) + BM25 ──────────────────────────────────────
    # vLLM uses spawn (VLLM_WORKER_MULTIPROC_METHOD), so touching CUDA in the
    # parent here does not fork a corrupted context into vLLM workers.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
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
    query_embs = encode_texts(embed_model, [q["query"] for q in queries])
    del embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Embeddings + BM25 done ({device}).", flush=True)

    # ── Stage 1b: candidate pools ─────────────────────────────────────────
    pools = []   # per query: list of (para_id, text)
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
        pools.append([(valid[j]["para_id"], valid[j]["text"]) for j in pool_idx])
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

    # ── assemble: top-GEN_K reranked paragraphs per query ─────────────────
    items = []   # (ID, gen_pids, gen_texts, messages_or_None, query_text)
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool = pools[i]
        q_text = query["query"]
        if pool:
            scores = scores_by_q.get(i, [])
            order = sorted(range(len(pool)), key=lambda j: -scores[j])[:GEN_K]
            gen_pids  = [pool[j][0] for j in order]
            gen_texts = [pool[j][1] for j in order]
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": build_prompt(q_text, gen_texts)},
            ]
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((query["ID"], gen_pids, gen_texts, messages, q_text))

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=8192,
              gpu_memory_utilization=0.90, dtype="half")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[3] if it[3] is not None else [{"role": "user", "content": it[4]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    outputs = llm.generate(prompts, sampling)

    # ── Stage 3: parse answer + citations ─────────────────────────────────
    results = []
    n_explicit = 0
    ref_counts = []
    for it, out in zip(items, outputs):
        qid, gen_pids, gen_texts, _, q_text = it
        answer, cited = parse_output(out.outputs[0].text, len(gen_pids))
        if cited:
            ref_ids = [gen_pids[c - 1] for c in cited]
            n_explicit += 1
        elif gen_pids:
            ref_ids = [gen_pids[0]]          # fallback: top-1 reranked
        else:
            ref_ids = []
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
        ref_counts.append(len(ref_ids))
        results.append({"ID": qid, "abstractive": answer, "refs": ",".join(ref_ids)})

    print(f"citations: {n_explicit}/{len(results)} parsed explicit, "
          f"{len(results) - n_explicit} fell back to top-1", flush=True)
    print(f"avg refs/query: {sum(ref_counts) / len(ref_counts):.2f}", flush=True)

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
