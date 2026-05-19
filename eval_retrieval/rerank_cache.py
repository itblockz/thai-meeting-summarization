"""
E1 — cross-encoder rerank scoring step.

For each query, builds a stage-1 candidate pool (dense top-N union bm25
top-N), scores every (query, paragraph) pair with a cross-encoder, and
caches the scores to JSON. eval.py --method rerank then ranks by these
scores. Slow (GPU); run once per (pool size, model) config.

  sbatch eval_retrieval/submit_rerank.sh
  POOL_N=30 sbatch eval_retrieval/submit_rerank.sh    # wider pool
"""
import os
import time
import json
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from retrievers import rank_dense, rank_bm25, tokenize_th

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
POOL_N       = int(os.environ.get("POOL_N", "20"))   # top-N from each stage-1 retriever

HERE  = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "train.npz"
TEST  = HERE.parent / "textsum" / "eval_train" / "test.json"
OUT   = HERE / "cache" / "rerank_train.json"


def main() -> None:
    if not CACHE.exists():
        raise SystemExit(f"embedding cache not found: {CACHE}\n"
                         f"run first:  sbatch eval_retrieval/submit_embed.sh")

    z         = np.load(CACHE, allow_pickle=False)
    para_emb  = z["para_emb"]
    para_doc  = [str(x) for x in z["para_doc"]]
    para_pid  = [str(x) for x in z["para_pid"]]
    query_emb = z["query_emb"]
    query_id  = [str(x) for x in z["query_id"]]

    with open(TEST, encoding="utf-8") as f:
        data = json.load(f)
    text_map = {}
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            text_map[(doc["doc_id"], p["para_id"])] = p["text"]
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}

    doc_idx = defaultdict(list)
    for i, d in enumerate(para_doc):
        doc_idx[d].append(i)

    doc_bundle = {}
    for d, idxs in doc_idx.items():
        pe   = para_emb[idxs]
        pids = [para_pid[i] for i in idxs]
        bm25 = BM25Okapi([tokenize_th(text_map.get((d, pid), "")) for pid in pids])
        doc_bundle[d] = (pe, pids, bm25)

    # build candidate pools, flatten (query, paragraph) pairs
    all_pairs, pair_qid, pair_pid = [], [], []
    for i, qid in enumerate(query_id):
        d = qdoc_map.get(qid)
        if d not in doc_bundle:
            continue
        pe, pids, bm25 = doc_bundle[d]
        qtext     = qtxt_map[qid]
        dense_top = rank_dense(query_emb[i], pe, pids)[:POOL_N]
        bm25_top  = rank_bm25(tokenize_th(qtext), bm25, pids)[:POOL_N]
        pool      = list(dict.fromkeys(dense_top + bm25_top))   # union, order-preserving
        for pid in pool:
            all_pairs.append((qtext, text_map.get((d, pid), "")))
            pair_qid.append(qid)
            pair_pid.append(pid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  queries={len(query_id)}  pairs={len(all_pairs)}  "
          f"pool_n={POOL_N}  model={RERANK_MODEL}", flush=True)

    ce = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
    t0 = time.time()
    scores = ce.predict(all_pairs, batch_size=64, show_progress_bar=False)
    print(f"scored {len(all_pairs)} pairs in {time.time() - t0:.1f}s", flush=True)

    cache = defaultdict(list)
    for qid, pid, s in zip(pair_qid, pair_pid, scores):
        cache[qid].append([pid, float(s)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"cached -> {OUT}  ({len(cache)} queries, pool_n={POOL_N})", flush=True)


if __name__ == "__main__":
    main()
