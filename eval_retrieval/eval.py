"""
Retrieval harness — eval step (E0 baseline + E1 rerank).

Loads cached embeddings (embed_cache.py) and, when present, cross-encoder
rerank scores (rerank_cache.py), runs a retrieval config, and reports
recall / MRR / IoU against the train-set gold refs. No GPU, no LLM —
instant for dense, ~20s when BM25 is involved (Thai tokenization).

  python eval_retrieval/eval.py                  # dense + bm25 + rrf (+ rerank if cached)
  python eval_retrieval/eval.py --method dense
  python eval_retrieval/eval.py --method rerank

Metric reference:
  hit@k    query has >=1 gold ref within top-k        (retrieval recall)
  recall@5 fraction of gold refs found within top-5
  MRR      mean reciprocal rank of the first gold ref
  iou@k    IoU if exactly top-k were reported as refs (simulates REF_K=k)
  iou@oK   IoU if top-(true ref count) reported       (perfect-K ceiling)
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrievers import rank_dense, rank_bm25, rrf_fuse, rank_rerank, tokenize_th

HERE           = Path(__file__).resolve().parent
DEFAULT_CACHE  = HERE / "cache" / "train.npz"
DEFAULT_RERANK = HERE / "cache" / "rerank_train.json"
DEFAULT_TEST   = HERE.parent / "textsum" / "eval_train" / "test.json"

HIT_K = [1, 5, 10, 20]
IOU_K = [1, 3]
COLS  = ["hit@1", "hit@5", "hit@10", "hit@20", "recall@5", "MRR", "iou@1", "iou@3", "iou@oK"]


def as_list(refs) -> list:
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def metrics_for(ranked: list, gold: list) -> dict:
    gold = set(gold)
    m = {}
    first = next((i + 1 for i, p in enumerate(ranked) if p in gold), None)
    m["MRR"] = 1.0 / first if first else 0.0
    for k in HIT_K:
        m[f"hit@{k}"] = 1.0 if gold & set(ranked[:k]) else 0.0
    m["recall@5"] = len(gold & set(ranked[:5])) / len(gold) if gold else 0.0
    for k in IOU_K:
        top = set(ranked[:k])
        union = gold | top
        m[f"iou@{k}"] = len(gold & top) / len(union) if union else 0.0
    oracle = set(ranked[:len(gold)])
    union = gold | oracle
    m["iou@oK"] = len(gold & oracle) / len(union) if union else 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="all",
                    choices=["all", "dense", "bm25", "rrf", "rerank"])
    ap.add_argument("--cache", default=str(DEFAULT_CACHE))
    ap.add_argument("--rerank-cache", default=str(DEFAULT_RERANK))
    ap.add_argument("--test", default=str(DEFAULT_TEST))
    ap.add_argument("--rrf-k", type=int, default=60)
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise SystemExit(
            f"cache not found: {cache_path}\n"
            f"run the embed step first:  sbatch eval_retrieval/submit_embed.sh"
        )

    z         = np.load(cache_path, allow_pickle=False)
    para_emb  = z["para_emb"]
    para_doc  = [str(x) for x in z["para_doc"]]
    para_pid  = [str(x) for x in z["para_pid"]]
    query_emb = z["query_emb"]
    query_id  = [str(x) for x in z["query_id"]]

    with open(args.test, encoding="utf-8") as f:
        data = json.load(f)

    text_map = {}
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            text_map[(doc["doc_id"], p["para_id"])] = p["text"]
    gold_map = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}

    rerank_path = Path(args.rerank_cache)
    if args.method == "all":
        methods = ["dense", "bm25", "rrf"]
        if rerank_path.exists():
            methods.append("rerank")
    else:
        methods = [args.method]

    rerank_scores = {}
    if "rerank" in methods:
        if not rerank_path.exists():
            raise SystemExit(
                f"rerank cache not found: {rerank_path}\n"
                f"run the rerank step first:  sbatch eval_retrieval/submit_rerank.sh"
            )
        rerank_scores = json.loads(rerank_path.read_text())

    doc_idx = defaultdict(list)
    for i, d in enumerate(para_doc):
        doc_idx[d].append(i)

    cached_pids = defaultdict(set)
    for d, pid in zip(para_doc, para_pid):
        cached_pids[d].add(pid)
    total_gold = unreachable = 0
    for qid, gold in gold_map.items():
        for g in gold:
            total_gold += 1
            if g not in cached_pids[qdoc_map[qid]]:
                unreachable += 1

    need_bm25 = any(m in ("bm25", "rrf") for m in methods)

    doc_bundle = {}
    for d, idxs in doc_idx.items():
        pe   = para_emb[idxs]
        pids = [para_pid[i] for i in idxs]
        bm25 = None
        if need_bm25:
            from rank_bm25 import BM25Okapi
            bm25 = BM25Okapi([tokenize_th(text_map.get((d, pid), "")) for pid in pids])
        doc_bundle[d] = (pe, pids, bm25)

    qtok = {}
    if need_bm25:
        qtok = {qid: tokenize_th(qtxt_map[qid]) for qid in query_id}

    results = {}
    for method in methods:
        agg, n = defaultdict(float), 0
        for i, qid in enumerate(query_id):
            d = qdoc_map.get(qid)
            if d not in doc_bundle:
                continue
            pe, pids, bm25 = doc_bundle[d]
            if method == "dense":
                ranked = rank_dense(query_emb[i], pe, pids)
            elif method == "bm25":
                ranked = rank_bm25(qtok[qid], bm25, pids)
            elif method == "rrf":
                ranked = rrf_fuse(
                    [rank_bm25(qtok[qid], bm25, pids),
                     rank_dense(query_emb[i], pe, pids)],
                    k=args.rrf_k,
                )
            else:  # rerank
                ranked = rank_rerank(rerank_scores.get(qid, []))
            for key, val in metrics_for(ranked, gold_map[qid]).items():
                agg[key] += val
            n += 1
        results[method] = {key: val / n for key, val in agg.items()}

    print(f"\n=== retrieval eval — {Path(args.test).name} ===")
    print(f"queries={len(query_id)}  docs={len(doc_idx)}  "
          f"gold refs unreachable (filtered as invalid): {unreachable}/{total_gold}\n")
    header = f"{'method':<8}" + "".join(f"{c:>10}" for c in COLS)
    print(header)
    print("-" * len(header))
    for method in methods:
        r = results[method]
        print(f"{method:<8}" + "".join(f"{r[c]:>10.4f}" for c in COLS))

    if "dense" in results:
        d = results["dense"]
        print(f"\ncheck  : dense should reproduce the textsum baseline — "
              f"hit@1~0.575 iou@1~0.474  (got hit@1={d['hit@1']:.4f} iou@1={d['iou@1']:.4f})")
        print(f"ceiling: dense hit@20={d['hit@20']:.4f}  "
              f"<- max recall@1 a reranker over a top-20 pool could reach")
    if "rerank" in results and "dense" in results:
        rr, dd = results["rerank"], results["dense"]
        print(f"E1     : rerank vs dense — "
              f"hit@1 {dd['hit@1']:.4f} -> {rr['hit@1']:.4f} ({rr['hit@1'] - dd['hit@1']:+.4f}) | "
              f"iou@1 {dd['iou@1']:.4f} -> {rr['iou@1']:.4f} ({rr['iou@1'] - dd['iou@1']:+.4f})")

    out = HERE / "result" / "e0_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"queries": len(query_id), "docs": len(doc_idx),
                  "unreachable_gold": unreachable, "total_gold": total_gold,
                  "rrf_k": args.rrf_k},
         "results": results}, indent=2, ensure_ascii=False))
    print(f"\nsaved  -> {out}")


if __name__ == "__main__":
    main()
