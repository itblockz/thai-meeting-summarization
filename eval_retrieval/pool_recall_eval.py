"""
E4 quick test — pool-recall sweep for hybrid fusion tuning.

exp30 feeds the FULL dense∪BM25 union pool to the LLM, so what matters for
IoU is **pool inclusion of gold**, not rank-1. This script sweeps several
candidate-pool construction strategies on CPU using train.npz, instantly,
no GPU/LLM. If any strategy lifts pool-recall by >=0.005 over the current
exp30 default, that's a green light for a full LLM run (exp33).

Metrics per strategy:
  pool_size     mean candidate pool size (smaller = less LLM context = faster)
  pool_recall   fraction of queries with >=1 gold ref in the pool
  ref_recall    fraction of *gold refs* covered (multi-ref aware)
  hit@5/10/20   smaller-cut recall (matches truncated configs)
  iou@K=avg     iou if we reported the whole pool as refs (lower bound, sanity)

Reference numbers from CLAUDE.md:
  Current exp30 default = union_20 (dense top-20 ∪ BM25 top-20, set-union)
  exp30 measured pool size: mean 28.57, median 28, max 39
  Failure analysis: 9.9% of IoU=0 = "gold not in pool at all" -> ceiling for E4
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from retrievers import rank_dense, rank_bm25, rrf_fuse, tokenize_th

HERE          = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "cache" / "train.npz"
DEFAULT_TEST  = HERE.parent / "textsum" / "eval_train" / "test.json"


def as_list(refs):
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def union_pool(dense_ranked, bm25_ranked, n_each):
    """Concat-union (current exp30 default): preserves dense order first."""
    seen, out = set(), []
    for src in (dense_ranked[:n_each], bm25_ranked[:n_each]):
        for pid in src:
            if pid not in seen:
                seen.add(pid)
                out.append(pid)
    return out


def rrf_pool(dense_ranked, bm25_ranked, k_rrf, cap):
    fused = rrf_fuse([dense_ranked, bm25_ranked], k=k_rrf)
    return fused[:cap]


def weighted_pool(dense_scores, bm25_scores, pids, alpha, cap):
    """norm(dense) + alpha * norm(bm25); top-cap by combined score."""
    def normz(x):
        x = np.asarray(x, dtype=np.float32)
        mn, mx = x.min(), x.max()
        return (x - mn) / max(mx - mn, 1e-9)
    combined = (1.0 - alpha) * normz(dense_scores) + alpha * normz(bm25_scores)
    order = np.argsort(-combined)[:cap]
    return [pids[i] for i in order]


def metrics_for_pool(pool, gold):
    gold_set = set(gold)
    in_pool = gold_set & set(pool)
    return {
        "pool_size": len(pool),
        "pool_recall": 1.0 if in_pool else 0.0,
        "ref_recall":  len(in_pool) / len(gold_set) if gold_set else 0.0,
        "iou_full":    (len(in_pool) / len(gold_set | set(pool))) if (gold_set | set(pool)) else 0.0,
    }


def cut_hits(ranked, gold, ks=(1, 5, 10, 20)):
    g = set(gold)
    return {f"hit@{k}": 1.0 if g & set(ranked[:k]) else 0.0 for k in ks}


def main():
    z         = np.load(DEFAULT_CACHE, allow_pickle=False)
    para_emb  = z["para_emb"]
    para_doc  = [str(x) for x in z["para_doc"]]
    para_pid  = [str(x) for x in z["para_pid"]]
    query_emb = z["query_emb"].astype(np.float32)
    query_id  = [str(x) for x in z["query_id"]]

    with open(DEFAULT_TEST, encoding="utf-8") as f:
        data = json.load(f)
    text_map = {(d["doc_id"], p["para_id"]): p["text"]
                for d in data["docs"] for p in d["paragraphs"]}
    gold_map = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}

    doc_idx = defaultdict(list)
    for i, d in enumerate(para_doc):
        doc_idx[d].append(i)

    from rank_bm25 import BM25Okapi
    doc_bundle = {}
    for d, idxs in doc_idx.items():
        pe   = para_emb[idxs]
        pids = [para_pid[i] for i in idxs]
        bm25 = BM25Okapi([tokenize_th(text_map.get((d, pid), "")) for pid in pids])
        doc_bundle[d] = (pe, pids, bm25)
    qtok = {qid: tokenize_th(qtxt_map[qid]) for qid in query_id}

    # Precompute per-query rankings & scores (so each strategy is O(1))
    print(f"Pre-computing dense + BM25 rankings for {len(query_id)} queries...",
          flush=True)
    cache = []
    for i, qid in enumerate(query_id):
        d = qdoc_map.get(qid)
        if d not in doc_bundle:
            cache.append(None)
            continue
        pe, pids, bm25 = doc_bundle[d]
        dense_scores = pe @ query_emb[i]
        dense_order  = np.argsort(-dense_scores)
        dense_ranked = [pids[j] for j in dense_order]
        bm25_scores = np.asarray(bm25.get_scores(qtok[qid]))
        bm25_order  = np.argsort(-bm25_scores)
        bm25_ranked = [pids[j] for j in bm25_order]
        cache.append({
            "pids": pids,
            "dense_ranked": dense_ranked,
            "bm25_ranked":  bm25_ranked,
            "dense_scores": dense_scores,
            "bm25_scores":  bm25_scores,
        })

    STRATEGIES = []
    # (label, build_fn)  build_fn -> list of pids
    for n in (10, 15, 20, 25, 30, 40):
        STRATEGIES.append(
            (f"union_{n:02d}",
             lambda c, n=n: union_pool(c["dense_ranked"], c["bm25_ranked"], n)))
    for k_rrf in (10, 30, 60, 100):
        for cap in (20, 28, 40):
            STRATEGIES.append(
                (f"rrf_k{k_rrf:03d}_top{cap:02d}",
                 lambda c, k=k_rrf, cap=cap: rrf_pool(
                     c["dense_ranked"], c["bm25_ranked"], k, cap)))
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        STRATEGIES.append(
            (f"weighted_a{alpha:.2f}_top28",
             lambda c, a=alpha: weighted_pool(
                 c["dense_scores"], c["bm25_scores"], c["pids"], a, 28)))

    results = {}
    for label, builder in STRATEGIES:
        agg = defaultdict(float); n = 0
        for i, qid in enumerate(query_id):
            c = cache[i]
            if c is None:
                continue
            pool = builder(c)
            m = metrics_for_pool(pool, gold_map[qid])
            for k, v in m.items():
                agg[k] += v
            n += 1
        results[label] = {k: v / n for k, v in agg.items()}

    cols = ["pool_size", "pool_recall", "ref_recall", "iou_full"]
    print(f"\n=== E4 pool-recall sweep ({len(query_id)} queries) ===")
    header = f"{'strategy':<28}" + "".join(f"{c:>14}" for c in cols)
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        row = f"{label:<28}"
        for c in cols:
            row += f"{r[c]:>14.4f}"
        print(row)

    base = results["union_20"]
    print(f"\nbaseline (exp30 default = union_20): "
          f"pool_size={base['pool_size']:.2f}  "
          f"pool_recall={base['pool_recall']:.4f}  "
          f"ref_recall={base['ref_recall']:.4f}")

    deltas = []
    for label, r in results.items():
        if label == "union_20":
            continue
        d_pr = r["pool_recall"] - base["pool_recall"]
        d_rr = r["ref_recall"]  - base["ref_recall"]
        d_ps = r["pool_size"]   - base["pool_size"]
        deltas.append((label, d_pr, d_rr, d_ps, r))

    print(f"\n=== deltas vs union_20 (positive = better) ===")
    print(f"{'strategy':<28}{'d_pool_recall':>16}{'d_ref_recall':>16}{'d_pool_size':>14}")
    for label, d_pr, d_rr, d_ps, _ in sorted(deltas, key=lambda x: -x[1]):
        print(f"{label:<28}{d_pr:>+16.4f}{d_rr:>+16.4f}{d_ps:>+14.2f}")

    out = HERE / "result" / "pool_recall_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
