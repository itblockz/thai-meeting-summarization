"""
E2 — evaluate bge-m3 multi-vector (ColBERT) reranking on exp03's pool.

Compares 4 strategies on the train set, all on the same dense∪bm25 top-20
candidate pool used by exp03's rerank cache:

  baseline_ce  cross-encoder only (= exp03 rerank, current best)
  pure_colbert ColBERT MaxSim only (Strategy B)
  ce_colbert   weighted sum: α × ce_z + (1-α) × cb_z  (Strategy A — tie-break / fusion)
  pool_C       upper bound: dense ∪ bm25 ∪ colbert top-20 pool, scored by oracle.
               (No new cross-encoder run yet — reports pool recall@20 ceiling.)

All scores are min-max-normalized within each query before combining, so
α weights have intuitive meaning. CPU only (~10 s).
"""
import os
import pickle
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")

from retrievers import (
    rank_dense, rank_bm25, rank_rerank, rank_colbert, colbert_score, tokenize_th,
)

HERE         = Path(__file__).resolve().parent
EMBED_CACHE  = HERE / "cache" / "train.npz"
COLBERT_PKL  = HERE / "cache" / "colbert_train.pkl"
RERANK_CACHE = HERE / "cache" / "rerank_train.json"
TEST_JSON    = HERE.parent / "textsum" / "eval_train" / "test.json"
OUT_PATH     = HERE / "result" / "e2_colbert.json"

POOL_N = 20
HIT_KS = (1, 5, 10, 20)


def as_list(refs):
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def hit_at(ranked, gold, k):
    return 1.0 if set(gold) & set(ranked[:k]) else 0.0


def mrr(ranked, gold):
    gold = set(gold)
    for i, p in enumerate(ranked):
        if p in gold:
            return 1.0 / (i + 1)
    return 0.0


def iou_at(ranked, gold, k):
    g = set(gold)
    top = set(ranked[:k])
    union = g | top
    return len(g & top) / len(union) if union else 0.0


def per_query_metrics(ranked, gold):
    m = {f"hit@{k}": hit_at(ranked, gold, k) for k in HIT_KS}
    m["MRR"]    = mrr(ranked, gold)
    m["iou@1"]  = iou_at(ranked, gold, 1)
    return m


def minmax(scores):
    s = np.asarray(scores, dtype=np.float32)
    if len(s) == 0:
        return s
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-9:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def aggregate(rows):
    keys = rows[0].keys() if rows else []
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    # ── load embedding + ColBERT + rerank caches ──────────────────────────
    if not COLBERT_PKL.exists():
        raise SystemExit(f"ColBERT cache not found: {COLBERT_PKL}\n"
                         f"run first:  sbatch eval_retrieval/submit_colbert_cache.sh")
    if not RERANK_CACHE.exists():
        raise SystemExit(f"rerank cache not found: {RERANK_CACHE}\n"
                         f"run first:  sbatch eval_retrieval/submit_rerank.sh")
    if not EMBED_CACHE.exists():
        raise SystemExit(f"embedding cache not found: {EMBED_CACHE}\n"
                         f"run first:  sbatch eval_retrieval/submit_embed.sh")

    print(f"loading caches...", flush=True)
    with open(COLBERT_PKL, "rb") as f:
        cb = pickle.load(f)
    cb_para_doc = cb["para_doc"]
    cb_para_pid = cb["para_pid"]
    cb_para_vec = cb["para_vecs"]
    cb_query_id = cb["query_id"]
    cb_query_vec = cb["query_vecs"]
    cb_para_index = {(d, p): i for i, (d, p) in enumerate(zip(cb_para_doc, cb_para_pid))}
    cb_query_index = {qid: i for i, qid in enumerate(cb_query_id)}

    z = np.load(EMBED_CACHE, allow_pickle=False)
    para_emb  = z["para_emb"]
    para_doc  = [str(x) for x in z["para_doc"]]
    para_pid  = [str(x) for x in z["para_pid"]]
    query_emb = z["query_emb"]
    query_id  = [str(x) for x in z["query_id"]]

    raw_rerank = json.loads(RERANK_CACHE.read_text())
    # rerank_cache stores in pool-build order — we need the original pool, NOT
    # sorted by score, since that's the candidate set we're rescoring.
    rerank_pool = {qid: [(pid, s) for pid, s in entries] for qid, entries in raw_rerank.items()}

    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    text_map = {}
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            text_map[(doc["doc_id"], p["para_id"])] = p["text"]
    gold_map = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}

    # group paragraphs by doc (for Strategy C pool expansion)
    doc_para = defaultdict(list)
    for i, d in enumerate(para_doc):
        doc_para[d].append((para_pid[i], i))

    # ── compute ColBERT scores per query, only on the existing rerank pool ─
    print(f"scoring {len(rerank_pool)} queries with ColBERT on existing pool...",
          flush=True)
    cb_scores = {}        # qid -> {pid: score}
    cb_pool_scores = {}   # qid -> [(pid, score), ...] in pool-build order
    for qid, pool in rerank_pool.items():
        if qid not in cb_query_index:
            continue
        qv = cb_query_vec[cb_query_index[qid]]
        d  = qdoc_map[qid]
        per_pid = {}
        out_pool = []
        for pid, _ in pool:
            key = (d, pid)
            if key not in cb_para_index:
                per_pid[pid] = 0.0
                out_pool.append((pid, 0.0))
                continue
            pv = cb_para_vec[cb_para_index[key]]
            s = colbert_score(qv, pv)
            per_pid[pid] = s
            out_pool.append((pid, s))
        cb_scores[qid] = per_pid
        cb_pool_scores[qid] = out_pool

    # ── evaluate ───────────────────────────────────────────────────────────
    qids = [q for q in qtxt_map if q in rerank_pool and q in cb_scores and gold_map.get(q)]
    print(f"queries with gold + both caches: {len(qids)}\n", flush=True)

    def eval_strategy(rank_fn):
        rows = []
        for qid in qids:
            ranked = rank_fn(qid)
            rows.append(per_query_metrics(ranked, gold_map[qid]))
        return aggregate(rows)

    # baseline: cross-encoder
    def ranker_ce(qid):
        return rank_rerank(rerank_pool[qid])

    # B — pure ColBERT
    def ranker_colbert(qid):
        return rank_rerank(cb_pool_scores[qid])

    # A — fusion with α (sweep below)
    def make_fusion_ranker(alpha):
        def f(qid):
            pool = rerank_pool[qid]
            pids = [p for p, _ in pool]
            ce_s = minmax([s for _, s in pool])
            cb_s = minmax([cb_scores[qid].get(p, 0.0) for p in pids])
            fused = alpha * ce_s + (1 - alpha) * cb_s
            order = np.argsort(-fused)
            return [pids[i] for i in order]
        return f

    results = {}
    print("Baseline strategies on the existing dense∪bm25 top-20 pool:")
    results["baseline_ce"] = eval_strategy(ranker_ce)
    results["pure_colbert"] = eval_strategy(ranker_colbert)

    print(f"  {'strategy':<14}{'hit@1':>9}{'hit@5':>9}{'MRR':>9}{'iou@1':>9}")
    for name, r in [("baseline_ce", results["baseline_ce"]),
                    ("pure_colbert", results["pure_colbert"])]:
        print(f"  {name:<14}{r['hit@1']:>9.4f}{r['hit@5']:>9.4f}"
              f"{r['MRR']:>9.4f}{r['iou@1']:>9.4f}")

    # A — sweep α (cross-encoder weight, 1.0 = pure CE, 0.0 = pure ColBERT)
    print(f"\nStrategy A — α × ce_z + (1-α) × cb_z  (1.0 = pure CE, 0.0 = pure ColBERT):")
    print(f"  {'α':>6}{'hit@1':>9}{'hit@5':>9}{'MRR':>9}{'iou@1':>9}")
    best_alpha, best_iou = None, 0.0
    fusion_rows = {}
    for alpha in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        r = eval_strategy(make_fusion_ranker(alpha))
        fusion_rows[f"alpha_{alpha:.2f}"] = r
        marker = ""
        if r["iou@1"] > best_iou:
            best_iou = r["iou@1"]
            best_alpha = alpha
            marker = " ← best iou@1"
        print(f"  {alpha:>6.2f}{r['hit@1']:>9.4f}{r['hit@5']:>9.4f}"
              f"{r['MRR']:>9.4f}{r['iou@1']:>9.4f}{marker}")
    results["fusion_sweep"] = fusion_rows
    results["best_fusion"] = {"alpha": best_alpha, "iou@1": best_iou}

    # ── Strategy C — pool expansion (recall ceiling) ──────────────────────
    # Build dense∪bm25∪colbert top-20 union per query; check pool recall@20.
    # We DON'T re-score with the cross-encoder here (would need a GPU pass);
    # instead we compute the pool's gold-coverage ceiling.
    print(f"\nStrategy C — pool expansion (recall ceiling, no rerun of CE):")

    # bm25 needs Thai-tokenized texts per doc
    from rank_bm25 import BM25Okapi
    doc_bundle = {}
    for d, items in doc_para.items():
        pids = [pid for pid, _ in items]
        embs = para_emb[[i for _, i in items]]
        texts = [text_map.get((d, pid), "") for pid in pids]
        bm25 = BM25Okapi([tokenize_th(t) for t in texts])
        doc_bundle[d] = (pids, embs, bm25)

    rows_ce_pool, rows_c_pool = [], []
    extra_per_query = []
    for qid in qids:
        d = qdoc_map[qid]
        if d not in doc_bundle:
            continue
        pids_all, embs, bm25 = doc_bundle[d]

        # current pool (dense ∪ bm25 top-20)
        qi = query_id.index(qid) if qid in query_id else None
        if qi is None:
            continue
        d_top = rank_dense(query_emb[qi], embs, pids_all)[:POOL_N]
        b_top = rank_bm25(tokenize_th(qtxt_map[qid]), bm25, pids_all)[:POOL_N]
        ce_pool = list(dict.fromkeys(d_top + b_top))

        # add ColBERT top-20 on the WHOLE doc (not just the existing pool)
        qv = cb_query_vec[cb_query_index[qid]]
        cb_doc_scores = []
        for pid in pids_all:
            key = (d, pid)
            if key in cb_para_index:
                cb_doc_scores.append((pid, colbert_score(qv, cb_para_vec[cb_para_index[key]])))
            else:
                cb_doc_scores.append((pid, 0.0))
        cb_top = [pid for pid, _ in sorted(cb_doc_scores, key=lambda x: -x[1])[:POOL_N]]

        c_pool = list(dict.fromkeys(ce_pool + cb_top))
        extra_per_query.append(len(c_pool) - len(ce_pool))

        g = set(gold_map[qid])
        rows_ce_pool.append({
            "pool_size":    len(ce_pool),
            "pool_recall":  len(g & set(ce_pool)) / len(g) if g else 0.0,
            "pool_hit":     1.0 if g & set(ce_pool) else 0.0,
        })
        rows_c_pool.append({
            "pool_size":    len(c_pool),
            "pool_recall":  len(g & set(c_pool)) / len(g) if g else 0.0,
            "pool_hit":     1.0 if g & set(c_pool) else 0.0,
        })

    if rows_ce_pool:
        agg_ce = aggregate(rows_ce_pool)
        agg_c  = aggregate(rows_c_pool)
        print(f"  baseline (dense∪bm25):   "
              f"pool size={agg_ce['pool_size']:.2f}  "
              f"hit={agg_ce['pool_hit']:.4f}  recall={agg_ce['pool_recall']:.4f}")
        print(f"  + ColBERT top-20:        "
              f"pool size={agg_c['pool_size']:.2f}  "
              f"hit={agg_c['pool_hit']:.4f}  recall={agg_c['pool_recall']:.4f}")
        print(f"  Δ pool hit  : {agg_c['pool_hit'] - agg_ce['pool_hit']:+.4f}")
        print(f"  Δ pool recall: {agg_c['pool_recall'] - agg_ce['pool_recall']:+.4f}")
        print(f"  avg extra candidates added: {np.mean(extra_per_query):.2f}")
        results["pool_C"] = {
            "baseline_pool": agg_ce,
            "expanded_pool": agg_c,
            "avg_extra_candidates": float(np.mean(extra_per_query)),
        }

    # ── persist ────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
