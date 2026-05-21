"""
E6 — adaptive K from rerank score distribution.

For each query, decide how many paragraphs to report as `refs` based on
the bge-reranker score profile, instead of always reporting K=1. The
goal is to recover IoU on the 28.3% of queries that have ≥2 gold refs
without hurting the 71.7% single-ref queries.

CPU-only, runs in seconds. Sweeps several strategies × thresholds
against gold refs on train set and reports the best IoU achievable.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RERANK_CACHE = HERE / "cache/rerank_train.json"
TEST_JSON    = PROJECT / "textsum/eval_train/test.json"
RESULT_DIR   = HERE / "result"

K_CAP = 5   # never report more than this many refs


def as_list(refs):
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def iou(gold, pred):
    g, p = set(gold), set(pred)
    union = g | p
    return len(g & p) / len(union) if union else 0.0


def main():
    with open(RERANK_CACHE) as f:
        raw_cache = json.load(f)   # qid -> [[pid, score], ...] in pool-build order

    # rerank_cache.py writes entries in candidate-pool order, NOT score order.
    # Sort by score descending so cache[qid][0] is the rank-1 reranked pick.
    rerank_cache = {qid: sorted(items, key=lambda x: -x[1]) for qid, items in raw_cache.items()}

    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    gold = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}

    # only queries that have rerank scores AND gold refs
    qids = [qid for qid in gold if qid in rerank_cache and gold[qid]]
    print(f"queries with gold + rerank cache: {len(qids)}", flush=True)

    # ref-count distribution (for context)
    rc = defaultdict(int)
    for qid in qids:
        rc[min(len(gold[qid]), 5)] += 1
    print("ref count distribution:")
    for k in sorted(rc):
        share = rc[k] / len(qids)
        print(f"  {k} refs: {rc[k]:>4} ({share:.1%})")
    print()

    # ── baseline: K=1 always ──────────────────────────────────────────────
    iou_k1 = np.mean([iou(gold[qid], [rerank_cache[qid][0][0]]) for qid in qids])
    iou_k2 = np.mean([iou(gold[qid], [c[0] for c in rerank_cache[qid][:2]]) for qid in qids])
    iou_k3 = np.mean([iou(gold[qid], [c[0] for c in rerank_cache[qid][:3]]) for qid in qids])
    print(f"baseline K=1 (current): IoU = {iou_k1:.4f}")
    print(f"baseline K=2 always   : IoU = {iou_k2:.4f}")
    print(f"baseline K=3 always   : IoU = {iou_k3:.4f}")

    # oracle: pick K = len(gold) for each query
    iou_oracle = np.mean([
        iou(gold[qid], [c[0] for c in rerank_cache[qid][:len(gold[qid])]])
        for qid in qids
    ])
    print(f"oracle K=|gold|       : IoU = {iou_oracle:.4f}  (ceiling for adaptive K)")
    print()

    # ── Strategy A: absolute gap top1 − topK ──────────────────────────────
    # K is the smallest j such that score[0] - score[j] > T  (i.e., elbow).
    # Equivalently: keep candidates whose score is within T of top1.
    def k_from_abs_gap(scores, T):
        s0 = scores[0]
        k = 1
        for s in scores[1:K_CAP]:
            if s0 - s <= T:
                k += 1
            else:
                break
        return k

    print("Strategy A — keep candidates within absolute gap T of top1:")
    print(f"  {'T':>6} {'IoU':>8} {'mean K':>8}")
    best_a = (None, 0.0, 0)
    for T in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]:
        ious, ks = [], []
        for qid in qids:
            scores = [c[1] for c in rerank_cache[qid]]
            k = k_from_abs_gap(scores, T)
            pred = [c[0] for c in rerank_cache[qid][:k]]
            ious.append(iou(gold[qid], pred))
            ks.append(k)
        mean_iou = float(np.mean(ious))
        mean_k = float(np.mean(ks))
        marker = ""
        if mean_iou > best_a[1]:
            best_a = (T, mean_iou, mean_k)
            marker = " ← best"
        print(f"  {T:>6.2f} {mean_iou:>8.4f} {mean_k:>8.2f}{marker}")

    print()

    # ── Strategy B: hard score threshold (keep all candidates score > T) ──
    def k_from_threshold(scores, T):
        k = 0
        for s in scores[:K_CAP]:
            if s > T:
                k += 1
            else:
                break
        return max(k, 1)   # always return at least top-1

    print("Strategy B — keep all candidates with absolute score > T:")
    print(f"  {'T':>6} {'IoU':>8} {'mean K':>8}")
    best_b = (None, 0.0, 0)
    for T in [-5.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]:
        ious, ks = [], []
        for qid in qids:
            scores = [c[1] for c in rerank_cache[qid]]
            k = k_from_threshold(scores, T)
            pred = [c[0] for c in rerank_cache[qid][:k]]
            ious.append(iou(gold[qid], pred))
            ks.append(k)
        mean_iou = float(np.mean(ious))
        mean_k = float(np.mean(ks))
        marker = ""
        if mean_iou > best_b[1]:
            best_b = (T, mean_iou, mean_k)
            marker = " ← best"
        print(f"  {T:>6.2f} {mean_iou:>8.4f} {mean_k:>8.2f}{marker}")

    print()

    # ── Strategy C: ratio (relative to top1 magnitude) ────────────────────
    # K is the largest j such that score[j] / score[0] >= R (only when top1 > 0).
    def k_from_ratio(scores, R):
        s0 = scores[0]
        if s0 <= 0:
            return 1
        k = 1
        for s in scores[1:K_CAP]:
            if s / s0 >= R:
                k += 1
            else:
                break
        return k

    print("Strategy C — keep candidates whose score / top1 >= R (only when top1 > 0):")
    print(f"  {'R':>6} {'IoU':>8} {'mean K':>8}")
    best_c = (None, 0.0, 0)
    for R in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95, 0.97, 0.99]:
        ious, ks = [], []
        for qid in qids:
            scores = [c[1] for c in rerank_cache[qid]]
            k = k_from_ratio(scores, R)
            pred = [c[0] for c in rerank_cache[qid][:k]]
            ious.append(iou(gold[qid], pred))
            ks.append(k)
        mean_iou = float(np.mean(ious))
        mean_k = float(np.mean(ks))
        marker = ""
        if mean_iou > best_c[1]:
            best_c = (R, mean_iou, mean_k)
            marker = " ← best"
        print(f"  {R:>6.2f} {mean_iou:>8.4f} {mean_k:>8.2f}{marker}")

    print()
    print("=== summary ===")
    print(f"current (K=1 always)    : IoU = {iou_k1:.4f}")
    print(f"best Strategy A T={best_a[0]:.2f}  : IoU = {best_a[1]:.4f}  meanK={best_a[2]:.2f}  Δ={best_a[1]-iou_k1:+.4f}")
    print(f"best Strategy B T={best_b[0]:.2f}  : IoU = {best_b[1]:.4f}  meanK={best_b[2]:.2f}  Δ={best_b[1]-iou_k1:+.4f}")
    print(f"best Strategy C R={best_c[0]:.2f}  : IoU = {best_c[1]:.4f}  meanK={best_c[2]:.2f}  Δ={best_c[1]-iou_k1:+.4f}")
    print(f"oracle ceiling          : IoU = {iou_oracle:.4f}  meanK={np.mean([min(len(gold[q]),5) for q in qids]):.2f}")
    print()
    composite_weight_iou = 0.20
    best_delta = max(best_a[1], best_b[1], best_c[1]) - iou_k1
    print(f"best Δcomposite (IoU × 0.20) ≈ {best_delta * composite_weight_iou:+.4f}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / "adaptive_k.json"
    out.write_text(json.dumps({
        "baseline_k1": iou_k1,
        "baseline_k2": iou_k2,
        "oracle": iou_oracle,
        "best_strategy_A": {"T": best_a[0], "iou": best_a[1], "mean_k": best_a[2]},
        "best_strategy_B": {"T": best_b[0], "iou": best_b[1], "mean_k": best_b[2]},
        "best_strategy_C": {"R": best_c[0], "iou": best_c[1], "mean_k": best_c[2]},
    }, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
