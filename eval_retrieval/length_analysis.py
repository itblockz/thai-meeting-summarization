"""
E8 — length calibration analysis.

Compare lengths (chars and Thai-word tokens via pythainlp newmm) of
exp03 predictions vs gold abstractive answers on the train set. Goal:
decide whether outputs are systematically too long/short and whether
RougeL correlates with the length gap. If pred is systematically more
verbose, capping max_tokens or adding a length hint to the prompt is a
cheap candidate fix.

CPU only, no model load.
"""
import csv
import json
from pathlib import Path

import numpy as np
from pythainlp.tokenize import word_tokenize

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SUBMISSION = PROJECT / "exp03/eval_result/submission.csv"
DETAIL     = PROJECT / "exp03/eval_result/train_eval_detail.csv"
TEST_JSON  = PROJECT / "textsum/eval_train/test.json"


def tok(text):
    return word_tokenize(text or "", engine="newmm", keep_whitespace=False)


def percentiles(arr, ps=(10, 25, 50, 75, 90, 95, 99)):
    return {p: float(np.percentile(arr, p)) for p in ps}


def fmt_dist(label, arr):
    arr = np.asarray(arr)
    print(f"{label:<22} n={len(arr):>4}  mean={arr.mean():>6.1f}  "
          f"median={np.median(arr):>6.1f}  p90={np.percentile(arr, 90):>6.1f}  "
          f"max={arr.max():>6.0f}")


def main():
    # ── load gold ─────────────────────────────────────────────────────────
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    gold_abs = {q["ID"]: (q.get("abstractive") or "") for q in data["queries"]}

    # ── load predictions ──────────────────────────────────────────────────
    pred_abs = {}
    with open(SUBMISSION, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred_abs[row["ID"]] = row.get("abstractive") or ""

    # ── load per-query rougeL ─────────────────────────────────────────────
    rouge = {}
    with open(DETAIL, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rouge[row["ID"]] = float(row["rougeL"])
            except (KeyError, ValueError):
                pass

    qids = sorted(set(gold_abs) & set(pred_abs) & set(rouge))
    print(f"queries matched: {len(qids)}\n")

    # ── length distributions ──────────────────────────────────────────────
    pred_chars = np.array([len(pred_abs[q]) for q in qids])
    gold_chars = np.array([len(gold_abs[q]) for q in qids])

    pred_toks  = np.array([len(tok(pred_abs[q])) for q in qids])
    gold_toks  = np.array([len(tok(gold_abs[q])) for q in qids])

    print("=== character length ===")
    fmt_dist("gold chars", gold_chars)
    fmt_dist("pred chars", pred_chars)
    print(f"  pred / gold ratio (median): {np.median(pred_chars)/np.median(gold_chars):.2f}x")
    print(f"  pred / gold ratio (mean)  : {pred_chars.mean()/gold_chars.mean():.2f}x")

    print("\n=== Thai-word token length (newmm) ===")
    fmt_dist("gold tokens", gold_toks)
    fmt_dist("pred tokens", pred_toks)
    print(f"  pred / gold ratio (median): {np.median(pred_toks)/np.median(gold_toks):.2f}x")
    print(f"  pred / gold ratio (mean)  : {pred_toks.mean()/gold_toks.mean():.2f}x")

    # ── overshoot / undershoot ────────────────────────────────────────────
    over  = (pred_toks > gold_toks * 1.5).sum()
    much  = (pred_toks > gold_toks * 2.0).sum()
    under = (pred_toks < gold_toks * 0.5).sum()
    print(f"\n  pred > 1.5x gold tokens: {over:>4} ({over/len(qids):.1%})")
    print(f"  pred > 2.0x gold tokens: {much:>4} ({much/len(qids):.1%})")
    print(f"  pred < 0.5x gold tokens: {under:>4} ({under/len(qids):.1%})")

    # ── RougeL vs length gap (tokens) ─────────────────────────────────────
    rl = np.array([rouge[q] for q in qids])
    diff = pred_toks - gold_toks
    abs_diff = np.abs(diff)
    ratio = pred_toks / np.maximum(gold_toks, 1)

    def corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    print(f"\n=== correlations with per-query RougeL ===")
    print(f"  RougeL ↔ pred_tokens         : {corr(rl, pred_toks):+.3f}")
    print(f"  RougeL ↔ gold_tokens         : {corr(rl, gold_toks):+.3f}")
    print(f"  RougeL ↔ (pred - gold)       : {corr(rl, diff):+.3f}")
    print(f"  RougeL ↔ |pred - gold|       : {corr(rl, abs_diff):+.3f}")
    print(f"  RougeL ↔ (pred / gold)       : {corr(rl, ratio):+.3f}")

    # ── RougeL by pred-length bucket (relative to gold) ───────────────────
    print(f"\n=== mean RougeL by pred/gold token ratio bucket ===")
    buckets = [
        ("<0.5x       ", ratio < 0.5),
        ("0.5–1.0x    ", (ratio >= 0.5) & (ratio < 1.0)),
        ("1.0–1.5x    ", (ratio >= 1.0) & (ratio < 1.5)),
        ("1.5–2.0x    ", (ratio >= 1.5) & (ratio < 2.0)),
        ("2.0–3.0x    ", (ratio >= 2.0) & (ratio < 3.0)),
        (">=3.0x      ", ratio >= 3.0),
    ]
    for label, mask in buckets:
        n = int(mask.sum())
        if n == 0:
            print(f"  {label}  n={n:>4}   (empty)")
            continue
        print(f"  {label}  n={n:>4}   mean RougeL = {rl[mask].mean():.4f}")

    # ── RougeL by absolute pred length bucket ─────────────────────────────
    print(f"\n=== mean RougeL by absolute pred-token length bucket ===")
    bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, 400), (400, 1000)]
    for lo, hi in bins:
        mask = (pred_toks >= lo) & (pred_toks < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"  [{lo:>4}, {hi:>4})  n={n:>4}   (empty)")
            continue
        print(f"  [{lo:>4}, {hi:>4})  n={n:>4}   mean RougeL = {rl[mask].mean():.4f}  "
              f"(gold p50={np.median(gold_toks[mask]):.0f})")

    # ── max_tokens recommendation (look at gold distribution) ─────────────
    print(f"\n=== gold-token percentiles (for max_tokens calibration) ===")
    for p in [50, 75, 90, 95, 99]:
        v = np.percentile(gold_toks, p)
        print(f"  p{p:>2} = {v:>5.0f} tokens")
    # rough conversion: pythainlp tokens ≈ vLLM tokens × 0.5–0.7 for Thai
    # current MAX_NEW_TOKENS = 512 (model tokens). Most likely too lenient.
    cap_gold_p95 = int(np.percentile(gold_toks, 95))
    print(f"\nNOTE: current MAX_NEW_TOKENS = 512 (model BPE tokens). "
          f"gold p95 ≈ {cap_gold_p95} pythainlp-tokens.")


if __name__ == "__main__":
    main()
