"""
Per-query loss analysis: exp06 (2-shot) vs exp03 (no shot).

Both pipelines share retrieval (TOP_K=1 + same rerank), so IoU is identical
and the only signal in Δ rougeL / Δ SS-score comes from the LLM seeing the
2 worked examples. exp06 excluded doc_050; we compare on its 1218-query
held-out subset.

Goals:
  1. Distribution of Δ rougeL: how many wins vs losses, magnitude.
  2. Identify queries where few-shot HURT (top 20 by negative Δ).
  3. Test whether losses cluster on a detectable feature:
       - query length / gold length
       - query type (factoid vs explanation, presence of WH-words / digits)
       - retrieval correctness (IoU=1 vs IoU=0)
  4. Surface example losing-query texts so we can eyeball patterns.

CPU only.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
from pythainlp.tokenize import word_tokenize

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EXP03_DETAIL = PROJECT / "exp03/eval_result/train_eval_detail.csv"
EXP06_DETAIL = PROJECT / "exp06/eval_result/train_eval_detail.csv"
EXP03_SUB    = PROJECT / "exp03/eval_result/submission.csv"
EXP06_SUB    = PROJECT / "exp06/eval_result/submission.csv"
TEST_JSON    = PROJECT / "textsum/eval_train/test.json"


def tok(text):
    return word_tokenize(text or "", engine="newmm", keep_whitespace=False)


def load_detail(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ID"]] = {
                "rougeL": float(row["rougeL"]),
                "ss":     float(row["SS-score"]),
                "iou":    float(row["IoU"]),
            }
    return out


def load_pred(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ID"]] = {
                "abs":  row.get("abstractive") or "",
                "refs": row.get("refs") or "",
            }
    return out


WH_WORDS = ("ใคร", "อะไร", "ที่ใด", "ที่ไหน", "เมื่อใด", "เมื่อไหร่",
            "อย่างไร", "ทำไม", "เพราะ", "เท่าใด", "กี่")
DIGIT_RE = re.compile(r"\d")


def query_type(q):
    """Light heuristic — sufficient to bucket factoid-ish vs explanation-ish."""
    has_digit = bool(DIGIT_RE.search(q))
    wh = [w for w in WH_WORDS if w in q]
    # crude: "อย่างไร" / "ทำไม" / "เพราะ" → explanation; "ใคร" / "อะไร" / digit → factoid
    if any(w in q for w in ("อย่างไร", "ทำไม", "เพราะ", "เป็นอย่างไร")):
        return "explanation"
    if any(w in q for w in ("ใคร", "ที่ใด", "ที่ไหน", "เมื่อใด", "เมื่อไหร่", "เท่าใด", "กี่")):
        return "factoid"
    if has_digit:
        return "factoid"
    if any(w in q for w in ("อะไร",)):
        return "factoid"
    return "other"


def bucket_summary(label, mask, drl):
    n = int(mask.sum())
    if n == 0:
        print(f"  {label:<22}  n={n:>4}   (empty)")
        return
    sub = drl[mask]
    wins = int((sub > 0).sum())
    losses = int((sub < 0).sum())
    print(f"  {label:<22}  n={n:>4}   mean Δ={sub.mean():+.4f}   "
          f"median={np.median(sub):+.4f}   wins={wins} losses={losses}")


def main():
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    q_meta = {q["ID"]: q for q in data["queries"]}

    d03 = load_detail(EXP03_DETAIL)
    d06 = load_detail(EXP06_DETAIL)
    p03 = load_pred(EXP03_SUB)
    p06 = load_pred(EXP06_SUB)

    # apples-to-apples held-out subset: queries that exist in both
    qids = sorted(set(d03) & set(d06))
    print(f"queries in both: {len(qids)}\n")

    drl = np.array([d06[q]["rougeL"] - d03[q]["rougeL"] for q in qids])
    dss = np.array([d06[q]["ss"]     - d03[q]["ss"]     for q in qids])
    diou = np.array([d06[q]["iou"]   - d03[q]["iou"]    for q in qids])

    # ── headline ──────────────────────────────────────────────────────────
    wins = int((drl > 0).sum())
    losses = int((drl < 0).sum())
    zero = int((drl == 0).sum())
    print(f"=== Δ RougeL (exp06 − exp03) ===")
    print(f"  mean    = {drl.mean():+.5f}")
    print(f"  median  = {np.median(drl):+.5f}")
    print(f"  wins    = {wins:>4}  ({wins/len(qids):.1%})")
    print(f"  losses  = {losses:>4}  ({losses/len(qids):.1%})")
    print(f"  no chg  = {zero:>4}  ({zero/len(qids):.1%})")
    print(f"  Δ SS    = {dss.mean():+.5f}")
    print(f"  Δ IoU   = {diou.mean():+.5f}   (expected 0; pipeline shares retrieval)")

    # ── magnitude distribution of losses ──────────────────────────────────
    print(f"\n=== Δ RougeL distribution (percentiles) ===")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2}  Δ = {np.percentile(drl, p):+.4f}")

    big_loss = drl < -0.10
    big_win  = drl >  0.10
    print(f"\n  losses  < -0.10  : {int(big_loss.sum()):>4} ({big_loss.mean():.1%})")
    print(f"  losses  < -0.20  : {int((drl < -0.20).sum()):>4}")
    print(f"  wins    > +0.10  : {int(big_win.sum()):>4} ({big_win.mean():.1%})")
    print(f"  wins    > +0.20  : {int((drl > 0.20).sum()):>4}")

    # ── bucket by query type ──────────────────────────────────────────────
    qtype = np.array([query_type(q_meta[q]["query"]) for q in qids])
    print(f"\n=== Δ RougeL by query type (heuristic) ===")
    for t in ("factoid", "explanation", "other"):
        bucket_summary(t, qtype == t, drl)

    # ── bucket by IoU (retrieval correctness) ─────────────────────────────
    iou = np.array([d03[q]["iou"] for q in qids])
    print(f"\n=== Δ RougeL by exp03 IoU (retrieval correctness) ===")
    bucket_summary("IoU = 1.0 (hit)", iou >= 0.999, drl)
    bucket_summary("IoU = 0.0 (miss)", iou < 0.001, drl)
    bucket_summary("IoU partial",    (iou > 0.001) & (iou < 0.999), drl)

    # ── bucket by gold-answer length ──────────────────────────────────────
    gold_toks = np.array([len(tok(q_meta[q].get("abstractive") or "")) for q in qids])
    print(f"\n=== Δ RougeL by gold-answer token length ===")
    bins = [(0, 30), (30, 60), (60, 100), (100, 150), (150, 250), (250, 9999)]
    for lo, hi in bins:
        bucket_summary(f"gold [{lo},{hi})", (gold_toks >= lo) & (gold_toks < hi), drl)

    # ── bucket by query length ────────────────────────────────────────────
    q_toks = np.array([len(tok(q_meta[q]["query"])) for q in qids])
    print(f"\n=== Δ RougeL by query token length ===")
    bins = [(0, 8), (8, 15), (15, 25), (25, 9999)]
    for lo, hi in bins:
        bucket_summary(f"query [{lo},{hi})", (q_toks >= lo) & (q_toks < hi), drl)

    # ── bucket by digit-in-query ──────────────────────────────────────────
    has_digit = np.array([bool(DIGIT_RE.search(q_meta[q]["query"])) for q in qids])
    print(f"\n=== Δ RougeL by digit-in-query ===")
    bucket_summary("digit in query",     has_digit, drl)
    bucket_summary("no digit",          ~has_digit, drl)

    # ── prediction length change: did few-shot make output shorter? ───────
    pred03_toks = np.array([len(tok(p03[q]["abs"])) for q in qids])
    pred06_toks = np.array([len(tok(p06[q]["abs"])) for q in qids])
    dlen = pred06_toks - pred03_toks
    print(f"\n=== Pred length change (exp06 − exp03), Thai tokens ===")
    print(f"  mean Δlen = {dlen.mean():+.2f}  median Δlen = {np.median(dlen):+.0f}")
    print(f"  pred shorter in exp06: {int((dlen < 0).sum())} ({(dlen<0).mean():.1%})")
    print(f"  pred longer  in exp06: {int((dlen > 0).sum())} ({(dlen>0).mean():.1%})")

    # correlation between length shift and rougeL shift
    if dlen.std() > 0 and drl.std() > 0:
        c = float(np.corrcoef(drl, dlen)[0, 1])
        print(f"  corr(Δlen, Δ rougeL) = {c:+.3f}")

    # ── top losses (eyeball patterns) ─────────────────────────────────────
    print(f"\n=== Top 15 losses (exp06 dropped most) ===")
    order = np.argsort(drl)[:15]
    for k in order:
        q = qids[k]
        print(f"  {q}  Δrl={drl[k]:+.3f}  iou={d03[q]['iou']:.0f}  "
              f"|gold|={gold_toks[k]}  |q|={q_toks[k]}  type={qtype[k]}")
        print(f"    Q: {q_meta[q]['query'][:120]}")
        print(f"    gold:  {(q_meta[q].get('abstractive') or '')[:140]}")
        print(f"    03:    {p03[q]['abs'][:140]}")
        print(f"    06:    {p06[q]['abs'][:140]}")
        print()

    # ── top wins ──────────────────────────────────────────────────────────
    print(f"\n=== Top 10 wins (exp06 gained most) ===")
    order = np.argsort(-drl)[:10]
    for k in order:
        q = qids[k]
        print(f"  {q}  Δrl={drl[k]:+.3f}  iou={d03[q]['iou']:.0f}  "
              f"|gold|={gold_toks[k]}  type={qtype[k]}")
        print(f"    Q: {q_meta[q]['query'][:120]}")
        print(f"    gold:  {(q_meta[q].get('abstractive') or '')[:140]}")
        print(f"    03:    {p03[q]['abs'][:140]}")
        print(f"    06:    {p06[q]['abs'][:140]}")
        print()


if __name__ == "__main__":
    main()
