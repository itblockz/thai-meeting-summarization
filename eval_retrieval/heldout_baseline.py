"""
Apples-to-apples baseline for held-out few-shot experiments.

A few-shot experiment (exp06+) holds out the doc(s) its examples come
from, so it is scored on a subset of the 1239 train queries. Comparing
its composite to exp03's full-set 0.6256 is invalid — the subset differs.

exp03/eval_result/train_eval_detail.csv stores per-query rougeL,
SS-score and IoU for all 1239 queries, so the matching exp03 baseline
on any subset is pure arithmetic (filter + re-average) — no GPU needed.

Usage:
  python3 heldout_baseline.py <exp_dir> <held-out-doc> [<held-out-doc> ...]
  e.g.  python3 heldout_baseline.py exp16 doc_045 doc_031
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TEST_JSON = PROJECT / "textsum/eval_train/test.json"
EXP03_DETAIL = PROJECT / "exp03/eval_result/train_eval_detail.csv"

W_SS, W_ROUGE, W_IOU = 0.45, 0.35, 0.20


def composite(ss, rouge, iou):
    return W_SS * ss + W_ROUGE * rouge + W_IOU * iou


def load_detail(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["ID"]] = (
                float(r["rougeL"]), float(r["SS-score"]), float(r["IoU"]))
    return rows


def avg(rows, ids):
    ids = [i for i in ids if i in rows]
    n = len(ids)
    ss = sum(rows[i][1] for i in ids) / n
    rouge = sum(rows[i][0] for i in ids) / n
    iou = sum(rows[i][2] for i in ids) / n
    return n, ss, rouge, iou


def line(tag, n, ss, rouge, iou):
    return (f"{tag:<22} ({n:>4} q)  RougeL={rouge:.4f}  SS={ss:.4f}  "
            f"IoU={iou:.4f}  ->  {composite(ss, rouge, iou):.4f}")


def main():
    exp_dir = sys.argv[1]
    heldout = set(sys.argv[2:])

    data = json.load(open(TEST_JSON, encoding="utf-8"))
    id2doc = {q["ID"]: q["doc_id"] for q in data["queries"]}

    exp03 = load_detail(EXP03_DETAIL)
    all_ids = list(exp03)
    subset_ids = [i for i in all_ids if id2doc.get(i) not in heldout]

    exp_detail = load_detail(PROJECT / exp_dir / "eval_result/train_eval_detail.csv")

    print(f"held-out docs: {sorted(heldout)}\n")
    print(line("exp03 FULL", *avg(exp03, all_ids)))
    n_b, ss_b, r_b, i_b = avg(exp03, subset_ids)
    print(line("exp03 baseline", n_b, ss_b, r_b, i_b))
    n_e, ss_e, r_e, i_e = avg(exp_detail, list(exp_detail))
    print(line(exp_dir, n_e, ss_e, r_e, i_e))

    base = composite(ss_b, r_b, i_b)
    exp = composite(ss_e, r_e, i_e)
    print(f"\nDelta {exp_dir} vs exp03 (same {n_e}-q subset): {exp - base:+.4f}")
    print(f"  RougeL {r_e - r_b:+.4f}   SS {ss_e - ss_b:+.4f}   IoU {i_e - i_b:+.4f}")
    if n_e != n_b:
        print(f"  ! subset size mismatch: exp03 {n_b} vs {exp_dir} {n_e}")


if __name__ == "__main__":
    main()
