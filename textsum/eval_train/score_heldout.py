"""
Leak-free scoring — score a submission excluding every query from a
held-out document (the document the few-shot examples were drawn from).

The E5 + few-shot experiments draw their worked examples from train
doc_050. Scoring the full 1239 train queries lets the model's exposure
to doc_050 gold answers (inside the few-shot prompt) leak into the
doc_050 eval queries. This re-scores on the 1239 - |held-out doc|
remaining queries, reusing score.py's evaluation code unchanged.

  python score_heldout.py <submission.csv> [heldout_doc]

heldout_doc defaults to doc_050.
"""
import sys
import json
from pathlib import Path

import score   # textsum/eval_train/score.py — same directory

HELDOUT_DEFAULT = "doc_050"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: score_heldout.py <submission.csv> [heldout_doc]")
    submission = Path(sys.argv[1])
    heldout = sys.argv[2] if len(sys.argv) > 2 else HELDOUT_DEFAULT

    with open(score.TRAIN_JSON, encoding="utf-8") as f:
        data = json.load(f)
    heldout_ids = {q["ID"] for q in data["queries"] if q["doc_id"] == heldout}

    gt   = score.load_ground_truth(score.TRAIN_JSON)
    pred = score.load_submission(submission)

    # Drop held-out queries from the ground truth; run_evaluation's inner
    # merge on ID then trims pred to the same kept set automatically.
    gt_kept = gt[~gt["ID"].isin(heldout_ids)].reset_index(drop=True)
    n_scored = len(gt_kept)

    print(f"Submission : {submission}")
    print(f"Held-out   : {heldout} ({len(heldout_ids)} queries excluded)")
    print(f"Scoring on : {n_scored} queries")

    metrics, _ = score.run_evaluation(gt_kept, pred)

    print(f"\n=== Leak-free evaluation (excl. {heldout}) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    metrics["n_scored"]    = n_scored
    metrics["heldout_doc"] = heldout
    out = submission.parent / "train_eval_score_heldout.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
