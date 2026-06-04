"""
exp51 length-cap sweep — post-hoc, NO LLM re-run.

Takes exp51's already-generated submission.csv and truncates each
`abstractive` answer to a maximum length — both by Thai-word count
(pythainlp newmm) and by character count — for a range of caps, then
recomputes the composite score for every cap. Answers already under a
cap pass through untouched; only the over-length ones are trimmed.

Why this is cheap: capping only touches the answer text, so it moves
RougeL and SS-score; it does NOT touch `refs`, so IoU is cap-invariant
(computed once). All capped variants are embedded in a single bge-m3
pass (gt once + N_caps × preds), so the GPU cost is one batched encode,
not one job per cap.

Leak-free by default: holds out doc_050 (the few-shot source doc), same
as score_heldout.py, so the numbers line up with the exp51 leak-free
row.

Usage:
  python cap_sweep.py [submission.csv] [heldout_doc]
    submission.csv  default $PROJECT/exp51/eval_result/submission.csv
    heldout_doc     default doc_050  (pass "" to score the full 1239)

Caveat (the prior verdict): CLAUDE.md "E8 length calibration / truncation"
already found verbose preds carry recall and truncation loses RougeL at
every cap. This script re-confirms that on exp51's own outputs and prints
the full curve so the trade-off is visible rather than asserted.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer

# Reuse score.py's exact metric definitions (same dir on LANTA via the
# textsum/eval_train path appended below).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "textsum" / "eval_train"))
import score  # noqa: E402

# Caps to sweep as (unit, n) — unit "w" = newmm words, "c" = characters.
# The single ("none", None) entry is the uncapped exp51 baseline.
CAPS = (
    [("w", n) for n in (8, 12, 16, 20, 25, 30, 40, 50, 75, 100)]
    + [("c", n) for n in (40, 60, 80, 120, 160, 200, 300, 400)]
    + [("none", None)]
)

WSS, WRL, WJ = 0.45, 0.35, 0.20


def cap_label(unit, n):
    return "∞" if unit == "none" else f"{unit}{n}"


def cap_answer(text, unit, n):
    """Truncate `text` to a maximum length. unit "w" caps newmm content
    words (whitespace kept, faithfully reconstructed); "c" caps raw
    characters. Shorter answers are returned unchanged."""
    if unit == "none" or not isinstance(text, str) or not text.strip():
        return text
    if unit == "c":
        return text[:n].strip()
    toks = word_tokenize(text, engine="newmm", keep_whitespace=True)
    # Count only non-whitespace tokens toward the cap, but keep the
    # whitespace that falls between kept content tokens.
    kept, content = [], 0
    for t in toks:
        if t.strip():
            if content >= n:
                break
            content += 1
        kept.append(t)
    return "".join(kept).strip()


def main():
    submission = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).parent / "eval_result" / "submission.csv"
    heldout = sys.argv[2] if len(sys.argv) > 2 else "doc_050"

    if not submission.exists():
        raise SystemExit(f"submission not found: {submission}\n"
                         f"Run exp51 first (sbatch exp51/submit_eval_train.sh).")

    gt   = score.load_ground_truth(score.TRAIN_JSON)
    pred = score.load_submission(submission)

    if heldout:
        with open(score.TRAIN_JSON, encoding="utf-8") as f:
            data = json.load(f)
        heldout_ids = {q["ID"] for q in data["queries"] if q["doc_id"] == heldout}
        gt = gt[~gt["ID"].isin(heldout_ids)].reset_index(drop=True)
        print(f"Held-out  : {heldout} ({len(heldout_ids)} queries excluded)")

    df = pd.merge(gt, pred, on="ID", suffixes=("_sol", "_pred"))
    n = len(df)
    print(f"Submission: {submission}")
    print(f"Scoring on: {n} queries\n")

    # ---- IoU: cap-invariant, compute once ----
    iou = df.apply(lambda r: score.calculate_iou(r["refs_pred"], r["refs_sol"]),
                   axis=1).mean()

    # ---- baseline answer-length distribution (newmm content words) ----
    base_len = df["abstractive_pred"].apply(
        lambda t: sum(1 for w in word_tokenize(str(t), engine="newmm",
                                               keep_whitespace=False) if w.strip()))
    print("Baseline answer length (newmm content words): "
          f"mean={base_len.mean():.1f} median={base_len.median():.0f} "
          f"p90={base_len.quantile(0.9):.0f} max={base_len.max()}\n")

    # ---- build every capped pred variant ----
    sol_texts = df["abstractive_sol"].tolist()
    variants = {}  # (unit, n) -> list[str]
    for unit, ncap in CAPS:
        variants[(unit, ncap)] = [cap_answer(t, unit, ncap)
                                  for t in df["abstractive_pred"]]

    # ---- RougeL per cap (reuse score.py's Thai-space tokenizer) ----
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=score.ThaiSpaceTokenizer())
    sol_toks = [score.tokenize_thai(t) for t in sol_texts]
    rouge_by_cap = {}
    for cap, preds in variants.items():
        pred_toks = [score.tokenize_thai(t) for t in preds]
        rouge_by_cap[cap] = float(np.mean(
            [scorer.score(g, p)["rougeL"].fmeasure
             for g, p in zip(sol_toks, pred_toks)]))

    # ---- SS-score: one batched bge-m3 pass (gt once + all caps) ----
    model = SentenceTransformer("BAAI/bge-m3")
    all_texts = list(sol_texts)
    spans = {}
    for cap, preds in variants.items():
        spans[cap] = (len(all_texts), len(all_texts) + n)
        all_texts.extend(preds)
    embs = model.encode(all_texts, batch_size=32, convert_to_tensor=True,
                        normalize_embeddings=True)
    sol_emb = embs[:n]
    ss_by_cap = {}
    for cap, (lo, hi) in spans.items():
        ss_by_cap[cap] = float(
            F.cosine_similarity(embs[lo:hi], sol_emb, dim=1).mean().cpu())

    # ---- report (word caps, then char caps, baseline last) ----
    base_comp = WSS * ss_by_cap[("none", None)] \
        + WRL * rouge_by_cap[("none", None)] + WJ * iou
    rows = []
    for unit, ncap in CAPS:
        cap = (unit, ncap)
        rl, ss = rouge_by_cap[cap], ss_by_cap[cap]
        comp = WSS * ss + WRL * rl + WJ * iou
        # avg length reported in the cap's own unit
        if unit == "c":
            avg_len = float(np.mean([len(t) for t in variants[cap]]))
        else:
            avg_len = float(np.mean([sum(1 for w in word_tokenize(
                t, engine="newmm", keep_whitespace=False) if w.strip())
                for t in variants[cap]]))
        rows.append({"unit": unit, "n": ncap, "label": cap_label(unit, ncap),
                     "rougeL": rl, "SS": ss, "IoU": iou,
                     "composite": comp, "avg_len": avg_len})

    print(f"{'cap':>6}  {'RougeL':>7}  {'SS':>7}  {'IoU':>7}  {'composite':>9}  "
          f"{'Δvs∞':>8}  {'avg_len':>7}")
    for r in rows:
        delta = f"{r['composite']-base_comp:+.4f}"
        unit_tag = "w" if r["unit"] != "c" else "c"
        print(f"{r['label']:>6}  {r['rougeL']:>7.4f}  {r['SS']:>7.4f}  "
              f"{r['IoU']:>7.4f}  {r['composite']:>9.4f}  {delta:>8}  "
              f"{r['avg_len']:>6.1f}{unit_tag}")

    best = max(rows, key=lambda r: r["composite"])
    print(f"\nbaseline (∞) composite = {base_comp:.4f}")
    print(f"best cap = {best['label']}  composite = {best['composite']:.4f}  "
          f"({best['composite']-base_comp:+.4f} vs ∞)")

    out = submission.parent / "cap_sweep.json"
    out.write_text(json.dumps(
        [{k: (v if v is not None else "inf") for k, v in r.items()} for r in rows],
        indent=2, ensure_ascii=False))
    print(f"curve → {out}")


if __name__ == "__main__":
    main()
