"""
Post-hoc test: how much RougeL does exp06 recover if we strip the
"คำตอบ:" prefix and/or convert Thai numerals to Arabic?

Both regressions are formatting artifacts taught by the 2 worked examples
(format marker bleed + Thai-numeral style). RougeL only — CPU, no model
load. Reports per-strategy mean RougeL on the same 1218-query held-out
subset as exp06.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
EXP03_SUB    = PROJECT / "exp03/eval_result/submission.csv"
EXP06_SUB    = PROJECT / "exp06/eval_result/submission.csv"
TEST_JSON    = PROJECT / "textsum/eval_train/test.json"

THAI_TO_ARABIC = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
PREFIX_RE = re.compile(r"^\s*(คำตอบ\s*[:：])\s*")


class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split(" ")


def tok(text):
    return " ".join(word_tokenize(text or "", engine="newmm", keep_whitespace=False))


def strip_prefix(s):
    return PREFIX_RE.sub("", s, count=1)


def to_arabic_digits(s):
    return s.translate(THAI_TO_ARABIC)


def load_csv(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ID"]] = row.get("abstractive") or ""
    return out


def score_set(qids, golds, preds, transform):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=ThaiSpaceTokenizer())
    rls = []
    for q in qids:
        g = tok(golds[q])
        p = tok(transform(preds[q]))
        rls.append(scorer.score(g, p)["rougeL"].fmeasure)
    return np.array(rls)


def main():
    data = json.load(open(TEST_JSON, encoding="utf-8"))
    golds = {q["ID"]: (q.get("abstractive") or "") for q in data["queries"]}

    p03 = load_csv(EXP03_SUB)
    p06 = load_csv(EXP06_SUB)

    qids = sorted(set(p03) & set(p06) & set(golds))
    print(f"queries: {len(qids)}\n")

    # baselines (apples-to-apples on held-out subset)
    rl_03 = score_set(qids, golds, p03, lambda s: s)
    rl_06 = score_set(qids, golds, p06, lambda s: s)

    print(f"exp03 baseline                       RougeL = {rl_03.mean():.4f}")
    print(f"exp06 (no fix)                       RougeL = {rl_06.mean():.4f}   "
          f"Δ vs exp03 = {rl_06.mean() - rl_03.mean():+.4f}")

    # fixes
    rl_06_strip   = score_set(qids, golds, p06, strip_prefix)
    rl_06_arabic  = score_set(qids, golds, p06, to_arabic_digits)
    rl_06_both    = score_set(qids, golds, p06,
                              lambda s: to_arabic_digits(strip_prefix(s)))

    print(f"exp06 + strip prefix                 RougeL = {rl_06_strip.mean():.4f}   "
          f"Δ vs exp06 raw = {rl_06_strip.mean() - rl_06.mean():+.4f}")
    print(f"exp06 + ๐-๙ → 0-9                    RougeL = {rl_06_arabic.mean():.4f}   "
          f"Δ vs exp06 raw = {rl_06_arabic.mean() - rl_06.mean():+.4f}")
    print(f"exp06 + strip + numerals (combined)  RougeL = {rl_06_both.mean():.4f}   "
          f"Δ vs exp06 raw = {rl_06_both.mean() - rl_06.mean():+.4f}")
    print(f"                                                   Δ vs exp03 = {rl_06_both.mean() - rl_03.mean():+.4f}")

    # how many queries gain from prefix strip alone
    delta_strip = rl_06_strip - rl_06
    gained = int((delta_strip > 1e-6).sum())
    unchanged = int(np.isclose(delta_strip, 0).sum())
    print(f"\nstrip-prefix per-query: {gained} gained, {unchanged} unchanged")
    print(f"  mean gain among those that gained: {delta_strip[delta_strip > 1e-6].mean():.4f}")
    print(f"  max gain: {delta_strip.max():.4f}")

    # composite recompute (assuming SS unchanged and IoU unchanged)
    # we can recompute composite for the both-fix case using exp06's stored SS and IoU
    detail_06 = {}
    with open(PROJECT / "exp06/eval_result/train_eval_detail.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            detail_06[row["ID"]] = (float(row["SS-score"]), float(row["IoU"]))

    ss = np.array([detail_06[q][0] for q in qids])
    iou = np.array([detail_06[q][1] for q in qids])
    composite_raw    = 0.45 * ss.mean() + 0.35 * rl_06.mean()      + 0.20 * iou.mean()
    composite_both   = 0.45 * ss.mean() + 0.35 * rl_06_both.mean() + 0.20 * iou.mean()

    # exp03 composite on same subset
    detail_03 = {}
    with open(PROJECT / "exp03/eval_result/train_eval_detail.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            detail_03[row["ID"]] = (float(row["SS-score"]), float(row["IoU"]))
    ss03 = np.array([detail_03[q][0] for q in qids])
    iou03 = np.array([detail_03[q][1] for q in qids])
    composite_03 = 0.45 * ss03.mean() + 0.35 * rl_03.mean() + 0.20 * iou03.mean()

    print(f"\n=== composite (RougeL only changes; SS held at exp06 value, IoU identical) ===")
    print(f"  exp03 baseline                : {composite_03:.4f}")
    print(f"  exp06 raw                     : {composite_raw:.4f}   "
          f"Δ vs exp03 = {composite_raw - composite_03:+.4f}")
    print(f"  exp06 + strip + numerals      : {composite_both:.4f}   "
          f"Δ vs exp03 = {composite_both - composite_03:+.4f}")
    print(f"\n  NOTE: SS-score would also change if we re-embed the fixed strings; "
          f"this is a lower bound on the recovery.")


if __name__ == "__main__":
    main()
