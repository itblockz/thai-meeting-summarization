"""
E8 post-hoc ablation — truncate exp03 predictions at various length caps
and recompute RougeL (CPU only).

The goal is to check if a static length cap would improve RougeL without
the cost of re-running the LLM. Caps tested:
  - hard pythainlp-token cap (10, 15, 20, 25, 30, 40, 50, 60, 80, 100)
  - sentence-level cap (1, 2, 3 sentences via "." / "ฯ" / newline splits)

We do not recompute SS-score here (needs GPU + bge-m3). If RougeL shows
a clear improvement at some cap, run the full scorer on the truncated
submission to get a true composite delta.
"""
import csv
import json
from pathlib import Path

import numpy as np
from pythainlp.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
SUBMISSION = PROJECT / "exp03/eval_result/submission.csv"
TEST_JSON  = PROJECT / "textsum/eval_train/test.json"
OUT_DIR    = HERE / "result"


class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split(" ")


def tok_thai_space(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return " ".join(word_tokenize(text, engine="newmm", keep_whitespace=False))


def truncate_tokens(text, n):
    toks = word_tokenize(text or "", engine="newmm", keep_whitespace=True)
    if len(toks) <= n:
        return text
    return "".join(toks[:n]).rstrip()


def truncate_sentences(text, n):
    if not text:
        return text
    try:
        sents = sent_tokenize(text, engine="crfcut")
    except Exception:
        sents = text.split("\n")
    if len(sents) <= n:
        return text
    return "".join(sents[:n]).rstrip()


def score_rougeL(pred_map, gold_map, qids):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=ThaiSpaceTokenizer())
    scores = []
    for qid in qids:
        g = tok_thai_space(gold_map[qid])
        p = tok_thai_space(pred_map[qid])
        scores.append(scorer.score(g, p)["rougeL"].fmeasure)
    return float(np.mean(scores)), scores


def main():
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    gold = {q["ID"]: (q.get("abstractive") or "") for q in data["queries"]}

    pred = {}
    with open(SUBMISSION, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pred[row["ID"]] = row.get("abstractive") or ""

    qids = sorted(set(gold) & set(pred))
    print(f"queries: {len(qids)}\n", flush=True)

    base_mean, _ = score_rougeL(pred, gold, qids)
    print(f"baseline RougeL = {base_mean:.4f}\n", flush=True)

    # ── token caps ─────────────────────────────────────────────────────────
    print("=== hard pythainlp-token cap ===")
    print(f"  {'cap':>5}  {'RougeL':>8}  Δ vs baseline")
    rows = [("baseline", base_mean, 0.0)]
    for cap in [10, 15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 150]:
        trunc = {qid: truncate_tokens(pred[qid], cap) for qid in qids}
        mean, _ = score_rougeL(trunc, gold, qids)
        delta = mean - base_mean
        marker = " ← improves" if delta > 0 else ""
        print(f"  {cap:>5}  {mean:>8.4f}  {delta:>+8.4f}{marker}", flush=True)
        rows.append((f"tok_cap_{cap}", mean, delta))

    # ── sentence caps ──────────────────────────────────────────────────────
    print(f"\n=== sentence cap (pythainlp crfcut) ===")
    print(f"  {'sents':>5}  {'RougeL':>8}  Δ vs baseline")
    for n in [1, 2, 3, 4]:
        trunc = {qid: truncate_sentences(pred[qid], n) for qid in qids}
        mean, _ = score_rougeL(trunc, gold, qids)
        delta = mean - base_mean
        marker = " ← improves" if delta > 0 else ""
        print(f"  {n:>5}  {mean:>8.4f}  {delta:>+8.4f}{marker}", flush=True)
        rows.append((f"sent_cap_{n}", mean, delta))

    # ── adaptive: cap at gold-length percentile (cheating; oracle) ─────────
    print(f"\n=== oracle (cap at gold token length) ===")
    trunc = {}
    for qid in qids:
        gold_toks = len(word_tokenize(gold[qid] or "", engine="newmm",
                                       keep_whitespace=False))
        trunc[qid] = truncate_tokens(pred[qid], max(gold_toks, 3))
    mean, _ = score_rougeL(trunc, gold, qids)
    print(f"  oracle cap = |gold tokens|  RougeL = {mean:.4f}  Δ = {mean-base_mean:+.4f}",
          flush=True)
    rows.append(("oracle_gold_len", mean, mean - base_mean))

    # ── persist ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "length_cap_ablation.json"
    out.write_text(json.dumps([{"strategy": r[0], "rougeL": r[1], "delta": r[2]}
                               for r in rows], indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
