"""
Score candidate few-shot examples by 4 explicit criteria — to test whether
exp08's hand-picked pair (Q0745+Q0747) was actually near-optimal among
doc_050 candidates, or whether a criteria-based search finds a better pair.

Criteria (derived from cross-experiment analysis exp08/12/13):
  1. Restate score   — how strongly the answer starts by echoing the query
  2. Length proximity — answer length close to 1.5x global gold median (~50 tok)
  3. Centrality      — bge-m3 cos-sim of this gold answer vs all other golds
  4. Query type freq — favor common patterns, penalize rare ones (esp. YN 0.4%)

For each candidate, the overall score is a weighted sum. The top-2 (with a
diversity constraint requiring different paragraphs) are reported.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from pythainlp.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
TEST_JSON = PROJECT / "textsum/eval_train/test.json"

GOLD_MEDIAN_TOK = 33   # measured from full dataset earlier
IDEAL_LEN_TOK   = int(1.5 * GOLD_MEDIAN_TOK)   # 50 tokens — matches exp08's average example length

# Query-type → frequency mapping (1239 dataset)
QTYPE_FREQS = {
    "expl":     0.445,    # อย่างไร/ทำไม/เพราะ
    "list":     0.174,    # มีอะไรบ้าง/รายละเอียด/เท่าใด/กี่/จำนวน
    "what":     0.164,    # คืออะไร/อะไร
    "other":    0.150,
    "who":      0.030,    # ใคร
    "location": 0.018,    # ที่ใด/ที่ไหน
    "when":     0.015,    # เมื่อใด/เมื่อไหร่
    "yn":       0.004,    # หรือไม่
}


def tok(t):
    return word_tokenize(t or "", engine="newmm", keep_whitespace=False)


def classify_query(q):
    if any(w in q for w in ("อย่างไร", "ทำไม", "เพราะ", "เหตุใด")):
        return "expl"
    if any(w in q for w in ("ที่ใด", "ที่ไหน", "สถานที่")):
        return "location"
    if any(w in q for w in ("ใคร",)):
        return "who"
    if any(w in q for w in ("เมื่อใด", "เมื่อไหร่")):
        return "when"
    if any(w in q for w in ("หรือไม่",)):
        return "yn"
    if any(w in q for w in ("มีอะไรบ้าง", "รายละเอียด", "เท่าใด", "กี่", "จำนวน")):
        return "list"
    if any(w in q for w in ("คืออะไร", "อะไร")):
        return "what"
    return "other"


def restate_score(query, answer, k=5):
    """How many of the first k question-tokens appear in the first k answer-tokens."""
    q_toks = set(tok(query)[:k])
    a_toks = tok(answer)[:k]
    if not q_toks:
        return 0.0
    return sum(1 for t in a_toks if t in q_toks) / k


def length_score(answer_tok, ideal=IDEAL_LEN_TOK):
    """1.0 at ideal length, falls off symmetrically."""
    return 1.0 / (1.0 + abs(answer_tok - ideal) / ideal)


def main():
    data = json.load(open(TEST_JSON, encoding="utf-8"))
    queries = data["queries"]

    # ── 1. Compute centrality (cos sim of each gold answer to all other golds) ──
    print("Embedding 1239 gold answers on CPU (~2 min) ...", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    valid_qs = [q for q in queries if (q.get("abstractive") or "").strip()]
    embs = model.encode([q["abstractive"] for q in valid_qs],
                        batch_size=32, normalize_embeddings=True,
                        convert_to_tensor=True, show_progress_bar=False)

    centrality = {}
    sim_matrix = (embs @ embs.T).numpy()  # 1239 × 1239
    for i, q in enumerate(valid_qs):
        # exclude self
        sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        centrality[q["ID"]] = float(sims.mean())

    # ── 2. Score all doc_050 single-ref candidates ──
    held_out = "doc_050"
    print(f"\nScoring all {held_out} candidates ...")
    rouge03 = {row["ID"]: float(row["rougeL"])
               for row in csv.DictReader(open(PROJECT / "exp03/eval_result/train_eval_detail.csv", encoding="utf-8"))}

    candidates = []
    for q in queries:
        if q["doc_id"] != held_out:
            continue
        refs = q.get("refs", [])
        if isinstance(refs, str):
            refs = [refs]
        if len(refs) != 1:    # only single-ref candidates (exp08's choice)
            continue
        gold = (q.get("abstractive") or "").strip()
        if not gold:
            continue
        ans_tok = len(tok(gold))
        rs = restate_score(q["query"], gold)
        ls = length_score(ans_tok)
        cs = centrality.get(q["ID"], 0.0)
        qt = classify_query(q["query"])
        qs = QTYPE_FREQS.get(qt, 0.01)

        # weights chosen so each component is in [0,1] roughly and weighted by
        # the cross-exp evidence: restate strongest, length next, then centrality + qtype
        weights = {"restate": 0.40, "length": 0.30, "centrality": 0.15, "qtype": 0.15}
        total = (weights["restate"] * rs +
                 weights["length"]   * ls +
                 weights["centrality"] * cs +
                 weights["qtype"]    * qs)

        candidates.append({
            "id": q["ID"], "doc_id": q["doc_id"], "ref_para": refs[0],
            "query": q["query"], "gold": gold,
            "qtype": qt, "ans_tok": ans_tok,
            "restate": rs, "length_s": ls, "centrality": cs, "qtype_freq": qs,
            "rouge03": rouge03.get(q["ID"], 0.0),
            "score": total,
        })

    candidates.sort(key=lambda c: -c["score"])
    print(f"\nTop 8 candidates from {held_out} (sorted by total score):")
    print(f'{"ID":<6} {"qtype":<8} {"ans_tok":>7} {"restate":>7} {"length":>7} {"central":>7} {"qfreq":>6} {"score":>6} {"para":>5} {"rl03":>5}')
    for c in candidates[:8]:
        print(f'{c["id"]:<6} {c["qtype"]:<8} {c["ans_tok"]:>7} {c["restate"]:>7.2f} {c["length_s"]:>7.2f} {c["centrality"]:>7.3f} {c["qtype_freq"]:>6.3f} {c["score"]:>6.3f} {c["ref_para"]:>5} {c["rouge03"]:>5.2f}')
        print(f'       Q: {c["query"][:90]}')
        print(f'       A: {c["gold"][:90]}')

    # ── 3. Pick top diverse pair (different qtype, different paragraph) ──
    print()
    print("=== Best diverse pair (different qtype + different paragraph) ===")
    best_pair = None
    best_score = -1
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            if c1["qtype"] == c2["qtype"]:
                continue
            if c1["ref_para"] == c2["ref_para"]:
                continue
            s = c1["score"] + c2["score"]
            if s > best_score:
                best_score = s
                best_pair = (c1, c2)
    if best_pair:
        c1, c2 = best_pair
        print(f"  Example 1: {c1['id']}  qtype={c1['qtype']}  score={c1['score']:.3f}")
        print(f"    Q: {c1['query']}")
        print(f"    A: {c1['gold']}")
        print(f"  Example 2: {c2['id']}  qtype={c2['qtype']}  score={c2['score']:.3f}")
        print(f"    Q: {c2['query']}")
        print(f"    A: {c2['gold']}")
        print(f"  Pair score: {best_score:.3f}")

    # ── 4. Compare with exp08's pair ──
    exp08_ids = ["Q0745", "Q0747"]
    print()
    print("=== exp08's pair for reference ===")
    for cand in candidates:
        if cand["id"] in exp08_ids:
            print(f"  {cand['id']}  qtype={cand['qtype']}  score={cand['score']:.3f}  "
                  f"(restate={cand['restate']:.2f} length={cand['length_s']:.2f} "
                  f"central={cand['centrality']:.3f} qfreq={cand['qtype_freq']:.3f})")
    exp08_score = sum(c["score"] for c in candidates if c["id"] in exp08_ids)
    print(f"  exp08 pair score: {exp08_score:.3f}")


if __name__ == "__main__":
    main()
