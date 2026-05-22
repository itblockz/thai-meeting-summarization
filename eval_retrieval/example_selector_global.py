"""
Global criteria-based example selection — apply the findings from
exp08–exp15 to find the best few-shot pair from the entire labeled
train set (no held-out restriction).

Key lesson from exp15 (criteria pair beat exp03 +0.0049 but LOST to
exp08 -0.0042): restate-pattern strength is the single most-predictive
criterion, so we raise its weight and add a hard constraint (both
examples must have restate ≥ 4/5).

Constraints:
  • restate ≥ 4/5 for BOTH examples (the hard floor for "strong teacher")
  • different paragraphs (avoid context coupling)
  • different docs (avoid topic coupling)
  • different qtypes (cover at least 2 patterns)
  • single-ref gold (clean 1:1 mapping for the example)

Within those, score by:
  0.50 * restate + 0.25 * length_proximity + 0.15 * centrality + 0.10 * qtype_freq

Length-proximity peaks at 1.5x global gold median (~50 tok) — matches
exp08's avg example length where the model produces output near gold.

For deployment context: held-out validation is impossible if examples
come from arbitrary docs, so this script outputs the top 5 candidate
pairs and one suggestion to validate via a held-out-pair experiment
(holding out both example-source docs).
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

GOLD_MEDIAN_TOK = 33
IDEAL_LEN_TOK   = GOLD_MEDIAN_TOK   # 33, exp08's empirical good range was 20-30 per example
LEN_LO, LEN_HI  = 20, 35            # HARD constraint — match exp08 (avg 25 per ex)

QTYPE_FREQS = {
    "expl":     0.445, "list":     0.174, "what":     0.164, "other":    0.150,
    "who":      0.030, "location": 0.018, "when":     0.015, "yn":       0.004,
}


def tok(t):
    return word_tokenize(t or "", engine="newmm", keep_whitespace=False)


def classify_query(q):
    if any(w in q for w in ("อย่างไร", "ทำไม", "เพราะ", "เหตุใด")): return "expl"
    if any(w in q for w in ("ที่ใด", "ที่ไหน", "สถานที่")):      return "location"
    if any(w in q for w in ("ใคร",)):                            return "who"
    if any(w in q for w in ("เมื่อใด", "เมื่อไหร่")):           return "when"
    if any(w in q for w in ("หรือไม่",)):                        return "yn"
    if any(w in q for w in ("มีอะไรบ้าง", "รายละเอียด", "เท่าใด", "กี่", "จำนวน")): return "list"
    if any(w in q for w in ("คืออะไร", "อะไร")):                 return "what"
    return "other"


def restate_score(query, answer, k=5):
    q_toks = set(tok(query)[:k])
    a_toks = tok(answer)[:k]
    if not q_toks: return 0.0
    return sum(1 for t in a_toks if t in q_toks) / k


def length_score(answer_tok, ideal=IDEAL_LEN_TOK):
    return 1.0 / (1.0 + abs(answer_tok - ideal) / ideal)


def main():
    data = json.load(open(TEST_JSON, encoding="utf-8"))
    queries = data["queries"]
    doc_index = {d["doc_id"]: d for d in data["docs"]}

    # Centrality
    print("Embedding 1239 gold answers on CPU (~2 min) ...", flush=True)
    model = SentenceTransformer("BAAI/bge-m3", device="cpu")
    valid_qs = [q for q in queries if (q.get("abstractive") or "").strip()]
    embs = model.encode([q["abstractive"] for q in valid_qs],
                        batch_size=32, normalize_embeddings=True,
                        convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = (embs @ embs.T).numpy()
    centrality = {}
    for i, q in enumerate(valid_qs):
        sims = np.concatenate([sim_matrix[i, :i], sim_matrix[i, i+1:]])
        centrality[q["ID"]] = float(sims.mean())

    # Score all globally — but only those that pass hard constraints
    candidates = []
    for q in queries:
        refs = q.get("refs", [])
        if isinstance(refs, str): refs = [refs]
        if len(refs) != 1: continue
        gold = (q.get("abstractive") or "").strip()
        if not gold: continue
        para_text = next((p["text"] for p in doc_index[q["doc_id"]]["paragraphs"]
                         if p["para_id"] == refs[0]), "")
        if not para_text.strip(): continue

        ans_tok = len(tok(gold))
        rs = restate_score(q["query"], gold)
        # HARD CONSTRAINTS
        if rs < 0.8: continue                       # strong restate teacher
        if not (LEN_LO <= ans_tok <= LEN_HI): continue   # length match exp08 empirical good
        ls = length_score(ans_tok)
        cs = centrality.get(q["ID"], 0.0)
        qt = classify_query(q["query"])
        qf = QTYPE_FREQS.get(qt, 0.01)

        weights = {"restate": 0.50, "length": 0.25, "centrality": 0.15, "qtype": 0.10}
        total = (weights["restate"] * rs + weights["length"] * ls +
                 weights["centrality"] * cs + weights["qtype"] * qf)

        candidates.append({
            "id": q["ID"], "doc_id": q["doc_id"], "ref_para": refs[0],
            "query": q["query"], "gold": gold, "paragraph": para_text,
            "qtype": qt, "ans_tok": ans_tok,
            "restate": rs, "length_s": ls, "centrality": cs, "qtype_freq": qf,
            "score": total,
        })
    candidates.sort(key=lambda c: -c["score"])
    print(f"\n{len(candidates)} candidates passed restate ≥ 0.8 hard constraint")

    # Show top 15 individuals
    print(f'\nTop 15 individual candidates:')
    print(f'{"ID":<6} {"doc":<8} {"qtype":<8} {"ans_tok":>7} {"restate":>7} {"length":>7} {"central":>7} {"score":>6}')
    for c in candidates[:15]:
        print(f'{c["id"]:<6} {c["doc_id"]:<8} {c["qtype"]:<8} {c["ans_tok"]:>7} {c["restate"]:>7.2f} {c["length_s"]:>7.2f} {c["centrality"]:>7.3f} {c["score"]:>6.3f}')
        print(f'       Q: {c["query"][:90]}')
        print(f'       A: {c["gold"][:90]}')

    # Find top 5 diverse pairs (different doc + qtype)
    print()
    print('=== Top 5 diverse pairs (different doc + qtype) ===')
    pairs = []
    seen = set()
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            if c1["doc_id"] == c2["doc_id"]: continue
            if c1["qtype"] == c2["qtype"]: continue
            pair_key = tuple(sorted([c1["id"], c2["id"]]))
            if pair_key in seen: continue
            seen.add(pair_key)
            pairs.append((c1, c2, c1["score"] + c2["score"]))
    pairs.sort(key=lambda x: -x[2])

    for i, (c1, c2, ps) in enumerate(pairs[:5], 1):
        print(f"\n--- Pair #{i} (score {ps:.3f}) ---")
        print(f"  {c1['id']} {c1['doc_id']} {c1['qtype']:<6} restate={c1['restate']:.2f} ans_tok={c1['ans_tok']} central={c1['centrality']:.3f}")
        print(f"    Q: {c1['query']}")
        print(f"    A: {c1['gold']}")
        print(f"  {c2['id']} {c2['doc_id']} {c2['qtype']:<6} restate={c2['restate']:.2f} ans_tok={c2['ans_tok']} central={c2['centrality']:.3f}")
        print(f"    Q: {c2['query']}")
        print(f"    A: {c2['gold']}")

    # For reference: where does exp08's pair score in this global ranking?
    print()
    print("=== exp08 pair (Q0745+Q0747) reference ===")
    e8_ids = ["Q0745", "Q0747"]
    e8 = [c for c in candidates if c["id"] in e8_ids]
    if len(e8) == 2:
        e8_score = sum(c["score"] for c in e8)
        print(f"  pair score = {e8_score:.3f}")
        # rank in global pair list
        e8_pair = tuple(sorted(e8_ids))
        for i, (c1, c2, ps) in enumerate(pairs, 1):
            if tuple(sorted([c1["id"], c2["id"]])) == e8_pair:
                print(f"  rank in global diverse-pair list: #{i} of {len(pairs)}")
                break
        else:
            print(f"  exp08 pair NOT in diverse-pair list (Q0747 type='what', Q0745 type='location' — both diverse; check why excluded)")


if __name__ == "__main__":
    main()
