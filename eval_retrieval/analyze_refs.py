"""
Per-query analysis of LLM citation quality, exp22 vs exp23.

For each query, compare:
  - gold refs (from test.json)
  - bge top-5 rerank (rerank_train.json)
  - qwen3 top-5 rerank (rerank_qwen3_train.json)
  - exp22 cited refs (submission.csv, bge rerank in pipeline)
  - exp23 cited refs (submission.csv, qwen3 rerank in pipeline)

Reports:
  - retrieval-side: was gold in top-5? (the upper bound the LLM can cite)
  - citation-side: given gold was reachable, did the LLM cite it?
  - over-citation: queries where LLM cited extra refs not in gold

Run on LANTA where submission.csv files exist:
  python3 eval_retrieval/analyze_refs.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

PROJECT = Path("/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047")
TEST    = PROJECT / "textsum/eval_train/test.json"
BGE_RR  = PROJECT / "eval_retrieval/cache/rerank_train.json"
QW3_RR  = PROJECT / "eval_retrieval/cache/rerank_qwen3_train.json"
EXP22   = PROJECT / "exp22/eval_result/submission.csv"
EXP23   = PROJECT / "exp23/eval_result/submission.csv"

GEN_K = 5


def as_list(refs):
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def parse_refs(field):
    return [r.strip() for r in field.split(",") if r.strip()]


def load_subm(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["ID"]] = parse_refs(row["refs"])
    return out


def load_topk(path, k=GEN_K):
    raw = json.loads(Path(path).read_text())
    out = {}
    for qid, scored in raw.items():
        ordered = sorted(scored, key=lambda x: -x[1])[:k]
        out[qid] = [pid for pid, _ in ordered]
    return out


def jaccard(a, b):
    a, b = set(a), set(b)
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def main():
    with open(TEST, encoding="utf-8") as f:
        data = json.load(f)
    gold = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}
    qdoc = {q["ID"]: q["doc_id"] for q in data["queries"]}

    bge_top5 = load_topk(BGE_RR)
    qw3_top5 = load_topk(QW3_RR)
    e22 = load_subm(EXP22)
    e23 = load_subm(EXP23)

    qids = [qid for qid in gold if qid in e22 and qid in e23]
    print(f"queries: {len(qids)}  (gold={len(gold)})")

    # Leak-free subset for fair comparison
    leakfree = [qid for qid in qids if qdoc[qid] != "doc_050"]
    print(f"leak-free (excl doc_050): {len(leakfree)}\n")

    def report(name, qid_set):
        rows = [(qid, set(gold[qid]), set(bge_top5.get(qid, [])),
                 set(qw3_top5.get(qid, [])), set(e22[qid]), set(e23[qid]))
                for qid in qid_set]

        # Retrieval-side: gold ⊂ top-5? (% queries where ALL gold is reachable)
        # And: any gold ∈ top-5?
        bge_any  = sum(1 for _, g, b5, _, _, _ in rows if g & b5)
        qw3_any  = sum(1 for _, g, _, q5, _, _ in rows if g & q5)
        bge_full = sum(1 for _, g, b5, _, _, _ in rows if g and g <= b5)
        qw3_full = sum(1 for _, g, _, q5, _, _ in rows if g and g <= q5)
        n = len(rows)

        # Citation-side: when gold was reachable, did the LLM cite it?
        # We restrict to queries where the RESPECTIVE reranker put gold in top-5.
        def cite_quality(rerank_set_idx, cited_set_idx):
            tp = fp = fn = 0
            reachable_n = cited_when_reachable = 0
            cited_correct = 0
            for r in rows:
                g = r[1]
                top5 = r[rerank_set_idx]
                cited = r[cited_set_idx]
                # only consider gold paragraphs that were reachable
                reachable_gold = g & top5
                if reachable_gold:
                    reachable_n += 1
                    if cited & reachable_gold:
                        cited_when_reachable += 1
                tp += len(cited & g)
                fp += len(cited - g)
                fn += len(g - cited)
                if cited & g:
                    cited_correct += 1
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec  = tp / (tp + fn) if (tp + fn) else 0
            return {
                "reachable_n": reachable_n,
                "cited_when_reachable_pct": cited_when_reachable / reachable_n if reachable_n else 0,
                "cited_correct_pct": cited_correct / n,
                "micro_precision": prec,
                "micro_recall":    rec,
                "micro_f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0,
                "avg_refs": sum(len(r[i]) for r in rows for i in [cited_set_idx]) / n,
                "avg_gold": sum(len(r[1]) for r in rows) / n,
            }

        e22_q = cite_quality(2, 4)   # bge top-5, exp22 cited
        e23_q = cite_quality(3, 5)   # qw3 top-5, exp23 cited

        # Over-citation: did the model cite more refs than gold had?
        over22 = sum(1 for r in rows if len(r[4]) > len(r[1]))
        over23 = sum(1 for r in rows if len(r[5]) > len(r[1]))

        # Per-query IoU
        iou22 = sum(jaccard(r[4], r[1]) for r in rows) / n
        iou23 = sum(jaccard(r[5], r[1]) for r in rows) / n

        # Where exp23 lost: queries where exp22 IoU > exp23 IoU
        regression = 0; improvement = 0; tie = 0
        for r in rows:
            i22 = jaccard(r[4], r[1]); i23 = jaccard(r[5], r[1])
            if i23 < i22 - 1e-9: regression += 1
            elif i23 > i22 + 1e-9: improvement += 1
            else: tie += 1

        print(f"=== {name} (n={n}) ===")
        print(f"  reachable in top-5 (hit@5):  bge {bge_any/n:.4f}   qw3 {qw3_any/n:.4f}   Δ={qw3_any/n - bge_any/n:+.4f}")
        print(f"  ALL gold in top-5         :  bge {bge_full/n:.4f}   qw3 {qw3_full/n:.4f}   Δ={qw3_full/n - bge_full/n:+.4f}")
        print()
        print(f"  citation behaviour:")
        print(f"  {'metric':<35} {'exp22(bge)':>12} {'exp23(qw3)':>12} {'Δ':>10}")
        for k in ["cited_when_reachable_pct", "cited_correct_pct",
                  "micro_precision", "micro_recall", "micro_f1",
                  "avg_refs", "avg_gold"]:
            v22, v23 = e22_q[k], e23_q[k]
            print(f"  {k:<35} {v22:>12.4f} {v23:>12.4f} {v23 - v22:>+10.4f}")
        print()
        print(f"  over-citation (|cited|>|gold|): exp22 {over22}/{n} ({over22/n:.1%})   "
              f"exp23 {over23}/{n} ({over23/n:.1%})")
        print(f"  per-query IoU mean:             exp22 {iou22:.4f}   exp23 {iou23:.4f}   "
              f"Δ={iou23 - iou22:+.4f}")
        print(f"  per-query IoU changes:          regression={regression}   "
              f"improvement={improvement}   tie={tie}")
        print()

        # Sample over-cited refs distribution
        ref_count_22 = Counter(len(r[4]) for r in rows)
        ref_count_23 = Counter(len(r[5]) for r in rows)
        gold_count   = Counter(len(r[1]) for r in rows)
        print(f"  cited-count distribution:")
        keys = sorted(set(ref_count_22) | set(ref_count_23) | set(gold_count))
        print(f"    {'#refs':<8}" + "".join(f"{k:>10}" for k in keys))
        print(f"    {'gold':<8}" + "".join(f"{gold_count.get(k,0):>10}" for k in keys))
        print(f"    {'exp22':<8}" + "".join(f"{ref_count_22.get(k,0):>10}" for k in keys))
        print(f"    {'exp23':<8}" + "".join(f"{ref_count_23.get(k,0):>10}" for k in keys))
        print()

    report("FULL 1239", qids)
    report("LEAK-FREE 1218", leakfree)


if __name__ == "__main__":
    main()
