"""
GEN_K re-analysis: pool recall ceiling, oracle IoU, simulated LLM behavior,
and prompt length cost for K ∈ {1, 3, 5, 7, 10, 15, 20, 25}.
"""
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

P = Path("/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047")
with open(P / "textsum/eval_train/test.json", encoding="utf-8") as f:
    data = json.load(f)
gold = {q["ID"]: set([q["refs"]] if isinstance(q.get("refs"), str)
                     else (q.get("refs") or [])) for q in data["queries"]}
qdoc = {q["ID"]: q["doc_id"] for q in data["queries"]}
text_map = {}
for doc in data["docs"]:
    for p in doc["paragraphs"]:
        text_map[(doc["doc_id"], p["para_id"])] = p["text"]


def load_scored(p):
    raw = json.loads(Path(p).read_text())
    return {q: sorted(s, key=lambda x: -x[1]) for q, s in raw.items()}


qw3 = load_scored(P / "eval_retrieval/cache/rerank_qwen3_train.json")
bge = load_scored(P / "eval_retrieval/cache/rerank_train.json")

leakfree = [q for q in gold if qdoc[q] != "doc_050"]
n = len(leakfree)


def jaccard(a, b):
    a, b = set(a), set(b); u = a | b
    return len(a & b) / len(u) if u else 0.0


# Observed cite-rate-by-rank from exp23 actual (GEN_K=5)
e23_cite_rate = {1: 0.794, 2: 0.269, 3: 0.188, 4: 0.129, 5: 0.102}
# Precision when cited (% of cited at rank R that ARE gold)
e23_prec = {1: 0.878, 2: 0.463, 3: 0.336, 4: 0.408, 5: 0.290}


# ─── Pool recall at each K ───────────────────────────────────────────────
print("=" * 92)
print("GEN_K SWEEP — pool-level statistics")
print("=" * 92)

print(f"\n  {'K':<5}{'%gold qw3':>11}{'%gold bge':>11}"
      f"{'recall@K qw3':>14}{'hit@K qw3':>11}"
      f"{'IoU oracle qw3':>16}{'avg ctx chars':>15}")
print(f"  {'-'*5}{'-'*11}{'-'*11}{'-'*14}{'-'*11}{'-'*16}{'-'*15}")

total_gold = sum(len(gold[q]) for q in leakfree)
for K in [1, 3, 5, 7, 10, 15, 20, 25]:
    # pool-level
    qw3_found = sum(len(gold[q] & set(p for p, _ in qw3.get(q, [])[:K])) for q in leakfree)
    bge_found = sum(len(gold[q] & set(p for p, _ in bge.get(q, [])[:K])) for q in leakfree)
    rec_qw3 = sum(len(gold[q] & set(p for p, _ in qw3.get(q, [])[:K])) / len(gold[q])
                  for q in leakfree if gold[q]) / n
    hit_qw3 = sum(1 for q in leakfree if gold[q] & set(p for p, _ in qw3.get(q, [])[:K])) / n
    # IoU oracle: best K' ∈ {1..K} per-query that maximizes IoU
    iou_oracle = 0
    for q in leakfree:
        topK = [p for p, _ in qw3.get(q, [])[:K]]
        g = gold[q]
        best = 0
        # try each K' ≤ K (pick first K' from rank list, then maximize IoU by
        # rearranging to put golds first if possible)
        for k_use in range(1, K + 1):
            # oracle pick: take as many golds as possible from top-K, fill rest
            golds_in_top = [p for p in topK if p in g][:k_use]
            non_golds = [p for p in topK if p not in g][:max(0, k_use - len(golds_in_top))]
            pred = set(golds_in_top + non_golds)
            iou = jaccard(pred, g)
            if iou > best:
                best = iou
        iou_oracle += best
    # avg context chars (qw3 top-K paragraphs)
    avg_ctx = 0
    for q in leakfree:
        topK = [p for p, _ in qw3.get(q, [])[:K]]
        avg_ctx += sum(len(text_map.get((qdoc[q], p), "")) for p in topK)
    avg_ctx /= n

    print(f"  {K:<5}{qw3_found/total_gold:>10.1%} {bge_found/total_gold:>10.1%} "
          f"{rec_qw3:>14.4f}{hit_qw3:>11.4f}{iou_oracle/n:>16.4f}{avg_ctx:>15.0f}")


# ─── Estimated LLM IoU assuming "constant avg cited" behavior ────────────
print("\n" + "=" * 92)
print("ESTIMATED IoU under 2 LLM models")
print("=" * 92)
print(f"\n  Model 1: 'rank-1 anchored' — cite top-1 always, plus top-2 with rate observed at GEN_K=5")
print(f"  Model 2: 'avg-K constant' — pick K' refs/query = avg observed at GEN_K=5 (≈1.49)")
print(f"\n  {'GEN_K':<6}{'pool recall':>13}{'rank1-anchor IoU':>20}{'avg-K constant IoU':>22}")
print(f"  {'-'*6}{'-'*13}{'-'*20}{'-'*22}")
np.random.seed(0)
TARGET_CITES = 1.49  # observed avg in exp23
for K in [1, 3, 5, 7, 10, 15, 20]:
    # Model 1: cite rank-1 always; then for rank 2..K, sample with declining rate
    # We extrapolate cite rate beyond rank 5 by linear-log decline
    rates = {1: 0.794}
    for r in range(2, K + 1):
        # observed: r=2: 0.269, r=3: 0.188, r=4: 0.129, r=5: 0.102
        # rough exponential decay
        if r <= 5:
            rates[r] = e23_cite_rate[r]
        else:
            rates[r] = max(0.05, e23_cite_rate[5] * (0.8 ** (r - 5)))
    iou_m1 = 0
    iou_m2 = 0
    for q in leakfree:
        topK = [p for p, _ in qw3.get(q, [])[:K]]
        if not topK: continue
        g = gold[q]
        # Model 1: rank-1 always + sample others
        pred1 = {topK[0]}
        for r in range(1, len(topK)):
            if np.random.random() < rates[r + 1]:
                pred1.add(topK[r])
        iou_m1 += jaccard(pred1, g)
        # Model 2: pick TARGET_CITES refs from top-K (rounded probabilistically)
        k_pick = int(TARGET_CITES) + (1 if np.random.random() < (TARGET_CITES % 1) else 0)
        k_pick = min(k_pick, len(topK))
        pred2 = set(topK[:k_pick])
        iou_m2 += jaccard(pred2, g)
    pool_rec = sum(len(gold[q] & set(p for p, _ in qw3.get(q, [])[:K])) / len(gold[q])
                   for q in leakfree if gold[q]) / n
    print(f"  {K:<6}{pool_rec:>13.4f}{iou_m1/n:>20.4f}{iou_m2/n:>22.4f}")


# ─── Realistic projection: what would IoU look like with bigger K? ───────
# Use exp23 actual citation as anchor: when GEN_K=5, LLM cited from positions 1-5
# At GEN_K=10, LLM still sees 10. We don't know what it'd cite, but we can bound it.
print("\n" + "=" * 92)
print("REALISTIC PROJECTION (exp23 anchor + observed cite-rate decay)")
print("=" * 92)
print(f"\n  exp23 actual (GEN_K=5):  IoU = 0.6513   avg_cited = 1.49")
print()
print(f"  If GEN_K = 7:")
print(f"    pool recall:  0.849 (vs 0.816 at K=5)  +0.033 headroom")
print(f"    LLM behavior likely: cite ~1.5-1.7 refs (slight increase)")
print(f"    Best case (oracle): IoU = {0.0:.4f}  ← see oracle column above")
print(f"    Realistic est:      IoU ≈ 0.65-0.67  (small gain, similar precision drag)")
print()
print(f"  If GEN_K = 10:")
print(f"    pool recall:  0.876 (vs 0.816 at K=5)  +0.060 headroom")
print(f"    LLM behavior risk: longer context, may cite more (1.6-2.0)")
print(f"    Best case (oracle): see column above")
print(f"    Realistic est:      IoU ≈ 0.64-0.67  (recall up, but more wrong cites)")
print(f"    Cost: max_model_len 16384 → still fits; prompt 5×→2× chars")
print()
print(f"  If GEN_K = 3:")
print(f"    pool recall:  0.777 (vs 0.816)  −0.039 headroom (worse)")
print(f"    LLM behavior: cite 1.0-1.2 (less to choose from)")
print(f"    Realistic est:      IoU ≈ 0.61-0.63 (loses headroom)")

# ─── Prompt length impact ────────────────────────────────────────────────
print("\n" + "=" * 92)
print("PROMPT LENGTH at each GEN_K (avg chars, leak-free 1218)")
print("=" * 92)
print(f"\n  {'K':<5}{'avg ctx chars':>15}{'avg ctx tokens (~est)':>23}"
      f"{'pct of 16384':>16}{'fits max_model_len?':>22}")
print(f"  {'-'*5}{'-'*15}{'-'*23}{'-'*16}{'-'*22}")
SYSTEM_FEWSHOT_TOKENS = 1200  # estimate of system + 2 few-shots + query
for K in [1, 3, 5, 7, 10, 15, 20]:
    avg_ctx_chars = 0
    for q in leakfree:
        topK = [p for p, _ in qw3.get(q, [])[:K]]
        avg_ctx_chars += sum(len(text_map.get((qdoc[q], p), "")) for p in topK)
    avg_ctx_chars /= n
    # Thai roughly: 3-4 chars per token; conservative 3
    est_ctx_tokens = avg_ctx_chars / 3
    total_tokens = est_ctx_tokens + SYSTEM_FEWSHOT_TOKENS
    pct = total_tokens / 16384
    fits = "✓ comfortable" if pct < 0.5 else "✓ ok" if pct < 0.8 else "⚠ tight"
    print(f"  {K:<5}{avg_ctx_chars:>15.0f}{est_ctx_tokens:>23.0f}"
          f"{pct:>15.1%} {fits:>22}")

# ─── Per-bucket recall: where K matters most ─────────────────────────────
print("\n" + "=" * 92)
print("Per-|gold| bucket: %gold@K (qw3) — where bigger K helps most")
print("=" * 92)
buckets = defaultdict(list)
for q in leakfree:
    b = len(gold[q]) if len(gold[q]) <= 5 else "6+"
    buckets[b].append(q)
print(f"\n  {'|gold|':<8}{'n':>6}{'sum_g':>8}", end="")
for K in [3, 5, 7, 10, 15, 20]:
    print(f"  K={K:<4}", end="")
print()
for bucket in [1, 2, 3, 4, 5, "6+"]:
    qids = buckets[bucket]
    if not qids: continue
    sg = sum(len(gold[q]) for q in qids)
    print(f"  {str(bucket):<8}{len(qids):>6}{sg:>8}", end="")
    for K in [3, 5, 7, 10, 15, 20]:
        found = sum(len(gold[q] & set(p for p, _ in qw3.get(q, [])[:K])) for q in qids)
        print(f"  {found/sg:>6.1%}", end="")
    print()
