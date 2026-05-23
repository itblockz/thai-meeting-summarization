"""
Pool analysis: hit@pool / recall@pool / %gold@pool for the candidate pool
(dense + BM25 union) before reranking. Splits the union into dense-only,
bm25-only contributions to see which retriever brings gold the other misses.
Sweeps POOL_N from 10..60 to see if widening the pool would recover gold.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

P = Path("/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047")
CACHE = P / "eval_retrieval/cache/train.npz"
TEST  = P / "textsum/eval_train/test.json"

with open(TEST, encoding="utf-8") as f:
    data = json.load(f)
gold = {q["ID"]: set([q["refs"]] if isinstance(q.get("refs"), str)
                     else (q.get("refs") or [])) for q in data["queries"]}
qdoc = {q["ID"]: q["doc_id"] for q in data["queries"]}

z = np.load(CACHE, allow_pickle=False)
para_emb  = z["para_emb"]
para_doc  = [str(x) for x in z["para_doc"]]
para_pid  = [str(x) for x in z["para_pid"]]
query_emb = z["query_emb"]
query_id  = [str(x) for x in z["query_id"]]

text_map = {}
for doc in data["docs"]:
    for p in doc["paragraphs"]:
        text_map[(doc["doc_id"], p["para_id"])] = p["text"]
qtxt = {q["ID"]: q["query"] for q in data["queries"]}

doc_idx = defaultdict(list)
for i, d in enumerate(para_doc):
    doc_idx[d].append(i)

from pythainlp.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


def tokenize_th(t): return word_tokenize(t, engine="newmm", keep_whitespace=False)


print("building per-doc dense + BM25 indices ...", flush=True)
doc_bundle = {}
for d, idxs in doc_idx.items():
    pe = para_emb[idxs]
    pids = [para_pid[i] for i in idxs]
    bm25 = BM25Okapi([tokenize_th(text_map.get((d, p), "")) for p in pids])
    doc_bundle[d] = (pe, pids, bm25)

leakfree = [q for q in query_id if qdoc.get(q) != "doc_050" and q in gold]
n = len(leakfree)
total_gold = sum(len(gold[q]) for q in leakfree)
print(f"leak-free queries: {n}   total gold: {total_gold}\n", flush=True)


def topN(pool_n):
    """Return dict qid -> (dense_set, bm25_set, union_set, intersect_set)."""
    out = {}
    for i, qid in enumerate(query_id):
        if qid not in gold or qdoc[qid] == "doc_050":
            continue
        d = qdoc[qid]
        if d not in doc_bundle:
            out[qid] = (set(), set(), set(), set())
            continue
        pe, pids, bm25 = doc_bundle[d]
        sims = pe @ query_emb[i]
        dense_idx = np.argsort(-sims)[:pool_n]
        bm25_sc = np.asarray(bm25.get_scores(tokenize_th(qtxt[qid])))
        bm25_idx = np.argsort(-bm25_sc)[:pool_n]
        dense = {pids[j] for j in dense_idx}
        bm25_s = {pids[j] for j in bm25_idx}
        out[qid] = (dense, bm25_s, dense | bm25_s, dense & bm25_s)
    return out


def metrics(pred_for_qid):
    hit = rec = found = total_g = 0
    for qid in leakfree:
        g = gold[qid]
        p = pred_for_qid(qid)
        inter = g & p
        if inter: hit += 1
        rec += len(inter)/len(g) if g else 0
        found += len(inter)
        total_g += len(g)
    return hit/n, rec/n, found/total_g


print("=" * 88)
print("METHOD COMPARISON at POOL_N=20 (current setup)")
print("=" * 88)
sets20 = topN(20)
print(f"\n  {'method':<22}{'pool size':>11}{'hit@pool':>11}{'recall@pool':>14}{'%gold@pool':>13}")
print(f"  {'-'*22}{'-'*11}{'-'*11}{'-'*14}{'-'*13}")
for label, idx in [("dense only top-20", 0), ("BM25 only top-20", 1),
                   ("UNION (= pool)", 2), ("INTERSECTION", 3)]:
    avg_size = sum(len(sets20[q][idx]) for q in leakfree) / n
    h, r, g = metrics(lambda q, i=idx: sets20[q][i])
    print(f"  {label:<22}{avg_size:>11.1f}{h:>11.1%}{r:>14.4f}{g:>13.1%}")

# What does bm25 add?
print("\n  --- gold contribution by retriever ---")
only_dense = only_bm25 = both = 0
for q in leakfree:
    d, b, _, _ = sets20[q]
    g = gold[q]
    for p in g:
        in_d = p in d; in_b = p in b
        if in_d and in_b: both += 1
        elif in_d: only_dense += 1
        elif in_b: only_bm25 += 1

print(f"  gold refs found by:")
print(f"    BOTH dense+BM25  : {both:>5}  ({both/total_gold:>6.1%} of total gold)")
print(f"    DENSE only       : {only_dense:>5}  ({only_dense/total_gold:>6.1%})")
print(f"    BM25 only        : {only_bm25:>5}  ({only_bm25/total_gold:>6.1%})  ← BM25 contribution")
print(f"    NEITHER (lost)   : {total_gold - both - only_dense - only_bm25:>5}  "
      f"({1 - (both+only_dense+only_bm25)/total_gold:>6.1%}) ← UNREACHABLE")

print("\n" + "=" * 88)
print("POOL_N SWEEP — does widening help?")
print("=" * 88)
print(f"\n  {'POOL_N':<8}{'pool size':>12}{'hit@pool':>11}{'recall@pool':>14}{'%gold@pool':>13}")
print(f"  {'-'*8}{'-'*12}{'-'*11}{'-'*14}{'-'*13}")
for pn in [5, 10, 15, 20, 25, 30, 40, 60, 100, 999]:
    s = topN(pn)
    avg_size = sum(len(s[q][2]) for q in leakfree) / n
    h, r, g = metrics(lambda q: s[q][2])
    label = f"all" if pn == 999 else str(pn)
    print(f"  {label:<8}{avg_size:>12.1f}{h:>11.1%}{r:>14.4f}{g:>13.1%}")

# Per-|gold| bucket at POOL_N=20
print("\n" + "=" * 88)
print("POOL RECALL by |gold| bucket  (POOL_N=20, dense ∪ BM25)")
print("=" * 88)
buckets = defaultdict(list)
for q in leakfree:
    b = len(gold[q]) if len(gold[q]) <= 5 else "6+"
    buckets[b].append(q)

print(f"\n  {'|gold|':<8}{'n':>5}{'sum_gold':>10}{'hit':>8}{'recall':>10}{'%gold':>9}{'avg_pool':>10}")
for bucket in [1, 2, 3, 4, 5, "6+"]:
    qids = buckets[bucket]
    if not qids: continue
    sg = sum(len(gold[q]) for q in qids)
    h = sum(1 for q in qids if gold[q] & sets20[q][2])
    r = sum(len(gold[q] & sets20[q][2])/len(gold[q]) for q in qids)/len(qids)
    g = sum(len(gold[q] & sets20[q][2]) for q in qids) / sg
    pool_sz = sum(len(sets20[q][2]) for q in qids) / len(qids)
    print(f"  {str(bucket):<8}{len(qids):>5}{sg:>10}{h/len(qids):>8.1%}{r:>10.3f}{g:>9.1%}{pool_sz:>10.1f}")

# Per-|gold| bucket at POOL_N=40 (does widening help worst bucket?)
print(f"\n  --- same but POOL_N=40 ---")
sets40 = topN(40)
print(f"  {'|gold|':<8}{'n':>5}{'sum_gold':>10}{'hit':>8}{'recall':>10}{'%gold':>9}{'avg_pool':>10}")
for bucket in [1, 2, 3, 4, 5, "6+"]:
    qids = buckets[bucket]
    if not qids: continue
    sg = sum(len(gold[q]) for q in qids)
    h = sum(1 for q in qids if gold[q] & sets40[q][2])
    r = sum(len(gold[q] & sets40[q][2])/len(gold[q]) for q in qids)/len(qids)
    g = sum(len(gold[q] & sets40[q][2]) for q in qids) / sg
    pool_sz = sum(len(sets40[q][2]) for q in qids) / len(qids)
    print(f"  {str(bucket):<8}{len(qids):>5}{sg:>10}{h/len(qids):>8.1%}{r:>10.3f}{g:>9.1%}{pool_sz:>10.1f}")
