"""
exp31 — HyDE retrieval evaluation.

Embeds the hypothetical answers produced by hyde_generate.py with bge-m3,
then sweeps blend weights between (original query embedding) and (HyDE
embedding) and reports retrieval metrics on the train set — reusing
eval.py's metrics_for so numbers are directly comparable to
eval_retrieval/result/e0_eval.json.

Methods tested:
  blend{alpha}   — pure-dense ranking with embedding = norm((1-a)*q + a*h)
                   a=0.0 reproduces the baseline; a=1.0 is pure HyDE.
  rrf+blend{a}   — same blended dense + BM25 (original query tokens),
                   fused with RRF k=60 — matches exp30's stage-1 pool.

What we want to see:
  - hit@20 lift   -> larger candidate pool with gold included
                     (matters for exp30 which feeds the full pool to the LLM)
  - hit@1 / MRR   -> if dense already ranks gold higher, rerank has an easier job
  - iou@1         -> proxy for production composite

Negative signal:
  - HyDE drops numbers / changes named entities -> hit@1 falls on numeric
    queries; in that case retry with a stricter generation prompt.
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrievers import rank_dense, rank_bm25, rrf_fuse, tokenize_th
from eval import metrics_for, as_list, COLS

HERE          = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "cache" / "train.npz"
DEFAULT_HYDE  = HERE / "cache" / "hyde_train.json"
DEFAULT_TEST  = HERE.parent / "textsum" / "eval_train" / "test.json"
EMBED_MODEL   = "BAAI/bge-m3"

BLENDS     = [0.0, 0.25, 0.5, 0.75, 1.0]   # weight on HyDE
RRF_BLENDS = [0.0, 0.5, 1.0]


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def run_dense(query_emb_for_q, doc_bundle, query_id, qdoc_map, gold_map):
    agg, n = defaultdict(float), 0
    for i, qid in enumerate(query_id):
        d = qdoc_map.get(qid)
        if d not in doc_bundle:
            continue
        pe, pids, _ = doc_bundle[d]
        ranked = rank_dense(query_emb_for_q[i], pe, pids)
        for k, v in metrics_for(ranked, gold_map[qid]).items():
            agg[k] += v
        n += 1
    return {k: v / n for k, v in agg.items()}


def run_rrf(query_emb_for_q, doc_bundle, qtok, query_id, qdoc_map, gold_map, k=60):
    agg, n = defaultdict(float), 0
    for i, qid in enumerate(query_id):
        d = qdoc_map.get(qid)
        if d not in doc_bundle:
            continue
        pe, pids, bm25 = doc_bundle[d]
        r_dense = rank_dense(query_emb_for_q[i], pe, pids)
        r_bm25  = rank_bm25(qtok[qid], bm25, pids)
        ranked  = rrf_fuse([r_bm25, r_dense], k=k)
        for kk, vv in metrics_for(ranked, gold_map[qid]).items():
            agg[kk] += vv
        n += 1
    return {k: v / n for k, v in agg.items()}


def main() -> None:
    cache_path = Path(os.environ.get("EMBED_CACHE", DEFAULT_CACHE))
    hyde_path  = Path(os.environ.get("HYDE_JSON",   DEFAULT_HYDE))
    test_path  = Path(os.environ.get("TEST_JSON",   DEFAULT_TEST))

    if not cache_path.exists():
        raise SystemExit(
            f"embed cache not found: {cache_path}\n"
            f"run:  sbatch eval_retrieval/submit_embed.sh"
        )
    if not hyde_path.exists():
        raise SystemExit(
            f"HyDE cache not found: {hyde_path}\n"
            f"run:  sbatch eval_retrieval/submit_hyde.sh"
        )

    z         = np.load(cache_path, allow_pickle=False)
    para_emb  = z["para_emb"]
    para_doc  = [str(x) for x in z["para_doc"]]
    para_pid  = [str(x) for x in z["para_pid"]]
    query_emb = z["query_emb"].astype(np.float32)
    query_id  = [str(x) for x in z["query_id"]]

    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)
    text_map = {(d["doc_id"], p["para_id"]): p["text"]
                for d in data["docs"] for p in d["paragraphs"]}
    gold_map = {q["ID"]: as_list(q.get("refs")) for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}

    hyde     = json.loads(hyde_path.read_text(encoding="utf-8"))
    missing  = [qid for qid in query_id if qid not in hyde]
    if missing:
        raise SystemExit(
            f"{len(missing)} queries missing from HyDE cache, e.g. {missing[:5]}"
        )
    hyde_texts = [hyde[qid] for qid in query_id]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding {len(hyde_texts)} HyDE answers on {device}...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=device)
    hyde_emb = model.encode(
        hyde_texts, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    doc_idx = defaultdict(list)
    for i, d in enumerate(para_doc):
        doc_idx[d].append(i)

    from rank_bm25 import BM25Okapi
    doc_bundle = {}
    for d, idxs in doc_idx.items():
        pe   = para_emb[idxs]
        pids = [para_pid[i] for i in idxs]
        bm25 = BM25Okapi([tokenize_th(text_map.get((d, pid), "")) for pid in pids])
        doc_bundle[d] = (pe, pids, bm25)
    qtok = {qid: tokenize_th(qtxt_map[qid]) for qid in query_id}

    results = {}
    for a in BLENDS:
        blended = normalize((1 - a) * query_emb + a * hyde_emb)
        results[f"dense_blend{a:.2f}"] = run_dense(
            blended, doc_bundle, query_id, qdoc_map, gold_map)
    for a in RRF_BLENDS:
        blended = normalize((1 - a) * query_emb + a * hyde_emb)
        results[f"rrf+blend{a:.2f}"] = run_rrf(
            blended, doc_bundle, qtok, query_id, qdoc_map, gold_map)

    print(f"\n=== HyDE retrieval eval — {test_path.name} ({len(query_id)} queries) ===")
    header = f"{'method':<18}" + "".join(f"{c:>10}" for c in COLS)
    print(header)
    print("-" * len(header))
    for label, r in results.items():
        print(f"{label:<18}" + "".join(f"{r[c]:>10.4f}" for c in COLS))

    base   = results["dense_blend0.00"]
    pureh  = results["dense_blend1.00"]
    rrf0   = results["rrf+blend0.00"]
    rrf_h  = results["rrf+blend1.00"]
    print(f"\nHyDE-only vs query-only  (dense): "
          f"hit@1 {base['hit@1']:.4f} -> {pureh['hit@1']:.4f} "
          f"({pureh['hit@1']-base['hit@1']:+.4f}) | "
          f"hit@20 {base['hit@20']:.4f} -> {pureh['hit@20']:.4f} "
          f"({pureh['hit@20']-base['hit@20']:+.4f})")
    print(f"HyDE-only vs query-only  (rrf  ): "
          f"hit@1 {rrf0['hit@1']:.4f} -> {rrf_h['hit@1']:.4f} "
          f"({rrf_h['hit@1']-rrf0['hit@1']:+.4f}) | "
          f"hit@20 {rrf0['hit@20']:.4f} -> {rrf_h['hit@20']:.4f} "
          f"({rrf_h['hit@20']-rrf0['hit@20']:+.4f})")

    best_dense = max(BLENDS, key=lambda a: results[f"dense_blend{a:.2f}"]["hit@1"])
    print(f"best dense blend by hit@1: alpha={best_dense:.2f} "
          f"hit@1={results[f'dense_blend{best_dense:.2f}']['hit@1']:.4f}")

    out = HERE / "result" / "hyde_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"queries": len(query_id),
                  "blends": BLENDS, "rrf_blends": RRF_BLENDS},
         "results": results}, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
