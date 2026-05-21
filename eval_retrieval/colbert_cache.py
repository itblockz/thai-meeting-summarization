"""
E2 — bge-m3 multi-vector (ColBERT) embedding cache.

For each valid paragraph and each query, encode with bge-m3's multi-vector
head (FlagEmbedding BGEM3FlagModel) and save the per-token embeddings to
a pickle. Multi-vector encodings are variable-length (one vector per token),
so npz with stack doesn't fit — we store dicts of {id -> ndarray}.

  sbatch eval_retrieval/submit_colbert_cache.sh
  python eval_retrieval/colbert_cache.py <test.json> <out.pkl>
"""
import os
import sys
import time
import json
import pickle
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np

EMBED_MODEL  = "BAAI/bge-m3"
HERE         = Path(__file__).resolve().parent
DEFAULT_TEST = HERE.parent / "textsum" / "eval_train" / "test.json"
DEFAULT_OUT  = HERE / "cache" / "colbert_train.pkl"
BATCH_SIZE   = 16


def is_valid_para(p: dict) -> bool:
    text = p["text"].strip()
    return bool(text) and not (set(text) <= set("_-=. \t\n"))


def encode_colbert(model, texts, batch_size=BATCH_SIZE):
    """Encode texts with bge-m3 multi-vector head. Returns list of (n_tok, 1024) arrays."""
    out = model.encode(
        texts, batch_size=batch_size,
        return_dense=False, return_sparse=False, return_colbert_vecs=True,
        max_length=512,
    )
    return [v.astype(np.float16) for v in out["colbert_vecs"]]


def main() -> None:
    from FlagEmbedding import BGEM3FlagModel
    import torch

    test_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST
    out_path  = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)

    para_doc, para_pid, para_text = [], [], []
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            if is_valid_para(p):
                para_doc.append(doc["doc_id"])
                para_pid.append(p["para_id"])
                para_text.append(p["text"])

    queries = data["queries"]
    q_id  = [q["ID"] for q in queries]
    q_txt = [q["query"] for q in queries]

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = torch.cuda.is_available()
    print(f"source     : {test_path}", flush=True)
    print(f"device     : {device_str}  fp16={use_fp16}", flush=True)
    print(f"paragraphs : {len(para_text)}   queries: {len(q_txt)}", flush=True)

    model = BGEM3FlagModel(EMBED_MODEL, use_fp16=use_fp16, device=device_str)

    t0 = time.time()
    para_vecs  = encode_colbert(model, para_text)
    print(f"  paragraphs encoded in {time.time() - t0:.1f}s", flush=True)
    t1 = time.time()
    query_vecs = encode_colbert(model, q_txt)
    print(f"  queries encoded in {time.time() - t1:.1f}s", flush=True)

    payload = {
        "para_doc":  para_doc,
        "para_pid":  para_pid,
        "para_vecs": para_vecs,    # list of (n_tok, 1024) f16
        "query_id":  q_id,
        "query_vecs": query_vecs,  # list of (n_tok, 1024) f16
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    total_bytes = sum(v.nbytes for v in para_vecs) + sum(v.nbytes for v in query_vecs)
    avg_para_tok  = np.mean([v.shape[0] for v in para_vecs])
    avg_query_tok = np.mean([v.shape[0] for v in query_vecs])
    print(f"cached     : {out_path}  ({total_bytes / 1e6:.1f} MB)", flush=True)
    print(f"avg tokens : para={avg_para_tok:.1f}  query={avg_query_tok:.1f}", flush=True)


if __name__ == "__main__":
    main()
