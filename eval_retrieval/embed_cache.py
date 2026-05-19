"""
E0 retrieval harness — embed & cache step.

Encodes every valid paragraph and every query in a test.json with bge-m3
(dense) and writes one .npz, so eval.py can score retrieval configs
without re-embedding. Slow; run once per dataset.

  sbatch eval_retrieval/submit_embed.sh                 # train set (default)
  python eval_retrieval/embed_cache.py <test.json> <out.npz>
"""
import os
import sys
import time
import json
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

EMBED_MODEL  = "BAAI/bge-m3"
HERE         = Path(__file__).resolve().parent
DEFAULT_TEST = HERE.parent / "textsum" / "eval_train" / "test.json"
DEFAULT_OUT  = HERE / "cache" / "train.npz"


def is_valid_para(p: dict) -> bool:
    text = p["text"].strip()
    return bool(text) and not (set(text) <= set("_-=. \t\n"))


def main() -> None:
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"source            : {test_path}", flush=True)
    print(f"device            : {device}", flush=True)
    print(f"paragraphs (valid): {len(para_text)}   queries: {len(q_txt)}", flush=True)

    model = SentenceTransformer(EMBED_MODEL, device=device)

    def enc(texts):
        return model.encode(
            texts, batch_size=64, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

    t0 = time.time()
    para_emb  = enc(para_text)
    query_emb = enc(q_txt)
    print(f"encoded in {time.time() - t0:.1f}s", flush=True)

    np.savez(
        out_path,
        para_emb=para_emb,
        para_doc=np.array(para_doc),
        para_pid=np.array(para_pid),
        query_emb=query_emb,
        query_id=np.array(q_id),
    )
    print(f"cached            : {out_path}  "
          f"({(para_emb.nbytes + query_emb.nbytes) / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
