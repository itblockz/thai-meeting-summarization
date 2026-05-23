"""
One-time download of Qwen3-Reranker-8B into the shared HF cache.

Run this on the LANTA *login* node (compute nodes have no internet):

  module load cray-python/3.11.7
  source /lustrefs/disk/project/zz991000-zdeva/zz991021/venv/bin/activate
  python eval_retrieval/download_qwen3_reranker.py

~16 GB on disk. Once done, the offline-mode `rerank_qwen3_cache.py`
can load it from $HF_HOME.
"""
import os
os.environ["HF_HOME"] = "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache"
# explicitly enable network for this script
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from huggingface_hub import snapshot_download

MODEL_ID = os.environ.get("QWEN_RERANKER", "Qwen/Qwen3-Reranker-8B")

print(f"downloading {MODEL_ID} -> {os.environ['HF_HOME']} ...")
path = snapshot_download(MODEL_ID)
print(f"done: {path}")
