"""Measure prompt token-length distribution for exp75 with the gemma-4
tokenizer, so we can pick a max_model_len that fits the nvidia ModelOpt
build's smaller KV budget (weights load at ~30 GiB, not exp73's ~22 GiB,
because self_attn* is excluded from NVFP4 → bf16) without truncating any
query. Reuses run.py's prompt construction. CPU only, no vLLM."""
import os
import json
from pathlib import Path
from transformers import AutoTokenizer

import run  # build_messages / filter_valid_paragraphs / load_data

MODEL_NAME = os.environ.get("LLM_MODEL", "nvidia/Gemma-4-31B-IT-NVFP4")
TEST_DIR   = os.environ.get("TEST_DIR", "/model/test")
MAX_NEW    = run.MAX_NEW_TOKENS

tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
data = run.load_data(TEST_DIR)
doc_index = {d["doc_id"]: d["paragraphs"] for d in data["docs"]}
doc_paras = {k: run.filter_valid_paragraphs(v) for k, v in doc_index.items()}

lengths = []
for q in data["queries"]:
    valid = doc_paras.get(q["doc_id"], [])
    if not valid:
        continue
    texts = [p["text"] for p in valid]
    msgs = run.build_messages(q["query"], texts)
    s = tok.apply_chat_template(msgs, tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    lengths.append(len(tok(s, add_special_tokens=False)["input_ids"]))

lengths.sort()
n = len(lengths)
def pct(p): return lengths[min(n - 1, int(p * n))]
print(f"prompts measured: {n}")
print(f"prompt tokens  — min={lengths[0]}  p50={pct(0.50)}  p90={pct(0.90)} "
      f" p99={pct(0.99)}  max={lengths[-1]}")
print(f"+ MAX_NEW_TOKENS={MAX_NEW}  → worst total = {lengths[-1] + MAX_NEW}")
for cap in (16384, 20480, 22528, 24576, 32768):
    over = sum(1 for L in lengths if L + MAX_NEW > cap)
    print(f"  max_model_len={cap:6d}: {over:4d}/{n} queries would truncate "
          f"({100*over/n:.1f}%)")
