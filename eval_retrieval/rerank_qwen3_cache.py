"""
Qwen3-Reranker-8B scoring step — compare against bge-reranker-v2-m3 on Thai.

Reuses the *exact same candidate pool* as `rerank_cache.py` (read from
`cache/rerank_train.json`) so the only variable is the reranker model.
Output is a drop-in replacement cache that `eval.py --rerank-cache <path>`
can rank.

  # one-time: download model to shared HF cache (login node, internet on)
  python eval_retrieval/download_qwen3_reranker.py

  # then: score the pool (GPU job)
  sbatch eval_retrieval/submit_rerank_qwen3.sh

  # finally: rank + compare
  python eval_retrieval/eval.py --method rerank \
      --rerank-cache eval_retrieval/cache/rerank_qwen3_train.json

Qwen3-Reranker is a *generative* reranker: it's an 8B causal-LM that
answers "yes"/"no" to a prompt template; the relevance score is the
yes-token probability after softmax(yes, no). Slower per pair than a
0.6B cross-encoder but reportedly stronger on MMTEB-R (+14.6 over
bge-reranker-v2-m3 multilingually).
"""
import os
import time
import json
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID    = os.environ.get("QWEN_RERANKER", "Qwen/Qwen3-Reranker-8B")
MAX_LENGTH  = int(os.environ.get("MAX_LENGTH", "2048"))
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE", "8"))
TASK_INSTR  = os.environ.get(
    "TASK_INSTR",
    "Given a Thai question about a parliamentary meeting document, "
    "retrieve the paragraph that answers the question.",
)

HERE     = Path(__file__).resolve().parent
TEST     = HERE.parent / "textsum" / "eval_train" / "test.json"
POOL_IN  = HERE / "cache" / "rerank_train.json"        # bge-reranker pool (source of pids)
OUT      = HERE / "cache" / "rerank_qwen3_train.json"

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_prompt(query: str, doc: str) -> str:
    return (f"<Instruct>: {TASK_INSTR}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}")


def main() -> None:
    if not POOL_IN.exists():
        raise SystemExit(
            f"bge rerank cache not found: {POOL_IN}\n"
            f"run first:  sbatch eval_retrieval/submit_rerank.sh"
        )

    # paragraph text + query text
    with open(TEST, encoding="utf-8") as f:
        data = json.load(f)
    text_map = {}
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            text_map[(doc["doc_id"], p["para_id"])] = p["text"]
    qtxt_map = {q["ID"]: q["query"] for q in data["queries"]}
    qdoc_map = {q["ID"]: q["doc_id"] for q in data["queries"]}

    # reuse the bge candidate pool — only the pids; we re-score with Qwen3
    pool = json.loads(POOL_IN.read_text())
    pairs, pair_qid, pair_pid = [], [], []
    for qid, scored in pool.items():
        d = qdoc_map.get(qid)
        if d is None:
            continue
        for pid, _ in scored:
            pairs.append(format_prompt(qtxt_map[qid], text_map.get((d, pid), "")))
            pair_qid.append(qid)
            pair_pid.append(pid)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("Qwen3-Reranker-8B needs a GPU — abort.")

    print(f"device={device}  model={MODEL_ID}  queries={len(pool)}  "
          f"pairs={len(pairs)}  batch={BATCH_SIZE}  max_len={MAX_LENGTH}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")

    # flash-attn if available — else sdpa (both bf16)
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    print(f"attn_impl={attn_impl}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    ).to(device).eval()

    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(SUFFIX, add_special_tokens=False)
    body_cap   = MAX_LENGTH - len(prefix_ids) - len(suffix_ids)

    @torch.no_grad()
    def score_batch(batch_pairs):
        enc = tokenizer(
            batch_pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=body_cap,
        )
        for i, ids in enumerate(enc["input_ids"]):
            enc["input_ids"][i] = prefix_ids + ids + suffix_ids
        enc = tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits   = model(**enc).logits[:, -1, :]
        yes_vec  = logits[:, token_true_id]
        no_vec   = logits[:, token_false_id]
        stacked  = torch.stack([no_vec, yes_vec], dim=1)
        probs    = torch.nn.functional.log_softmax(stacked, dim=1)
        return probs[:, 1].exp().float().cpu().tolist()

    t0 = time.time()
    all_scores = []
    for i in range(0, len(pairs), BATCH_SIZE):
        chunk = pairs[i:i + BATCH_SIZE]
        all_scores.extend(score_batch(chunk))
        if (i // BATCH_SIZE) % 50 == 0:
            elapsed = time.time() - t0
            done    = i + len(chunk)
            eta     = elapsed / max(done, 1) * (len(pairs) - done)
            print(f"  {done}/{len(pairs)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)
    print(f"scored {len(pairs)} pairs in {time.time() - t0:.1f}s", flush=True)

    cache = defaultdict(list)
    for qid, pid, s in zip(pair_qid, pair_pid, all_scores):
        cache[qid].append([pid, float(s)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"cached -> {OUT}  ({len(cache)} queries)", flush=True)


if __name__ == "__main__":
    main()
