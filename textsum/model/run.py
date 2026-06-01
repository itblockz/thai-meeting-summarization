"""
v16.1 — exp51 port (single-model A3B + V10_factual prompt), H100-safe.

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to
Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, 30 GB FP8 weights) in document
order. The model is shown the paragraphs as a numbered [1..N] context,
answers in Thai, then cites which paragraphs it used as [อ้างอิง: X];
the cited paragraphs become `refs` (E5 self-citation, adaptive count).
Two worked few-shot examples are prepended as multi-turn chat turns.

v16.1 = exp51 port. Single-variable change from v16 (exp42, 0.7087): the
instruction prompt is swapped from exp38's E5 wording ("กระชับและ
ครอบคลุม") to **V10_factual** ("สั้นและตรงประเด็น, ระบุข้อเท็จจริงเท่านั้น,
ห้ามตีความ"). Same model, same shots, same decoding. Leak-free composite
**0.7110** (venv run, exp51 — RougeL 0.4858 / SS 0.8574 / IoU 0.7754),
+0.0023 over v16's 0.7087. The sharper "facts only, no interpretation"
directive lifts IoU (0.7410 → 0.7754) by trimming over-cited refs.

This is the SINGLE-MODEL v16 lineage (one ~30 GB A3B image), kept as a
proven-buildable, H100-safe alternative to the v17–v21 two-stage hybrid
(~50 GB, both 27B-FP8 + 32B-AWQ weights) that lives on `master`. The
hybrid scores higher (0.7215) but has been hard to build/ship; this
branch is the low-risk fallback.

Why this model on a 40 GB GPU:
- A3B = 30 B params but 3 B activated per token (MoE). FP8-e4m3 weights
  total 29.54 GiB → fits 40 GB GPU with ~6 GB KV-cache headroom (vLLM
  reported 73,744-token KV size at gpu_memory_utilization=0.95).
- A100 has no native FP8 cores → vLLM picks MarlinFP8ScaledMMLinearKernel
  (linear) + MARLIN Fp8 MoE backend, dequantises FP8 → FP16 on the fly.
- Multimodal vision blocks load idle for text-only chat;
  limit_mm_per_prompt={"image":0,"video":0} stops vLLM from reserving
  the encoder cache budget (was eating ~5 GB on prior 35B-A3B-FP8 try).
- gpu_memory_utilization 0.95: confirmed safe in the exp42/exp51 venv
  runs (peak ~36 GB on a 40 GB GPU).

H100-safe (carry over the v20 fix): the benchmark backend runs on H100
(SM 9.0); LANTA is A100-only so no local test exercises the Hopper path.
On H100 vLLM's Hopper-only FP8 fast path — DeepGEMM (block-FP8 GEMM) —
JIT-compiles CUDA kernels with nvcc at first use, which dies on the
runtime base image (`nvcc: not found`). VLLM_USE_DEEP_GEMM=0 (set below
before any torch/vllm import) + the default kv_cache_dtype="auto" (no
FlashInfer fp8 KV) force the precompiled Marlin/Triton FP8 path that the
A100 sim already validates. Costs the H100 FP8 speed bonus; output
unchanged. This image needs NO nvcc — keeps the SIF lean (~30 GB).

v14→v16 infra (carry over):
- Sort queries by doc_id before submission so the ~14K-token full-doc
  prefix hits vLLM's prefix cache across all queries from the same
  doc (cache-hit ceiling ~90%).
- LLMEngine.step() streaming instead of llm.generate() — the benchmark
  `progress` binary fires per finished request, so the backend sees a
  real heartbeat.
- enforce_eager=True keeps the V1-engine torch.compile path off
  (Apptainer-incompatible) and costs ~0 latency at this batch size.
- MAX_NEW_TOKENS 1024 (matches exp51 venv config; long multi-ref
  answers from shot 2 sometimes truncate at 512).

Output: submission.csv with columns ID, abstractive, refs — written in
the *original* queries order (sort is internal only).
"""
from pathlib import Path
import os
import re
import json
import csv
import time

# --- Hopper/H100: force the precompiled FP8 path (no runtime nvcc) -----------
# The base image is a CUDA *runtime* (no nvcc / no CUDA toolkit). On the
# benchmark's H100, vLLM's Hopper-only FP8 fast path — DeepGEMM (block-FP8
# GEMM) — JIT-compiles CUDA kernels with nvcc+ninja at first use, which dies
# with `nvcc: not found` / `ninja: build stopped` and the EngineCore exits.
# The A100 backend-sim never hits this: SM 8.0 has no native FP8, so vLLM uses
# the precompiled Marlin/Triton FP8 path. Disabling DeepGEMM here (env read at
# engine init) + keeping kv_cache_dtype="auto" forces that same precompiled
# path on H100, so the shared SIF runs identically on A100 and H100. Must be
# set BEFORE the first torch/vllm import. (Mirrors the v20 fix, 085e314.)
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from transformers import AutoTokenizer
from vllm import LLMEngine, EngineArgs, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_NEW_TOKENS         = 1024
MAX_MODEL_LEN          = int(os.environ.get("MAX_MODEL_LEN", "32768"))
# 16384 = sweet spot for the prior 32B-AWQ build. Kept for A3B-FP8 too:
# the MoE intermediate dim per expert is smaller (~2048) than 32B-AWQ's
# 27648, so MLP activations are cheaper; the 14K-tok median prompt is
# still single-chunk and the 28K-tok max prompt still fits 2 chunks.
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"))
# 0.95 confirmed safe on the exp42/exp51 venv runs: 29.54 GiB weights +
# 6.75 GiB KV cache + activations on a 40 GB GPU → peak ~36 GB.
GPU_MEM_UTIL           = float(os.environ.get("GPU_MEM_UTIL", "0.95"))
MODEL_NAME             = os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Two worked few-shot examples (both from held-out doc_050), rendered in
# E5 form: a 5-paragraph numbered context and an answer ending with the
# [อ้างอิง: N] tag.
#
# Shot 1 (exp08 carry-over): single-ref — matches the 71.8% single-ref
# dataset prior, teaches "cite exactly the source paragraph."
# Shot 2 (v15-K new): multi-ref subset — replaces exp08's single-ref
# shot2 because the 153 missed-tag queries in the v15 train eval were
# overwhelmingly multi-ref gold (model emits a comprehensive answer but
# forgets to cite anything). Q0746 from doc_050: 4 of 5 context paragraphs
# are gold (P21-P24, absentee list); P20 is a same-section distractor
# (attendee #12). Teaches "structured answer covers several paragraphs
# → cite them all" + "don't cite paragraphs the answer didn't use."
_SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
_SHOT1_PARAS = [
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "_________________________",
    "กรรมาธิการผู้มาประชุม",
]
_SHOT1_ANSWER = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา [อ้างอิง: 3]"

_SHOT2_QUERY = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมมีจำนวนกี่คน"
_SHOT2_PARAS = [
    "๑๒. นางสาวแอนศิริ วลัยกนก กรรมาธิการ",
    "กรรมาธิการผู้ไม่มาประชุม",
    "๑. นายพิบูลย์ รัชกิจประการ (ลาการประชุม)",
    "๒. นายธนยศ ทิมสุวรรณ (ลาการประชุม)",
    "๓. นายอัคร ทองใจสด (ลาการประชุม)",
]
_SHOT2_ANSWER = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมจำนวน 3 คน [อ้างอิง: 2, 3, 4, 5]"


def benchmark_lib(i):
    os.system(f"{PROGRESS_LIB} {i}")


def load_data(test_dir):
    with open(Path(test_dir) / "test.json", encoding="utf-8") as f:
        return json.load(f)


def filter_valid_paragraphs(paragraphs):
    def is_valid(p):
        text = p["text"].strip()
        if not text:
            return False
        if set(text) <= set("_-=. \t\n"):
            return False
        return True
    return [p for p in paragraphs if is_valid(p)]


def build_prompt(query, paras):
    """V10_factual prompt — CONTEXT FIRST, then query, then instruction.

    The exp51 wording: answer "สั้นและตรงประเด็น" (short, to the point),
    "ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความ" (state only facts
    present in the paragraphs, no interpretation). Sharper than v16's
    exp38 "กระชับและครอบคลุม" — lifts IoU by trimming over-cited refs.

    Context-first lets vLLM's prefix cache match the full ~14K-token doc
    block across all queries from the same doc (the query is the
    divergence point, not the cache-killer at the start). Instruction
    stays at the end for recency bias on the citation directive.
    """
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages(query, paras):
    """System + 2 few-shot turns + the final user turn.

    Every user turn uses the same build_prompt; the few-shot assistant
    turns carry the worked [อ้างอิง: N] answer.
    """
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user", "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user", "content": build_prompt(query, paras)},
    ]


def parse_citation(text, n_paras):
    """0-indexed paragraph indices from ALL [อ้างอิง...] tags.

    The model emits one tag per item in a multi-part answer; re.search
    (first tag only) under-counts refs, so collect every tag's numbers.
    """
    nums = []
    for grp in re.findall(r'\[อ้างอิง[:\s]+([0-9,\s]+)\]', text):
        nums += [int(x) for x in re.findall(r'\d+', grp)]
    valid, seen = [], set()
    for num in nums:
        if 1 <= num <= n_paras and num not in seen:
            seen.add(num)
            valid.append(num - 1)
    return valid or [0]  # fallback: first paragraph


def split_answer_citation(text):
    """Return (answer, raw_citation_tag).

    answer = text with EVERY [อ้างอิง...] tag removed. The model emits a
    tag inline after each item in multi-part answers, not only at the
    end, so stripping from the last tag alone (rfind) leaks the earlier
    tags into abstractive — gold answers carry no such tag.
    """
    idx = text.rfind('[อ้างอิง')
    raw_tag = text[idx:] if idx != -1 else ""
    answer = re.sub(r'\s*\[อ้างอิง[^\]]*\]', '', text).strip()
    return answer, raw_tag


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"v16.1 (exp51: A3B + V10_factual) — {n} queries, {len(doc_index)} docs "
          f"(NO RETRIEVAL — full doc, max_model_len={MAX_MODEL_LEN}, "
          f"gpu_mem_util={GPU_MEM_UTIL}, max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS})",
          flush=True)
    benchmark_lib(0)

    # Pre-build full-doc paragraph lists (in document order) once per doc.
    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # Sort indices by doc_id so queries from the same doc are submitted
    # contiguously — vLLM's prefix cache then reuses the full-doc prefilled
    # KV blocks across them. CSV is written in original queries order at
    # the end (sort is internal only).
    order = sorted(range(n), key=lambda i: queries[i]["doc_id"])

    items = []  # one tuple per query, in sorted submission order
    pool_sizes = []
    for idx in order:
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        if valid:
            gen_pids  = [p["para_id"] for p in valid]
            gen_texts = [p["text"]    for p in valid]
            pool_sizes.append(len(valid))
            messages = build_messages(q_text, gen_texts)
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((idx, query["ID"], gen_pids, gen_texts, messages, q_text))

    if pool_sizes:
        print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # FP8 quantization auto-detected from the model's config.json
    # (quant_method=fp8); vLLM picks MarlinFP8ScaledMMLinearKernel on A100
    # and (with VLLM_USE_DEEP_GEMM=0) the same precompiled path on H100.
    # kv_cache_dtype defaults to "auto" (FP16 KV) — no FlashInfer fp8 path,
    # so nothing JIT-compiles at runtime. limit_mm_per_prompt={"image":0,
    # "video":0} stops vLLM from reserving the multimodal encoder cache
    # (was ~5 GB on the prior 35B-A3B-FP8 attempt → KV cache went negative).
    engine_args = EngineArgs(
        model=MODEL_NAME,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        dtype="bfloat16", enforce_eager=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    # Load tokenizer separately via AutoTokenizer — vllm 0.19.1's
    # engine.get_tokenizer() exists but AutoTokenizer keeps init order
    # identical between container (0.19.1) and venv (0.19.1).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    engine = LLMEngine.from_engine_args(engine_args)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                              repetition_penalty=1.05)

    # Add all requests up-front; request_id encodes the submission position
    # so we can map outputs back to items[].
    for k, it in enumerate(items):
        msgs = it[4] if it[4] is not None else [{"role": "user", "content": it[5]}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        engine.add_request(request_id=str(k), prompt=prompt, params=sampling)

    # Stream completions: call benchmark_lib on each finished request so the
    # benchmark backend sees a real heartbeat (was: one batch ping at the
    # very end after llm.generate() returned, ~30-60min of silence).
    raw_by_k = {}
    n_done = 0
    t0 = time.time()
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                raw_by_k[int(o.request_id)] = o.outputs[0].text.strip()
                n_done += 1
                benchmark_lib(n_done)
                if n_done == 1 or n_done % 50 == 0 or n_done == n:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (n - n_done) / max(rate, 1e-6)
                    print(f"  [{n_done}/{n}] done — {rate:.2f} q/s, eta {eta:.0f}s",
                          flush=True)

    # Parse outputs (still in sorted item order)
    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    n_explicit = 0
    ref_counts = []
    results_by_qid = {}
    for k, it in enumerate(items):
        _idx, qid, gen_pids, gen_texts, _, q_text = it
        raw = raw_by_k[k]
        answer, _ = split_answer_citation(raw)
        cited_idx = parse_citation(raw, len(gen_pids))
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]
        else:
            ref_ids = []
        if cite_re.search(raw):
            n_explicit += 1
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
        ref_counts.append(len(ref_ids))
        results_by_qid[qid] = {"ID": qid, "abstractive": answer,
                               "refs": ",".join(ref_ids)}

    print(f"citations: {n_explicit}/{len(items)} emitted an [อ้างอิง: …] tag, "
          f"{len(items) - n_explicit} fell back to top-1", flush=True)
    if ref_counts:
        print(f"avg refs/query: {sum(ref_counts) / len(ref_counts):.2f}", flush=True)

    # Write CSV in ORIGINAL queries order (not sorted submission order)
    results = [results_by_qid[q["ID"]] for q in queries]
    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Written {len(results)} rows to {out_path}", flush=True)
    return n


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)   # final "fully done, CSV ready" ping
