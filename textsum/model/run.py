"""
v22 — exp56 hybrid with the Stage-A ref-picker swapped to gemma-4-31B-NVFP4
(the exp73 model). H100 40GB single-GPU optimised.

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to
both stages in document order. Two sequential LLMs share one 40 GB GPU
(weights together ≈ 22 + 18 = 40 GB at the edge, so they cannot coexist —
loaded one at a time with del + empty_cache between):

  Stage A — RedHatAI/gemma-4-31B-it-NVFP4 (4-bit NVFP4 weights ≈ 22 GB via
            NvFp4LinearBackend.MARLIN) with the V10_factual prompt + the
            exp38 multi-ref shot pair. Picks *refs* via [อ้างอิง: …] tags
            from the full doc. (v21 used Qwen3.6-27B-FP8 here.)
  --- free weights, gc.collect(), torch.cuda.empty_cache() ---
  Stage B — Qwen/Qwen3-32B-AWQ (INT4 AWQ-Marlin, ≈ 18 GB) with exp38's
            E5 prompt + a "เน้นย่อหน้าหมายเลข [X, Y, Z]" hint pointing
            at Stage A's selection. Writes the abstractive answer on
            the FULL doc context. Its emitted citations are parsed but
            DISCARDED — final `refs` are FIXED to Stage A's selection.

Leak-free composite **0.7215** on A100 (exp56, +0.0228 over v16's 0.7087).

H100 40 GB optimisations (vs the A100-targeted exp56 reference):
  - Native FP8 GEMM: A100 has no FP8 tensor cores, so 27B-FP8 falls back
    to MarlinFP8 dequant kernels (~1.5–2× slower). H100 SM 9.0 runs the
    FP8 GEMM directly; vLLM auto-selects the native path.
  - FlashAttention 3: H100-only TMA + WGMMA attention path, 1.5–2× faster
    than FA2 on the long-context (~14K-tok) doc prompts.
  - FP8 KV cache (`kv_cache_dtype="fp8"`): H100-only, halves KV cache
    memory → roughly 2× concurrent prompts in flight.
  - gpu_memory_utilization 0.92 (vs A100's 0.90): H100 firmware is
    stable enough at higher mem util. Stage A budget: 36.8 GB → 30 GB
    weights + ~5–6 GB FP8 KV. Stage B: 18 GB weights + ~17 GB FP8 KV.
  - enforce_eager remains True: vllm 0.19.1's V1 engine still segfaults
    inside Apptainer (verified at v15 bump attempt, job 5787048 —
    FLASHINFER / TRITON_ATTN / FLEX_ATTENTION all SIGSEGV after model
    load). `VLLM_USE_V1=0` (set in textsum.def) pins V0 in-process
    workers. Eager mode in V0 costs ~0 latency at this batch size.
  - max_num_batched_tokens 16384: long-context (full doc ≈ 14K tok)
    throughput is bound by prefill chunking, not decode.

A100 fallback (auto-detected): `kv_cache_dtype="auto"` keeps FP16 KV
and we stay on MarlinFP8 dequant. Runtime ≈ exp56's measured 2h45m on
A100. The same SIF runs on either GPU — no rebuild required.

Doc-grouping (carry over from v16): both stages submit queries sorted
by doc_id so vLLM's prefix cache reuses the full-doc prefilled KV blocks
across all queries from the same doc (~5% → ~90% cache hit ceiling).
CSV is written in the ORIGINAL queries order at the end.

Output: submission.csv with columns ID, abstractive, refs.
"""
from pathlib import Path
import os
import re
import json
import csv
import gc
import time
import shutil

# --- Writable scratch for Triton's JIT kernel cache (MUST run before torch/
# vllm/triton import) ---------------------------------------------------------
# Qwen3.6-27B is a qwen3_next (linear-attention / Mamba-GDN) model: it
# runtime-compiles many Triton FLA kernels at startup and writes them to a
# cache dir. The benchmark backend launches the container with `--containall`,
# whose session /tmp (and read-only /root/.triton) is a tiny tmpfs — the cache
# write hits `OSError: [Errno 28] No space left on device` and the vLLM
# EngineCore dies during profile_run (verified: job 5815781).
#
# v18 briefly anchored TMPDIR under RESULT_DIR. That fixed ENOSPC but polluted
# the result upload tree with runtime temp/IPC files (e.g. sockets under
# .cache/tmp), and the benchmark uploader later failed with "no such device or
# address" before any progress was reported. v19 therefore keeps all runtime
# scratch outside RESULT_DIR. Local Apptainer tests use a sibling of RESULT_DIR;
# the Docker image provides /scratch for the benchmark backend.
_result_dir_for_cache = Path(os.environ.get("RESULT_DIR", "/result/")).resolve()


def _is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
    except OSError:
        return False


def _is_writable_dir(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        marker = path / ".write_test"
        with open(marker, "w", encoding="utf-8") as f:
            f.write("ok")
        marker.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _runtime_cache_candidates(result_dir):
    for var in ("TEXTSUM_SCRATCH_DIR", "AIBENCHMARK_SCRATCH_DIR", "SCRATCH_DIR"):
        root = os.environ.get(var)
        if root:
            yield Path(root) / "textsum-runtime-cache"

    # If RESULT_DIR is a real path such as /lustre/.../result, its parent is the
    # safest large writable filesystem while staying outside the upload root.
    if result_dir.parent != result_dir and str(result_dir.parent) != os.sep:
        yield result_dir.parent / "textsum-runtime-cache"

    # Docker benchmark path. The Dockerfile creates /scratch with mode 1777.
    yield Path("/scratch/textsum-runtime-cache")
    yield Path("/var/tmp/textsum-runtime-cache")
    yield Path("/tmp/textsum-runtime-cache")
    yield Path("/textsum-runtime-cache")


def _configure_runtime_cache():
    seen = set()
    for candidate in _runtime_cache_candidates(_result_dir_for_cache):
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if _is_relative_to(candidate, _result_dir_for_cache):
            continue
        if _is_writable_dir(candidate):
            break
    else:
        # Last resort: allow the job to attempt inference, but keep this path
        # clean before exit so successful runs upload only submission.csv.
        candidate = _result_dir_for_cache / ".cache" / "runtime"
        candidate.mkdir(parents=True, exist_ok=True)

    # The first three are the original Triton/XDG/tmp anchors. The rest (v21)
    # are for the Hopper FP8 JIT path — DeepGEMM, FlashInfer and torch's
    # cpp_extension all compile CUDA at first use and write caches under
    # HOME/.cache, TORCH_EXTENSIONS_DIR, or CUDA_CACHE_PATH; under --containall
    # those default to read-only or tiny tmpfs → ENOSPC/permission failure.
    # Setting HOME here also redirects any ~/.cache/<tool> path we didn't name
    # explicitly. HF_HOME is set separately in the image, so the baked weights
    # are unaffected by the HOME change.
    for _sub, _var in (("triton",     "TRITON_CACHE_DIR"),
                       ("xdg",        "XDG_CACHE_HOME"),
                       ("tmp",        "TMPDIR"),
                       ("torch_ext",  "TORCH_EXTENSIONS_DIR"),
                       ("cuda",       "CUDA_CACHE_PATH"),
                       ("flashinfer", "FLASHINFER_WORKSPACE_BASE"),
                       ("home",       "HOME")):
        _d = candidate / _sub
        _d.mkdir(parents=True, exist_ok=True)
        os.environ[_var] = str(_d)
    print(f"Runtime cache dir: {candidate}", flush=True)
    return candidate


def _cleanup_result_cache():
    cache_dir = _result_dir_for_cache / ".cache"
    if cache_dir.exists() and _is_relative_to(cache_dir, _result_dir_for_cache):
        shutil.rmtree(cache_dir, ignore_errors=True)


def _cleanup_runtime_cache():
    if _runtime_cache_dir.exists():
        shutil.rmtree(_runtime_cache_dir, ignore_errors=True)


_cleanup_result_cache()
_runtime_cache_dir = _configure_runtime_cache()
for _sub, _var in (("triton", "TRITON_CACHE_DIR"),
                   ("xdg",    "XDG_CACHE_HOME"),
                   ("tmp",    "TMPDIR")):
    _d = Path(os.environ[_var])
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

# --- Hopper/H100: KEEP the native FP8 fast path (DeepGEMM + FlashInfer) -------
# v19 crashed on the H100 benchmark backend because those Hopper-only kernels
# JIT-compile with nvcc+ninja at first use and the runtime base image had no
# nvcc (`nvcc: not found` / `ninja: build stopped`). v20 sidestepped it by
# forcing the precompiled path; v21 instead KEEPS the FP8 path and makes it
# work: the Dockerfile bakes cuda-toolkit (nvcc) into the image, and
# _configure_runtime_cache() below routes EVERY JIT/compile cache
# (TORCH_EXTENSIONS_DIR, CUDA_CACHE_PATH, FlashInfer workspace, HOME/.cache)
# to writable scratch so the compile succeeds under --containall. The A100
# backend-sim still exercises the precompiled path (SM 8.0 has no native FP8),
# so the Hopper JIT branch is validated only by the real H100 submission.
# (No VLLM_USE_DEEP_GEMM override — vLLM's Hopper default enables it.)

import torch
from transformers import AutoTokenizer
from vllm import LLMEngine, EngineArgs, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_NEW_TOKENS         = int(os.environ.get("MAX_NEW_TOKENS",         "1024"))
MAX_MODEL_LEN          = int(os.environ.get("MAX_MODEL_LEN",          "32768"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"))
GPU_MEM_UTIL           = float(os.environ.get("GPU_MEM_UTIL",         "0.92"))

# v22: Stage A ref-picker swapped Qwen3.6-27B-FP8 → gemma-4-31B-it-NVFP4
# (the exp73 model). Loads at ~22 GB via vLLM's NvFp4LinearBackend.MARLIN
# (auto-detected from config.json — no `quantization` arg). The Stage A
# extra_kwargs below (dtype="bfloat16" + limit_mm_per_prompt to skip the
# gemma-4 vision encoder) are already exactly exp73's loader, so only the
# model name changes. Stage B (answer-writer) stays 32B-AWQ.
MODEL_A   = os.environ.get("LLM_MODEL_STAGE_A", "RedHatAI/gemma-4-31B-it-NVFP4")
MODEL_AWQ = os.environ.get("LLM_MODEL_STAGE_B", "Qwen/Qwen3-32B-AWQ")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot pair — both from held-out doc_050, both stages reuse the same
# shot QUERY/PARA/ANSWER triples; only the wrapping prompt differs.
# Shot 1 = exp08 single-ref. Shot 2 = exp38 multi-ref (Q0746, 4 of 5
# context paras are gold).
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


def is_hopper_or_newer():
    """Return True if the visible GPU is H100 (SM 9.0) or newer.

    Drives the FP8-KV-cache + V1-friendly decisions below. We probe at
    runtime rather than at build time because the SIF is shared across
    A100 (LANTA local test) and H100 (benchmark backend) without rebuild.
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 9


def build_prompt_v10(query, paras):
    """Stage A — exp50/V10_factual prompt for ref-picking."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_exp38(query, paras):
    """Stage B shot template (no hint) — exp38's E5 prompt."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_exp38_with_hint(query, paras, hint_idx):
    """Stage B — exp38's E5 prompt + the 'focus on paragraphs [...]' hint
    pointing at Stage A's 1-based picks within the full doc.
    """
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    hint_str = ", ".join(str(i) for i in hint_idx) if hint_idx else "—"
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"**โดยเน้นย่อหน้าหมายเลข [{hint_str}] เป็นข้อมูลหลัก** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_v10(query, paras):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10(query, paras)},
    ]


def build_messages_exp38_hint(query, paras, hint_idx):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_exp38(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_exp38(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_exp38_with_hint(query, paras, hint_idx)},
    ]


def parse_citation(text, n_paras):
    """Collect 0-indexed paragraph indices from ALL [อ้างอิง...] tags."""
    nums = []
    for grp in re.findall(r'\[อ้างอิง[:\s]+([0-9,\s]+)\]', text):
        nums += [int(x) for x in re.findall(r'\d+', grp)]
    valid, seen = [], set()
    for num in nums:
        if 1 <= num <= n_paras and num not in seen:
            seen.add(num)
            valid.append(num - 1)
    return valid or [0]


def split_answer(text):
    """Strip every [อ้างอิง...] tag inline."""
    return re.sub(r'\s*\[อ้างอิง[^\]]*\]', '', text).strip()


def make_engine_args(model_name, extra_kwargs):
    """Build EngineArgs with H100-aware defaults.

    Both stages share gpu_memory_utilization, max_model_len,
    max_num_batched_tokens, enable_prefix_caching, enforce_eager,
    and (on H100) kv_cache_dtype="fp8". Stage-specific knobs
    (quantization, dtype, MM caps) come in via extra_kwargs.
    """
    # v21: re-enable the Hopper FP8 KV path (fp8 on H100, auto on A100). On
    # Hopper this pulls in FlashInfer, which JIT-compiles with nvcc — now
    # present in the image (cuda-toolkit baked in the Dockerfile) and with all
    # JIT caches routed to writable scratch (see _configure_runtime_cache). On
    # A100 "auto" keeps the precompiled path (the config that passed the sim).
    kv = "fp8" if is_hopper_or_newer() else "auto"
    base = dict(
        model=model_name,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=True,
        enforce_eager=True,        # V0 in Apptainer; no torch.compile path
        trust_remote_code=True,
        kv_cache_dtype=kv,
    )
    base.update(extra_kwargs)
    return EngineArgs(**base)


def run_stage(stage_label, model_name, extra_kwargs, items, progress_fn):
    """Load model, stream all `items` through engine.step(), free GPU.

    items: list of (key, messages) tuples — key is the stable index used
    to map the raw output back to the caller's data structures.

    progress_fn(n_done_in_stage): called once per finished request so
    the benchmark `progress` binary keeps emitting heartbeats during
    the stage. The caller decides how to translate stage-local progress
    into overall 0..N progress (see split between Stages A and B in
    main()).
    """
    print(f"\n=== {stage_label}: {model_name} ===", flush=True)
    print(f"  H100 detected: {is_hopper_or_newer()} | "
          f"kv_cache_dtype: {'fp8' if is_hopper_or_newer() else 'auto'} | "
          f"gpu_mem_util: {GPU_MEM_UTIL} | "
          f"max_num_batched_tokens: {MAX_NUM_BATCHED_TOKENS}",
          flush=True)

    engine = LLMEngine.from_engine_args(make_engine_args(model_name, extra_kwargs))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                              repetition_penalty=1.05)

    n = len(items)
    for key, msgs in items:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        engine.add_request(request_id=str(key), prompt=prompt, params=sampling)

    raw_by_key = {}
    n_done = 0
    t0 = time.time()
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                raw_by_key[int(o.request_id)] = o.outputs[0].text.strip()
                n_done += 1
                progress_fn(n_done)
                if n_done == 1 or n_done % 50 == 0 or n_done == n:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (n - n_done) / max(rate, 1e-6)
                    print(f"  [{stage_label} {n_done}/{n}] {rate:.2f} q/s, eta {eta:.0f}s",
                          flush=True)

    # Free GPU before the next stage's weights land.
    del engine, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return raw_by_key


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    half = n // 2
    print(f"v22 hybrid (exp56 port, gemma Stage A): {n} queries, {len(doc_index)} docs | "
          f"Stage A = {MODEL_A} | Stage B = {MODEL_AWQ} | "
          f"max_model_len={MAX_MODEL_LEN}",
          flush=True)
    benchmark_lib(0)

    # Pre-build full-doc paragraph lists (in document order) once per doc.
    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # Sort indices by doc_id so queries from the same doc are submitted
    # contiguously to BOTH stages — vLLM's prefix cache then reuses the
    # full-doc prefilled KV blocks across them. CSV is written in
    # ORIGINAL queries order at the end (sort is internal only).
    order = sorted(range(n), key=lambda i: queries[i]["doc_id"])

    # ----- Stage A items (key = position in sorted order) -----
    pool_sizes = []
    stage_a_items = []
    stage_a_gen_pids = {}
    stage_a_gen_texts = {}
    for k, idx in enumerate(order):
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        if valid:
            gen_pids  = [p["para_id"] for p in valid]
            gen_texts = [p["text"]    for p in valid]
            pool_sizes.append(len(valid))
            msgs = build_messages_v10(q_text, gen_texts)
        else:
            gen_pids, gen_texts = [], []
            msgs = [{"role": "user", "content": q_text}]
        stage_a_items.append((k, msgs))
        stage_a_gen_pids[k] = gen_pids
        stage_a_gen_texts[k] = gen_texts
    if pool_sizes:
        print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # Progress: Stage A occupies the 0..n/2 half of the overall heartbeat
    # range. Map stage-local n_done → overall via integer halving so the
    # benchmark sees a smooth monotonic count.
    raws_A = run_stage(
        "Stage A (refs)", MODEL_A,
        dict(dtype="bfloat16",
             limit_mm_per_prompt={"image": 0, "video": 0}),
        stage_a_items,
        progress_fn=lambda i: benchmark_lib(i // 2),
    )

    # Parse Stage A → refs + hint indices (per submission-order key k).
    stage_a_refs = {}   # k -> list of para_id strings
    stage_a_hint = {}   # k -> list of 1-based positions within full doc
    for k in range(n):
        raw = raws_A.get(k, "")
        gen_pids = stage_a_gen_pids[k]
        cited_idx = parse_citation(raw, len(gen_pids))
        valid_idx = [j for j in cited_idx if j < len(gen_pids)]
        ref_pids = [gen_pids[j] for j in valid_idx]
        if not ref_pids and gen_pids:
            ref_pids = [gen_pids[0]]
            valid_idx = [0]
        stage_a_refs[k] = ref_pids
        stage_a_hint[k] = [j + 1 for j in valid_idx]
    avg_refs_A = (sum(len(r) for r in stage_a_refs.values()) /
                  max(len(stage_a_refs), 1))
    print(f"Stage A: avg refs/query = {avg_refs_A:.2f}", flush=True)
    del stage_a_items, raws_A

    # ----- Stage B items (same submission order; carries the hint) -----
    stage_b_items = []
    for k, idx in enumerate(order):
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_texts = stage_a_gen_texts[k]
        if valid:
            msgs = build_messages_exp38_hint(q_text, gen_texts, stage_a_hint[k])
        else:
            msgs = [{"role": "user", "content": q_text}]
        stage_b_items.append((k, msgs))

    # Progress: Stage B occupies n/2..n half of the heartbeat range.
    raws_B = run_stage(
        "Stage B (answer)", MODEL_AWQ,
        dict(quantization="awq_marlin", dtype="half"),
        stage_b_items,
        progress_fn=lambda i: benchmark_lib(half + i // 2),
    )

    # ----- Assemble final results (refs FIXED to Stage A) -----
    empty_answers = 0
    results_by_qid = {}
    for k, idx in enumerate(order):
        query = queries[idx]
        qid = query["ID"]
        q_text = query["query"]
        gen_texts = stage_a_gen_texts[k]
        ref_pids = stage_a_refs[k]
        raw = raws_B.get(k, "")
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty_answers += 1
        results_by_qid[qid] = {"ID": qid, "abstractive": answer,
                               "refs": ",".join(ref_pids)}
    print(f"empty_answers={empty_answers}", flush=True)

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
    try:
        n = main()
        _cleanup_result_cache()
        benchmark_lib(n)   # final "fully done, CSV ready" ping
    finally:
        _cleanup_result_cache()
        _cleanup_runtime_cache()
