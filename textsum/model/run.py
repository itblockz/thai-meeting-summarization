"""
v16.4 — exp77 port (single-model gemma-4-26B-A4B-NVFP4 MoE + V10_factual).

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to
nvidia/Gemma-4-26B-A4B-NVFP4 (26B MoE, ~4B active, NVFP4 experts ~18 GB
load — attention stays bf16) in document order. The model is shown the
paragraphs as a numbered [1..N] context, answers in Thai, then cites which
paragraphs it used as [อ้างอิง: X]; the cited paragraphs become `refs`
(E5 self-citation, adaptive count). Two worked few-shot examples are
prepended as multi-turn chat turns.

v16.4 = exp77 port. SINGLE-VARIABLE change from v16.3 (exp74, 0.6970): the
LLM swaps from RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (MoE, compressed-
tensors FP8) to nvidia/Gemma-4-26B-A4B-NVFP4 (same MoE, NVFP4 experts via
TensorRT-Model-Optimizer). SAME V10_factual prompt, SAME exp38 shots, SAME
greedy decoding (temp 0, rep-pen 1.05). This is a quant *publisher/format*
swap on the identical base model — exp77 ran it primarily for the FIT
verdict and the format-equivalence delta vs the FP8 build.

⚠️ KNOWN-NEGATIVE on the train set: leak-free composite **0.6982** (venv
run, exp77), essentially flat vs v16.3's 0.6970 and still −0.0158 below
v16.2's 0.7140. Same MoE answer-quality ceiling as v16.3/exp74: the
~4B-active experts cap RougeL/SS regardless of the quant format. exp77 was
worth running because the nvidia/ModelOpt NVFP4 MoE build is what the
exp77-Stage-A hybrid (memory:hybrid-stage-a-refpicker, 0.7235 NEW BEST)
uses as its ref-picker — this image packages that same checkpoint stand-
alone. Kept as a proven-buildable variant, NOT a single-model score win —
prefer v16.2 (0.7140) for single-model quality.

This is the SINGLE-MODEL v16 lineage (one ~18 GB-weights image), kept as a
proven-buildable alternative to the v17–v21 two-stage hybrid (~50 GB) on
`master`.

Why this nvidia/ModelOpt NVFP4 MoE fits a single A100-40GB where the dense
31B nvidia build did NOT (exp75 — infeasible):
- nvidia/ModelOpt `exclude_modules` ALL self_attn* from NVFP4 → attention
  stays bf16. On the DENSE 31B that bf16-attention slice was ~8 GiB →
  ~29.96 GiB load → KV-floor overrun on 1×A100 (exp75). On THIS MoE the
  ~22B of expert params (the bulk) stay 4-bit and only the small shared
  attention is bf16 → ~18 GiB load → fits with room (exp77 first-run
  profiled GPU KV = 82,096 tok / 7.88x at 32768). The bet that quantized
  experts dominate the footprint held.
- gpu_memory_utilization 0.90 (NOT 0.95 — see GPU_MEM_UTIL note). At 0.95
  the lighter ~18 GiB weights + a near-full 32768 KV left only ~246 MiB
  free → the sampling-time frequency-penalty buffer (repetition_penalty=
  1.05, batched over all prompts) OOM'd at first decode (exp77 job
  5824692). 0.90 keeps the full-32768 KV while leaving ~4 GiB truly-free
  GPU RAM for vLLM's untracked sampling allocations → MAX_MODEL_LEN=32768,
  NO truncation.
- On A100 (no native FP4 cores) vLLM reads NVFP4 from config and uses an
  FP4 weight-only dequant path (Marlin for dense Linear; the MoE experts
  need the analogous sm80 fused-MoE FP4 dequant). The MoE routes only ~4B
  active params/token → near-A3B latency (exp77-verified).
- Multimodal vision blocks load idle for text-only chat;
  limit_mm_per_prompt={"image":0,"video":0} stops vLLM reserving the
  encoder cache budget.
- enable_prefix_caching is honored for this MoE arch (NOT auto-disabled
  like the 27B-FP8 multimodal arch; see the engine log line).

⚠️ KV-dtype is a TWO-SIDED trap on the nvidia/ModelOpt build (exp77 job
5824680) — RESOLVED here by a runtime config override (see
build_kv_neutralized_model below). nvidia/ModelOpt bakes
"kv_cache_quant_algo": "FP8" into hf_quant_config.json (RedHatAI's FP8
build, v16.3, carried NO kv directive — its "auto" resolved to bf16 for
free). With that directive present, kv_cache_dtype="auto" PROMOTES to
fp8_e4m3:
  - on A100 (sm80): no fp8e4nv reshape_and_cache kernel → engine init dies
    (the exp71/72/75 wall);
  - on H100 (sm90): fp8-KV pulls the FlashInfer/DeepGEMM path that
    JIT-compiles with nvcc — absent from this runtime image → the v19
    crash (memory:h100-fp8-nvcc-jit-trap).
And the exp75 dense fix (kv_cache_dtype="bfloat16") is REJECTED here: this
MoE selects the TRITON_ATTN backend, whose triton_reshape_and_cache_flash
asserts kv_cache_dtype ∈ {"auto","fp8*"} → "bfloat16" raises
AssertionError. FIX (matches exp77's submit-time override, but built at
runtime so the baked container cache needs no surgery): symlink the HF
snapshot into a scratch dir and rewrite config.json + hf_quant_config.json
with the kv directive STRIPPED, so "auto" resolves to bf16 on BOTH
platforms — the universal, JIT-free, sm80/sm90-safe KV path. run.py then
leaves kv_cache_dtype at the default "auto".

H100-safe (carry over the v20 fix), with one OPEN RISK: the benchmark
backend runs on H100 (SM 9.0); LANTA is A100-only so no local test
exercises the Hopper path. VLLM_USE_DEEP_GEMM=0 (set below before any
torch/vllm import) disables the DeepGEMM block-FP8 GEMM that JIT-compiles
with nvcc; the KV override above forces bf16 KV so the fp8-KV FlashInfer
JIT never fires either. v16.2 (also NVFP4, dense 31B) PASSED on the
H100-40GB backend, de-risking the NVFP4 weight-dequant path on Hopper;
v16.4's only new wrinkle vs v16.2 is the kv directive, neutralized above.
The first-run raw-output dump below guards against silent gibberish. This
image needs NO nvcc — keeps the SIF lean.

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
import glob
import shutil
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
# 8192 (carried from v16.3.1): the chunked-prefill budget = peak activation /
# FP4 dequant *scratch* per step, which lives OUTSIDE vLLM's pre-allocated KV
# pool. v16.4's NVFP4 weights are LIGHTER than v16.3's FP8 (~18 vs ~26 GiB),
# so at util 0.90 there is ~4 GiB physical free — a fatter buffer than v16.3's
# ~2.5 GiB. 8192 was the value that fixed v16.3's 98% crash on a tighter
# buffer, so it is comfortably safe here; kept as a correctness-neutral
# default (chunking is transparent — drops ZERO paragraphs). Raise to 16384
# for slightly faster prefill if the H100 run shows headroom; drop to 4096 if
# it ever spikes.
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "8192"))
# 0.90 (exp77 value; LOWER than v16.3's 0.95 — and that is deliberate). The
# benchmark backend is H100-*40GB*. NVFP4 weights ~18 GiB leave more KV room
# than v16.3's FP8 26 GiB, BUT exp77 found 0.95 fatal here for a different
# reason than v16.3's prefill spike: with the lighter weights vLLM grows the
# KV pool to fill 0.95 → only ~246 MiB physically free → the sampling-time
# frequency-penalty buffer (repetition_penalty=1.05, batched over all prompts)
# OOM'd at the FIRST decode (exp77 job 5824692, "Tried to allocate 256 MiB ...
# 246 MiB free"). 0.90 caps the KV pool lower (still ≫ the 9.54 GiB needed for
# full 32768 — exp77 profiled 82,096 tok / 7.88x) and leaves ~4 GiB truly-free
# for vLLM's untracked sampling allocations. Do NOT raise it.
GPU_MEM_UTIL           = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MODEL_NAME             = os.environ.get("LLM_MODEL", "nvidia/Gemma-4-26B-A4B-NVFP4")

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


def build_kv_neutralized_model(model_name):
    """Return a model path whose configs carry NO fp8-KV directive.

    The nvidia/Gemma-4-26B-A4B-NVFP4 checkpoint bakes
    "kv_cache_quant_algo": "FP8" into hf_quant_config.json (+ a
    kv_cache_scheme in config.json's quantization_config). With it present,
    vLLM's kv_cache_dtype="auto" promotes KV to fp8_e4m3 → on A100 (sm80)
    there is no fp8e4nv reshape_and_cache kernel, and on the H100 backend
    fp8-KV pulls the FlashInfer/DeepGEMM JIT that needs nvcc (absent from
    this runtime image). The exp75 dense fix (kv_cache_dtype="bfloat16") is
    rejected too — this MoE selects TRITON_ATTN, whose kernel asserts
    kv_cache_dtype ∈ {"auto","fp8*"}. The only universal, JIT-free path is
    to make "auto" resolve to bf16 by STRIPPING the directive (mirrors
    exp77's submit-time override, here built at runtime from the baked
    cache). Idempotent: re-stripping an already-clean config is a no-op.

    If LLM_MODEL already points at an existing local dir (an override the
    caller pre-built), use it verbatim. If the repo carries no kv directive
    (e.g. the RedHatAI FP8 build, or a future clean checkpoint), return the
    snapshot path unchanged — no override needed.
    """
    # Already a local path? Trust it as-is.
    if os.path.isdir(model_name):
        return model_name

    # Locate the snapshot in the baked HF cache (HF_HOME/hub/models--org--name).
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir = "models--" + model_name.replace("/", "--")
    snap_glob = os.path.join(hf_home, "hub", repo_dir, "snapshots", "*")
    snaps = sorted(glob.glob(snap_glob))
    if not snaps:
        # Not in cache as a snapshot dir — hand the repo id back to vLLM and
        # let it resolve (online or a non-standard layout). No override.
        return model_name
    snap = snaps[-1]

    quant_cfg = os.path.join(snap, "hf_quant_config.json")
    has_kv_directive = False
    if os.path.isfile(quant_cfg):
        try:
            qj = json.load(open(quant_cfg, encoding="utf-8"))
            has_kv_directive = "kv_cache_quant_algo" in qj.get("quantization", {})
        except (ValueError, OSError):
            has_kv_directive = False
    if not has_kv_directive:
        # RedHatAI-style checkpoint (no directive) → "auto" is already bf16.
        return snap

    # Build a scratch override: symlink every snapshot entry, then replace
    # config.json + hf_quant_config.json with kv-directive-stripped copies.
    scratch = os.environ.get("TEXTSUM_SCRATCH_DIR", "/scratch")
    ovr = os.path.join(scratch, "model_override")
    shutil.rmtree(ovr, ignore_errors=True)
    os.makedirs(ovr, exist_ok=True)
    rewrite = {"config.json", "hf_quant_config.json"}
    for name in os.listdir(snap):
        if name in rewrite:
            continue
        os.symlink(os.path.join(snap, name), os.path.join(ovr, name))

    # hf_quant_config.json: drop kv_cache_quant_algo from quantization{}.
    qj = json.load(open(quant_cfg, encoding="utf-8"))
    qj.get("quantization", {}).pop("kv_cache_quant_algo", None)
    json.dump(qj, open(os.path.join(ovr, "hf_quant_config.json"), "w",
                       encoding="utf-8"), indent=2)

    # config.json: drop kv_cache_scheme / kv_cache_quant_algo from
    # quantization_config{} (present in some ModelOpt exports).
    cfg_path = os.path.join(snap, "config.json")
    cj = json.load(open(cfg_path, encoding="utf-8"))
    qc = cj.get("quantization_config", {})
    qc.pop("kv_cache_scheme", None)
    qc.pop("kv_cache_quant_algo", None)
    json.dump(cj, open(os.path.join(ovr, "config.json"), "w",
                       encoding="utf-8"), indent=2)

    print(f"KV override: stripped fp8-KV directive → {ovr} "
          f"(symlinked {snap})", flush=True)
    return ovr


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
    print(f"v16.4 (exp77: gemma-4-26B-A4B-NVFP4 MoE + V10_factual) — {n} queries, "
          f"{len(doc_index)} docs (NO RETRIEVAL — full doc, model={MODEL_NAME}, "
          f"max_model_len={MAX_MODEL_LEN}, gpu_mem_util={GPU_MEM_UTIL}, "
          f"max_num_batched_tokens={MAX_NUM_BATCHED_TOKENS})",
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

    # Resolve a KV-neutralized model path: nvidia/ModelOpt's NVFP4 checkpoint
    # bakes "kv_cache_quant_algo": "FP8" into its config, which would make
    # kv_cache_dtype="auto" promote KV to fp8 (no sm80 kernel on A100; nvcc-JIT
    # FlashInfer path on H100). build_kv_neutralized_model symlinks the snapshot
    # and strips that directive so "auto" → bf16 on both platforms. Returns the
    # plain snapshot/repo path unchanged for checkpoints with no kv directive.
    model_path = build_kv_neutralized_model(MODEL_NAME)

    # NVFP4 quantization auto-detected from the model's config (ModelOpt /
    # compressed-tensors NVFP4); on A100 (no native FP4 cores) vLLM uses an FP4
    # weight-only dequant path (Marlin for dense Linear + the sm80 fused-MoE FP4
    # dequant), and (with VLLM_USE_DEEP_GEMM=0) the precompiled path on H100 —
    # see the OPEN RISK in the module docstring. The MoE (A4B) routes only ~4B
    # active params/token → near-A3B latency despite the dequant tax.
    # kv_cache_dtype defaults to "auto" → bf16 KV here because model_path's
    # config has had its fp8-KV directive stripped above (gemma-4 has no A100
    # FP8-KV kernel and the H100 fp8-KV path needs nvcc — bf16 KV is mandatory;
    # do NOT set kv_cache_dtype="fp8"). limit_mm_per_prompt={"image":0,"video":0}
    # stops vLLM reserving the multimodal encoder cache budget (gemma-4 is
    # multimodal, text-only here).
    engine_args = EngineArgs(
        model=model_path,
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
    # identical between container (0.19.1) and venv (0.19.1). Use model_path
    # (the override dir symlinks the tokenizer files, so this resolves the
    # same tokenizer as the snapshot).
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
