"""
v17.4 — COUPLED ref-hint pipe: gemma refs (Stage 1) HINT the 32B-AWQ answer
(Stage 2). Final = ANSWER from Stage 2, REFS from Stage 1 (= exp86 `ansB_refA`).

v17.4 = v17.3 + ref-INDEX hint to Stage 2. v17.3 ran the two single-model
images **independently** (no handoff) and column-merged their CSVs — that
realizes the "paper" column-merge ceiling (~0.7243) where exp37's cold answer
never saw a hint. v17.4 is the TRUE exp86 recipe: Stage 2's answer is actually
generated WITH Stage 1's cited paragraph indices as a hint line (exp81's
"ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]"), exactly as exp86 measured 0.7235
leak-free. Ref-hinting costs ~−0.0008 vs the invalid column-merge ceiling but
is the recipe that was actually scored (the answer was produced under the hint).

  Stage 1 — REFS source = v16.4 = nvidia/Gemma-4-26B-A4B-NVFP4 (26B MoE,
            ~4B active, NVFP4 experts ~18 GB load). Cold V10_factual + exp38
            shots, full doc. Record single-model citation IoU (exp77 0.8155).
            Emits refs AND a per-query HINT = 1-based positions of the paragraphs
            it cited (a sidecar JSON the orchestrator hands to Stage 2). Still
            byte-identical to v16.4 standalone (cold — the hint is derived from
            its citations, it does not change Stage 1's generation).
  Stage 2 — ANSWER source = Qwen/Qwen3-32B-AWQ (awq_marlin, dense). exp37 E5
            prompt + exp08 shot1 + exp38 multi-ref shot2, full doc, max_new 512,
            util 0.90. The final query turn now carries the exp81 hint line built
            from Stage 1's indices (the SHOTS stay non-hinted, as in exp86). NO
            LONGER byte-identical to v15.2 standalone — the hint is the point.

Final = 32B-AWQ hinted answer + gemma refs (`ansB_refA`, exp86's best cell).

The two stages do NOT share a prompt: Stage 1 (gemma) keeps V10_factual, Stage
2 (32B-AWQ) uses exp37's E5 prompt + the hint line. The shots (shot1 single-ref
+ shot2 multi-ref Q0746) are byte-identical across stages.

PROCESS MODEL — one OS process PER STAGE (the bit-identical fix)
---------------------------------------------------------------
This file runs in two modes:
  * ORCHESTRATOR (default, `python3 run.py`): imports NOTHING CUDA-related.
    It spawns `python3 run.py --stage 1`, waits for it to FULLY exit (Stage 1
    has by then written its refs CSV + the hint sidecar), spawns `--stage 2`,
    then merges the two stage CSVs into submission.csv.
  * WORKER (`python3 run.py --stage N`): loads exactly ONE model. Stage 1 writes
    a full single-model submission (== v16.4) PLUS the hint sidecar; Stage 2
    reads the hint sidecar and writes its hinted answer CSV. vLLM/torch/
    transformers are imported ONLY here, so the orchestrator never creates a
    CUDA context.

Why a subprocess per stage instead of `del engine; empty_cache()` in one
process: in V1 multiprocessing the EngineCore already runs in a child, so the
model's VRAM is reclaimed when that child exits — NOT by the parent's
`torch.cuda.empty_cache()`, which only ever created a ~0.4 GiB context in the
parent and left it resident. That residual context made Stage 2 measure
slightly less free VRAM than a standalone run → a marginally smaller KV pool →
different scheduler batching → rare greedy token flips. Running each stage as
its own process means the GPU is genuinely fresh for the second model: the
worker is born, loads one model on a clean card, writes its CSV, and dies. The
Stage 2 → Stage 1 handoff is the hint sidecar JSON on disk (RESULT_DIR), not
shared process state, so the GPU is still reclaimed by OS process teardown.

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed in
document order. Stage 1 (gemma) uses the cold V10_factual prompt; Stage 2
(32B-AWQ) uses exp37's E5 prompt + Stage 1's ref-index hint.

Container infra (carried from v15.2/v16.4):
 - VLLM_USE_DEEP_GEMM=0 (before any torch/vllm import): forces the precompiled
   CUTLASS/Marlin/NVFP4 path on the benchmark H100 (no nvcc JIT).
 - Stage 1 (gemma NVFP4) strips the baked fp8-KV directive via
   build_kv_neutralized_model so kv_cache_dtype="auto" → bf16 on A100 + H100.
   A no-op for the 32B-AWQ checkpoint.
 - Each worker streams via LLMEngine.step() and pings the benchmark `progress`
   binary per finished query (Stage 1 → 0..n/2, Stage 2 → n/2..n). The
   orchestrator pings 0 at the start and n at the very end.
 - Queries sorted by doc_id before submission (prefix-cache reuse); CSVs
   written in ORIGINAL query order.

Output: submission.csv with columns ID, abstractive, refs.
"""
from pathlib import Path
import os
import re
import json
import csv
import sys
import glob
import shutil
import time
import subprocess

# --- H100/no-nvcc: force the precompiled FP8/NVFP4 path (see docstring) ------
# Set in BOTH orchestrator and worker, BEFORE any torch/vllm import (the worker
# imports vllm lazily; this guarantees the env is in place first).
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")


def _pick_writable_dir(candidates):
    """First candidate we can actually create+write a file in.

    Under apptainer --containall the image rootfs is read-only, so the baked
    /scratch is NOT writable unless the host binds a dir over it; on the real
    (docker) backend it IS. RESULT_DIR is the one location guaranteed writable
    everywhere (the benchmark mounts it for submission.csv). Probe in
    preference order and fall back to it so the gemma KV override never fails
    for lack of a writable scratch."""
    for d in candidates:
        if not d:
            continue
        try:
            os.makedirs(d, exist_ok=True)
            probe = os.path.join(d, ".w_probe")
            with open(probe, "w") as f:
                f.write("x")
            os.remove(probe)
            return d
        except OSError:
            continue
    return candidates[-1]


# SCRATCH is ONLY for the gemma KV-override dir (tiny: symlinks + 2 small JSON).
# stage CSVs go to RESULT_DIR directly (see stage_csv_path).
SCRATCH = _pick_writable_dir([
    os.environ.get("TEXTSUM_SCRATCH_DIR"),
    "/scratch",
    os.path.join(RESULT_DIR, "_scratch"),
    "/tmp",
])

MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN",  "32768"))

# Stage 1 = REFS model (v16.4 NVFP4 gemma); Stage 2 = ANSWER model (v15.2
# Qwen3-32B-AWQ). Each worker is its own process loading one model on a fresh
# GPU, so the per-stage util/mnbt/max_new are EXACTLY the single-model values
# v16.4 / v15.2 validated → byte-identical engine config:
#   Stage 1 gemma NVFP4 — v16.4: util 0.90 + mnbt 8192 + max_new 1024 (0.90
#     avoids the sampling-buffer OOM at 0.95 with the lighter ~18 GiB weights).
#   Stage 2 32B-AWQ     — v15.2: util 0.90 + mnbt 16384 + max_new 512 (0.95
#     OOM'd at MLP activation; 16384 = 1 prefill chunk for the median prompt).
# Memory/throughput knobs only — greedy decode output unchanged.
MODEL_STAGE1 = os.environ.get("LLM_MODEL_STAGE_1", "nvidia/Gemma-4-26B-A4B-NVFP4")  # refs  (v16.4)
MODEL_STAGE2 = os.environ.get("LLM_MODEL_STAGE_2", "Qwen/Qwen3-32B-AWQ")            # answer(v15.2)
STAGE1_UTIL  = float(os.environ.get("STAGE1_UTIL", "0.90"))   # gemma NVFP4 (v16.4)
STAGE2_UTIL  = float(os.environ.get("STAGE2_UTIL", "0.90"))   # 32B-AWQ (v15.2)
STAGE1_MNBT  = int(os.environ.get("STAGE1_MNBT", "8192"))     # gemma NVFP4 (v16.4)
STAGE2_MNBT  = int(os.environ.get("STAGE2_MNBT", "16384"))    # 32B-AWQ (v15.2)
STAGE1_MAXNEW = int(os.environ.get("STAGE1_MAX_NEW_TOKENS", "1024"))  # gemma (v16.4)
STAGE2_MAXNEW = int(os.environ.get("STAGE2_MAX_NEW_TOKENS", "512"))   # 32B-AWQ (v15.2)

# Per-stage: (model, util, mnbt, max_new, prompt_style, label). prompt_style
# selects the instruction wording — "v10" (V10_factual, gemma/v16.4) vs
# "e5_hinted" (exp37's E5 prompt + Stage 1's ref-index hint on the query turn).
# Stage 2 reads the hint sidecar Stage 1 wrote; the shot turns stay non-hinted.
STAGE_CFG = {
    1: (MODEL_STAGE1, STAGE1_UTIL, STAGE1_MNBT, STAGE1_MAXNEW, "v10",       "gemma NVFP4 refs (v16.4)"),
    2: (MODEL_STAGE2, STAGE2_UTIL, STAGE2_MNBT, STAGE2_MAXNEW, "e5_hinted", "32B-AWQ hinted answer (exp86 Stage B)"),
}

# Sidecar JSON the orchestrator hands Stage 1 → Stage 2: {ID: [1-based positions
# of the paragraphs Stage 1 cited]}. In RESULT_DIR (writable on every backend).
HINT_PATH = os.path.join(RESULT_DIR, "_v17_4_hint.json")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot pair — both from held-out doc_050. Shot 1 = exp08 single-ref,
# Shot 2 = exp38 multi-ref (Q0746). The SHOT text is byte-identical across
# stages; only the per-stage instruction wording differs (see build_prompt).
# The shot turns are ALWAYS non-hinted — the ref-index hint (exp81/exp86) is
# added to the FINAL query turn only, and only for Stage 2 ("e5_hinted").
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

    nvidia/Gemma-4-26B-A4B-NVFP4 bakes "kv_cache_quant_algo": "FP8" into
    hf_quant_config.json. With it present, vLLM's kv_cache_dtype="auto" promotes
    KV to fp8_e4m3 → on A100 (sm80) there is no fp8e4nv reshape_and_cache
    kernel, and on the H100 backend fp8-KV pulls the FlashInfer/DeepGEMM JIT
    that needs nvcc. The exp75 dense fix (kv_cache_dtype="bfloat16") is rejected
    too — this MoE selects TRITON_ATTN, whose kernel asserts kv_cache_dtype ∈
    {"auto","fp8*"}. The only universal, JIT-free path is to make "auto"
    resolve to bf16 by STRIPPING the directive. Idempotent. A no-op for
    checkpoints with no kv directive (e.g. the 32B-AWQ Stage-2 model) → returns
    the snapshot/repo path unchanged.
    """
    if os.path.isdir(model_name):
        return model_name

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    repo_dir = "models--" + model_name.replace("/", "--")
    snap_glob = os.path.join(hf_home, "hub", repo_dir, "snapshots", "*")
    snaps = sorted(glob.glob(snap_glob))
    if not snaps:
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
        return snap

    ovr = os.path.join(SCRATCH, "model_override_" + model_name.replace("/", "--"))
    shutil.rmtree(ovr, ignore_errors=True)
    os.makedirs(ovr, exist_ok=True)
    rewrite = {"config.json", "hf_quant_config.json"}
    for name in os.listdir(snap):
        if name in rewrite:
            continue
        os.symlink(os.path.join(snap, name), os.path.join(ovr, name))

    qj = json.load(open(quant_cfg, encoding="utf-8"))
    qj.get("quantization", {}).pop("kv_cache_quant_algo", None)
    json.dump(qj, open(os.path.join(ovr, "hf_quant_config.json"), "w",
                       encoding="utf-8"), indent=2)

    cfg_path = os.path.join(snap, "config.json")
    cj = json.load(open(cfg_path, encoding="utf-8"))
    qc = cj.get("quantization_config", {})
    qc.pop("kv_cache_scheme", None)
    qc.pop("kv_cache_quant_algo", None)
    json.dump(cj, open(os.path.join(ovr, "config.json"), "w",
                       encoding="utf-8"), indent=2)

    print(f"KV override: stripped fp8-KV directive → {ovr} (symlinked {snap})",
          flush=True)
    return ovr


def build_prompt_v10(query, paras):
    """V10_factual — CONTEXT FIRST, then query, then instruction. Cold (no hint).

    Stage 1 (gemma / v16.4): answer "สั้นและตรงประเด็น", state only facts
    present, then cite [อ้างอิง: X]. The terse V10 instruction is the ref-picker
    prompt — it maximises citation IoU.
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


def build_prompt_e5(query, paras):
    """E5 prompt — CONTEXT FIRST, then query, then instruction (v15.2 verbatim).

    Stage 2 (32B-AWQ / v15.2): answer "อย่างกระชับและครอบคลุม" (concise AND
    comprehensive), then cite. This is the strong answer-writer prompt — it
    maximises RougeL/SS (exp37/exp38). Byte-identical to v15.2's build_prompt.
    """
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_e5_hinted(query, paras, hint_idx):
    """exp37 E5 prompt + exp81/exp86 ref-INDEX hint line (Stage 2, v17.4).

    Inserts "ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]" between the query and the
    instruction, and adds the "ใช้ย่อหน้าที่เกี่ยวข้องข้างต้นเป็นแนวทาง" clause —
    byte-identical to exp86's build_prompt_B_hinted. hint_idx = Stage 1's 1-based
    cited positions; empty → "—". The hint guides the answer; refs are still
    fixed to Stage 1 (refA) in the merge, so a wrong hint can't corrupt refs.
    """
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    hint_str = ", ".join(str(i) for i in hint_idx) if hint_idx else "—"
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [{hint_str}]\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยใช้ย่อหน้าที่เกี่ยวข้องข้างต้นเป็นแนวทาง "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


# "e5_hinted" reuses build_prompt_e5 for the (non-hinted) shot turns; the hinted
# builder is applied to the FINAL query turn only (see build_messages).
_PROMPT_BUILDERS = {"v10": build_prompt_v10, "e5": build_prompt_e5,
                    "e5_hinted": build_prompt_e5}


def build_messages(query, paras, prompt_style, hint_idx=None):
    """System + 2 few-shot turns + the final user turn.

    prompt_style picks the instruction wording ("v10" gemma / "e5" cold /
    "e5_hinted" Stage 2). The shot turns are ALWAYS non-hinted; for "e5_hinted"
    the FINAL query turn carries Stage 1's ref-index hint (exp81/exp86).
    """
    build_prompt = _PROMPT_BUILDERS[prompt_style]
    if prompt_style == "e5_hinted":
        final_turn = build_prompt_e5_hinted(query, paras, hint_idx)
    else:
        final_turn = build_prompt(query, paras)
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": final_turn},
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


def stage_csv_path(stage):
    # In RESULT_DIR (guaranteed writable on every backend), not SCRATCH.
    # stage1.csv is a full single-model submission == v16.4 standalone (cold);
    # stage2.csv is the 32B-AWQ submission generated UNDER the stage-1 hint (NOT
    # == v15.2 anymore — the hint is the v17.4 change).
    return os.path.join(RESULT_DIR, f"_v17_4_stage{stage}.csv")


# ============================================================================
# WORKER — loads ONE model, writes a full single-model submission CSV.
# Stage 1 == v16.4 standalone (cold) + writes the ref-index hint sidecar;
# Stage 2 = 32B-AWQ generated under that hint (exp86 Stage B).
# ============================================================================
def run_worker(stage):
    # Heavy imports live ONLY here so the orchestrator process never touches
    # CUDA / imports vllm. (torch is NOT imported — process exit frees the GPU,
    # so no empty_cache() is needed; vLLM pulls its own torch internally.)
    from transformers import AutoTokenizer
    from vllm import LLMEngine, EngineArgs, SamplingParams

    model_name, gpu_mem_util, mnbt, max_new, prompt_style, label = STAGE_CFG[stage]
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    half = n // 2
    print(f"[worker stage {stage}: {label}] {model_name} | {n} queries, "
          f"{len(doc_index)} docs | util={gpu_mem_util} mnbt={mnbt} "
          f"max_model_len={MAX_MODEL_LEN}", flush=True)

    doc_paras = {doc_id: filter_valid_paragraphs(paras)
                 for doc_id, paras in doc_index.items()}

    # Stage 2 reads Stage 1's ref-index hint (written when stage 1 fully exited).
    hint_by_qid = {}
    if prompt_style == "e5_hinted":
        with open(HINT_PATH, encoding="utf-8") as f:
            hint_by_qid = json.load(f)
        print(f"  loaded ref-index hints for {len(hint_by_qid)} queries "
              f"← {HINT_PATH}", flush=True)

    # Sort by doc_id so same-doc queries submit contiguously (prefix-cache).
    # CSV is written in ORIGINAL query order at the end. TEXTSUM_SUBMIT_ORDER=
    # original disables the sort (matches exp77's unsorted llm.generate path) —
    # a diagnostic toggle; "doc_id" (default) is the production prefix-cache mode.
    submit_order = os.environ.get("TEXTSUM_SUBMIT_ORDER", "doc_id")
    if submit_order == "original":
        order = list(range(n))
    else:
        order = sorted(range(n), key=lambda i: queries[i]["doc_id"])
    print(f"  submit order = {submit_order}", flush=True)

    items = []   # (k, qid, gen_pids, gen_texts, msgs, q_text) in submission order
    pool_sizes = []
    for k, idx in enumerate(order):
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        if valid:
            gen_pids  = [p["para_id"] for p in valid]
            gen_texts = [p["text"]    for p in valid]
            pool_sizes.append(len(valid))
            hint_idx = hint_by_qid.get(query["ID"]) if prompt_style == "e5_hinted" else None
            msgs = build_messages(q_text, gen_texts, prompt_style, hint_idx)
        else:
            gen_pids, gen_texts, msgs = [], [], None
        items.append((k, query["ID"], gen_pids, gen_texts, msgs, q_text))
    if pool_sizes:
        print(f"  pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # Each stage builds the EXACT EngineArgs/tokenizer its single-model image
    # validated, so each stage CSV is byte-identical to that standalone image.
    if stage == 1:
        # gemma NVFP4 (v16.4): strip the baked fp8-KV directive → override dir;
        # bf16 KV, multimodal arch (trust_remote_code + limit_mm to disable the
        # image/video towers), NVFP4 weights load via MARLIN.
        model_path = build_kv_neutralized_model(model_name)
        engine_args = EngineArgs(
            model=model_path,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=gpu_mem_util,
            max_num_batched_tokens=mnbt,
            dtype="bfloat16",
            enforce_eager=True,
            enable_prefix_caching=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    else:
        # 32B-AWQ (v15.2 verbatim): awq_marlin quant, dtype="half", no
        # trust_remote_code / limit_mm (text-only dense model). Bare repo id to
        # EngineArgs/AutoTokenizer — no override resolver — so the answer stage
        # is the v15.2 process byte-for-byte.
        model_path = model_name
        engine_args = EngineArgs(
            model=model_path,
            quantization="awq_marlin",
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=gpu_mem_util,
            max_num_batched_tokens=mnbt,
            dtype="half",
            enforce_eager=True,
            enable_prefix_caching=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    engine = LLMEngine.from_engine_args(engine_args)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new,
                              repetition_penalty=1.05)

    for k, it in enumerate(items):
        msgs = it[4] if it[4] is not None else [{"role": "user", "content": it[5]}]
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        engine.add_request(request_id=str(k), prompt=prompt, params=sampling)

    # Heartbeat: stage 1 fills 0..n/2, stage 2 fills n/2..n.
    def ping(done):
        benchmark_lib(done // 2 if stage == 1 else half + done // 2)

    raw_by_k = {}
    n_done = 0
    t0 = time.time()
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                raw_by_k[int(o.request_id)] = o.outputs[0].text.strip()
                n_done += 1
                ping(n_done)
                if n_done == 1 or n_done % 50 == 0 or n_done == n:
                    elapsed = time.time() - t0
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (n - n_done) / max(rate, 1e-6)
                    print(f"  [stage {stage} {n_done}/{n}] {rate:.2f} q/s, eta {eta:.0f}s",
                          flush=True)

    # Parse → full single-model submission (answer + refs), like v16.x.
    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    n_explicit = 0
    ref_counts = []
    results_by_qid = {}
    hint_by_qid = {}   # stage 1 only: ID → 1-based cited positions (exp86 hint)
    for k, it in enumerate(items):
        _idx_k, qid, gen_pids, gen_texts, _m, q_text = it
        raw = raw_by_k.get(k, "")
        answer = split_answer(raw)
        cited_idx = parse_citation(raw, len(gen_pids))
        if gen_pids:
            cited = [j for j in cited_idx if j < len(gen_pids)]
            if not cited:
                cited = [0]   # fallback to para 1, like v16.x / exp86
            ref_ids = [gen_pids[j] for j in cited]
        else:
            cited, ref_ids = [], []
        # exp86 hint = 1-based positions of the paragraphs Stage 1 cited.
        hint_by_qid[qid] = [j + 1 for j in cited]
        if cite_re.search(raw):
            n_explicit += 1
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
        ref_counts.append(len(ref_ids))
        results_by_qid[qid] = {"ID": qid, "abstractive": answer,
                               "refs": ",".join(ref_ids)}
    print(f"  [stage {stage}] citations: {n_explicit}/{n} tagged | "
          f"avg refs/query {sum(ref_counts)/max(n,1):.2f}", flush=True)

    # Write this stage's full submission in ORIGINAL query order.
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = stage_csv_path(stage)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results_by_qid[q["ID"]] for q in queries)
    print(f"  [stage {stage}] wrote {n} rows → {out}", flush=True)

    # Stage 1 hands its ref-index hint to Stage 2 via a sidecar JSON. Written
    # AFTER the CSV so its presence implies stage 1 fully succeeded.
    if stage == 1:
        with open(HINT_PATH, "w", encoding="utf-8") as f:
            json.dump(hint_by_qid, f, ensure_ascii=False)
        print(f"  [stage 1] wrote ref-index hints for {len(hint_by_qid)} "
              f"queries → {HINT_PATH}", flush=True)


# ============================================================================
# ORCHESTRATOR — spawns one process per stage, merges columns. NO CUDA here.
# ============================================================================
def read_stage_csv(stage):
    with open(stage_csv_path(stage), encoding="utf-8") as f:
        return {row["ID"]: row for row in csv.DictReader(f)}


def orchestrate():
    data = load_data(TEST_DIR)
    queries = data["queries"]
    n = len(queries)
    print(f"v17.4 orchestrator: {n} queries | "
          f"REFS+hint(stage1)={MODEL_STAGE1} → hinted ANSWER(stage2)={MODEL_STAGE2} | "
          f"one process per stage (fresh GPU each)", flush=True)
    benchmark_lib(0)

    # Run each stage in its own process. check=True → any stage crash aborts
    # before a half-baked submission is written. Each child fully exits before
    # the next spawns, so the GPU is genuinely fresh for stage 2. Stage 1 writes
    # the ref-index hint sidecar before it exits; stage 2 reads it on startup.
    for stage in (1, 2):
        print(f"\n=== spawning worker for stage {stage} ===", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--stage", str(stage)], check=True)

    # Merge = exp86 ansB_refA: abstractive from stage 2 (32B-AWQ, hinted), refs
    # from stage 1 (gemma). Refs stay fixed to Stage 1 — Stage 2's own citations
    # (refB) are discarded — so the hint can only move the answer, never the refs.
    refs_src = read_stage_csv(1)   # gemma         → refs
    ans_src  = read_stage_csv(2)   # 32B-AWQ hinted → answer
    results = [{"ID": q["ID"],
                "abstractive": ans_src[q["ID"]]["abstractive"],
                "refs":        refs_src[q["ID"]]["refs"]}
               for q in queries]

    out_path = Path(RESULT_DIR) / "submission.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nMerged {len(results)} rows (32B-AWQ hinted answer + gemma refs, "
          f"exp86 ansB_refA) → {out_path}", flush=True)
    benchmark_lib(n)   # final "fully done, CSV ready" ping


if __name__ == "__main__":
    if "--stage" in sys.argv:
        run_worker(int(sys.argv[sys.argv.index("--stage") + 1]))
    else:
        orchestrate()
