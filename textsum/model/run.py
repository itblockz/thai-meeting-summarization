"""
v17.2 — independent column-merge: ANSWER from v16.1, REFS from v16.4.

NOT v17.1. v17.1 was a *coupled* two-stage pipe — Stage 1's refs were fed to
Stage 2 as a hint line. v17.2 runs the two single-model images **independently**
(no hint, no handoff) and merges their CSVs column-wise: abstractive from one
model, refs from the other. Score decomposes linearly under greedy decode, so
the merged composite = answerModel(RougeL,SS) + refModel(IoU).

  Stage 1 — REFS source = v16.4 = nvidia/Gemma-4-26B-A4B-NVFP4 (26B MoE,
            ~4B active, NVFP4 experts ~18 GB load). Cold V10_factual + exp38
            shots, full doc. Record single-model citation IoU (exp77 0.8155).
  Stage 2 — ANSWER source = v16.1 = Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
            (the v16/exp42 model). SAME cold V10_factual + exp38 shots, full
            doc. Strongest ~4B-active answer-writer here (exp51 0.7110).

Each stage's CSV column we keep is its strength; the other column is discarded.
Final = A3B answer + gemma refs (the exp80-85 "best of both" cell, A3B-answer
variant of exp86's recipe).

PROCESS MODEL — one OS process PER STAGE (the bit-identical fix)
---------------------------------------------------------------
This file runs in two modes:
  * ORCHESTRATOR (default, `python3 run.py`): imports NOTHING CUDA-related.
    It spawns `python3 run.py --stage 1`, waits for it to FULLY exit, spawns
    `--stage 2`, then merges the two stage CSVs into submission.csv.
  * WORKER (`python3 run.py --stage N`): loads exactly ONE model and writes a
    FULL single-model submission (answer + refs) — byte-for-byte the v16.4
    (stage 1) / v16.1 (stage 2) image. vLLM/torch/transformers are imported
    ONLY here, so the orchestrator never creates a CUDA context.

Why a subprocess per stage instead of `del engine; empty_cache()` in one
process: in V1 multiprocessing the EngineCore already runs in a child, so the
model's VRAM is reclaimed when that child exits — NOT by the parent's
`torch.cuda.empty_cache()`, which only ever created a ~0.4 GiB context in the
parent and left it resident. That residual context made Stage 2 measure
slightly less free VRAM than a standalone v16.1 run → a marginally smaller KV
pool → different scheduler batching → rare greedy token flips. Running each
stage as its own process means the GPU is genuinely fresh for the second
model: the worker is born, loads one model on a clean card, writes its CSV, and
dies — identical to v16.1/v16.4 standalone BY CONSTRUCTION (no shared
process/CUDA state, independent of vLLM teardown timing). Each stage CSV is
therefore an independently scoreable v16.x submission: stage1 ↔ v16.4, stage2
↔ v16.1.

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed in
document order. Both stages use the IDENTICAL cold V10_factual prompt.

Container infra (carried from v16.1/v16.4):
 - VLLM_USE_DEEP_GEMM=0 (before any torch/vllm import): forces the precompiled
   CUTLASS/Marlin/NVFP4 path on the benchmark H100 (no nvcc JIT).
 - Stage 1 (gemma NVFP4) strips the baked fp8-KV directive via
   build_kv_neutralized_model so kv_cache_dtype="auto" → bf16 on A100 + H100.
   A no-op for the A3B FP8 checkpoint.
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

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN",  "32768"))

# Stage 1 = REFS model (v16.4 NVFP4 gemma); Stage 2 = ANSWER model (v16.1 A3B).
# Each worker is its own process loading one model on a fresh GPU, so the
# per-stage util/mnbt are EXACTLY the single-model values v16.4 / v16.1
# validated → byte-identical engine config:
#   Stage 1 gemma NVFP4 — v16.4: util 0.90 + mnbt 8192 (0.90 avoids the
#     sampling-buffer OOM at 0.95 with the lighter ~18 GiB NVFP4 weights).
#   Stage 2 A3B         — v16.1: util 0.95 + mnbt 16384.
# Memory/throughput knobs only — greedy decode output unchanged.
MODEL_STAGE1 = os.environ.get("LLM_MODEL_STAGE_1", "nvidia/Gemma-4-26B-A4B-NVFP4")          # refs  (v16.4)
MODEL_STAGE2 = os.environ.get("LLM_MODEL_STAGE_2", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")  # answer(v16.1)
STAGE1_UTIL  = float(os.environ.get("STAGE1_UTIL", "0.90"))   # gemma NVFP4 (v16.4)
STAGE2_UTIL  = float(os.environ.get("STAGE2_UTIL", "0.95"))   # A3B (v16.1)
STAGE1_MNBT  = int(os.environ.get("STAGE1_MNBT", "8192"))     # gemma NVFP4 (v16.4)
STAGE2_MNBT  = int(os.environ.get("STAGE2_MNBT", "16384"))    # A3B (v16.1)

STAGE_CFG = {
    1: (MODEL_STAGE1, STAGE1_UTIL, STAGE1_MNBT, "gemma NVFP4 refs"),
    2: (MODEL_STAGE2, STAGE2_UTIL, STAGE2_MNBT, "A3B answer"),
}

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot pair — both from held-out doc_050. Shot 1 = exp08 single-ref,
# Shot 2 = exp38 multi-ref (Q0746). IDENTICAL to v16.1 and v16.4 (both stages
# run the SAME cold prompt — no hint variant, that is the whole point of v17.2).
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
    checkpoints with no kv directive (e.g. the A3B FP8 Stage-2 model) → returns
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


def build_prompt(query, paras):
    """V10_factual — CONTEXT FIRST, then query, then instruction. Cold (no hint).

    IDENTICAL to v16.1 and v16.4: answer "สั้นและตรงประเด็น", state only facts
    present, then cite [อ้างอิง: X]. Both stages use this exact prompt — v17.2's
    whole premise is two INDEPENDENT cold runs.
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
    """System + 2 few-shot turns + the final user turn (cold)."""
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt(query, paras)},
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
    # In RESULT_DIR (guaranteed writable on every backend), not SCRATCH. Each
    # is a full single-model submission — score _v17_2_stage1.csv against v16.4
    # and _v17_2_stage2.csv against v16.1 to confirm per-column bit-identity.
    return os.path.join(RESULT_DIR, f"_v17_2_stage{stage}.csv")


# ============================================================================
# WORKER — loads ONE model, writes a full single-model submission CSV.
# Byte-for-byte the v16.4 (stage 1) / v16.1 (stage 2) image.
# ============================================================================
def run_worker(stage):
    # Heavy imports live ONLY here so the orchestrator process never touches
    # CUDA / imports vllm. (torch is NOT imported — process exit frees the GPU,
    # so no empty_cache() is needed; vLLM pulls its own torch internally.)
    from transformers import AutoTokenizer
    from vllm import LLMEngine, EngineArgs, SamplingParams

    model_name, gpu_mem_util, mnbt, label = STAGE_CFG[stage]
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

    # Sort by doc_id so same-doc queries submit contiguously (prefix-cache).
    # CSV is written in ORIGINAL query order at the end.
    order = sorted(range(n), key=lambda i: queries[i]["doc_id"])

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
            msgs = build_messages(q_text, gen_texts)
        else:
            gen_pids, gen_texts, msgs = [], [], None
        items.append((k, query["ID"], gen_pids, gen_texts, msgs, q_text))
    if pool_sizes:
        print(f"  pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # Stage 1 (gemma NVFP4) needs the fp8-KV directive stripped → override dir
    # (exactly v16.4). Stage 2 (A3B) carries no kv directive and v16.1 passed
    # the bare repo id to EngineArgs/AutoTokenizer → do the SAME here (don't
    # route it through the override resolver) so the answer stage is the v16.1
    # process verbatim.
    model_path = build_kv_neutralized_model(model_name) if stage == 1 else model_name
    engine = LLMEngine.from_engine_args(EngineArgs(
        model=model_path,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_mem_util,
        max_num_batched_tokens=mnbt,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    ))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
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
    for k, it in enumerate(items):
        _idx_k, qid, gen_pids, gen_texts, _m, q_text = it
        raw = raw_by_k.get(k, "")
        answer = split_answer(raw)
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
    print(f"v17.2 orchestrator: {n} queries | "
          f"REFS(stage1)={MODEL_STAGE1} → ANSWER(stage2)={MODEL_STAGE2} | "
          f"one process per stage (fresh GPU each)", flush=True)
    benchmark_lib(0)

    # Run each stage in its own process. check=True → any stage crash aborts
    # before a half-baked submission is written. Each child fully exits before
    # the next spawns, so the GPU is genuinely fresh for stage 2.
    for stage in (1, 2):
        print(f"\n=== spawning worker for stage {stage} ===", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--stage", str(stage)], check=True)

    # Merge: abstractive from stage 2 (A3B), refs from stage 1 (gemma).
    refs_src = read_stage_csv(1)   # gemma → refs
    ans_src  = read_stage_csv(2)   # A3B   → answer
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
    print(f"\nMerged {len(results)} rows (A3B answer + gemma refs) → {out_path}",
          flush=True)
    benchmark_lib(n)   # final "fully done, CSV ready" ping


if __name__ == "__main__":
    if "--stage" in sys.argv:
        run_worker(int(sys.argv[sys.argv.index("--stage") + 1]))
    else:
        orchestrate()
