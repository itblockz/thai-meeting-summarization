"""
v17.2 — independent column-merge: ANSWER from v16.1, REFS from v16.4.

NOT v17.1. v17.1 was a *coupled* two-stage pipe — Stage 1's refs were fed to
Stage 2 as a `ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]` hint line. v17.2 runs the
two single-model images **independently** (no hint, no handoff): each stage
sees the IDENTICAL cold V10_factual prompt, neither knows the other ran. We
then merge the two CSVs column-wise — abstractive from one model, refs from the
other. This is the score-decomposes-linearly trick done at inference time:
RougeL/SS come entirely from the answer model, IoU entirely from the ref model,
so the merged composite = answerModel(RougeL,SS) + refModel(IoU).

  Stage 1 — REFS source = v16.4 = nvidia/Gemma-4-26B-A4B-NVFP4 (26B MoE,
            ~4B active, NVFP4 experts ~18 GB load, attention bf16). Cold
            V10_factual + exp38 shots, full doc. Its [อ้างอิง: …] picks become
            the FINAL refs (its answer text is discarded). gemma's citation is
            the record single-model IoU (exp77 0.8155 leak-free).
  --- free weights, gc.collect(), torch.cuda.empty_cache() ---
  Stage 2 — ANSWER source = v16.1 = Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
            (the v16/exp42 model). SAME cold V10_factual + exp38 shots, full
            doc. Its abstractive answer becomes the FINAL answer (its refs are
            discarded). A3B is the strongest ~4B-active answer-writer here
            (exp51 0.7110 single-model).

Why this beats either single model: gemma's answers are weak (~4B-active
ceiling, exp77 0.6982) but its refs are sharp (IoU 0.8155); A3B's answers are
strong (exp51) but its refs are softer (single-stage A3B IoU ~0.75). Take the
best column from each. Leak-free expectation = A3B's RougeL/SS (exp51 line)
+ gemma's IoU 0.8155 — the exp80–85 "best of both" cell (A3B cold answer +
gemma cold refs, ~0.715–0.7205). This is the A3B-answer variant of exp86's
recipe (exp86 used a 32B-AWQ answer → 0.7235; here the answer model is A3B per
the brief "answer from v16.1").

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to both
stages in document order. The two models share one 40 GB GPU loaded one at a
time (del + gc + empty_cache between — V1 child-process teardown frees the GPU;
they cannot coexist).

Container infra (carried from v16.1/v16.4/v17.1):
 - Emit ONE submission.csv (A3B answer + gemma refs).
 - LLMEngine.step() streaming so the benchmark `progress` binary keeps a
   heartbeat (Stage 1 → 0..n/2, Stage 2 → n/2..n of the range).
 - Queries sorted by doc_id before submission so vLLM's prefix cache reuses
   the ~14K-token full-doc prefilled KV across same-doc queries (CSV written
   back in ORIGINAL order).
 - VLLM_USE_DEEP_GEMM=0 (before any torch/vllm import): on the benchmark H100,
   vLLM's Hopper DeepGEMM block-FP8 path JIT-compiles with nvcc, absent from
   this runtime image. Disabling it forces the precompiled CUTLASS/Marlin path,
   identical to the A100 local test.
 - Stage 1 (gemma NVFP4) needs a KV-directive strip: the nvidia/ModelOpt
   checkpoint bakes "kv_cache_quant_algo": "FP8" → kv_cache_dtype="auto" would
   promote KV to fp8 (no sm80 kernel on A100; nvcc-JIT FlashInfer on H100).
   build_kv_neutralized_model symlinks the snapshot with that directive
   stripped so "auto" → bf16 on both platforms. A no-op for the A3B FP8
   checkpoint (carries no kv directive).

The handoff DEPENDS on V1 multiprocessing (VLLM_ENABLE_V1_MULTIPROCESSING=1,
pinned in the Dockerfile): each stage's EngineCore is a child process, so
`del engine` tears it down and the OS reclaims its GPU memory before Stage 2
loads. In-process mode leaves Stage 1's weights resident → Stage 2 OOMs.

Output: submission.csv with columns ID, abstractive, refs.
"""
from pathlib import Path
import os
import re
import json
import csv
import glob
import shutil
import gc
import time

# --- H100/no-nvcc: force the precompiled FP8/NVFP4 path (see docstring) ------
# Must be set BEFORE the first torch/vllm import.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

import torch
from transformers import AutoTokenizer
from vllm import LLMEngine, EngineArgs, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN",  "32768"))

# Stage 1 = REFS model (v16.4 NVFP4 gemma); Stage 2 = ANSWER model (v16.1 A3B).
# Per-stage util/MNBT are the PROVEN-in-container single-model values — each
# stage loads alone after the prior stage's teardown, i.e. the single-model
# situation v16.1/v16.4 validated:
#   Stage 1 gemma NVFP4 — v16.4: util 0.90 + MNBT 8192. 0.90 (NOT 0.95) is
#     deliberate: with the lighter ~18 GiB NVFP4 weights vLLM grows the KV pool
#     to fill 0.95 → only ~246 MiB physically free → the sampling frequency-
#     penalty buffer (rep-pen 1.05, batched) OOM'd at first decode (exp77 job
#     5824692). 0.90 keeps full-32768 KV and leaves ~4 GiB truly-free.
#   Stage 2 A3B — v16.1 used util 0.95; we keep 0.90 here (SECOND load after a
#     gemma teardown → a touch more headroom for residual fragmentation, as in
#     v17.1). MNBT 16384 (v16.1) — A3B's small per-expert MLP makes activations
#     cheap.
# Memory/throughput knobs only — greedy decode output is unchanged.
MODEL_STAGE1 = os.environ.get("LLM_MODEL_STAGE_1", "nvidia/Gemma-4-26B-A4B-NVFP4")          # refs  (v16.4)
MODEL_STAGE2 = os.environ.get("LLM_MODEL_STAGE_2", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")  # answer(v16.1)
STAGE1_UTIL  = float(os.environ.get("STAGE1_UTIL", "0.90"))  # gemma NVFP4 (v16.4)
STAGE2_UTIL  = float(os.environ.get("STAGE2_UTIL", "0.90"))  # A3B (v16.1 used 0.95; 0.90 = 2nd-load headroom)
STAGE1_MNBT  = int(os.environ.get("STAGE1_MNBT", "8192"))    # gemma NVFP4 (v16.4)
STAGE2_MNBT  = int(os.environ.get("STAGE2_MNBT", "16384"))   # A3B (v16.1)

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
    that needs nvcc (absent from this runtime image). The exp75 dense fix
    (kv_cache_dtype="bfloat16") is rejected too — this MoE selects TRITON_ATTN,
    whose kernel asserts kv_cache_dtype ∈ {"auto","fp8*"}. The only universal,
    JIT-free path is to make "auto" resolve to bf16 by STRIPPING the directive.
    Idempotent. A no-op for checkpoints with no kv directive (e.g. the A3B FP8
    Stage-2 model) → returns the snapshot/repo path unchanged.
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

    # Build a scratch override unique per model (both stages may call this).
    scratch = os.environ.get("TEXTSUM_SCRATCH_DIR", "/scratch")
    ovr = os.path.join(scratch, "model_override_" + model_name.replace("/", "--"))
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


def make_engine_args(model_path, gpu_mem_util, max_num_batched_tokens):
    """Single-model loader (v16.1/v16.4 kwargs) as EngineArgs.

    dtype bf16 + enable_prefix_caching + enforce_eager + limit_mm_per_prompt
    (gemma-4 is multimodal, text-only here) — byte-identical to v16.x.
    kv_cache_dtype left at "auto" → bf16 (the gemma override dir has its fp8-KV
    directive stripped; A3B carries none).
    """
    return EngineArgs(
        model=model_path,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gpu_mem_util,
        max_num_batched_tokens=max_num_batched_tokens,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )


def run_stage(stage_label, model_name, gpu_mem_util, max_num_batched_tokens,
              items, progress_fn):
    """Load model, stream all `items` through engine.step(), free the GPU.

    items: list of (key, messages). progress_fn(n_done_in_stage) fires per
    finished request so the benchmark `progress` binary keeps a heartbeat.
    """
    model_path = build_kv_neutralized_model(model_name)
    print(f"\n=== {stage_label}: {model_name} "
          f"(util {gpu_mem_util}, mnbt {max_num_batched_tokens}) ===", flush=True)
    engine = LLMEngine.from_engine_args(
        make_engine_args(model_path, gpu_mem_util, max_num_batched_tokens))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

    # Free GPU before the next stage's weights land (V1 child-process teardown).
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
    print(f"v17.2 (independent merge: A3B answer + gemma-NVFP4 refs): {n} queries, "
          f"{len(doc_index)} docs | REFS={MODEL_STAGE1} → ANSWER={MODEL_STAGE2} | "
          f"max_model_len={MAX_MODEL_LEN}", flush=True)
    benchmark_lib(0)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # Sort indices by doc_id so same-doc queries submit contiguously to BOTH
    # stages (prefix-cache reuse). CSV is written in ORIGINAL order at the end.
    order = sorted(range(n), key=lambda i: queries[i]["doc_id"])

    # Both stages run the SAME cold prompt → build the items ONCE and reuse.
    pool_sizes = []
    gen_pids_by_k = {}
    gen_texts_by_k = {}
    items = []
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
            gen_pids, gen_texts = [], []
            msgs = [{"role": "user", "content": q_text}]
        gen_pids_by_k[k] = gen_pids
        gen_texts_by_k[k] = gen_texts
        items.append((k, msgs))
    if pool_sizes:
        print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # ----- Stage 1 = REFS (gemma NVFP4, v16.4). Occupies 0..n/2 of heartbeat. -----
    raws_ref = run_stage("Stage 1 (gemma NVFP4 refs)", MODEL_STAGE1,
                         STAGE1_UTIL, STAGE1_MNBT, items,
                         progress_fn=lambda i: benchmark_lib(i // 2))

    ref_ids_by_k = {}
    n_explicit_ref = 0
    ref_counts = []
    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    for k in range(n):
        raw = raws_ref.get(k, "")
        gen_pids = gen_pids_by_k[k]
        cited_idx = parse_citation(raw, len(gen_pids))
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]
        else:
            ref_ids = []
        if cite_re.search(raw):
            n_explicit_ref += 1
        ref_ids_by_k[k] = ref_ids
        ref_counts.append(len(ref_ids))
    print(f"Stage 1 (refs): {n_explicit_ref}/{n} emitted an [อ้างอิง: …] tag, "
          f"{n - n_explicit_ref} fell back to top-1 | "
          f"avg refs/query = {sum(ref_counts)/max(n,1):.2f}", flush=True)
    del raws_ref

    # ----- Stage 2 = ANSWER (A3B, v16.1). Occupies n/2..n of heartbeat. -----
    raws_ans = run_stage("Stage 2 (A3B answer)", MODEL_STAGE2,
                         STAGE2_UTIL, STAGE2_MNBT, items,
                         progress_fn=lambda i: benchmark_lib(half + i // 2))

    # ----- Merge: abstractive from A3B (Stage 2), refs from gemma (Stage 1) -----
    empty_answers = 0
    results_by_qid = {}
    for k, idx in enumerate(order):
        query = queries[idx]
        qid = query["ID"]
        q_text = query["query"]
        gen_texts = gen_texts_by_k[k]
        raw = raws_ans.get(k, "")
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty_answers += 1
        ref_ids = ref_ids_by_k[k]
        results_by_qid[qid] = {"ID": qid, "abstractive": answer,
                               "refs": ",".join(ref_ids)}
    print(f"Stage 2 (answer): empty_answers={empty_answers}", flush=True)

    # Write CSV in ORIGINAL queries order (not sorted submission order).
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
