"""
v17.1 — exp81 `s2ans_s2ref` port (two-stage, A3B writes the final answer).

Container port of exp81's grid winner. The exp ran 6 combos; v17.1 ships the
single best cell the user picked: **s2ans + s2ref** (Stage-2 answer AND Stage-2
refs — A3B's own hinted pass supplies both columns). Leak-free train composite
**0.7207** (RougeL 0.4879 / SS 0.8607 / IoU 0.8132), ties exp56's 0.7215 line.

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to both
stages in document order. Two FP8 models share one 40 GB GPU loaded one at a
time (del + gc + empty_cache between — they cannot coexist):

  Stage 1 — RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (26B MoE, ~4B active, FP8
            ~26 GB). NORMAL V10_factual prompt + exp38 shots → answer + cite.
            gemma's citation is the record single-model IoU (exp74 0.8139), so
            its [อ้างอิง: …] picks become the REF-INDEX hint for Stage 2.
            (Its answer text is discarded — see finding (1) below.)
  --- free weights, gc.collect(), torch.cuda.empty_cache() ---
  Stage 2 — Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 (the v16/exp42 model). SAME
            V10 prompt + a "ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]" line carrying
            ONLY Stage 1's ref *indices* → answer + cite. BOTH the abstractive
            answer (s2ans) AND the refs (s2ref) come from this pass.

Why this exact plumbing (exp80–85 grid findings):
 (1) hint as REF INDICES helps, hint as ANSWER TEXT contaminates — feeding
     gemma's weak ~4B-active answer to A3B drags A3B's strong answer down
     (−0.018 RougeL/SS); passing only the ref numbers leaves it intact.
 (2) direction gemma→A3B beats A3B→gemma — the 0.80-weighted answer quality
     dominates and A3B must write the final answer.

Container changes vs exp81/run.py (everything else is byte-identical: prompts,
shots, models, greedy decode temp 0 / rep-pen 1.05, ref-index hint):
 - Emit ONE submission.csv (s2ans + s2ref), not the 6-combo grid.
 - LLMEngine.step() streaming instead of llm.generate(): the benchmark
   `progress` binary fires per finished request, so the backend sees a real
   heartbeat through the long generation (Stage 1 → 0..n/2, Stage 2 → n/2..n).
 - Queries sorted by doc_id before submission so vLLM's prefix cache reuses
   the ~14K-token full-doc prefilled KV across same-doc queries (CSV is written
   back in ORIGINAL order). Keeps both stages inside the benchmark time budget.
 - VLLM_USE_DEEP_GEMM=0 (set before any torch/vllm import): both stages are FP8
   checkpoints; on the benchmark H100 vLLM's Hopper DeepGEMM block-FP8 path
   JIT-compiles with nvcc, absent from this runtime image (`nvcc: not found` →
   EngineCore dies). Disabling it forces the precompiled CUTLASS/Marlin FP8
   path, identical to what the A100 local test exercises. kv_cache_dtype stays
   at vLLM's default "auto" (→ bf16 KV) — no FP8-KV, no FlashInfer JIT, NO
   H100-specific tuning (per the brief).

The two-stage handoff DEPENDS on V1 multiprocessing being ON
(VLLM_ENABLE_V1_MULTIPROCESSING=1, pinned in the Dockerfile): each stage's
EngineCore is a child process, so `del engine` tears it down and the OS
reclaims its GPU memory before Stage 2 loads. In-process mode leaves Stage 1's
weights resident → Stage 2 OOMs.

Output: submission.csv with columns ID, abstractive, refs.
"""
from pathlib import Path
import os
import re
import json
import csv
import gc
import time

# --- H100/no-nvcc: force the precompiled FP8 path (see docstring) ------------
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
# exp81 model pairing. Per-stage util + max_num_batched_tokens are the
# PROVEN-in-container values from the single-model v16 lineage (each stage
# loads alone after the prior stage's teardown, so each sees a fresh 40 GB
# GPU = the single-model situation those images validated):
#   Stage 1 gemma — v16.3: util 0.95 + MNBT 8192. The default 16384 caused a
#     98% OOM on the H100-40GB backend (gemma's ~26 GB FP8 weights + ~9.54 GiB
#     KV leave only ~2.5 GiB physical buffer; the prefill-scratch spike at
#     16384 overran it). 8192 halves the spike, drops zero paragraphs.
#   Stage 2 A3B   — v16.1: util 0.95 + MNBT 16384 passed on the backend; A3B's
#     small per-expert MLP makes activations cheap. We keep exp81's slightly
#     more conservative util 0.90 here (this is the SECOND load after a gemma
#     teardown, so a touch more headroom for any residual fragmentation).
# These knobs are memory/throughput only — greedy decode output is unchanged,
# so the 0.7207 score is preserved. All env-overridable.
MODEL_STAGE1 = os.environ.get("LLM_MODEL_STAGE_1", "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")
MODEL_STAGE2 = os.environ.get("LLM_MODEL_STAGE_2", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
STAGE1_UTIL  = float(os.environ.get("STAGE1_UTIL", "0.95"))  # gemma (v16.3)
STAGE2_UTIL  = float(os.environ.get("STAGE2_UTIL", "0.90"))  # A3B   (exp81; v16.1 used 0.95)
STAGE1_MNBT  = int(os.environ.get("STAGE1_MNBT", "8192"))    # gemma (v16.3 crash fix)
STAGE2_MNBT  = int(os.environ.get("STAGE2_MNBT", "16384"))   # A3B   (v16.1)

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot pair — both from held-out doc_050. Shot 1 = exp08 single-ref,
# Shot 2 = exp38 multi-ref (Q0746). Stage 1 (cold) and Stage 2 (hinted) reuse
# the same QUERY/PARA triples; the wrapping prompt differs (hint line on S2).
_SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
_SHOT1_PARAS = [
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "_________________________",
    "กรรมาธิการผู้มาประชุม",
]
_SHOT1_ANSWER_TEXT = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา"
_SHOT1_HINT_IDX = [3]
_SHOT1_ANSWER = f"{_SHOT1_ANSWER_TEXT} [อ้างอิง: 3]"

_SHOT2_QUERY = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมมีจำนวนกี่คน"
_SHOT2_PARAS = [
    "๑๒. นางสาวแอนศิริ วลัยกนก กรรมาธิการ",
    "กรรมาธิการผู้ไม่มาประชุม",
    "๑. นายพิบูลย์ รัชกิจประการ (ลาการประชุม)",
    "๒. นายธนยศ ทิมสุวรรณ (ลาการประชุม)",
    "๓. นายอัคร ทองใจสด (ลาการประชุม)",
]
_SHOT2_ANSWER_TEXT = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมจำนวน 3 คน"
_SHOT2_HINT_IDX = [2, 3, 4, 5]
_SHOT2_ANSWER = f"{_SHOT2_ANSWER_TEXT} [อ้างอิง: 2, 3, 4, 5]"


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


def build_prompt_v10(query, paras):
    """V10_factual — normal answer+cite prompt (Stage 1)."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_v10_hinted(query, paras, hint_idx):
    """Normal V10 + Stage 1's REF INDICES only as the hint (Stage 2)."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    hint_str = ", ".join(str(i) for i in hint_idx) if hint_idx else "—"
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [{hint_str}]\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"โดยใช้ย่อหน้าที่เกี่ยวข้องข้างต้นเป็นแนวทาง "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_stage1(query, paras):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10(query, paras)},
    ]


def build_messages_stage2(query, paras, hint_idx):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10_hinted(_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_HINT_IDX)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10_hinted(_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_HINT_IDX)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10_hinted(query, paras, hint_idx)},
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


def make_engine_args(model_name, gpu_mem_util, max_num_batched_tokens):
    """exp81's loader (LLM(...) kwargs) expressed as EngineArgs.

    dtype bfloat16 + enable_prefix_caching + enforce_eager + the
    limit_mm_per_prompt cap (gemma-4 is multimodal, text-only here) are
    byte-identical to exp81. kv_cache_dtype is left at vLLM's default "auto"
    (bf16 KV) — no FP8-KV, no H100 tuning. max_num_batched_tokens is the
    PROVEN-in-container per-stage value (see the STAGE*_MNBT notes above).
    """
    return EngineArgs(
        model=model_name,
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

    items: list of (key, messages) — key maps the raw output back to the
    caller's structures. progress_fn(n_done_in_stage) fires per finished
    request so the benchmark `progress` binary keeps a heartbeat.
    """
    print(f"\n=== {stage_label}: {model_name} "
          f"(util {gpu_mem_util}, mnbt {max_num_batched_tokens}) ===", flush=True)
    engine = LLMEngine.from_engine_args(
        make_engine_args(model_name, gpu_mem_util, max_num_batched_tokens))
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
    print(f"v17.1 (exp81 s2ans_s2ref): {n} queries, {len(doc_index)} docs | "
          f"S1={MODEL_STAGE1} → S2={MODEL_STAGE2} | max_model_len={MAX_MODEL_LEN}",
          flush=True)
    benchmark_lib(0)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # Sort indices by doc_id so same-doc queries submit contiguously to BOTH
    # stages (prefix-cache reuse). CSV is written in ORIGINAL order at the end.
    order = sorted(range(n), key=lambda i: queries[i]["doc_id"])

    # ----- Stage 1 items (key = position in sorted order) -----
    pool_sizes = []
    gen_pids_by_k = {}
    gen_texts_by_k = {}
    stage1_items = []
    for k, idx in enumerate(order):
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        if valid:
            gen_pids  = [p["para_id"] for p in valid]
            gen_texts = [p["text"]    for p in valid]
            pool_sizes.append(len(valid))
            msgs = build_messages_stage1(q_text, gen_texts)
        else:
            gen_pids, gen_texts = [], []
            msgs = [{"role": "user", "content": q_text}]
        gen_pids_by_k[k] = gen_pids
        gen_texts_by_k[k] = gen_texts
        stage1_items.append((k, msgs))
    if pool_sizes:
        print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
              f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # Stage 1 occupies the 0..n/2 half of the heartbeat range.
    raws1 = run_stage("Stage 1 (gemma refs)", MODEL_STAGE1, STAGE1_UTIL, STAGE1_MNBT,
                      stage1_items, progress_fn=lambda i: benchmark_lib(i // 2))

    # Parse Stage 1 → ref-index hint (1-based positions within the full doc).
    s1_hint = {}
    avg_refs1 = 0
    for k in range(n):
        raw = raws1.get(k, "")
        gen_pids = gen_pids_by_k[k]
        cited_idx = parse_citation(raw, len(gen_pids))
        valid_idx = [j for j in cited_idx if j < len(gen_pids)]
        if not valid_idx and gen_pids:
            valid_idx = [0]
        s1_hint[k] = [j + 1 for j in valid_idx]
        avg_refs1 += len(s1_hint[k])
    print(f"Stage 1: avg hint refs/query = {avg_refs1/max(n,1):.2f}", flush=True)
    del stage1_items, raws1

    # ----- Stage 2 items (same submission order; carries the ref-index hint) -----
    stage2_items = []
    for k, idx in enumerate(order):
        query = queries[idx]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_texts = gen_texts_by_k[k]
        if valid:
            msgs = build_messages_stage2(q_text, gen_texts, s1_hint[k])
        else:
            msgs = [{"role": "user", "content": q_text}]
        stage2_items.append((k, msgs))

    # Stage 2 occupies the n/2..n half of the heartbeat range.
    raws2 = run_stage("Stage 2 (A3B answer+refs)", MODEL_STAGE2, STAGE2_UTIL, STAGE2_MNBT,
                      stage2_items, progress_fn=lambda i: benchmark_lib(half + i // 2))

    # ----- Assemble final results: s2ans (A3B answer) + s2ref (A3B refs) -----
    empty_answers = 0
    n_explicit = 0
    ref_counts = []
    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    results_by_qid = {}
    for k, idx in enumerate(order):
        query = queries[idx]
        qid = query["ID"]
        q_text = query["query"]
        gen_pids = gen_pids_by_k[k]
        gen_texts = gen_texts_by_k[k]
        raw = raws2.get(k, "")
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty_answers += 1
        cited_idx = parse_citation(raw, len(gen_pids))
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]
        else:
            ref_ids = []
        if cite_re.search(raw):
            n_explicit += 1
        ref_counts.append(len(ref_ids))
        results_by_qid[qid] = {"ID": qid, "abstractive": answer,
                               "refs": ",".join(ref_ids)}
    print(f"Stage 2 citations: {n_explicit}/{n} emitted an [อ้างอิง: …] tag, "
          f"{n - n_explicit} fell back to top-1 | empty_answers={empty_answers}",
          flush=True)
    if ref_counts:
        print(f"avg refs/query: {sum(ref_counts)/len(ref_counts):.2f}", flush=True)

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
