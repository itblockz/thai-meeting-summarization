"""
Thai document summarization pipeline (production / Docker submission).

NO RETRIEVAL: the full list of *valid* paragraphs from `doc_id` is fed to
Qwen3-32B-AWQ in original document order. The model is shown the
paragraphs as a numbered [1..N] context, answers in Thai, then cites
which paragraphs it used as [อ้างอิง: X]; the cited paragraphs become
`refs` (E5 self-citation, adaptive count). Two worked few-shot examples
are prepended as multi-turn chat turns.

This matches exp37 — full-doc context (no retrieval) + E5 self-citation
+ 2-shot few-shot (exp08's example pair, from held-out doc_050) +
**context-first prompt order** (doc context placed BEFORE the per-query
text, so vLLM's prefix cache can reuse the ~14K-token full-doc prefix
across all queries from the same doc — was ~5% cache hit with the
query-first order of exp35).
Train-set composite **0.6944 leak-free** (1218 queries, doc_050
excluded) — +0.0015 over exp35 and +0.0297 over exp22.

v15.2 = the v15 container base (deadsnakes python 3.11.x, surgical Triton
bypass, the v13 infra below) with the EXACT exp38 model+prompt: Qwen3-32B-AWQ
+ exp08's shot1 (single-ref) PLUS the multi-ref shot2 (Q0746). This restores
v15-K's multi-ref shot2 that v15.1 had reverted — i.e. v15.2 = exp38 (0.6987
leak-free), v15.1 = exp37 (0.6944). The two are byte-identical except for
shot2, so v15.2 vs v15.1 isolates the exp37→exp38 delta inside the container,
holding the infra (and the never-shipped v15-K image) constant.

v14 = v13 infra (sort by doc_id, streaming progress, KV opts) PLUS
context-first prompt order. The infra changes alone were no-op for
output (verified container-vs-venv drift ≤1/50). The prompt-order swap
in v14 is the actual score win.

v13 throughput optimisations (carry over):
- Sort queries by doc_id before submitting to vLLM — queries from the
  same doc share the ~14K-token full-doc prefix, so vLLM's
  enable_prefix_caching reuses the prefilled KV blocks across them.
  With v14's context-first prompt, cache hit ceiling rises from ~5%
  (v13) to ~90% (the doc context is now in the shared prefix).
- max_num_batched_tokens 8192 → 16384 — single-chunk prefill for the
  median ~14K-tok prompt (was 2 chunks). Bigger (32768) OOM'd at MLP
  forward because Qwen3-32B's intermediate dim 27648 needs ~1.5GB
  activation per chunk; 16384 halves that to a safe ~750MB.
- LLMEngine.step() streaming instead of llm.generate() — the
  benchmark `progress` binary is called as each request *finishes* (not
  during prompt prep, which was instant) so the backend sees real-time
  progress and a SLURM log shows a heartbeat.
- gpu_memory_utilization kept at 0.90 (0.95 left no room for
  activations and OOM'd — v13a failure on lanta-g-006).

Container-specific settings (vllm 0.9.2 in the image): enforce_eager=True
skips the V1-engine torch.compile path that crashes silently inside
Apptainer/Docker.

Output: submission.csv with columns ID, abstractive, refs — written in
the *original* queries order (sort is internal only).
"""
from pathlib import Path
import os
import re
import json
import csv
import time

from transformers import AutoTokenizer
from vllm import LLMEngine, EngineArgs, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_NEW_TOKENS         = 512
MAX_MODEL_LEN          = int(os.environ.get("MAX_MODEL_LEN", "32768"))
# 16384 = sweet spot: 1 prefill chunk for the median ~14K-tok prompt, 2
# chunks for the max ~28K-tok prompt. 32768 OOM'd at MLP activation
# (~1.5 GB per chunk for Qwen3-32B's 27648 intermediate dim) — see v13a
# failure on lanta-g-006 (job 5798218).
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", "16384"))
# 0.90 = same as v12 (proven safe). 0.95 left only ~100 MiB headroom
# after model (18GB) + KV cache + activations → OOM during MLP forward.
GPU_MEM_UTIL           = float(os.environ.get("GPU_MEM_UTIL", "0.90"))
MODEL_NAME             = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-AWQ")

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
# Shot 2 (v15-K / exp38, restored in v15.2): multi-ref subset — replaces
# exp08's single-ref shot2 because the 153 missed-tag queries in the v15
# train eval were overwhelmingly multi-ref gold (model emits a comprehensive
# answer but forgets to cite anything). Q0746 from doc_050: 4 of 5 context
# paragraphs are gold (P21-P24, absentee list); P20 is a same-section
# distractor (attendee #12). Teaches "structured answer covers several
# paragraphs → cite them all" + "don't cite paragraphs the answer didn't
# use." This multi-ref shot2 IS the exp37→exp38 delta (IoU 0.6669→0.6906);
# v15.1 dropped it (= exp37), v15.2 keeps it (= exp38).
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
    """E5 prompt — CONTEXT FIRST, then query, then instruction.

    Context-first lets vLLM's prefix cache match the full ~14K-token doc
    block across all queries from the same doc (the query is the
    divergence point, not the cache-killer at the start). Instruction
    stays at the end for recency bias on the citation directive.
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
    print(f"{n} queries, {len(doc_index)} docs "
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

    engine_args = EngineArgs(
        model=MODEL_NAME, quantization="awq_marlin",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        dtype="half", enforce_eager=True,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=True,
    )
    # Load tokenizer separately via AutoTokenizer — engine.get_tokenizer() was
    # added in vllm 0.10+ and is absent in 0.9.2 (container).
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
