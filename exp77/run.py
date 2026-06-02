"""
exp77 — NVFP4 publisher swap on the MoE: RedHatAI/gemma-4-26B-A4B-it-NVFP4
(exp76) → nvidia/Gemma-4-26B-A4B-NVFP4 (single A100-40GB). Same exp51
pipeline (V10_factual prompt + exp38 2-shot multi-turn shots, full doc,
single-stage); everything is identical to exp76 except the NVFP4
*publisher/format* (compressed-tensors → TensorRT-Model-Optimizer).

WHY both publishers: on the DENSE 31B line the two NVFP4 builds were NOT
interchangeable — RedHatAI (compressed-tensors, targets:['Linear'])
quantized the decoder's self_attn too → 18.54 GiB load → fit a single A100
(exp73 = 0.7140); nvidia/ModelOpt `exclude_modules` ALL self_attn* → attention
stays bf16 → 29.96 GiB load → INFEASIBLE on 1×A100-40GB (exp75, KV floor
overrun). This run asks whether that wall still bites on the **MoE**: here
the ~22B of expert params (the bulk) are still NVFP4-quantized; only the
shared attention stays bf16, a far smaller slice than in the dense 31B. So
the nvidia build SHOULD fit where its dense sibling did not — but the load
size is the open question (see fit comment + first-run log).

KV — the exp75 LOAD-BEARING fix carries over: nvidia/ModelOpt bakes
"kv_cache_quant_algo": "FP8" into hf_quant_config.json (RedHatAI does NOT),
so kv_cache_dtype="auto" would resolve to fp8_e4m3 → sm80 has no fp8e4nv
reshape_and_cache kernel → engine init dies (the exp71/72/75 wall). Forcing
kv_cache_dtype="bfloat16" makes resolve_kv_cache_dtype_string return it
verbatim, ignoring the FP8 directive → bf16 KV, exactly like exp76.

Quality bet: identical to exp76 — this is an MoE answer-quality ceiling task
(active-param count, cf. exp74 = 0.6970), so neither publisher is expected
to beat the dense 31B (exp73 = 0.7140). The only thing that could move is a
small drift from ModelOpt's per-tensor amax / scale placement vs RedHatAI's
compressed-tensors layout (cf. the exp73↔exp75 rationale). Run primarily to
(a) confirm the nvidia MoE build actually FITS where the dense one didn't,
and (b) get the format-equivalence delta vs exp76.

⚠️ OPEN RISKS (first-run raw dump guards against silent failure):
- NVFP4 *MoE* fused-expert dequant on A100/sm80 (same kernel concern as
  exp76) — check the load log.
- If attention-stays-bf16 makes the load too heavy to fit 32768 KV, trim
  MAX_MODEL_LEN (exp75 measured worst prompt ~19.3K tok → 20480 truncates
  0/1239) and/or raise util — never TP=2 (memory:prefer-single-gpu).
- enable_prefix_caching honored for this MoE arch unverified — check log.

Decision rule: format-equivalence check against exp76 / exp74 (0.6970); the
single-model bar to beat remains exp73 (0.7140). First-run priority is the
FIT verdict (does the nvidia MoE build avoid exp75's dense-31B OOM?).
"""
from pathlib import Path
import os
import re
import json
import csv

from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
TP_SIZE        = int(os.environ.get("TP_SIZE", "1"))
MODEL_NAME     = os.environ.get("LLM_MODEL", "nvidia/Gemma-4-26B-A4B-NVFP4")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Same shots as exp38/39/51/73/74/76
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
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user", "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user", "content": build_prompt(query, paras)},
    ]


def parse_citation(text, n_paras):
    nums = []
    for grp in re.findall(r'\[อ้างอิง[:\s]+([0-9,\s]+)\]', text):
        nums += [int(x) for x in re.findall(r'\d+', grp)]
    valid, seen = [], set()
    for num in nums:
        if 1 <= num <= n_paras and num not in seen:
            seen.add(num)
            valid.append(num - 1)
    return valid or [0]


def split_answer_citation(text):
    idx = text.rfind('[อ้างอิง')
    raw_tag = text[idx:] if idx != -1 else ""
    answer = re.sub(r'\s*\[อ้างอิง[^\]]*\]', '', text).strip()
    return answer, raw_tag


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp77: {n} queries, {len(doc_index)} docs "
          f"(model={MODEL_NAME}, TP={TP_SIZE}, "
          f"MAX_NEW_TOKENS={MAX_NEW_TOKENS}, max_model_len={MAX_MODEL_LEN})",
          flush=True)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        doc_paras[doc_id] = valid

    items = []
    pool_sizes = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        if valid:
            gen_pids  = [p["para_id"] for p in valid]
            gen_texts = [p["text"]    for p in valid]
            pool_sizes.append(len(valid))
            messages = build_messages(q_text, gen_texts)
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((query["ID"], gen_pids, gen_texts, messages, q_text))

    print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
          f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    # nvidia/Gemma-4-26B-A4B-NVFP4: ModelOpt-packed NVFP4 (quant_method=
    # modelopt). On A100/sm80 (no native FP4 cores) vLLM uses an FP4
    # weight-only dequant path (Marlin for dense Linear, exp73/75-confirmed;
    # the MoE experts need an analogous sm80 FP4 fused-MoE dequant — OPEN
    # RISK, same as exp76). MoE A4B routes ~4B active params/token. arch=
    # Gemma4ForConditionalGeneration; limit_mm_per_prompt skips vision cache.
    #
    # ⚠️ kv_cache_dtype="bfloat16" is LOAD-BEARING (vs exp76, which omits it).
    # nvidia/ModelOpt bakes "kv_cache_quant_algo": "FP8" into
    # hf_quant_config.json; with kv_cache_dtype="auto" vLLM's
    # resolve_kv_cache_dtype_string() reads that → fp8_e4m3 → sm80
    # reshape_and_cache has no fp8e4nv path ("type fp8e4nv not supported") →
    # engine init dies (the exp71/72/75 KV wall). Passing "bfloat16" (a valid
    # CacheDType) is returned verbatim → forces bf16 KV, ignoring the FP8
    # directive, matching exp76's RedHatAI behaviour.
    #
    # Fit — HEAVIER than exp76 (RedHatAI). nvidia/ModelOpt `exclude_modules`
    # the self_attn* layers from NVFP4 → attention stays bf16. In the dense
    # 31B that added ~8 GiB → 29.96 GiB → didn't fit (exp75). On THIS MoE the
    # quantized experts are the bulk and stay 4-bit, so the bf16-attention
    # penalty is a smaller fraction → expected to fit a single A100 (the
    # first-run load log is the verdict). util 0.95 (not 0.92 like exp76) to
    # claw back the KV the bf16 attention costs; MAX_MODEL_LEN=32768. If KV
    # won't fit 32768, trim MAX_MODEL_LEN (exp75: worst prompt ~19.3K →
    # 20480 truncates 0/1239) — never TP=2 (memory:prefer-single-gpu).
    # enable_prefix_caching=True reuses the doc-grouped few-shot+context
    # prefix (verify honored for this MoE arch in the engine log).
    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.95,
              enable_prefix_caching=True,
              dtype="bfloat16", kv_cache_dtype="bfloat16",
              enforce_eager=True,
              trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
                              repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[3] if it[3] is not None else [{"role": "user", "content": it[4]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    outputs = llm.generate(prompts, sampling)

    # FIRST-RUN SANITY CHECK: dump a couple of raw generations so the .out
    # reveals gibberish (vllm#39049) or leaked thinking-channel text early.
    for it, out in list(zip(items, outputs))[:3]:
        print(f"[raw {it[0]}] {out.outputs[0].text.strip()[:300]!r}", flush=True)

    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    results = []
    n_explicit = 0
    ref_counts = []
    for it, out in zip(items, outputs):
        qid, gen_pids, gen_texts, _, q_text = it
        raw = out.outputs[0].text.strip()
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
        results.append({"ID": qid, "abstractive": answer, "refs": ",".join(ref_ids)})

    print(f"citations: {n_explicit}/{len(results)} emitted an [อ้างอิง: …] tag, "
          f"{len(results) - n_explicit} fell back to top-1", flush=True)
    print(f"avg refs/query: {sum(ref_counts) / len(ref_counts):.2f}", flush=True)

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
    benchmark_lib(n)
