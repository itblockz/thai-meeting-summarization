"""
exp74 — model swap: gemma-4-31B-it-NVFP4 (exp73) →
RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic (single A100-40GB). Same exp51
pipeline (V10_factual prompt + exp38 2-shot multi-turn shots, full doc,
single-stage). Only the LLM changes.

WHY this model can use the FP8 checkpoint where the 31B could NOT (exp71/72):
exp71/72 failed because the 31B *dense* FP8 weights load at ~30.4 GB (≈1
byte/param on A100, kept FP8/marlin-packed), leaving only ~6.49 GiB for KV
→ vLLM capped context at ~7.7K → 89% truncated. This model is a **26B MoE
(A4B, ~4B active)**: ALL 26B params still resident, but FP8 weights are
~26 GB, not ~30 GB. At util 0.95 (~38 GB budget − 26 weights − ~2
activations) ≈ 10 GiB KV — enough for the full 32768 context (exp73's
gemma-4 KV cost was 9.54 GiB for 32768). So FP8-Dynamic is viable here and
we keep MAX_MODEL_LEN=32768 with NO truncation (matches exp51/exp73).

KV stays bf16 (default). Do NOT set kv_cache_dtype="fp8" — same A100 wall
as exp71/72: e4m3 has no sm80 reshape_and_cache kernel ("fp8e4nv not
supported") and e5m2 is rejected with fp8 checkpoints. There is ample KV
room without it.

On A100 (no native FP8 cores) vLLM reads quantization_config (fp8) from
config.json and picks fp8_marlin / fp8_w8a16 software dequant (~1.5–2×
slower than AWQ per the gotchas) — but the MoE routes only ~4B active
params/token, so wall-time should land near the A3B line, not the dense
27B-FP8 line. (FP8 marlin is weight-only; KV/activation compute is bf16.)

Compatibility (assumed same as exp73, vllm 0.19.1 / transformers 5.8.1):
- gemma-4 ConditionalGeneration arch registered; chat template supports the
  system role + enable_thinking=False → exp51's 2-shot multi-turn structure
  reused unchanged.
- limit_mm_per_prompt skips the vision encoder cache (gemma-4 multimodal).

⚠️ OPEN RISKS (the first-run raw dump below guards against silent failure):
- enable_prefix_caching honored for the MoE arch is unverified; if vLLM
  auto-disables it (multimodal arch, cf. 27B-FP8 / fp8-slow-prefix-caching
  memory) the run is correct but slow — check the "enable_prefix_caching"
  line in the engine log.
- If KV does NOT fit 32768 at util 0.95, trim MAX_MODEL_LEN (measure the
  worst prompt) and/or nudge util — do NOT go TP=2 (single-GPU policy).

Decision rule: accept if leak-free composite >= exp73 (0.7140). The bet is
A4B's larger active capacity (vs A3B's 3B) + gemma-4's near-perfect
V10_factual citation (exp73 IoU 0.8091) at a fraction of the 31B's latency.
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
MODEL_NAME     = os.environ.get("LLM_MODEL", "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Same shots as exp38/39/51/73
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
    print(f"exp74: {n} queries, {len(doc_index)} docs "
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

    # gemma-4-26B-A4B-it FP8-Dynamic on a single A100-40GB. config.json
    # quantization_config is quant_method="compressed-tensors" (llm-compressor
    # FP8 dynamic scheme); vLLM's CompressedTensors backend picks the
    # fp8_marlin / fp8_w8a16 software-dequant path on A100 (no native FP8
    # cores). MoE A4B routes only ~4B active params/token → near-A3B latency
    # despite the marlin tax. arch=Gemma4ForConditionalGeneration, 30 layers.
    #
    # Memory fit (vs exp73 NVFP4 ~22 GB weights): FP8 weights ~26 GB. At
    # util 0.95 (~38 GB budget − 26 weights − ~2 activations) ≈ 10 GiB KV;
    # gemma-4's full-32768 KV was 9.54 GiB (exp73) → fits with thin
    # headroom, NO truncation. util 0.95 (not 0.92) to claw back the KV the
    # heavier FP8 weights cost vs NVFP4. KV stays bf16 — do NOT set
    # kv_cache_dtype="fp8" (no sm80 e4m3 kernel; e5m2 rejected w/ fp8 ckpt,
    # see exp71/72). If KV still won't fit 32768, trim MAX_MODEL_LEN — never
    # TP=2 (single-GPU policy).
    # enable_prefix_caching=True reuses the doc-grouped few-shot+context
    # prefix across a doc's queries (verify it's honored for this MoE arch
    # in the engine log; if auto-disabled, run is correct but ~slow).
    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.95,
              enable_prefix_caching=True,
              dtype="bfloat16", enforce_eager=True,
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
