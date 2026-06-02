"""
exp79 — publisher swap to FIX exp78's A100 wall: nvidia/Qwen3.6-35B-A3B-NVFP4
→ **unsloth/Qwen3.6-35B-A3B-NVFP4** (single A100-40GB). Identical to exp78
(exp51 pipeline: V10_factual prompt + exp38 2-shot multi-turn shots, full
doc, single-stage; same arch, same limit_mm, enable_thinking=False) — ONLY
the checkpoint publisher changes.

WHY exp78 failed and why this should NOT: exp78's nvidia checkpoint is
`quant_algo:"MIXED_PRECISION"` (FP8 attention + NVFP4 experts) → vLLM
resolves it to quant method `modelopt_mixed`, which is HARD-GATED to compute
capability ≥89 (`VllmConfig` ValidationError: "modelopt_mixed … Minimum
capability: 89. Current capability: 80") with NO sm80 fallback → infeasible
on LANTA's A100, never scored. The unsloth build is a DIFFERENT format
(verified on HF before this run):
  - NO hf_quant_config.json (it is NOT a ModelOpt checkpoint).
  - config.quantization_config: quant_method=`compressed-tensors`,
    format=`nvfp4-pack-quantized`, config_groups targets=[['Linear']],
    kv_cache_scheme=None.
This is exactly the RedHatAI-style PURE-NVFP4 layout that ran on sm80 for
gemma-4 (exp73 dense / exp76 MoE) via the MARLIN NvFp4 backend — NOT the
sm89-gated modelopt_mixed. targets:['Linear'] quantizes the Linear layers
(attention projections + expert MLPs) to 4-bit, so the load is lighter than
nvidia's (which kept attention FP8). So exp79 should clear BOTH exp78 walls:
no modelopt_mixed capability gate, AND no baked FP8-KV directive.

KV — UNLIKE exp78/exp77, NO override dir is needed: the unsloth checkpoint
carries no `kv_cache_quant_algo`/`kv_cache_scheme`, so kv_cache_dtype="auto"
resolves to bf16 naturally (the exp71/72/75/77/78 fp8e4nv trap does not
apply). submit_eval_train.sh points LLM_MODEL straight at the cached snapshot
(like the exp73/76 RedHatAI runs), no config rewriting.

ARCH (confirmed from exp78's run + the unsloth config):
- Qwen3_5MoeForConditionalGeneration — a MULTIMODAL (vision/video
  preprocessor) HYBRID linear-attention MoE. vLLM DID support it on exp78
  (recognized `linear_attn` as Mamba layers — 3/4 layers Mamba + every 4th
  full self_attn — and enabled prefix caching in Mamba 'align' experimental
  mode). KEEP `limit_mm_per_prompt={"image":0,"video":0}` (exp41/48 + exp78
  all needed it). enable_thinking=False (comparable to exp73/74/76/77).
- Tiny KV footprint (only ~10 self_attn layers) → full 32768 ctx fits a
  single A100-40GB easily at 4-bit. util 0.90 (exp77 sampling-OOM-safe);
  raise toward 0.92-0.95 if the load log shows free RAM. Never TP=2
  (memory:prefer-single-gpu).

⚠️ OPEN RISKS (first-run raw dump guards against silent failure):
- compressed-tensors `nvfp4-pack-quantized` fused-MoE on the *hybrid
  linear-attn* arch on sm80 — pure-NVFP4 MARLIN ran for gemma-4 MoE
  (exp76/77); this is the first time on a Qwen3.5-MoE w/ Mamba layers. If
  the loader rejects it, that's the verdict — read the load log first.
- prefix caching in Mamba 'align' mode is experimental (exp78 enabled it).
- enable_thinking=False leaked-<think> check via the raw dump.

Decision rule: beat exp42/v16 (0.7087, the Qwen A3B production single model)
to be interesting; beat dense exp73 (0.7140) to become the new best single
model. Below 0.7087 → the active-param ceiling holds across Qwen generations
too (cf. exp74/77), and dense gemma-4-31B-NVFP4 stays the single-model
champion. (vs exp78 the only question that changed is FEASIBILITY — this is
the first time the Qwen3.6 A3B base actually gets to produce a score.)
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
MODEL_NAME     = os.environ.get("LLM_MODEL", "unsloth/Qwen3.6-35B-A3B-NVFP4")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Same shots as exp38/39/51/73/74/76/77
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
    print(f"exp79: {n} queries, {len(doc_index)} docs "
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

    # unsloth/Qwen3.6-35B-A3B-NVFP4: compressed-tensors PURE NVFP4
    # (quant_method=compressed-tensors, format=nvfp4-pack-quantized,
    # config_groups targets:['Linear']) — the RedHatAI-style layout that runs
    # on A100/sm80 via the MARLIN NvFp4 backend (exp73 dense / exp76 MoE), NOT
    # nvidia's sm89-gated modelopt_mixed that killed exp78. targets:['Linear']
    # quantizes attention projections + expert MLPs to 4-bit → lighter load
    # than nvidia's FP8-attention build. A3B routes ~3B active params/token.
    #
    # KV: the unsloth checkpoint has NO kv_cache_quant_algo/kv_cache_scheme
    # (verified on HF; no hf_quant_config.json at all), so kv_cache_dtype=
    # "auto" resolves to bf16 naturally — NO override dir needed (unlike
    # exp77/exp78). The fp8e4nv reshape_and_cache trap (exp71/72/75/77/78)
    # does not apply.
    #
    # ARCH (confirmed on exp78's run): Qwen3_5MoeForConditionalGeneration, a
    # MULTIMODAL (vision/video preprocessor) HYBRID linear-attention MoE —
    # vLLM supports it (recognized linear_attn as Mamba layers, 3/4 layers
    # Mamba + every 4th self_attn, prefix caching in Mamba 'align' mode).
    # limit_mm_per_prompt stops vLLM reserving the encoder cache (~5 GiB;
    # exp41/48/78 all needed it). Tiny KV footprint (~10 self_attn layers) →
    # full 32768 ctx fits a single A100-40GB at 4-bit. util 0.90 (exp77
    # sampling-OOM-safe; raise if load log shows free RAM).
    # ⚠️ Real risk = compressed-tensors NVFP4 fused-MoE on this hybrid
    # linear-attn arch on sm80 (first time on a Qwen3.5-MoE); read load log.
    # enable_prefix_caching=True reuses the doc-grouped few-shot+context
    # prefix (Mamba 'align' mode, experimental — verify in the engine log).
    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.90,
              enable_prefix_caching=True,
              dtype="bfloat16", kv_cache_dtype="auto",
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
