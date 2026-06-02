"""
exp78 — model swap on the exp77 base: nvidia/Gemma-4-26B-A4B-NVFP4 →
**nvidia/Qwen3.6-35B-A3B-NVFP4** (single A100-40GB). Same exp51 pipeline
(V10_factual prompt + exp38 2-shot multi-turn shots, full doc, single-stage);
everything is identical to exp77 except the MODEL and the few arch-specific
knobs below.

WHY this model: the single-model bar to beat is the DENSE gemma-4-31B-NVFP4
(exp73 = 0.7140). Every MoE we have tried — gemma-4-26B-A4B FP8 (exp74 =
0.6970) and its NVFP4 builds (exp76 = 0.6882 RedHatAI, exp77 = 0.6982
nvidia) — sits ~0.7B-0.0158 below it, capped by ~4B active params (the
"active-param ceiling", cf. exp74). BUT every MoE so far has been gemma-4.
The PRODUCTION single model is Qwen3-30B-A3B-Instruct-2507 (exp42/v16 =
0.7087) — a Qwen A3B MoE that already beat all the gemma MoEs and sits only
−0.0053 under dense gemma-4-31B. This run asks whether the *newer* Qwen3.6
A3B base, in NVFP4, can close (or beat) that remaining gap as a single model
— i.e. whether a stronger A3B base lifts the active-param ceiling enough to
matter, while NVFP4 keeps it on one A100.

KV — the nvidia/ModelOpt builds bake "kv_cache_quant_algo": "FP8" into
hf_quant_config.json (RedHatAI does NOT), so a default kv_cache_dtype="auto"
promotes to fp8_e4m3 → sm80 has no fp8e4nv reshape_and_cache kernel → engine
init dies (the exp71/72/75/77 wall). This is checkpoint-baked, NOT
arch-specific, so the Qwen build is expected to carry the same directive.
FIX (same as exp77, backend-agnostic): submit_eval_train.sh builds a local
override dir (symlinks the snapshot, strips kv_cache_quant_algo +
kv_cache_scheme from the configs) so "auto" resolves to bf16, and run.py
leaves kv_cache_dtype at the default "auto". The exp77 nuance — that the
gemma MoE's TRITON_ATTN backend ALSO rejects an explicit "bfloat16" override
— may or may not apply to Qwen3-MoE (it likely selects FLASH_ATTN on sm80
with GQA), but the strip-directive override works either way, so it is the
safe choice. (If first-run log shows no kv directive baked in, the override
is a harmless no-op.)

ARCH knobs (same as exp77, confirmed by exp41/exp48):
- The 35B-A3B family resolves to a MULTIMODAL arch in vLLM (vision blocks
  loaded, idle for text-only chat). exp41/exp48 (Qwen3.5-35B-A3B-GPTQ-Int4,
  the direct predecessor) BOTH kept `limit_mm_per_prompt={"image":0,"video":0}`
  to stop vLLM reserving the encoder cache (~5 GiB). KEPT here.
- enable_thinking=False kept (exp41 ran the same; exp48 was the thinking-on
  variant — chose off to stay comparable to exp73/74/76/77). First-run raw
  dump guards against leaked <think> text.

ARCH (confirmed from the checkpoint, NOT gemma-like): this is
Qwen3_5MoeForConditionalGeneration — a HYBRID linear-attention MoE.
hf_quant_config quant_algo=MIXED_PRECISION: the experts/shared-experts are
W4A16_NVFP4 (group_size 16, the bulk), but ATTENTION IS FP8, not bf16 — and
3 of every 4 layers use `linear_attn` (Mamba-style state, no standard KV
cache); only every 4th layer (3,7,…,39) is full `self_attn`. So the KV-cache
footprint is tiny (10 attention layers, not 40) → full 32768 ctx should fit
a single A100-40GB easily at 4-bit (cf. exp41/48 fit the 35B family at Int4
~22.78 GiB). util kept at 0.90 (exp77's sampling-OOM-safe value); raise
toward 0.92-0.95 if the load log shows ample free GPU RAM. Never TP=2
(memory:prefer-single-gpu).

⚠️ BIGGEST open risk is vLLM SUPPORT for this hybrid linear-attn MoE arch on
the installed version (linear_attn = SSM-style state; needs a mamba/hybrid
code path). If the engine rejects the arch or the linear-attn layers, that is
the verdict — read the load log first.

⚠️ OPEN RISKS (first-run raw dump guards against silent failure):
- NVFP4 *MoE* fused-expert dequant on A100/sm80 — exp76/exp77 confirmed the
  MARLIN NvFp4 MoE backend runs on sm80 for gemma-4; Qwen3-MoE uses the same
  fused-MoE path, but verify in the load log.
- KV directive baked in (see above) — check load log; override handles it.
- enable_prefix_caching honored for this arch — check log.

Decision rule: beat exp42/v16 (0.7087, the Qwen A3B production single model)
to be interesting; beat dense exp73 (0.7140) to become the new best single
model. Below 0.7087 → the active-param ceiling holds across Qwen generations
too, and dense gemma-4-31B-NVFP4 stays the single-model champion.
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
MODEL_NAME     = os.environ.get("LLM_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")

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
    print(f"exp78: {n} queries, {len(doc_index)} docs "
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

    # nvidia/Qwen3.6-35B-A3B-NVFP4: ModelOpt-packed NVFP4 (quant_method=
    # modelopt). On A100/sm80 (no native FP4 cores) vLLM uses an FP4
    # weight-only dequant path — Marlin for dense Linear (exp73/75-confirmed)
    # and the MARLIN NvFp4 fused-MoE backend for the experts (exp76/exp77-
    # confirmed on sm80 for gemma-4; Qwen3-MoE uses the same fused-MoE path —
    # verify in the load log). A3B routes ~3B active params/token.
    #
    # ⚠️ KV-dtype: ModelOpt typically bakes "kv_cache_quant_algo": "FP8" into
    # hf_quant_config.json → a default "auto" promotes to fp8_e4m3 → sm80 has
    # no fp8e4nv reshape_and_cache kernel → engine init dies (exp71/72/75/77).
    # FIX (submit_eval_train.sh): LLM_MODEL points at a local override dir that
    # symlinks the snapshot but STRIPS kv_cache_quant_algo / kv_cache_scheme
    # from the configs → "auto" resolves to bf16. So here kv_cache_dtype is
    # left at the default "auto" — the neutralized config, not a dtype
    # override, is what forces bf16 KV. (Unlike exp77's gemma TRITON_ATTN
    # path, Qwen3-MoE likely selects FLASH_ATTN on sm80; the strip-directive
    # override is backend-agnostic, so it is correct either way.)
    #
    # ARCH (confirmed from checkpoint): Qwen3_5MoeForConditionalGeneration, a
    # MULTIMODAL (vision/video preprocessor present) HYBRID linear-attention
    # MoE. exp41/exp48 (Qwen3.5-35B-A3B-GPTQ-Int4, predecessor) needed
    # limit_mm_per_prompt to stop vLLM reserving the encoder cache (~5 GiB) —
    # kept here. quant: experts=W4A16_NVFP4 (bulk), attention=FP8; 3/4 layers
    # are linear_attn (SSM state, no KV), only every 4th is self_attn → tiny
    # KV footprint → full 32768 ctx fits a single A100-40GB at 4-bit (exp41/48
    # fit the 35B family at Int4 ~22.78 GiB). util 0.90 (exp77 sampling-OOM-
    # safe; exp41/48 ran 0.95 on GPTQ — raise if load log shows free RAM).
    # ⚠️ Real risk = vLLM support for this hybrid linear-attn arch; read log.
    # enable_prefix_caching=True reuses the doc-grouped few-shot+context
    # prefix (verify honored for this arch in the engine log).
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
