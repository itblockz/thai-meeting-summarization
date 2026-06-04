"""
exp90 — PUBLISHER swap on exp89's dense Qwen3-32B-NVFP4 line:
nvidia/Qwen3-32B-NVFP4 (ModelOpt) → RedHatAI/Qwen3-32B-NVFP4 (compressed-tensors).
Same single-stage recipe as exp89/exp51 (E5/system + exp38 2-shot multi-turn
shots, full doc, single-stage); ONLY the checkpoint publisher changes.

WHY this run: exp89 ran the nvidia/ModelOpt build of Qwen3-32B-NVFP4. The
publisher choice is a KNOWN, arch-dependent lever on this task (CLAUDE.md):
- ModelOpt `exclude_modules` the self_attn* layers → attention stays bf16
  (heavier load, ~25 GiB est. on a dense 32B) and bakes a "kv_cache_quant_algo":
  "FP8" directive that the submit-script override dir must strip on sm80.
- RedHatAI ships compressed-tensors `nvfp4-pack-quantized` with
  `config_groups.targets:['Linear']` → quantizes the decoder's self_attn TOO
  → 4-bit attention → ~17 GiB load (lighter), and carries NO `hf_quant_config`
  / NO baked KV directive (`kv_cache_scheme:None`) → "auto" KV resolves to bf16
  with NO override dir (exp79's compressed-tensors path confirmed this).

For DENSE ≥30B models the verdict so far favors RedHatAI: on the dense
gemma-4-31B the nvidia/ModelOpt build was INFEASIBLE on a single A100-40GB
(bf16 attn → 29.96 GiB + gemma's expensive sliding-window KV floor, exp75),
while the RedHatAI build fit and won (exp73, 0.7140 — best single model). The
publisher REVERSES on the MoE (ModelOpt wins, exp77), but Qwen3-32B is DENSE,
so exp90 tests whether the dense rule (RedHatAI > ModelOpt) holds for Qwen3-32B
the same way it did for gemma-4-31B. Note: Qwen3-32B is text-only with cheap
GQA KV (8 KV heads, no gemma sliding-window floor), so unlike gemma-4 BOTH
publishers should FIT a single A100 here (exp89 already verifies the ModelOpt
side fits) — exp90 is therefore a clean ANSWER-QUALITY publisher comparison on
Qwen3-32B, not a feasibility test.

Two axes vs the immediate neighbors:
  (a) publisher axis — RedHatAI vs nvidia/ModelOpt on the SAME Qwen3-32B-NVFP4
      (bar = exp89's number);
  (b) quant axis — NVFP4 vs AWQ on the SAME Qwen3-32B base (cf. exp37 AWQ
      0.6944, exp37 recipe though, not exp51 — keep the comparison loose).

⚠️ OPEN RISKS / load-log guards (first-run raw dump below catches gibberish):
- Pure NVFP4 on sm80: compressed-tensors `nvfp4-pack-quantized` resolves to
  vLLM quant method `compressed-tensors` → NvFp4 GEMM via MARLIN weight-only
  dequant on A100/sm80 (exp79-confirmed for this exact layout). This is NOT
  nvidia's sm89-gated `modelopt_mixed` (exp78 wall) — RedHatAI is pure NVFP4,
  no MIXED_PRECISION. Confirm `quant method compressed-tensors` in the log.
- KV-dtype: RedHatAI carries no FP8-KV directive, so kv_cache_dtype="auto"
  resolves to bf16 with NO override dir. (submit_eval_train.sh points LLM_MODEL
  straight at the cached snapshot — contrast exp89's override-dir dance.)
- Fit: 4-bit attention (~17 GiB) + cheap GQA KV → full 32768 ctx fits with room
  to spare; util 0.92 as exp89. If KV ever won't fit, trim MAX_MODEL_LEN (worst
  prompt ~19.3K tok → 20480 truncates ~0) — never TP=2 (memory:prefer-single-gpu).

Compatibility (vllm 0.19.1 / transformers 5.8.1):
- Qwen3-32B = Qwen3ForCausalLM (text-only); chat template supports the system
  role and enable_thinking=False (no-think, exp03/37 lineage) → exp51's 2-shot
  multi-turn structure reused unchanged. limit_mm_per_prompt is a harmless
  no-op for a text model (exp51's A3B kept it).

Decision rule: accept if leak-free composite >= exp73 (0.7140, single-model
best). Headline comparisons: vs exp89 (publisher axis, same model+recipe) and
vs exp73 (model axis, dense NVFP4 best).
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
MODEL_NAME     = os.environ.get("LLM_MODEL", "RedHatAI/Qwen3-32B-NVFP4")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Same shots as exp38/39/51/73/74/76/77/89
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
    print(f"exp90: {n} queries, {len(doc_index)} docs "
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

    # RedHatAI/Qwen3-32B-NVFP4: compressed-tensors `nvfp4-pack-quantized` on a
    # DENSE text-only Qwen3-32B (Qwen3ForCausalLM). config_groups.targets:
    # ['Linear'] → the decoder's self_attn is quantized too (4-bit attention,
    # ~17 GiB load) — contrast exp89's nvidia/ModelOpt build which excludes
    # self_attn (bf16 attn, ~25 GiB). On A100/sm80 (no native FP4 cores) vLLM
    # uses the weight-only FP4 dequant path (NvFp4 MARLIN GEMM, exp79-confirmed
    # for this compressed-tensors layout).
    #
    # KV-dtype: RedHatAI carries NO hf_quant_config / NO baked FP8-KV directive
    # (kv_cache_scheme:None), so the default kv_cache_dtype="auto" resolves to
    # bf16 directly — NO override dir needed (exp89's override-dir strip was
    # only for nvidia/ModelOpt's baked "kv_cache_quant_algo":"FP8"). LLM_MODEL
    # points straight at the cached snapshot (submit_eval_train.sh).
    #
    # Fit (single A100-40GB): 4-bit attn (~17 GiB) + cheap GQA KV (8 KV heads,
    # no gemma sliding-window floor) → full 32768 KV fits with room to spare.
    # util 0.92 with full 32768 ctx; if KV ever won't fit, trim MAX_MODEL_LEN
    # (worst prompt ~19.3K tok → 20480 truncates ~0) — never TP=2
    # (memory:prefer-single-gpu). enable_prefix_caching reuses the doc-grouped
    # few-shot+context prefix across a doc's queries (verify honored in log).
    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.92,
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
