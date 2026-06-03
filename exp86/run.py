"""
exp86 — best-of-both two-stage grid. BASE = exp81 (4-combo, ref-INDEX hint
handoff). Stage 1 = exp77's model+prompt; Stage 2 = exp37's model+prompt.

Goal: realize the single-model best-of-both ceiling as a REAL ref-hinted
pipeline. Column-merging exp37-ans + exp77-ref (the 0.7243 paper number) is
INVALID — exp37's answer was never produced under a ref hint ("ต้อง ref
hinted"). Here Stage 2's answer is actually generated WITH Stage 1's ref
indices as a hint, exactly like exp81.

  Stage 1 = exp77: nvidia/Gemma-4-26B-A4B-NVFP4 (config-override dir) +
            V10_factual prompt + exp38 shots (shot2 = multi-ref "ไม่มาประชุม").
            Best single-model REF picker (leak-free IoU 0.8165). Its
            [อ้างอิง: …] picks = refA AND the 1-based ref-INDEX hint for Stage 2.
  --- del llm; gc.collect(); torch.cuda.empty_cache() ---
  Stage 2 = exp37: Qwen3-32B-AWQ + E5 context-first prompt + exp37 shots
            (shot2 = single-ref "แบบสำรวจ"). Best single-model ANSWER writer
            (Rouge 0.494 / SS 0.864). The final query turn carries exp81's
            hint line "ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]" (shots use the
            non-hinted prompt). Writes ansB; its own citations = refB.

Emits the exp81 4-combo grid: ans{A,B} x ref{A,B}.
  - ansB_refA = AWQ(exp37) answer hinted by exp77 refs + exp77 refs <- TARGET ~0.725
  - ansA_refA = Stage 1 standalone = exp77 reproduction (SANITY: refA should
                match exp77/eval_result/submission.csv refs → IoU 0.8165).

Two NVFP4/AWQ models share one A100-40GB, loaded one at a time (cannot coexist).
Stage 1 model = exp77 config-override dir (FP8-KV directive stripped so vLLM
"auto" KV resolves to bf16 on sm80). Stage 1 kwargs replicate exp77 EXACTLY
(util 0.90, enable_prefix_caching, dtype bf16, limit_mm, 1024 new tok). Stage 2
kwargs replicate exp37 EXACTLY (awq_marlin, dtype half, util 0.90, 512 new tok).
"""
from pathlib import Path
import os
import re
import json
import csv
import gc

import torch
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

MAX_MODEL_LEN   = int(os.environ.get("MAX_MODEL_LEN", "32768"))
STAGE_A_MAX_NEW = int(os.environ.get("STAGE_A_MAX_NEW", "1024"))  # exp77
STAGE_B_MAX_NEW = int(os.environ.get("STAGE_B_MAX_NEW", "512"))   # exp37

MODEL_STAGE_A = os.environ.get("LLM_MODEL_STAGE_A",
                               os.path.join(os.environ.get("PROJECT", ""),
                                            "exp86/model_override"))
MODEL_STAGE_B = os.environ.get("LLM_MODEL_STAGE_B", "Qwen/Qwen3-32B-AWQ")

# Shared across both stages (exp37 / exp77 use identical system msg + shot1).
SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Shot 1 — single-ref, IDENTICAL in exp37 and exp77/exp38.
_SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
_SHOT1_PARAS = [
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "_________________________",
    "กรรมาธิการผู้มาประชุม",
]
_SHOT1_ANSWER = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา [อ้างอิง: 3]"

# Stage A shot 2 — exp77/exp38 MULTI-ref ("กรรมการผู้ไม่มาประชุม").
_A_SHOT2_QUERY = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมมีจำนวนกี่คน"
_A_SHOT2_PARAS = [
    "๑๒. นางสาวแอนศิริ วลัยกนก กรรมาธิการ",
    "กรรมาธิการผู้ไม่มาประชุม",
    "๑. นายพิบูลย์ รัชกิจประการ (ลาการประชุม)",
    "๒. นายธนยศ ทิมสุวรรณ (ลาการประชุม)",
    "๓. นายอัคร ทองใจสด (ลาการประชุม)",
]
_A_SHOT2_ANSWER = "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมจำนวน 3 คน [อ้างอิง: 2, 3, 4, 5]"

# Stage B shot 2 — exp37 SINGLE-ref ("แบบสำรวจความพึงพอใจ").
_B_SHOT2_QUERY = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมาธิการจัดขึ้นเพื่ออะไร"
_B_SHOT2_PARAS = [
    "เริ่มประชุมเวลา ๐๙.๔๖ นาฬิกา",
    "เมื่อกรรมาธิการมาครบองค์ประชุมแล้ว ประธานคณะกรรมาธิการได้กล่าวเปิดประชุม และดำเนินการประชุมตามระเบียบวาระการประชุม สรุปสาระสำคัญได้ ดังนี้",
    "ระเบียบวาระที่ ๑ เรื่องที่ประธานแจ้งต่อที่ประชุม",
    "สำนักงานเลขาธิการสภาผู้แทนราษฎรขอความอนุเคราะห์ตอบแบบสำรวจความพึงพอใจและความไม่พึงพอใจของคณะกรรมาธิการต่อการบริหารจัดการด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา เพื่อนำผลการประเมินความพึงพอใจและความไม่พึงพอใจมาเป็นข้อมูลในการทบทวน ปรับปรุง และพัฒนาการปฏิบัติงานให้มีประสิทธิภาพต่อไป",
    "ที่ประชุมรับทราบ",
]
_B_SHOT2_ANSWER = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมการในครั้งนี้ มีการจัดทำขึ้นเพื่อนำข้อมูลที่ได้มาทบทวน ปรับปรุง รวมถึงนำไปพัฒนาการปฏิบัติงานให้มีประสิทธิภาพยิ่งขึ้น [อ้างอิง: 4]"


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


# ----- Stage A prompt: exp77 V10_factual ("สั้นและตรงประเด็น") -----
def build_prompt_A(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_A(query, paras):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_A(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_A(_A_SHOT2_QUERY, _A_SHOT2_PARAS)},
        {"role": "assistant", "content": _A_SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_A(query, paras)},
    ]


# ----- Stage B prompt: exp37 E5 ("กระชับและครอบคลุม"), + exp81 hint on query turn -----
def build_prompt_B(query, paras):
    """exp37's verbatim prompt (no hint) — used for the shot turns."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_B_hinted(query, paras, hint_idx):
    """exp37 prompt + exp81 ref-INDEX hint line (base = exp81)."""
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


def build_messages_B(query, paras, hint_idx):
    return [
        {"role": "system",    "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_B(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_B(_B_SHOT2_QUERY, _B_SHOT2_PARAS)},
        {"role": "assistant", "content": _B_SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_B_hinted(query, paras, hint_idx)},
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


def split_answer(text):
    return re.sub(r'\s*\[อ้างอิง[^\]]*\]', '', text).strip()


def run_stage(model_name, model_kwargs, messages_list, max_new_tokens):
    llm = LLM(model=model_name, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, enforce_eager=True, **model_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens,
                              repetition_penalty=1.05)
    rendered = [tokenizer.apply_chat_template(
        m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for m in messages_list]
    outputs = llm.generate(rendered, sampling)
    raws = [o.outputs[0].text.strip() for o in outputs]
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return raws


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp86 grid (base exp81): {n} queries, {len(doc_index)} docs", flush=True)
    print(f"Stage 1 (ref, exp77) = {MODEL_STAGE_A}", flush=True)
    print(f"Stage 2 (ans, exp37) = {MODEL_STAGE_B}", flush=True)

    doc_paras = {d: filter_valid_paragraphs(p) for d, p in doc_index.items()}

    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        valid = doc_paras.get(query["doc_id"], [])
        items.append({
            "ID":    query["ID"],
            "q":     query["query"],
            "pids":  [p["para_id"] for p in valid],
            "texts": [p["text"]    for p in valid],
        })

    # ---------- Stage 1 (exp77): NVFP4 gemma V10 -> ansA + refA + hint ----------
    print(f"\n=== Stage 1 (exp77): {MODEL_STAGE_A} -> refs ===", flush=True)
    msgsA = [build_messages_A(it["q"], it["texts"]) if it["texts"]
             else [{"role": "user", "content": it["q"]}] for it in items]
    raws_A = run_stage(
        MODEL_STAGE_A,
        dict(dtype="bfloat16", kv_cache_dtype="auto",
             enable_prefix_caching=True, trust_remote_code=True,
             limit_mm_per_prompt={"image": 0, "video": 0}),
        msgsA, STAGE_A_MAX_NEW)
    for it, raw in zip(items, raws_A):
        pids = it["pids"]
        ans = split_answer(raw)
        it["ansA"] = ans or (it["texts"][0] if it["texts"] else it["q"])
        cited = [j for j in parse_citation(raw, len(pids)) if j < len(pids)]
        refA = [pids[j] for j in cited]
        if not refA and pids:
            refA, cited = [pids[0]], [0]
        it["refA"] = refA
        it["hint"] = [j + 1 for j in cited]  # 1-based positions
    print(f"Stage 1 avg refs/query = "
          f"{sum(len(it['refA']) for it in items) / len(items):.2f}", flush=True)
    del raws_A, msgsA

    # ---------- Stage 2 (exp37): 32B-AWQ hinted by refA -> ansB + refB ----------
    print(f"\n=== Stage 2 (exp37): {MODEL_STAGE_B} (E5 + hint) -> answer ===", flush=True)
    msgsB = [build_messages_B(it["q"], it["texts"], it["hint"]) if it["texts"]
             else [{"role": "user", "content": it["q"]}] for it in items]
    raws_B = run_stage(MODEL_STAGE_B, dict(quantization="awq_marlin", dtype="half"),
                       msgsB, STAGE_B_MAX_NEW)
    for it, raw in zip(items, raws_B):
        pids = it["pids"]
        ans = split_answer(raw)
        it["ansB"] = ans or (it["texts"][0] if it["texts"] else it["q"])
        cited = [j for j in parse_citation(raw, len(pids)) if j < len(pids)]
        refB = [pids[j] for j in cited]
        if not refB and pids:
            refB = [pids[0]]
        it["refB"] = refB
    del raws_B, msgsB

    # ---------- emit the exp81 4-combo grid ----------
    combos = [("ansA_refA", "ansA", "refA"),
              ("ansA_refB", "ansA", "refB"),
              ("ansB_refA", "ansB", "refA"),   # <- best-of-both target
              ("ansB_refB", "ansB", "refB")]
    for name, akey, rkey in combos:
        od = Path(RESULT_DIR) / name
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "submission.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
            w.writeheader()
            for it in items:
                w.writerow({"ID": it["ID"], "abstractive": it[akey],
                            "refs": ",".join(it[rkey])})
        print(f"wrote {name} -> {od/'submission.csv'} ({len(items)} rows)", flush=True)
    return n


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)
