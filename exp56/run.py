"""
exp56 — Pipeline 2 (hint, refs fixed): two-stage LLM. Stage A picks refs;
Stage B writes answer on FULL context with a prompt hint pointing at
Stage A's refs.

Stage A: load Qwen3.6-27B-FP8, V10_factual + exp38 shots, full doc
         → fresh per-query refs.
Stage B: free 27B, load Qwen3-32B-AWQ. For each query, build FULL doc
         context plus a "เน้นย่อหน้า [X, Y, Z] เป็นข้อมูลหลัก" line where
         X/Y/Z are the 1-based positions of Stage A's refs in the full
         context. exp38 writes answer; its citations are PARSED but
         DISCARDED — final refs are FIXED to Stage A's selection.

No precomputed CSVs are read — both stages are fresh model output.
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

MAX_NEW_TOKENS = 512
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MODEL_27B   = "Qwen/Qwen3.6-27B-FP8"
MODEL_AWQ   = "Qwen/Qwen3-32B-AWQ"

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

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


def build_prompt_v10(query, paras):
    """Stage A — exp50's V10_factual prompt for ref-picking."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_exp38_with_hint(query, paras, hint_idx):
    """Stage B — exp38 prompt + 'focus on paragraphs [...]' hint."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    hint_str = ", ".join(str(i) for i in hint_idx) if hint_idx else "—"
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"**โดยเน้นย่อหน้าหมายเลข [{hint_str}] เป็นข้อมูลหลัก** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_exp38(query, paras):
    """Shot template (no hint) for few-shot examples."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_v10(query, paras):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10(query, paras)},
    ]


def build_messages_exp38_hint(query, paras, hint_idx):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_exp38(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_exp38(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_exp38_with_hint(query, paras, hint_idx)},
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


def run_stage(model_name, model_kwargs, messages_list):
    llm = LLM(model=model_name, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, enforce_eager=True,
              **model_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS,
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
    print(f"exp56 (hint, refs fixed): {n} queries, {len(doc_index)} docs", flush=True)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # ----- Stage A -----
    print(f"\n=== Stage A: {MODEL_27B} → fresh refs ===", flush=True)
    items_A = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_pids  = [p["para_id"] for p in valid]
        gen_texts = [p["text"]    for p in valid]
        msgs = build_messages_v10(q_text, gen_texts) if valid \
            else [{"role": "user", "content": q_text}]
        items_A.append((query["ID"], gen_pids, gen_texts, msgs, q_text))

    raws_A = run_stage(MODEL_27B, dict(dtype="bfloat16", trust_remote_code=True,
                                       limit_mm_per_prompt={"image": 0, "video": 0}),
                       [it[3] for it in items_A])

    stage_a_refs = {}
    stage_a_hint = {}
    for it, raw in zip(items_A, raws_A):
        qid, gen_pids, _, _, _ = it
        cited_idx = parse_citation(raw, len(gen_pids))
        valid_idx = [j for j in cited_idx if j < len(gen_pids)]
        ref_pids = [gen_pids[j] for j in valid_idx]
        if not ref_pids and gen_pids:
            ref_pids = [gen_pids[0]]
            valid_idx = [0]
        stage_a_refs[qid] = ref_pids
        stage_a_hint[qid] = [j + 1 for j in valid_idx]  # 1-based positions
    print(f"Stage A produced refs for {len(stage_a_refs)} queries; "
          f"avg refs/query = {sum(len(r) for r in stage_a_refs.values())/len(stage_a_refs):.2f}",
          flush=True)
    del items_A, raws_A

    # ----- Stage B: full context + hint -----
    print(f"\n=== Stage B: {MODEL_AWQ} → answer w/ hint (refs fixed) ===", flush=True)
    items_B = []
    for query in queries:
        qid = query["ID"]
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_pids  = [p["para_id"] for p in valid]
        gen_texts = [p["text"]    for p in valid]
        hint_idx = stage_a_hint.get(qid, [])
        msgs = build_messages_exp38_hint(q_text, gen_texts, hint_idx) if valid \
            else [{"role": "user", "content": q_text}]
        items_B.append((qid, gen_pids, gen_texts, msgs, q_text, stage_a_refs.get(qid, [])))

    raws_B = run_stage(MODEL_AWQ, dict(quantization="awq_marlin", dtype="half"),
                       [it[3] for it in items_B])

    results = []
    empty_answers = 0
    for it, raw in zip(items_B, raws_B):
        qid, gen_pids, gen_texts, _, q_text, ref_pids = it
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty_answers += 1
        # refs FIXED to Stage A
        results.append({"ID": qid, "abstractive": answer,
                        "refs": ",".join(ref_pids)})
    print(f"empty_answers={empty_answers}", flush=True)

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
