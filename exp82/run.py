"""
exp82 — S1=A3B → S2=gemma (hint = ANSWER text only); score ALL 4 combos.

Part of the 3×2 hint-type × direction grid (exp80–85). This cell:
  hint = preliminary answer text only (exp82/83), direction A3B→gemma.

Stage 1: A3B-Instruct-2507-FP8 does NORMAL V10 (answer + cite) → (S1 ans, S1 ref).
Stage 2: gemma-4-26B-A4B-it-FP8-Dynamic does NORMAL V10, its prompt carrying
         ONLY A3B's answer ("คำตอบเบื้องต้น: …") — no ref indices →
         (S2 ans, S2 ref).

Emits all 4 answer×ref combos (s1ans_s1ref / s1ans_s2ref / s2ans_s1ref /
s2ans_s2ref), scored full + leak-free. exp83 is the model-swapped twin.
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

MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MODEL_STAGE1 = os.environ.get("STAGE1_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
MODEL_STAGE2 = os.environ.get("STAGE2_MODEL", "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")
STAGE1_UTIL = float(os.environ.get("STAGE1_UTIL", "0.90"))  # A3B
STAGE2_UTIL = float(os.environ.get("STAGE2_UTIL", "0.95"))  # gemma

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
_SHOT1_ANSWER_TEXT = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา"
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


def build_prompt_v10_hinted(query, paras, prelim):
    """Normal V10 + Stage 1's ANSWER text only as the hint (Stage 2)."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำตอบเบื้องต้น: {prelim}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"โดยยึดตามคำตอบเบื้องต้นข้างต้น ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น "
        f"ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_stage1(query, paras):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10(query, paras)},
    ]


def build_messages_stage2(query, paras, prelim):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10_hinted(_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER_TEXT)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10_hinted(_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER_TEXT)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10_hinted(query, paras, prelim)},
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


def run_stage(model_name, model_kwargs, messages_list, gpu_mem_util=0.90):
    llm = LLM(model=model_name, max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=gpu_mem_util, enforce_eager=True,
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


def decode_outputs(items, raws):
    """Parse each raw output into (answer, ref_para_ids) keyed by query ID."""
    ans, refs = {}, {}
    empty = 0
    for it, raw in zip(items, raws):
        qid, gen_pids, gen_texts, _, q_text = it
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty += 1
        cited_idx = parse_citation(raw, len(gen_pids))
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]
        else:
            ref_ids = []
        ans[qid] = answer
        refs[qid] = ref_ids
    return ans, refs, empty


def write_submission(path, order, answers, refs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "abstractive", "refs"])
        writer.writeheader()
        for qid in order:
            writer.writerow({"ID": qid, "abstractive": answers[qid],
                             "refs": ",".join(refs[qid])})


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp82: S1={MODEL_STAGE1} → S2={MODEL_STAGE2} (hint=answer only); "
          f"emit 4 combos. {n} queries, {len(doc_index)} docs", flush=True)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # ----- Stage 1: normal V10 → (answer, ref) -----
    print(f"\n=== Stage 1: {MODEL_STAGE1} → answer + ref ===", flush=True)
    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_pids  = [p["para_id"] for p in valid]
        gen_texts = [p["text"]    for p in valid]
        msgs = build_messages_stage1(q_text, gen_texts) if valid \
            else [{"role": "user", "content": q_text}]
        items.append((query["ID"], gen_pids, gen_texts, msgs, q_text))

    s1_kwargs = dict(dtype="bfloat16", trust_remote_code=True,
                     enable_prefix_caching=True,
                     limit_mm_per_prompt={"image": 0, "video": 0})
    raws_s1 = run_stage(MODEL_STAGE1, s1_kwargs,
                        [it[3] for it in items], gpu_mem_util=STAGE1_UTIL)
    s1_ans, s1_ref, empty1 = decode_outputs(items, raws_s1)
    print(f"Stage 1: {len(s1_ans)} answers, empty={empty1}, "
          f"avg refs={sum(len(r) for r in s1_ref.values())/len(s1_ref):.2f}", flush=True)
    del raws_s1

    # ----- Stage 2: normal V10 hinted by S1's ANSWER → (answer, ref) -----
    print(f"\n=== Stage 2: {MODEL_STAGE2} → answer + ref (hinted by S1 answer) ===", flush=True)
    msgs_s2 = []
    for it in items:
        qid, gen_pids, gen_texts, _, q_text = it
        if gen_pids:
            msgs_s2.append(build_messages_stage2(q_text, gen_texts, s1_ans[qid]))
        else:
            msgs_s2.append([{"role": "user", "content": q_text}])

    s2_kwargs = dict(dtype="bfloat16", trust_remote_code=True,
                     enable_prefix_caching=True,
                     limit_mm_per_prompt={"image": 0, "video": 0})
    raws_s2 = run_stage(MODEL_STAGE2, s2_kwargs, msgs_s2, gpu_mem_util=STAGE2_UTIL)
    for it, raw in list(zip(items, raws_s2))[:3]:
        print(f"[S2 {it[0]}] {raw[:200]!r}", flush=True)
    s2_ans, s2_ref, empty2 = decode_outputs(items, raws_s2)
    print(f"Stage 2: {len(s2_ans)} answers, empty={empty2}, "
          f"avg refs={sum(len(r) for r in s2_ref.values())/len(s2_ref):.2f}", flush=True)
    del raws_s2

    # ----- Emit all 4 answer×ref combos -----
    order = [it[0] for it in items]
    combos = {
        "s1ans_s1ref": (s1_ans, s1_ref),
        "s1ans_s2ref": (s1_ans, s2_ref),
        "s2ans_s1ref": (s2_ans, s1_ref),
        "s2ans_s2ref": (s2_ans, s2_ref),
    }
    print(f"\nLegend: S1={MODEL_STAGE1}, S2={MODEL_STAGE2}", flush=True)
    for name, (ans, refs) in combos.items():
        out_path = Path(RESULT_DIR) / name / "submission.csv"
        write_submission(out_path, order, ans, refs)
        print(f"Written combo {name} → {out_path}", flush=True)
    return n


if __name__ == "__main__":
    n = main()
    benchmark_lib(n)
