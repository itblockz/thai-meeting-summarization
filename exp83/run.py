"""
exp83 — answer-FIRST attribution, answer+cite Stage 2.

Same as exp82 (A3B answers first → gemma produces the refs; A3B's answer is
the final abstractive) EXCEPT Stage 2's gemma prompt: instead of cite-only,
gemma does its FAMILIAR V10 answer+cite task with A3B's answer injected as a
"คำตอบเบื้องต้น" (preliminary answer). gemma re-writes an answer (DISCARDED)
and cites — we keep ONLY its citation as refs.

WHY two variants: cite-only (exp82) is the truest "attribution" framing but an
UNFAMILIAR output format for gemma. answer+cite (this) keeps gemma in the exact
V10 groove where it hit the record IoU 0.8139 (exp74) — the citation rides on
generating an answer, which is gemma's proven behavior. A/B tests whether the
unfamiliar cite-only prompt degrades gemma's citation vs the proven format.

Final abstractive = A3B's answer in BOTH variants; only gemma's refs differ.
See exp82 header for the full answer→refs rationale.
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
MODEL_ANSWER = os.environ.get("ANSWER_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
MODEL_REFS   = os.environ.get("REFS_MODEL",   "RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic")

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
    """V10_factual — Stage 1 (A3B answer) prompt."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_prompt_refine(query, paras, prelim):
    """Stage 2 (gemma) answer+cite: V10 with A3B's answer as preliminary."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำตอบเบื้องต้น: {prelim}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"โดยยึดตามคำตอบเบื้องต้นข้างต้น ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_messages_answer(query, paras):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_v10(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_v10(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_v10(query, paras)},
    ]


def build_messages_refine(query, paras, prelim):
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",      "content": build_prompt_refine(_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER_TEXT)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user",      "content": build_prompt_refine(_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER_TEXT)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user",      "content": build_prompt_refine(query, paras, prelim)},
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


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp83 (answer-first, answer+cite attribution): {n} queries, "
          f"{len(doc_index)} docs", flush=True)

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    # ----- Stage 1: A3B writes the answer FIRST (no hint) -----
    print(f"\n=== Stage 1: {MODEL_ANSWER} → answer (final abstractive) ===", flush=True)
    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        valid = doc_paras.get(query["doc_id"], [])
        q_text = query["query"]
        gen_pids  = [p["para_id"] for p in valid]
        gen_texts = [p["text"]    for p in valid]
        msgs = build_messages_answer(q_text, gen_texts) if valid \
            else [{"role": "user", "content": q_text}]
        items.append((query["ID"], gen_pids, gen_texts, msgs, q_text))

    a3b_kwargs = dict(dtype="bfloat16", trust_remote_code=True,
                      limit_mm_per_prompt={"image": 0, "video": 0})
    raws_answer = run_stage(MODEL_ANSWER, a3b_kwargs,
                            [it[3] for it in items], gpu_mem_util=0.90)

    answers = {}
    empty_answers = 0
    for it, raw in zip(items, raws_answer):
        qid, gen_pids, gen_texts, _, q_text = it
        answer = split_answer(raw)
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
            empty_answers += 1
        answers[qid] = answer
    print(f"Stage 1 wrote {len(answers)} answers; empty_answers={empty_answers}",
          flush=True)
    del raws_answer

    # ----- Stage 2: gemma re-answers (discarded) + cites → refs -----
    print(f"\n=== Stage 2: {MODEL_REFS} → refs via answer+cite (answer discarded) ===", flush=True)
    msgs_refine = []
    for it in items:
        qid, gen_pids, gen_texts, _, q_text = it
        if gen_pids:
            msgs_refine.append(build_messages_refine(q_text, gen_texts, answers[qid]))
        else:
            msgs_refine.append([{"role": "user", "content": q_text}])

    gemma_kwargs = dict(dtype="bfloat16", trust_remote_code=True,
                        enable_prefix_caching=True,
                        limit_mm_per_prompt={"image": 0, "video": 0})
    raws_refs = run_stage(MODEL_REFS, gemma_kwargs, msgs_refine, gpu_mem_util=0.95)

    # FIRST-RUN SANITY: dump a few raw gemma outputs (answer+cite; answer discarded).
    for it, raw in list(zip(items, raws_refs))[:3]:
        print(f"[refine {it[0]}] {raw[:200]!r}", flush=True)

    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    results = []
    n_explicit = 0
    ref_counts = []
    for it, raw in zip(items, raws_refs):
        qid, gen_pids, gen_texts, _, q_text = it
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
        results.append({"ID": qid, "abstractive": answers[qid],
                        "refs": ",".join(ref_ids)})
    print(f"attribution: {n_explicit}/{len(results)} emitted tag, "
          f"avg refs/query={sum(ref_counts)/len(ref_counts):.2f}", flush=True)

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
