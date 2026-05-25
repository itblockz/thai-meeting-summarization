"""
exp35 — ceiling test: NO RETRIEVAL, feed the entire doc to the LLM.

Drops every retrieval stage (dense, BM25, cross-encoder rerank) and
passes the full list of *valid* paragraphs from `doc_id` to Qwen3-32B-AWQ
in their original document order. E5 self-citation then has to pick the
right paragraph(s) from up to 271 candidates.

Purpose: measure the LLM's intrinsic ability and the true score
ceiling — strip out any retrieval bias / mistake, see what Qwen3-32B-AWQ
can do with maximum context. If composite >= exp34 (0.6790) then
retrieval is helping the model focus; if << exp34, retrieval is
critical signal-cleanup.

Token math (eval_retrieval/measure_full_doc_tokens.py on the train set,
Qwen3 tokenizer, including system + 2-shot + query + chat template):
  min=8916  median=14139  mean=14899  p90=19282  p95=23069  max=28646
  -> max_model_len must be >= 28646; bumped to 32768 to leave headroom.

Resource impact:
  * KV cache budget = ~64K tokens (same GPU mem 0.90), so max
    concurrency drops from ~4x (16K context) to ~2x (32K context).
  * Mean prompt 14.9K vs exp30's 4.2K -> ~3.5x more tokens to process
    per query -> expected runtime ~3.5h (vs exp30's 52min).
  * MAX_NEW_TOKENS unchanged at 512.

Risks (this is a ceiling/ablation experiment, not necessarily a winner):
  * LLM attention degrades at 15-29K context — gold paragraphs deep in
    the doc may be ignored.
  * avg refs/query may explode (6x more candidates -> spurious cites).
  * SS/RougeL likely flat or worse (more distractor noise); IoU could
    go either way (no gold-loss from retrieval mistakes, but harder to
    pick the right one out of 165).

Everything else identical to exp30/exp34:
  E5 self-cite, exp08 2-shot, Qwen3-32B-AWQ, enforce_eager=True.
  Valid-paragraph filter unchanged from exp30 (filter_valid_paragraphs).
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

MAX_NEW_TOKENS = 512
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"

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

_SHOT2_QUERY = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมาธิการจัดขึ้นเพื่ออะไร"
_SHOT2_PARAS = [
    "เริ่มประชุมเวลา ๐๙.๔๖ นาฬิกา",
    "เมื่อกรรมาธิการมาครบองค์ประชุมแล้ว ประธานคณะกรรมาธิการได้กล่าวเปิดประชุม และดำเนินการประชุมตามระเบียบวาระการประชุม สรุปสาระสำคัญได้ ดังนี้",
    "ระเบียบวาระที่ ๑ เรื่องที่ประธานแจ้งต่อที่ประชุม",
    "สำนักงานเลขาธิการสภาผู้แทนราษฎรขอความอนุเคราะห์ตอบแบบสำรวจความพึงพอใจและความไม่พึงพอใจของคณะกรรมาธิการต่อการบริหารจัดการด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา เพื่อนำผลการประเมินความพึงพอใจและความไม่พึงพอใจมาเป็นข้อมูลในการทบทวน ปรับปรุง และพัฒนาการปฏิบัติงานให้มีประสิทธิภาพต่อไป",
    "ที่ประชุมรับทราบ",
]
_SHOT2_ANSWER = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมการในครั้งนี้ มีการจัดทำขึ้นเพื่อนำข้อมูลที่ได้มาทบทวน ปรับปรุง รวมถึงนำไปพัฒนาการปฏิบัติงานให้มีประสิทธิภาพยิ่งขึ้น [อ้างอิง: 4]"


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
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
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
    print(f"exp35: {n} queries, {len(doc_index)} docs "
          f"(NO RETRIEVAL — full doc, max_model_len={MAX_MODEL_LEN})", flush=True)

    # Pre-build full-doc paragraph lists (in document order) once per doc.
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

    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.90, dtype="half", enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[3] if it[3] is not None else [{"role": "user", "content": it[4]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    outputs = llm.generate(prompts, sampling)

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
