"""
exp70 — exp51 + FULL-DOC few-shot, 2-shot (single-ref + multi-ref).

Sibling of exp68 (1-shot multi-ref) and exp69 (1-shot single-ref). Same
full-doc shot-context change vs exp51 (A3B-Instruct + V10_factual, 0.7110
leak-free), but keeps BOTH examples — the direct full-doc analog of
exp51's 2-shot setup:

  shot 1 — Q0745 single-ref (P6, 'where was meeting 49 held')
  shot 2 — Q0746 multi-ref  (P21–P24, 'how many absentees')

  Each shot displays the ENTIRE doc_050 (198 valid paras, ~22.9K tokens)
  numbered [1..198]; the [อ้างอิง: …] tags are recomputed to the gold
  refs' REAL positions in the full doc (P6 → its index; P21–P24 → theirs).

exp68/exp69/exp70 ablate the few-shot composition on top of the full-doc
realism change. exp70 carries both 71.8%-single-ref prior (shot 1) and
the multi-ref citation signal (shot 2).

Cost: doc_050 (22.9K tok) shown in EACH shot → shared shot prefix ≈ 45.7K.
Worst-case prompt = 2*22.9K + largest real doc(doc_006 27.6K) ≈ 73.3K
(+template/answer ≈ 75K), so max_model_len 32768 → 81920. Runs on a
SINGLE A100-40GB (user-chosen over TP=2): A3B-FP8 ~30 GB weights leave
~8 GB for KV; prefix caching stores the 45.7K shot prefix once (~4.4 GB)
and each query appends only its own doc, so the peak single-sequence KV
(~7 GB for the longest doc) is tight but fits with effectively batch≈1–2
on the largest docs. If it OOMs on the largest docs, fall back to TP=2
(set TP_SIZE=2 + gpus-per-node=2 in the submit script).

Leak-free: shots are doc_050, held out from scoring (same as exp51).

Decision rule: accept if leak-free composite >= exp51 (0.7110).
Hypothesis: full-doc realism + both-shot composition gives the best of
exp68 (multi-ref recall) and exp69 (single-ref precision); risk is the
doubled shot prefix dilutes attention / the model over-reads the
duplicated example doc.
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
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "81920"))
TP_SIZE        = int(os.environ.get("TP_SIZE", "1"))
MODEL_NAME     = os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

# The held-out doc the few-shot examples are drawn from.
SHOT_DOC_ID = "doc_050"

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Few-shot examples drawn from SHOT_DOC_ID. Each is (query, gold ref
# para_ids, answer prose WITHOUT the citation tag). The [อ้างอิง: …] tag
# is recomputed at runtime from each ref's position in the FULL filtered
# doc, so the demonstration uses realistic large indices.
_SHOT_MULTIREF = (
    "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมมีจำนวนกี่คน",
    ["P21", "P22", "P23", "P24"],
    "ในการประชุมคณะกรรมาธิการการเงิน การคลัง สถาบันการเงินและตลาดการเงิน ครั้งที่ 49 มีกรรมการผู้ที่ไม่มาประชุมจำนวน 3 คน",
)
_SHOT_SINGLEREF = (
    "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด",
    ["P6"],
    "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
)

# exp70 = 2-shot: single-ref first (71.8% prior), then multi-ref.
SHOTS = [_SHOT_SINGLEREF, _SHOT_MULTIREF]


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
    """V10_factual prompt (exp50/exp51) — CONTEXT FIRST, then query."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


def build_shot_messages(shot_doc_valid):
    """Build the few-shot user/assistant turns from the FULL shot doc.

    shot_doc_valid: filtered paragraph dicts of SHOT_DOC_ID. Each shot's
    context is the entire doc (numbered [1..N]); citation indices are the
    gold refs' 1-based positions in that same numbering.
    """
    pidpos = {p["para_id"]: i + 1 for i, p in enumerate(shot_doc_valid)}
    shot_texts = [p["text"] for p in shot_doc_valid]
    msgs = []
    for query, ref_ids, prose in SHOTS:
        positions = sorted(pidpos[r] for r in ref_ids if r in pidpos)
        if not positions:
            raise RuntimeError(f"shot refs {ref_ids} not found in {SHOT_DOC_ID}")
        tag = "[อ้างอิง: " + ", ".join(str(x) for x in positions) + "]"
        msgs.append({"role": "user", "content": build_prompt(query, shot_texts)})
        msgs.append({"role": "assistant", "content": f"{prose} {tag}"})
    return msgs


def build_messages(query, paras, shot_msgs):
    return (
        [{"role": "system", "content": SYSTEM_MSG}]
        + shot_msgs
        + [{"role": "user", "content": build_prompt(query, paras)}]
    )


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

    doc_paras = {}
    for doc_id, paragraphs in doc_index.items():
        doc_paras[doc_id] = filter_valid_paragraphs(paragraphs)

    shot_doc_valid = doc_paras.get(SHOT_DOC_ID)
    if not shot_doc_valid:
        raise RuntimeError(f"{SHOT_DOC_ID} not in dataset — cannot build few-shot")
    shot_msgs = build_shot_messages(shot_doc_valid)
    print(f"exp70: {n} queries, {len(doc_index)} docs "
          f"(2-shot FULL-DOC few-shot from {SHOT_DOC_ID} "
          f"[{len(shot_doc_valid)} paras x2], model={MODEL_NAME}, "
          f"TP={TP_SIZE}, max_model_len={MAX_MODEL_LEN})",
          flush=True)

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
            messages = build_messages(q_text, gen_texts, shot_msgs)
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((query["ID"], gen_pids, gen_texts, messages, q_text))

    print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
          f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.95,
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
