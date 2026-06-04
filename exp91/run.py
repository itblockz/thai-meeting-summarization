"""
exp91 — exp73 (gemma-4-31B-NVFP4, best single model 0.7140) + thinking ON.

Single-variable change from exp73: flip enable_thinking=False → True in
apply_chat_template. Everything else (RedHatAI/gemma-4-31B-it-NVFP4 on a
single A100-40GB, V10_factual prompt, exp38 2-shot, greedy + rep_pen 1.05,
full-doc context, max_model_len 32768) is exp73 unchanged.

WHY: exp73 is the best single model (0.7140) — its IoU 0.8091 / 99.7% tag
rate is already near-ceiling, so the only headroom is answer quality
(RougeL 0.4790 / SS 0.8546, a touch under the A3B line). Letting gemma-4
reason before answering is the one untried lever on this base (cf. the
exp79 note: thinking-on was flagged as the sole untried lever for the
Qwen MoE; this is the gemma analogue). Turning thinking on may sharpen the
abstractive answer (0.45R+0.35SS weight) without touching the strong refs.

THINKING-CHANNEL STRIP (delimiters VERIFIED from gemma-4's own
chat_template.jinja — see strip_thinking() below). gemma-4 wraps reasoning
as <|channel>thought…<channel|> then writes the answer; strip_thinking()
removes it before parse_citation. The 5 raw+post-strip dumps below confirm
the strip on the live output.

⚠️ ONE OPEN RISK: KV / TRUNCATED THINKING. thinking tokens are decode-side,
so peak ctx = prompt + think + answer. MAX_NEW_TOKENS 1024→4096 (exp48
precedent: thinking blocks run long). Worst prompt ~14K + 4096 ≈ 18K <<
32768 budget, inside exp73's KV headroom at util 0.92 → no prompt
truncation. But if the THINKING block itself overruns 4096 tokens, no
closing <channel|> is emitted → that query yields no answer and falls back
to the top paragraph. The finish_reason='length' counter below flags how
many hit the cap; if it's high, bump MAX_NEW_TOKENS and re-run.

Decision rule: accept if leak-free composite > exp73 (0.7140). If thinking
helps RougeL/SS while IoU holds, this becomes the new best single model;
if it only adds latency + leak risk for flat/negative score, keep exp73.
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

MAX_NEW_TOKENS = 4096  # thinking block can be long (exp48 precedent)
MAX_MODEL_LEN  = int(os.environ.get("MAX_MODEL_LEN", "32768"))
TP_SIZE        = int(os.environ.get("TP_SIZE", "1"))
MODEL_NAME     = os.environ.get("LLM_MODEL", "RedHatAI/gemma-4-31B-it-NVFP4")

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


# Thinking-channel strip — delimiters VERIFIED from gemma-4's own
# chat_template.jinja (RedHatAI/gemma-4-31B-it-NVFP4 snapshot c490598):
# the model wraps reasoning as  <|channel>thought\n …reasoning… <channel|>
# then emits the visible answer (note the mirrored brackets: <|channel>
# opens, <channel|> closes — same convention as <|turn>/<turn|>). The
# template's own strip_thinking macro splits on <channel|> and drops the
# text after each <|channel>; we mirror that with a regex. enable_thinking
# was always False before (exp71's "<|channel>…<turn|>" guess was wrong),
# so this is the first run that actually exercises the channel.
#
# Runs BEFORE parse_citation so any [อ้างอิง] the model writes mid-reasoning
# can't pollute refs. The second sub handles a thinking block truncated by
# MAX_NEW_TOKENS (no closing <channel|> ever emitted → no final answer):
# drop the dangling reasoning so we fall back to the top paragraph rather
# than dumping raw chain-of-thought into `abstractive`.
_THINK_CLOSED = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
_THINK_OPEN   = re.compile(r"<\|channel>.*$", re.DOTALL)


def strip_thinking(text):
    text = _THINK_CLOSED.sub("", text)
    text = _THINK_OPEN.sub("", text)   # unclosed (truncated) thought block
    return text.strip()


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
    print(f"exp91: {n} queries, {len(doc_index)} docs "
          f"(model={MODEL_NAME}, TP={TP_SIZE}, thinking=ON, "
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

    # Engine config identical to exp73 (NVFP4 ~22 GB weights → ~12.9 GiB KV
    # at util 0.92 → full 32768 ctx fits, no truncation). KV stays bf16
    # (no FP8-KV kernel on sm80). enable_prefix_caching reuses the
    # doc-grouped few-shot+context prefix; enforce_eager for the container
    # parity. Thinking is decode-side so it doesn't change the load fit —
    # only MAX_NEW_TOKENS grows (still << budget, see docstring risk 2).
    llm = LLM(model=MODEL_NAME, max_model_len=MAX_MODEL_LEN,
              tensor_parallel_size=TP_SIZE,
              gpu_memory_utilization=0.92,
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
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True))
    outputs = llm.generate(prompts, sampling)

    # FIRST-RUN SANITY CHECK: dump 5 raw generations (more than exp73's 3)
    # because this is the first run exercising the thinking channel —
    # eyeball the .out to see gemma-4's <|channel>…<channel|> wrapping and
    # verify strip_thinking() removed it (the post-strip line follows).
    for it, out in list(zip(items, outputs))[:5]:
        raw = out.outputs[0].text.strip()
        print(f"[raw  {it[0]}] {raw[:400]!r}", flush=True)
        print(f"[strip {it[0]}] {strip_thinking(raw)[:200]!r}", flush=True)

    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    results = []
    n_explicit = 0
    n_leaked = 0      # thinking marker survived the strip → format mismatch
    n_truncated = 0   # generation hit MAX_NEW_TOKENS (thinking may be cut)
    ref_counts = []
    for it, out in zip(items, outputs):
        qid, gen_pids, gen_texts, _, q_text = it
        raw = out.outputs[0].text.strip()
        if out.outputs[0].finish_reason == "length":
            n_truncated += 1
        clean = strip_thinking(raw)
        if "<|channel>" in clean or "<channel|>" in clean or "<|think" in clean:
            n_leaked += 1
        answer, _ = split_answer_citation(clean)
        cited_idx = parse_citation(clean, len(gen_pids))
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]
        else:
            ref_ids = []
        if cite_re.search(clean):
            n_explicit += 1
        if not answer:
            answer = gen_texts[0] if gen_texts else q_text
        ref_counts.append(len(ref_ids))
        results.append({"ID": qid, "abstractive": answer, "refs": ",".join(ref_ids)})

    print(f"citations: {n_explicit}/{len(results)} emitted an [อ้างอิง: …] tag, "
          f"{len(results) - n_explicit} fell back to top-1", flush=True)
    print(f"avg refs/query: {sum(ref_counts) / len(ref_counts):.2f}", flush=True)
    print(f"thinking: {n_truncated}/{len(results)} hit MAX_NEW_TOKENS "
          f"(finish_reason=length — thinking may be truncated, no answer)",
          flush=True)
    if n_leaked:
        print(f"⚠️ THINKING LEAK: {n_leaked}/{len(results)} answers still "
              f"contain a channel marker after strip — fix strip_thinking() "
              f"using the raw dump above and re-run.", flush=True)

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
