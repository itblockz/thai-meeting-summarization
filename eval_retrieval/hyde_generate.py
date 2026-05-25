"""
exp31 — HyDE (Hypothetical Document Embeddings) generator.

For each query in test.json, ask Qwen3-32B-AWQ to write a short hypothetical
paragraph that *would* appear in the Thai parliamentary minutes and would
answer that question. Save {qid: hypothetical_text} to cache/hyde_train.json
so hyde_eval.py can embed it and test retrieval lift.

Design notes:
  - Greedy decode (temperature=0) — deterministic, matches exp30 setup.
  - System prompt instructs the model to *preserve* numeric/named entities
    from the query (meeting numbers, dates, names). Numeric queries are the
    weak spot of bge-reranker; stripping numbers in the HyDE pass would
    hurt the very queries we want HyDE to help.
  - max_tokens=128 — paragraphs in this dataset are short (median 53
    chars). HyDE should match that distribution, not produce essays.
  - Same vLLM args as exp30 (enforce_eager, max_model_len 4096 is enough
    here — no few-shot context).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams

HERE         = Path(__file__).resolve().parent
DEFAULT_TEST = HERE.parent / "textsum" / "eval_train" / "test.json"
DEFAULT_OUT  = HERE / "cache" / "hyde_train.json"

MODEL_NAME = os.environ.get("LLM_MODEL", "Qwen/Qwen3-32B-AWQ")

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยทำ HyDE (Hypothetical Document Embeddings) "
    "สำหรับการค้นคืนเอกสาร "
    "เมื่อได้รับคำถาม จงเขียนข้อความสมมุติ 1-3 ประโยค "
    "ที่อาจปรากฏในบันทึกการประชุมรัฐสภาไทย "
    "และมีเนื้อหาเป็นคำตอบของคำถามนั้น "
    "ใส่ตัวเลข ชื่อ วันที่ และคำเฉพาะที่ปรากฏในคำถามไว้ในข้อความด้วยเสมอ "
    "ใช้ศัพท์ทางการแบบบันทึกการประชุม "
    "ห้ามขึ้นต้นด้วย 'คำตอบ' หรือ 'ตอบ' ห้ามใส่ข้อความนำ"
)


def build_prompt(query: str) -> str:
    return f"คำถาม: {query}\n\nข้อความสมมุติ:"


def main() -> None:
    test_path = Path(os.environ.get("TEST_JSON", DEFAULT_TEST))
    out_path  = Path(os.environ.get("HYDE_OUT",  DEFAULT_OUT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(test_path, encoding="utf-8") as f:
        data = json.load(f)
    queries = data["queries"]
    print(f"HyDE generate: {len(queries)} queries -> {out_path}", flush=True)

    llm = LLM(
        model=MODEL_NAME, quantization="awq_marlin", max_model_len=4096,
        gpu_memory_utilization=0.90, dtype="half", enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0, max_tokens=128, repetition_penalty=1.05,
    )

    prompts = []
    for q in queries:
        msgs = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": build_prompt(q["query"])},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    print(f"Generated {len(outputs)} HyDE answers in {time.time() - t0:.1f}s",
          flush=True)

    cache = {q["ID"]: out.outputs[0].text.strip()
             for q, out in zip(queries, outputs)}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    lens = [len(v) for v in cache.values()]
    lens.sort()
    print(f"HyDE length (chars): "
          f"min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]} "
          f"mean={sum(lens)/len(lens):.0f}", flush=True)
    print("\nFirst 3 samples:", flush=True)
    for q in queries[:3]:
        print(f"  Q[{q['ID']}]: {q['query'][:90]}", flush=True)
        print(f"     -> {cache[q['ID']][:140]}", flush=True)
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
