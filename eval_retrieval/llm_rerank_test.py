"""
Offline go/no-go for LLM-as-reranker.

For each query where exp03 (current best) picked the wrong paragraph
(IoU=0), show Qwen3-32B-AWQ the top-10 bge-reranked candidates and ask
it to pick the one that best answers the query. Compare against gold to
measure how many failures the LLM would have recovered.

Also runs a control sample of HIT cases (where exp03 was correct) to
estimate the risk of regression on cases bge-rerank already got right.
"""
import json
import csv
import re
import random
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent

EXP03_DETAIL = PROJECT / "exp03/eval_result/train_eval_detail.csv"
RERANK_CACHE = HERE / "cache/rerank_train.json"
TEST_JSON    = PROJECT / "textsum/eval_train/test.json"
RESULT_DIR   = HERE / "result"

MODEL_NAME = "Qwen/Qwen3-32B-AWQ"
TOP_N      = 10        # candidates shown to the LLM
MAX_CHARS  = 400       # per-paragraph cap inside the prompt
HIT_SAMPLE = 100       # control group from hit queries
SEED       = 42


def as_list(refs):
    if refs is None:
        return []
    return refs if isinstance(refs, list) else [refs]


def build_prompt(query, candidates):
    """candidates: list of paragraph texts, displayed as [1]..[N]."""
    lines = [f"[{i+1}] {t[:MAX_CHARS]}" for i, t in enumerate(candidates)]
    context = "\n".join(lines)
    return (
        f"คำถาม: {query}\n\n"
        f"ย่อหน้าผู้สมัคร {len(candidates)} ตัว:\n{context}\n\n"
        f"คำสั่ง: เลือกย่อหน้าที่ตอบคำถามได้ดีที่สุดเพียงหมายเลขเดียว "
        f"ตอบเป็นหมายเลขในวงเล็บ เช่น [3] เท่านั้น ห้ามอธิบายเพิ่ม\n"
        f"คำตอบ: ["
    )


def parse_pick(text, n):
    """Extract picked index (1-based) from LLM output. Returns None if invalid."""
    m = re.search(r"\d+", text)
    if not m:
        return None
    idx = int(m.group(0))
    if 1 <= idx <= n:
        return idx
    return None


def main():
    random.seed(SEED)

    # ── load exp03 per-query IoU ──────────────────────────────────────────
    detail = {}
    with open(EXP03_DETAIL) as f:
        for row in csv.DictReader(f):
            detail[row["ID"]] = float(row["IoU"])

    failures = [qid for qid, iou in detail.items() if iou == 0.0]
    hits     = [qid for qid, iou in detail.items() if iou >  0.0]
    hit_ctrl = random.sample(hits, min(HIT_SAMPLE, len(hits)))
    print(f"exp03: {len(failures)} failures, {len(hits)} hits "
          f"(sampled {len(hit_ctrl)} as control)", flush=True)

    # ── load data ────────────────────────────────────────────────────────
    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)
    queries  = {q["ID"]: q for q in data["queries"]}
    text_map = {}
    for doc in data["docs"]:
        for p in doc["paragraphs"]:
            text_map[(doc["doc_id"], p["para_id"])] = p["text"]
    gold = {q["ID"]: set(as_list(q.get("refs"))) for q in data["queries"]}

    with open(RERANK_CACHE) as f:
        rerank_cache = json.load(f)   # qid -> [[pid, score], ...]

    def top_candidates(qid, k=TOP_N):
        cands = rerank_cache.get(qid, [])
        return [pid for pid, _ in cands[:k]]

    # ── build prompts ─────────────────────────────────────────────────────
    eval_qids = failures + hit_ctrl
    items = []   # (qid, label, gold_set, cand_pids, prompt)
    for qid in eval_qids:
        q = queries[qid]
        cand_pids = top_candidates(qid, TOP_N)
        if not cand_pids:
            continue
        cand_texts = [text_map.get((q["doc_id"], pid), "") for pid in cand_pids]
        prompt = build_prompt(q["query"], cand_texts)
        label = "fail" if qid in set(failures) else "hit"
        items.append((qid, label, gold[qid], cand_pids, prompt))

    print(f"prepared {len(items)} prompts "
          f"({sum(1 for it in items if it[1]=='fail')} fail / "
          f"{sum(1 for it in items if it[1]=='hit')} hit)", flush=True)

    # ── run vLLM ──────────────────────────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=4096,
              gpu_memory_utilization=0.90, dtype="half")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=16)

    formatted = []
    for _, _, _, _, p in items:
        msgs = [{"role": "user", "content": p}]
        formatted.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    outputs = llm.generate(formatted, sampling)

    # ── score ─────────────────────────────────────────────────────────────
    bucket = {
        "fail": {"recovered": 0, "still_wrong": 0, "parse_fail": 0, "n": 0},
        "hit":  {"kept": 0,      "broke": 0,       "parse_fail": 0, "n": 0},
    }
    details = []
    bge_top1_correct_in_failures = 0   # sanity check
    for (qid, label, gold_set, cand_pids, _), out in zip(items, outputs):
        txt = out.outputs[0].text
        idx = parse_pick(txt, len(cand_pids))
        bucket[label]["n"] += 1
        bge_top1 = cand_pids[0]
        if label == "fail" and bge_top1 in gold_set:
            bge_top1_correct_in_failures += 1
        if idx is None:
            bucket[label]["parse_fail"] += 1
            picked = None
            outcome = "parse_fail"
        else:
            picked = cand_pids[idx - 1]
            if label == "fail":
                if picked in gold_set:
                    bucket["fail"]["recovered"] += 1
                    outcome = "recovered"
                else:
                    bucket["fail"]["still_wrong"] += 1
                    outcome = "still_wrong"
            else:
                if picked in gold_set:
                    bucket["hit"]["kept"] += 1
                    outcome = "kept"
                else:
                    bucket["hit"]["broke"] += 1
                    outcome = "broke"
        details.append({
            "ID": qid, "label": label, "outcome": outcome,
            "bge_pick": bge_top1, "llm_pick": picked,
            "gold": ",".join(sorted(gold_set)), "raw": txt.strip()[:80],
        })

    # ── report ────────────────────────────────────────────────────────────
    print("\n=== LLM-as-reranker eval ===\n")
    f = bucket["fail"]; h = bucket["hit"]
    print(f"FAILURES (exp03 IoU=0, n={f['n']}):")
    print(f"  recovered    : {f['recovered']:>4} ({f['recovered']/f['n']:.1%})")
    print(f"  still wrong  : {f['still_wrong']:>4} ({f['still_wrong']/f['n']:.1%})")
    print(f"  parse fail   : {f['parse_fail']:>4} ({f['parse_fail']/f['n']:.1%})")
    print(f"  (sanity) bge rank-1 actually in gold for {bge_top1_correct_in_failures}/{f['n']} "
          f"— should be ~0; non-zero means cache/exp03 mismatch")
    print()
    print(f"HITS (control, n={h['n']}):")
    print(f"  kept correct : {h['kept']:>4} ({h['kept']/h['n']:.1%})")
    print(f"  broke        : {h['broke']:>4} ({h['broke']/h['n']:.1%})")
    print(f"  parse fail   : {h['parse_fail']:>4} ({h['parse_fail']/h['n']:.1%})")
    print()

    # extrapolated impact: recover rate × #failures - break rate × #hits
    recover_rate = f["recovered"] / f["n"] if f["n"] else 0
    break_rate   = h["broke"]     / h["n"] if h["n"] else 0
    n_fail_total = sum(1 for v in detail.values() if v == 0)
    n_hit_total  = sum(1 for v in detail.values() if v >  0)
    delta_hit_at_1 = recover_rate * n_fail_total - break_rate * n_hit_total
    print(f"extrapolated Δhit@1 over full train set "
          f"(recover {recover_rate:.1%} × {n_fail_total} fails "
          f"− break {break_rate:.1%} × {n_hit_total} hits): "
          f"{delta_hit_at_1:+.1f} queries "
          f"({delta_hit_at_1 / len(detail):+.4f} pp)")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "llm_rerank_test.json"
    out_path.write_text(json.dumps({
        "config": {"model": MODEL_NAME, "top_n": TOP_N, "max_chars": MAX_CHARS,
                   "hit_sample": HIT_SAMPLE, "seed": SEED},
        "summary": bucket,
        "extrapolated_delta_hit_at_1_queries": delta_hit_at_1,
        "details": details,
    }, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
