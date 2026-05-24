"""
exp30 — exp27 with NO GEN_K cap (bge-reranker order + full pool).

Union of exp27 and exp29's strengths:
  exp27 (K=20 + rerank)        : 0.6777 — rerank puts gold at rank 1
                                          → best RougeL/SS
  exp29 (full pool + RRF only) : 0.6762 — LLM sees every candidate
                                          → best IoU
  exp30 (full pool + rerank)   : ???   — rerank order + full context

Hypothesis: rerank ordering keeps gold at rank 1 (LLM attends to it more
in answer generation → RougeL/SS lift), while the extra paragraphs at
ranks 21-28 expand IoU recall.

Pool stats (leak-free 1218):
  dense∪BM25 union — mean 28.57, median 28, max 39

Context cost: avg ~28 paras × ~150 tokens ≈ 4200 ctx + 2400 few-shot
≈ 6600 prompt tokens, well under max_model_len=16384.

Everything else identical to exp27 (POOL_N=20, bge-reranker-v2-m3,
E5 self-cite, exp08 few-shot, Qwen3-32B-AWQ, enforce_eager).
"""
from pathlib import Path
import os
import gc
import re
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

POOL_N         = 20     # top-N from each stage-1 retriever before rerank
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"

# NO GEN_K cap — feed the full reranked union to the LLM.

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


def tokenize_th(text):
    return word_tokenize(text, engine="newmm", keep_whitespace=False)


def encode_texts(model, texts, batch_size=EMBED_BATCH):
    return model.encode(
        texts, batch_size=batch_size, convert_to_tensor=True,
        normalize_embeddings=True, show_progress_bar=False,
    )


def build_prompt(query, paras):
    context = "\n".join(
        f"[{i + 1}] {t}" for i, t in enumerate(paras)
    )
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
    print(f"exp30: {n} queries, {len(doc_index)} docs (full pool, rerank ordered, no GEN_K cap)", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBED_MODEL, device=device)
    doc_para_data = {}
    for doc_id, paragraphs in doc_index.items():
        valid = filter_valid_paragraphs(paragraphs)
        if valid:
            embs = encode_texts(embed_model, [p["text"] for p in valid])
            bm25 = BM25Okapi([tokenize_th(p["text"]) for p in valid])
        else:
            embs = torch.zeros((0, embed_model.get_sentence_embedding_dimension()))
            bm25 = None
        doc_para_data[doc_id] = (valid, embs, bm25)
    query_embs = encode_texts(embed_model, [q["query"] for q in queries])
    del embed_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Embeddings + BM25 done ({device}).", flush=True)

    pools = []
    pair_texts, pair_qidx = [], []
    for i, query in enumerate(queries):
        valid, para_embs, bm25 = doc_para_data.get(
            query["doc_id"], ([], torch.zeros((0, 1)), None))
        if not valid:
            pools.append([])
            continue
        sims = F.cosine_similarity(query_embs[i].unsqueeze(0), para_embs, dim=1)
        dense_idx = torch.topk(sims, k=min(POOL_N, len(valid))).indices.tolist()
        bm25_scores = np.asarray(bm25.get_scores(tokenize_th(query["query"])))
        bm25_idx = np.argsort(-bm25_scores)[:POOL_N].tolist()
        pool_idx = list(dict.fromkeys(dense_idx + bm25_idx))
        pools.append([(valid[j]["para_id"], valid[j]["text"]) for j in pool_idx])
        for j in pool_idx:
            pair_texts.append((query["query"], valid[j]["text"]))
            pair_qidx.append(i)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker = CrossEncoder(RERANK_MODEL, max_length=512, device=device)
    pair_scores = reranker.predict(pair_texts, batch_size=64, show_progress_bar=False)
    del reranker
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Reranked {len(pair_texts)} pairs ({device}).", flush=True)

    scores_by_q = {}
    for qi, s in zip(pair_qidx, pair_scores):
        scores_by_q.setdefault(qi, []).append(float(s))

    items = []
    pool_sizes = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool = pools[i]
        q_text = query["query"]
        if pool:
            scores = scores_by_q.get(i, [])
            # FULL pool ordered by rerank score — NO [:GEN_K] cap
            order = sorted(range(len(pool)), key=lambda j: -scores[j])
            gen_pids  = [pool[j][0] for j in order]
            gen_texts = [pool[j][1] for j in order]
            pool_sizes.append(len(order))
            messages = build_messages(q_text, gen_texts)
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((query["ID"], gen_pids, gen_texts, messages, q_text))

    print(f"pool sizes — mean={sum(pool_sizes)/len(pool_sizes):.2f}, "
          f"min={min(pool_sizes)}, max={max(pool_sizes)}", flush=True)

    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=16384,
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
