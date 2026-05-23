"""
exp23 — exp22 (E5 + exp08 few-shot) with bge-reranker-v2-m3 swapped for
Qwen3-Reranker-8B.

Motivation: on the same train-set candidate pool, Qwen3-Reranker-8B
beat bge-reranker-v2-m3 on every retrieval metric
(eval_retrieval/rerank_qwen3_cache.py):
  hit@1   0.7401 -> 0.7579  (+0.0178)
  hit@5   0.9112 -> 0.9314  (+0.0202)
  iou@1   0.6190 -> 0.6344  (+0.0154)
  iou@oK  0.6414 -> 0.6723  (+0.0309)

For E5 (GEN_K=5) the relevant metric is hit@5: more often the gold
sits in the LLM's context window, so self-citation can pick it.

Everything else is exp22 identical:
  - exp03 retrieval (dense bge-m3 + BM25, POOL_N=20)
  - GEN_K=5, E5 numbered context
  - exp08 few-shot pair (Q0745 + Q0747, both doc_050)
  - vLLM Qwen3-32B-AWQ, max_model_len=16384, enforce_eager=True
  - Inline [อ้างอิง: N] parser

Memory: 8B reranker in bf16 ≈ 16 GB. We load it AFTER bge-m3 is freed
(same pattern as exp22) and free it AFTER scoring, BEFORE vLLM starts.
On a 40 GB A100 there is no overlap; on 80 GB H100 it is comfortable.
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
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from rank_bm25 import BM25Okapi
from pythainlp.tokenize import word_tokenize
from vllm import LLM, SamplingParams

TEST_DIR     = os.environ.get("TEST_DIR",     "/model/test")
RESULT_DIR   = os.environ.get("RESULT_DIR",   "/result/")
PROGRESS_LIB = os.environ.get("PROGRESS_LIB", "/benchmark_lib/progress")

POOL_N         = 20
GEN_K          = 5
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "Qwen/Qwen3-Reranker-8B"
RERANK_BATCH   = int(os.environ.get("RERANK_BATCH", "8"))
RERANK_MAXLEN  = int(os.environ.get("RERANK_MAXLEN", "2048"))
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

# Qwen3-Reranker prompt template (matches eval_retrieval/rerank_qwen3_cache.py).
_QWEN_RR_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements "
    "based on the Query and the Instruct provided. Note that the answer can "
    'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_RR_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_RR_INSTR = (
    "Given a Thai question about a parliamentary meeting document, "
    "retrieve the paragraph that answers the question."
)


def _qwen_rr_format(query, doc):
    return (f"<Instruct>: {_QWEN_RR_INSTR}\n"
            f"<Query>: {query}\n"
            f"<Document>: {doc}")


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


def qwen_rerank_score(pair_texts, device):
    """Score (query, paragraph) pairs with Qwen3-Reranker-8B via yes/no logits.

    Returns a list of float probabilities (yes-prob after softmax(yes, no)),
    in the same order as pair_texts.
    """
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL, padding_side="left")

    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    print(f"qwen3-reranker attn_impl={attn_impl}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        RERANK_MODEL, torch_dtype=torch.bfloat16, attn_implementation=attn_impl,
    ).to(device).eval()

    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    prefix_ids = tokenizer.encode(_QWEN_RR_PREFIX, add_special_tokens=False)
    suffix_ids = tokenizer.encode(_QWEN_RR_SUFFIX, add_special_tokens=False)
    body_cap   = RERANK_MAXLEN - len(prefix_ids) - len(suffix_ids)

    prompts = [_qwen_rr_format(q, d) for q, d in pair_texts]

    @torch.no_grad()
    def _score_batch(batch):
        enc = tokenizer(
            batch, padding=False, truncation="longest_first",
            return_attention_mask=False, max_length=body_cap,
        )
        for i, ids in enumerate(enc["input_ids"]):
            enc["input_ids"][i] = prefix_ids + ids + suffix_ids
        enc = tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=RERANK_MAXLEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        logits  = model(**enc).logits[:, -1, :]
        stacked = torch.stack([logits[:, token_false_id], logits[:, token_true_id]], dim=1)
        return torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().float().cpu().tolist()

    scores = []
    for i in range(0, len(prompts), RERANK_BATCH):
        scores.extend(_score_batch(prompts[i:i + RERANK_BATCH]))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp23: {n} queries, {len(doc_index)} docs", flush=True)

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

    pair_scores = qwen_rerank_score(pair_texts, device)
    print(f"Qwen3-Reranker scored {len(pair_texts)} pairs ({device}).", flush=True)

    scores_by_q = {}
    for qi, s in zip(pair_qidx, pair_scores):
        scores_by_q.setdefault(qi, []).append(float(s))

    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool = pools[i]
        q_text = query["query"]
        if pool:
            scores = scores_by_q.get(i, [])
            order = sorted(range(len(pool)), key=lambda j: -scores[j])[:GEN_K]
            gen_pids  = [pool[j][0] for j in order]
            gen_texts = [pool[j][1] for j in order]
            messages = build_messages(q_text, gen_texts)
        else:
            gen_pids, gen_texts, messages = [], [], None
        items.append((query["ID"], gen_pids, gen_texts, messages, q_text))

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
