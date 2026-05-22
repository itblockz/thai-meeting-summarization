"""
exp19 — exp18 (E5, toey [อ้างอิง:] citation) + 2-shot few-shot.

Step 1 of a 3-step ladder reproducing toey/exp01's E5 + few-shot config on
textsum's exp18 base, one variable per step:
  exp19 = exp18 + few-shot           (this file)
  exp20 = exp19 + toey's SYSTEM_MSG
  exp21 = exp20 + no paragraph cap   (= full toey/exp01 reproduction)

exp17/exp18 showed E5 alone is ~break-even vs exp03 (0.6234 / 0.6289 vs
0.6256): feeding 5 paragraphs lifts RougeL/SS but the LLM over-cites
(avg 1.5-1.7 refs vs 72% single-ref gold) so IoU drops -0.047. toey/exp01
(E5 + few-shot) scores 0.6522 — the few-shot calibrates the citation
count. exp19 adds toey's 2 worked examples (FEW_SHOT, from held-out
doc_050) as multi-turn chat turns; each assistant turn carries the
[อ้างอิง: N] tag, teaching both answer style and citation discipline.

max_model_len is raised 8192 -> 16384: the few-shot prompt (2 examples x
5 paragraphs + the query's 5 paragraphs) does not fit 8192. SYSTEM_MSG
and the MAX_PARA_CHARS=600 cap are still exp18's — exp20/exp21 change them.
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
GEN_K          = 5      # reranked paragraphs fed to the LLM as numbered context
MAX_PARA_CHARS = 600    # per-paragraph cap in the prompt
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยสรุปจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

# Two worked few-shot examples from held-out doc_050, copied verbatim from
# toey/exp01. Each answer ends with an [อ้างอิง: N] tag so the model learns
# the citation format and how many paragraphs to cite.
_SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
_SHOT1_PARAS = [
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "_________________________",
    "กรรมาธิการผู้มาประชุม",
]
_SHOT1_ANSWER = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา [อ้างอิง: 3]"

_SHOT2_QUERY = "ปัญหาหลักของผู้ประกอบการ SMEs ในการเข้าถึงแหล่งเงินทุนคืออะไร และมีการแก้ไขปัญหานี้อย่างไร"
_SHOT2_PARAS = [
    "ผู้แทนจากสำนักงานเศรษฐกิจการคลัง ได้ให้ข้อมูลต่อที่ประชุมว่า เนื่องจากโครงการค้ำประกันสินเชื่อส่วนใหญ่ของบรรษัทประกันสินเชื่ออุตสาหกรรมขนาดย่อม (บสย.) อยู่ในรูปแบบของพอร์ตสินเชื่อ โดยมีกระบวนการทำงานเริ่มต้นจากผู้ประกอบการ SMEs ติดต่อขอสินเชื่อกับสถาบันการเงิน และสถาบันการเงินจะเป็นผู้พิจารณาความเสี่ยงของลูกหนี้",
    "อีกทั้งในปัจจุบันการค้ำประกันของ บสย. บางส่วนเป็นการค้ำประกันสินเชื่อแบบรวมกลุ่ม (Portfolio Guarantee Scheme : PGS) โดยกลไกในการค้ำประกันแบบ PGS จะเป็นลักษณะที่สถาบันการเงินรับลูกหนี้หลาย ๆ รายรวมกันเป็นพอร์ตสินเชื่อ",
    "ทั้งนี้ กระทรวงการคลังได้เล็งเห็นความสำคัญของสภาพปัญหาดังกล่าวโดยเฉพาะอย่างยิ่งในเรื่องของผู้ประกอบการ SMEs ซึ่งมีจำนวนประมาณ ๓.๒ ล้านราย โดยมากกว่าร้อยละ ๔๐ ไม่สามารถเข้าถึงแหล่งเงินทุนในระบบสถาบันการเงินได้ โดยปัญหาหลักเนื่องมาจากการมีรายได้ที่ไม่แน่นอนและมีหลักทรัพย์ค้ำประกันที่ไม่เพียงพอ รวมทั้งไม่เคยมีประวัติข้อมูลเครดิตในระบบสถาบันการเงิน",
    "ดังนั้น กระทรวงการคลังจึงได้เสนอคณะรัฐมนตรีเพื่อให้ความเห็นชอบแนวทางการจัดตั้งสถาบันค้ำประกันเครดิตแห่งชาติ (NaCGA) ขึ้น ซึ่ง NaCGA จะมีสถานะเป็นนิติบุคคลที่เป็นหน่วยงานของภาครัฐ แต่ไม่ใช่หน่วยงานรัฐวิสาหกิจ ซึ่งจะทำให้มีความยืดหยุ่นในการดำเนินงาน และผู้ประกอบการ SMEs จะได้รับการค้ำประกันที่รวดเร็วขึ้น",
    "สำหรับแนวทางการปฏิรูปโครงสร้างการค้ำประกันสินเชื่อของประเทศไทย โดยการจัดตั้งสถาบันค้ำประกันเครดิตแห่งชาติ (NaCGA) มีดังนี้",
]
_SHOT2_ANSWER = "ปัญหาหลักของผู้ประกอบการ SMEs ในการเข้าถึงแหล่งเงินทุนคือมีรายได้ไม่แน่นอน หลักทรัพย์ค้ำประกันไม่เพียงพอ และไม่มีประวัติข้อมูลเครดิต กระทรวงการคลังแก้ไขปัญหาโดยเสนอจัดตั้งสถาบันค้ำประกันเครดิตแห่งชาติ (NaCGA) ที่มีความยืดหยุ่นในการดำเนินงาน เพื่อให้ผู้ประกอบการ SMEs เข้าถึงแหล่งเงินทุนได้มากขึ้น [อ้างอิง: 3, 4]"


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
    """E5 prompt — numbered [1..N] context + toey/exp01's inline citation
    request. Paragraphs are capped at MAX_PARA_CHARS in the prompt."""
    context = "\n".join(
        f"[{i + 1}] {t[:MAX_PARA_CHARS]}" for i, t in enumerate(paras)
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
    """System + toey/exp01's 2 few-shot turns + the final user turn.

    Every user turn uses the same build_prompt; the few-shot assistant
    turns carry the worked [อ้างอิง: N] answer.
    """
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(_SHOT1_QUERY, _SHOT1_PARAS)},
        {"role": "assistant", "content": _SHOT1_ANSWER},
        {"role": "user", "content": build_prompt(_SHOT2_QUERY, _SHOT2_PARAS)},
        {"role": "assistant", "content": _SHOT2_ANSWER},
        {"role": "user", "content": build_prompt(query, paras)},
    ]


def parse_citation(text, n_paras):
    """Extract 0-indexed paragraph indices from the LLM citation tag."""
    m = re.search(r'\[อ้างอิง[:\s]+([0-9,\s]+)\]', text)
    if m:
        nums = [int(x.strip()) for x in re.findall(r'\d+', m.group(1))]
        valid = [num - 1 for num in nums if 1 <= num <= n_paras]
        if valid:
            return valid
    return [0]  # fallback: top-ranked paragraph


def split_answer_citation(text):
    """Split LLM output into (answer, raw_citation_tag)."""
    idx = text.rfind('[อ้างอิง')
    if idx != -1:
        return text[:idx].strip(), text[idx:]
    return text.strip(), ""


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = data["queries"]
    n = len(queries)
    print(f"exp19: {n} queries, {len(doc_index)} docs", flush=True)

    # ── Stage 1a: embed (GPU) + BM25 ──────────────────────────────────────
    # vLLM uses spawn (VLLM_WORKER_MULTIPROC_METHOD), so touching CUDA in the
    # parent here does not fork a corrupted context into vLLM workers.
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

    # ── Stage 1b: candidate pools ─────────────────────────────────────────
    pools = []   # per query: list of (para_id, text)
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

    # ── Stage 1c: cross-encoder rerank (GPU) ──────────────────────────────
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

    # ── assemble: top-GEN_K reranked paragraphs per query ─────────────────
    items = []   # (ID, gen_pids, gen_texts, messages_or_None, query_text)
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

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    # enforce_eager=True: skip vLLM's torch.compile + CUDA-graph capture,
    # which OOMs the 40 GB A100 with bge-m3 embedding on GPU plus the
    # max_model_len=16384 KV cache. Matches toey/exp01; eager costs ~0
    # latency on this batched workload.
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

    # ── Stage 3: parse answer + citations ([อ้างอิง: X] format) ───────────
    cite_re = re.compile(r'\[อ้างอิง[:\s]+[0-9,\s]+\]')
    results = []
    n_explicit = 0
    ref_counts = []
    for it, out in zip(items, outputs):
        qid, gen_pids, gen_texts, _, q_text = it
        raw = out.outputs[0].text.strip()
        answer, _ = split_answer_citation(raw)
        cited_idx = parse_citation(raw, len(gen_pids))   # 0-indexed, fallback [0]
        if gen_pids:
            ref_ids = [gen_pids[j] for j in cited_idx if j < len(gen_pids)]
            if not ref_ids:
                ref_ids = [gen_pids[0]]      # fallback: top-1 reranked
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
    benchmark_lib(n)  # must be last operation
