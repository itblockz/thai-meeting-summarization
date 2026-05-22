"""
exp11 — 4-shot variant of exp08. exp10's three examples + Q0756 (PGS
limitations explanation, gold ~601 chars). Adds another long-answer
canonical example to test #shots saturation: if 4-shot ≈ 3-shot, the
marginal example contributes little; if 4-shot beats 3-shot, the model
keeps benefiting from more worked examples.
"""
from pathlib import Path
import os
import gc
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

TOP_K          = 1
POOL_N         = 20
EMBED_BATCH    = 64
MAX_NEW_TOKENS = 512
EMBED_MODEL    = "BAAI/bge-m3"
RERANK_MODEL   = "BAAI/bge-reranker-v2-m3"
MODEL_NAME     = "Qwen/Qwen3-32B-AWQ"

HELDOUT_DOC = "doc_050"

SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยสรุปจากข้อมูลที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

FEW_SHOT = [
    {
        "query": "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด",
        "paragraph": "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
        "answer": (
            "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น "
            "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา"
        ),
    },
    {
        "query": "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมาธิการจัดขึ้นเพื่ออะไร",
        "paragraph": (
            "สำนักงานเลขาธิการสภาผู้แทนราษฎรขอความอนุเคราะห์ตอบแบบสำรวจ"
            "ความพึงพอใจและความไม่พึงพอใจของคณะกรรมาธิการต่อการบริหารจัดการ"
            "ด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา "
            "เพื่อนำผลการประเมินความพึงพอใจและความไม่พึงพอใจที่ได้ "
            "ไปเป็นข้อมูลในการทบทวนปรับปรุงและพัฒนาการบริหารจัดการ"
            "ด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา"
        ),
        "answer": (
            "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมการในครั้งนี้ "
            "มีการจัดทำขึ้นเพื่อนำข้อมูลที่ได้มาทบทวน ปรับปรุง "
            "รวมถึงนำไปพัฒนาการปฏิบัติงานให้มีประสิทธิภาพยิ่งขึ้น"
        ),
    },
    {
        "query": (
            "บริษัทประกันสินเชื่ออุตสาหกรรมขนาดย่อม (บสย.) ใช้แหล่งเงินค่าชดเชย"
            "ความเสียหายประกอบด้วยกี่แหล่ง และแต่ละแหล่งคิดเป็นร้อยละเท่าใด"
        ),
        "paragraph": (
            "๑. ด้านการค้ำประกันสินเชื่อให้แก่ผู้ประกอบการวิสาหกิจขนาดกลาง"
            "และขนาดย่อม (SMEs) โดย บสย. ดำเนินการผ่านกลไกการค้ำประกันสินเชื่อ"
            "แบบรวมกลุ่ม (Portfolio Guarantee Scheme : PGS) "
            "ซึ่งสามารถช่วยค้ำประกันสินเชื่อให้กับผู้ประกอบการ SMEs "
            "ทั้งบุคคลธรรมดาหรือนิติบุคคลที่มีสินทรัพย์ถาวร (ไม่รวมที่ดิน) "
            "ได้ไม่เกิน ๒๐๐ ล้านบาท ในอัตราร้อยละ ๑๐๐ "
            "ซึ่ง บสย. แบ่งความเสี่ยงของลูกหนี้ร่วมกับสถาบันการเงิน "
            "คือ สถาบันการเงินร้อยละ ๗๐ ของพอร์ตสินเชื่อ "
            "และ บสย. ร้อยละ ๓๐ ของพอร์ตสินเชื่อโดยแหล่งเงินที่นำมาเป็น"
            "ค่าชดเชยความเสียหายของ บสย. ประกอบด้วย ๒ ส่วน "
            "คือ เงินค่าธรรมเนียมค้ำประกันสะสมจากผู้ประกอบการ SMEs "
            "ประมาณร้อยละ ๑.๕ ต่อปี ตลอดระยะเวลาค้ำประกัน ๑๐ ปี"
            "และเงินสนับสนุนจากภาครัฐ ประมาณร้อยละ ๑.๕ ของพอร์ตสินเชื่อ"
        ),
        "answer": (
            "แหล่งเงินที่บรรษัทประกันสินเชื่ออุตสาหกรรมขนาดย่อม (บสย.) "
            "ใช้สำหรับชดเชยความเสียหายจากการค้ำประกันสินเชื่อ มาจาก 2 แหล่งหลัก "
            "ได้แก่ ค่าธรรมเนียมค้ำประกันที่เก็บจากผู้ประกอบการ SMEs "
            "ซึ่งคิดเป็นประมาณร้อยละ 1.5 ต่อปี ตลอดระยะเวลา 10 ปีของการค้ำประกัน "
            "และเงินสนับสนุนจากภาครัฐซึ่งคิดเป็นประมาณร้อยละ 1.5 ของมูลค่าพอร์ตสินเชื่อ"
        ),
    },
    {
        "query": "การค้ำประกันสินเชื่อแบบ PGS ของ บสย. มีข้อจำกัดที่ส่งผลต่อการช่วยเหลือลูกหนี้อย่างไร",
        "paragraph": (
            "อีกทั้งในปัจจุบันการค้ำประกันของ บสย. บางส่วนเป็นการค้ำประกันสินเชื่อ"
            "แบบรวมกลุ่ม (Portfolio Guarantee Scheme : PGS) "
            "โดยกลไกในการค้ำประกันแบบ PGS จะเป็นลักษณะที่สถาบันการเงิน"
            "รับลูกหนี้หลาย ๆ รายรวมกันเป็นพอร์ตสินเชื่อ "
            "ซึ่ง บสย. จะชดเชยความเสียหายสูงสุด (Max Claim) "
            "ในอัตราร้อยละ ๓๐ ของพอร์ตสินเชื่อ "
            "เช่น หากในพอร์ตสินเชื่อหนึ่งมีลูกหนี้ จำนวน ๑๐๐ ราย ๆ ละ ๑ ล้านบาท"
            "รวมเป็นพอร์ตสินเชื่อ ๑๐๐ ล้านบาท บสย. จะค้ำประกันสินเชื่อให้กับ"
            "สถาบันการเงิน จำนวน ๓๐ ล้านบาท "
            "ทั้งนี้ กลไกการค้ำประกันที่ดำเนินการอยู่ในปัจจุบันมีข้อจำกัดบางประการ "
            "เช่น เงินชดเชยความเสียหายส่วนที่เป็นงบประมาณสนับสนุนจากภาครัฐ"
            "มีสัดส่วนที่ค่อนข้างสูง การขออนุมัติงบประมาณสำหรับโครงการในลักษณะนี้"
            "ทำได้ยากขึ้น เนื่องจากข้อจำกัดด้านงบประมาณที่เพิ่มมากขึ้น "
            "ซึ่งกลไกการค้ำประกันแบบ PGS และการชดเชยความเสียหายร้อยละ ๑๐๐ "
            "ต่อรายลูกหนี้ในพอร์ตสินเชื่อ อาจจะทำให้สถาบันการเงินไม่มีแรงจูงใจ"
            "ในการให้ความช่วยเหลือลูกหนี้ปรับโครงสร้างหนี้อย่างเหมาะสมได้"
        ),
        "answer": (
            "กลไกการค้ำประกันสินเชื่อแบบ PGS ของ บสย. มีข้อจำกัดที่กระทบต่อ"
            "ประสิทธิภาพในการช่วยเหลือลูกหนี้ โดยเฉพาะในกรณีที่รัฐต้องเป็น"
            "ผู้สนับสนุนเงินชดเชยความเสียหายเป็นสัดส่วนสูง "
            "ทำให้การขอจัดสรรงบประมาณในโครงการลักษณะนี้กลายเป็นเรื่องยากขึ้น "
            "เนื่องจากข้อจำกัดด้านงบประมาณของภาครัฐที่มีแนวโน้มเข้มงวดมากขึ้น "
            "นอกจากนี้ การที่ บสย. รับภาระค้ำประกันและชดเชยความเสียหาย"
            "เต็มจำนวน 100% ต่อรายลูกหนี้ในพอร์ตสินเชื่อ "
            "อาจทำให้สถาบันการเงินขาดแรงจูงใจในการช่วยลูกหนี้ปรับโครงสร้างหนี้"
            "อย่างเหมาะสม เพราะไม่มีความเสี่ยงที่ต้องแบกร่วม "
            "ส่งผลให้กลไกการค้ำประกันนี้ไม่สามารถกระตุ้นการแก้ปัญหาหนี้"
            "ได้อย่างมีประสิทธิภาพเท่าที่ควร"
        ),
    },
]


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


def build_user_turn(query, retrieved_texts):
    if len(retrieved_texts) == 1:
        paragraph_block = retrieved_texts[0]
    else:
        paragraph_block = "\n".join(
            f"ย่อหน้า {i + 1}: {t}" for i, t in enumerate(retrieved_texts)
        )
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n"
        f"ย่อหน้า 1: {paragraph_block}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น\n"
        f"คำตอบ:"
    )


def build_messages(query, retrieved_texts):
    messages = [{"role": "system", "content": SYSTEM_MSG}]
    for ex in FEW_SHOT:
        messages.append({"role": "user",
                         "content": build_user_turn(ex["query"], [ex["paragraph"]])})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user",
                     "content": build_user_turn(query, retrieved_texts)})
    return messages


def main():
    data = load_data(TEST_DIR)
    doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
    queries = [q for q in data["queries"] if q["doc_id"] != HELDOUT_DOC]
    n_held_out = len(data["queries"]) - len(queries)
    n = len(queries)
    print(f"{n} queries (excluded {n_held_out} from {HELDOUT_DOC}), "
          f"{len(doc_index)} docs", flush=True)

    # ── Stage 1a: embed paragraphs + queries (GPU; safe under spawn) ──────
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

    # ── Stage 1b: build candidate pools ───────────────────────────────────
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
        pools.append([valid[j]["para_id"] for j in pool_idx])
        for j in pool_idx:
            pair_texts.append((query["query"], valid[j]["text"]))
            pair_qidx.append(i)

    # ── Stage 1c: cross-encoder rerank (GPU) ──────────────────────────────
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

    # ── assemble retrieval results + LLM messages ─────────────────────────
    items = []
    for i, query in enumerate(queries):
        benchmark_lib(i)
        pool_pids = pools[i]
        q_text = query["query"]
        if pool_pids:
            scores = scores_by_q.get(i, [])
            order = sorted(range(len(pool_pids)), key=lambda j: -scores[j])
            ref_ids = [pool_pids[j] for j in order[:TOP_K]]
        else:
            ref_ids = []

        para_text_map = {p["para_id"]: p["text"] for p in doc_index.get(query["doc_id"], [])}
        retrieved_texts = [para_text_map[pid] for pid in ref_ids
                           if para_text_map.get(pid, "").strip()]

        if retrieved_texts:
            messages = build_messages(q_text, retrieved_texts)
        else:
            messages = None
        items.append((query["ID"], ref_ids, messages, q_text))

    # ── Stage 2: vLLM batch generation ────────────────────────────────────
    llm = LLM(model=MODEL_NAME, quantization="awq_marlin", max_model_len=8192,
              gpu_memory_utilization=0.85, dtype="half")
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, repetition_penalty=1.05)

    prompts = []
    for it in items:
        msgs = it[2] if it[2] is not None else [{"role": "user", "content": it[3]}]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    outputs = llm.generate(prompts, sampling)

    results = []
    for it, out in zip(items, outputs):
        summary = out.outputs[0].text.strip() or it[3]
        results.append({"ID": it[0], "abstractive": summary, "refs": ",".join(it[1])})

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
