# IDEAS — แผนยกระดับคะแนน 2026-textsum

แผนการทดลองเพื่อดันคะแนน composite ให้สูงกว่า 0.5584 (textsum baseline)
แต่ละข้อออกแบบให้ **ทดสอบแยกกันได้** และ **วัดผลเป็นตัวเลขได้**

> composite = 0.45 × SS-score + 0.35 × RougeL + 0.20 × IoU

---

## TL;DR

- **คอขวดหลักคือ retrieval** ไม่ใช่ generation — recall@1 ปัจจุบันแค่ **57.5%**
- 42.5% ของ query หยิบย่อหน้าผิดทั้งหมด (IoU=0) → ฉุดทั้ง 3 metric พร้อมกัน
- ลำดับความสำคัญ: **Tier A (retrieval) > Tier B (refs/IoU) > Tier C (generation)**
- เพดานเชิงทฤษฎี: ~0.68 (retrieval สมบูรณ์, TOP_K=1) / ~0.72 (รายงาน multi-ref ถูกด้วย)
- เป้าหมายที่ทำได้จริง: **0.62 – 0.68**

---

## 1. การวินิจฉัย — ทำไมคะแนนถึงตัน

ตัวเลขทั้งหมดมาจาก train set (1239 queries, 50 docs, ~168 ย่อหน้า/doc) และ
`textsum/eval_train/result/train_eval_detail.csv` (run ที่ดีที่สุด, composite 0.5584)

### 1.1 retrieval พลาดเกือบครึ่ง

| ผลลัพธ์ retrieval (TOP_K=1) | จำนวน query | สัดส่วน |
|---|---|---|
| IoU = 0 (หยิบย่อหน้าผิดทั้งหมด) | 527 | 42.5% |
| 0 < IoU < 1 (multi-ref โดนบางส่วน) | 177 | 14.3% |
| IoU = 1 (single-ref ถูกเป๊ะ) | 535 | 43.2% |

→ **recall@1 = (535+177)/1239 = 57.5%** ย่อหน้าอันดับ 1 อยู่ใน gold refs แค่ 57.5%

### 1.2 retrieval ฉุดทุก metric

แยก query เป็น 2 กลุ่มตามผล retrieval:

| กลุ่ม | n | RougeL | SS-score |
|---|---|---|---|
| HIT (IoU>0) | 712 | **0.414** | **0.839** |
| MISS (IoU=0) | 527 | 0.237 | 0.670 |

หยิบย่อหน้าผิด → LLM ได้ context ผิด → ตอบผิด → RougeL/SS ตกตาม
**การแก้ retrieval จึงได้กำไร 3 ทางพร้อมกัน** ไม่ใช่แค่ IoU

### 1.3 เพดานคะแนน

ถ้า retrieval สมบูรณ์ (ทุก query ทำคะแนนได้เท่ากลุ่ม HIT):
- TOP_K=1: composite ≈ 0.45(0.839) + 0.35(0.414) + 0.20(0.803*) ≈ **0.68**
- + รายงาน multi-ref ถูก (IoU→1.0): ≈ **0.72**

\* 0.803 = IoU สูงสุดที่เป็นไปได้เมื่อรายงานแค่ 1 ย่อหน้า (เพราะ 28.3% ของ query มี ≥2 refs)

### 1.4 ข้อเท็จจริงสำคัญอื่น ๆ

- **refs/query**: 1 ref = 889 (71.7%), ≥2 refs = 350 (28.3%), เฉลี่ย 2.09
  → การเดา K=1 เสมอเป็น prior ที่ดี แต่ทิ้งคะแนน IoU ของ 28% ที่เหลือ
- **abstractive เป็นการเรียบเรียงใหม่ 92%** (ไม่ใช่การคัดลอก — substring ของ gold para แค่ 0%)
  → prompt "คัดลอกคำต่อคำ ห้ามเปลี่ยนคำ" ของ exp02 **ผิดทาง** อธิบายได้ว่าทำไม exp02 แพ้ baseline
- **abstractive สั้น**: median 175 ตัวอักษร, mean 251 → เป็น QA ตอบตรงคำถาม ไม่ใช่ summary ยาว
- **ย่อหน้าสั้น**: median 53 ตัวอักษร, p90 = 361 → ใส่ context หลายย่อหน้าได้สบาย ไม่ชน max_model_len
- **exp02 hybrid RRF กลับแย่กว่า dense ล้วน** (IoU 0.4445 < textsum 0.4744)
  → RRF ที่ implement อยู่ตอนนี้ปรับจูนผิด ไม่ใช่ของฟรี
- **การทดลอง TOP_K=3 เดิม**: SS 0.794 / RougeL 0.349 (ดีขึ้น!) แต่ IoU ร่วงเหลือ 0.252
  → context เยอะ = generation ดีขึ้น; ที่พังคือรายงาน refs 3 อันเสมอ (ดู E5)
- **outlier**: มี query ที่ gold refs = 27, 28, 95 ย่อหน้า (รวมไม่กี่ข้อ) — กลุ่มนี้ยอมเสีย ไม่ต้องจูน

---

## 2. โครงสร้างพื้นฐานที่ต้องมีก่อน (ทำก่อน E1)

### E0 — Retrieval-only eval harness

**ปัญหา**: ทุกวันนี้การวัดผล retrieval ต้องรัน LLM เต็ม pipeline (SLURM 1–2 ชม.)
ทำให้ปรับจูน retrieval ช้ามาก

**ทำอะไร**: เขียนสคริปต์ที่รัน **เฉพาะขั้น retrieval** แล้วเทียบกับ gold `refs` ใน train set
คำนวณ recall@1, recall@5, recall@10, MRR, และ "IoU ceiling" — **ไม่ต้องใช้ LLM/GPU** รันใน CPU ไม่กี่วินาที

**ทำไมสำคัญ**: เปลี่ยน loop การทดลอง E1–E4 จาก "ชั่วโมง" เป็น "วินาที"
ทดสอบ retrieval config ได้สิบ ๆ แบบก่อนค่อยรัน LLM job จริง

**ส่ง output**: ตารางตัวเลข recall@K ต่อ retrieval config — เป็นเครื่องมือกลางของ Tier A และ B ทั้งหมด

---

## 3. รายการทดลอง

แต่ละข้อมี: **สมมติฐาน / ทำอะไร / prerequisite / วิธีวัด / คาดการณ์ delta / ต้นทุน**

### Tier A — Retrieval (ผลตอบแทนสูงสุด)

#### E1 — Cross-encoder reranking ⭐ สำคัญสุด

- **สมมติฐาน**: dense/RRF จัดอันดับหยาบ ได้ recall@1 แค่ 57.5%; cross-encoder ให้คะแนน
  query-paragraph แบบละเอียด (เห็นทั้งคู่พร้อมกัน) — งานวิจัย 2025 ชี้ pipeline สองชั้น
  (hybrid retrieve → neural rerank) ได้ recall@5 สูงถึง ~0.82
- **ทำอะไร**: retrieve top-N (N=20–50) ด้วย retriever เดิม → rerank ด้วย cross-encoder →
  เอา top-1 (หรือ top-K) ขั้นนี้แทน `retrieve_top_k` ที่ textsum/model/run.py:53
- **prerequisite**: ดาวน์โหลด `BAAI/bge-reranker-v2-m3` (~2.3GB) บน login node
  เพิ่มใน `textsum/download_models.sh` — compute node offline ดาวน์โหลดไม่ได้
- **วิธีวัด**: E0 harness — recall@1 ก่อน/หลัง rerank; แล้วรัน full eval เทียบ composite
- **คาดการณ์**: recall@1 57.5% → **72–80%**; composite **+0.04 ถึง +0.08**
- **ต้นทุน**: โหลดโมเดลเพิ่ม 1 ตัว, rerank N×1239 คู่ — บน GPU ไม่กี่นาที
- **ตัวเลือกที่ควร A/B**: `bge-reranker-v2-m3` (baseline ที่แนะนำ — เร็ว, multilingual, รองรับไทย),
  `Qwen3-Reranker-0.6B`, `jina-reranker-v3` (คุณภาพสูงสุดใน benchmark แต่ต้องโหลดเพิ่ม)

#### E2 — bge-m3 multi-vector (ColBERT) reranking

- **สมมติฐาน**: bge-m3 ผลิต dense + sparse + colbert (multi-vector) ได้ในตัว;
  คะแนน ColBERT แบบ late-interaction จัดอันดับดีกว่า cosine ของ single-vector —
  และ **ไม่ต้องโหลดโมเดลใหม่** (cache มี bge-m3 อยู่แล้ว)
- **ทำอะไร**: `pip install FlagEmbedding` → ใช้ `BGEM3FlagModel` ดึง 3 คะแนน รวมด้วยน้ำหนัก
  ที่แนะนำ (dense/sparse/colbert ≈ 0.4/0.2/0.4) — ทางนี้คือ `USE_TRIPLE` ที่ exp02/run.py:22-26
  เขียนเผื่อไว้แต่ปัจจุบัน fall back เพราะไม่ได้ install
- **prerequisite**: `pip install FlagEmbedding` (ไม่ต้องดาวน์โหลดโมเดล)
- **วิธีวัด**: E0 harness — recall@1 เทียบ dense ล้วน
- **คาดการณ์**: recall@1 **+5–10 จุด**; composite **+0.02 ถึง +0.04**
- **ต้นทุน**: ต่ำ ทำได้ทันที — **แนะนำให้ลองก่อน E1** เพราะไม่ต้องดาวน์โหลด

#### E3 — HyDE / query expansion

- **สมมติฐาน**: query เป็น "คำถาม" แต่ paragraph เป็น "ข้อความบอกเล่า" → คำศัพท์ไม่ตรงกัน
  HyDE ให้ LLM แต่งคำตอบสมมติแล้ว embed คำตอบนั้นแทนคำถาม
  **ข้อควรระวัง**: query จำนวนมากอ้างเลขเฉพาะ (ครั้งที่ประชุม, เลขหนังสือ, วันที่) —
  HyDE ช่วยน้อยกับ query เชิงตัวเลข จุดนั้นต้องพึ่ง lexical match (BM25) → อย่าทิ้ง BM25
- **ทำอะไร**: pre-pass ให้ LLM แต่งคำตอบสมมติต่อ query → embed → retrieve;
  หรือ multi-query expansion (แตก query เป็นหลายแบบ)
- **prerequisite**: ไม่มี (ใช้ LLM ที่มีอยู่)
- **วิธีวัด**: E0 harness — recall@1 แยกดูกลุ่ม query เชิงความหมาย vs เชิงตัวเลข
- **คาดการณ์**: composite **+0.01 ถึง +0.03**
- **ต้นทุน**: ปานกลาง — เพิ่ม LLM pass ครอบทุก query (~10–20 นาที)

#### E4 — ปรับจูน hybrid fusion

- **สมมติฐาน**: RRF ปัจจุบัน (`RRF_K=60`, น้ำหนัก BM25/dense เท่ากัน ที่ exp02/run.py:68-97)
  ยังไม่จูน — หลักฐาน: exp02 hybrid ได้ IoU **ต่ำกว่า** dense ล้วน fusion จึง net-negative
- **ทำอะไร**: grid search บน `RRF_K`, น้ำหนัก BM25/dense, `RETRIEVAL_POOL`;
  ลองใช้ bge-m3 sparse แทน BM25; ลอง weighted score fusion (normalize แล้วถ่วงน้ำหนัก) แทน RRF
- **prerequisite**: ไม่มี
- **วิธีวัด**: E0 harness — เกือบฟรี ทดสอบได้สิบ ๆ config
- **คาดการณ์**: composite **+0.005 ถึง +0.02**
- **ต้นทุน**: ต่ำมาก

### Tier B — Refs reporting / IoU

#### E5 — แยก context กับ refs + ให้ LLM อ้างอิงเอง ⭐

- **สมมติฐาน**: TOP_K=3 เดิมพิสูจน์แล้วว่า context เยอะ → generation ดีขึ้น (SS 0.794, RougeL 0.349)
  ที่พังคือรายงาน refs 3 อันเสมอ → IoU ตก ทางแก้: **ป้อน N ย่อหน้าให้ LLM แต่รายงานเฉพาะที่ใช้จริง**
- **ทำอะไร**: retrieve top-N (N=5) → ใส่ context พร้อมเลขกำกับ `[1]..[5]` → prompt ให้ LLM
  ตอบคำถาม **พร้อมบอกว่าใช้ย่อหน้าหมายเลขใดบ้าง** → parse เลขที่อ้าง → ใช้เป็น `refs`
  → generation ได้ context ครบ, `refs` กลายเป็น adaptive K อัตโนมัติ
- **prerequisite**: ไม่มี (ปรับ prompt + parsing)
- **หมายเหตุ**: นี่คือการเลิกกฎ "honest TOP_K=1" เดิม — **ทำได้ถูกต้องตามกติกา** เพราะ metric
  เทียบแค่ "เซ็ตของ refs" ไม่ได้บังคับว่า refs ต้องเท่าจำนวนที่ป้อน LLM
- **วิธีวัด**: full eval — ดู SS, RougeL (context) และ IoU (adaptive K) พร้อมกัน
- **คาดการณ์**: composite **+0.02 ถึง +0.05**
- **ต้นทุน**: ต่ำ — context ยาวขึ้นเล็กน้อย (ย่อหน้าสั้น ไม่ชน max_model_len)

#### E6 — Adaptive K predictor

- **สมมติฐาน**: 71.7% ของ query มี 1 ref; ทำนาย K จากการกระจายคะแนน retrieval —
  ถ้าคะแนน rank1 ทิ้งห่าง rank2 มาก ⇒ K=1; ถ้าคะแนนใกล้กัน ⇒ K≥2
- **ทำอะไร**: ตั้ง threshold บน "ช่องว่างคะแนน" ของ reranker หรือ "นับย่อหน้าที่คะแนนเกินค่าเกณฑ์"
  → calibrate threshold บน train set; ใช้แทนหรือเสริม E5
- **prerequisite**: ทำหลัง E1 (ใช้คะแนน reranker)
- **วิธีวัด**: E0 harness จำลอง IoU ได้ตรง ๆ กับ gold refs โดยไม่ต้องรัน LLM
- **คาดการณ์**: IoU ดีขึ้น; composite **+0.01 ถึง +0.03**
- **ต้นทุน**: ต่ำมาก

### Tier C — Generation (คุ้มหลังแก้ retrieval แล้ว — RougeL กลุ่ม HIT ยังแค่ 0.414)

#### E7 — Few-shot prompting

- **สมมติฐาน**: gold answer มีสไตล์สม่ำเสมอ (กระชับ ~175 ตัวอักษร ตอบตรงคำถาม);
  ตัวอย่าง few-shot ยึดสไตล์/ความยาวได้ดีกว่าคำสั่งบรรยาย
- **ทำอะไร**: เติม 2–3 ตัวอย่าง (query, ย่อหน้า, คำตอบ gold) จาก train set ก่อน prompt จริง
- **prerequisite**: ไม่มี
- **วิธีวัด**: full eval — RougeL + SS; **ระวัง**: ตอนวัดบน train อย่าใช้ตัวอย่างที่เป็น query ของตัวเอง
- **คาดการณ์**: RougeL **+0.01 ถึง +0.03**
- **ต้นทุน**: ต่ำ (prompt ยาวขึ้น)

#### E8 — Length calibration

- **สมมติฐาน**: RougeL เป็น F-measure — token เกินที่ไม่ match ทำลาย precision;
  ถ้าโมเดลเขียน 400 ตัวอักษรขณะ gold 175 → RougeL ตก
- **ทำอะไร**: วัดการกระจายความยาวคำตอบปัจจุบันเทียบ gold; ถ้าเขียนยาวเกิน → คุม prompt
  ("ตอบสั้น ไม่เกิน N ประโยค"), ลด `max_tokens`, หรือตัดท้าย; ถ้าสั้นไปสำหรับ multi-ref → กลับด้าน
- **prerequisite**: ไม่มี
- **วิธีวัด**: full eval — RougeL; plot ความยาว pred vs gold
- **คาดการณ์**: RougeL **+0.005 ถึง +0.02**
- **ต้นทุน**: ต่ำมาก

#### E9 — Model & sampling sweep

- **สมมติฐาน**: exp02 สรุปว่า Qwen3-32B ≤ Qwen2.5-7B แต่ **ข้อสรุปนั้น confounded** —
  ตอนนั้น retrieval แย่ทั้งคู่ (recall@1 ~57%) เลยวัดคุณภาพโมเดลไม่ออก ต้องเทสต์ใหม่หลังแก้ retrieval
- **ทำอะไร**: ตรึง retrieval ที่ดีแล้ว → เทียบ Qwen2.5-7B / Qwen3-32B-AWQ /
  (ดาวน์โหลด Qwen2.5-14B-Instruct); sweep `temperature`, `repetition_penalty` (run.py:138)
- **prerequisite**: ทำหลัง E1/E2; ถ้าจะเทสต์ 14B ต้องดาวน์โหลดบน login node
- **วิธีวัด**: full eval บนกลุ่ม HIT — RougeL + SS
- **คาดการณ์**: ไม่ทราบ — ต้องเทสต์ใหม่
- **ต้นทุน**: ปานกลาง (หลาย LLM job)

#### E10 — ออกแบบ prompt ใหม่ให้เป็น abstractive QA

- **สมมติฐาน**: gold เป็นการเรียบเรียงใหม่ 92% — prompt "คัดลอกคำต่อคำ" ของ exp02 ผิดทาง,
  prompt "สรุปกระชับครอบคลุม" ของ textsum กว้างไป; ต้องสั่งแบบ direct-QA
- **ทำอะไร**: เขียน prompt ใหม่: "ตอบคำถามให้ตรงประเด็น กระชับ อ้างข้อเท็จจริงจากเอกสาร เรียบเรียงใหม่ได้"
  A/B เทียบ prompt ปัจจุบัน (textsum/model/run.py:62-72, exp02/run.py:101-123)
- **prerequisite**: ไม่มี
- **วิธีวัด**: full eval — RougeL + SS
- **คาดการณ์**: composite **+0.01 ถึง +0.03**
- **ต้นทุน**: ต่ำมาก

---

## 4. สรุป expected impact

| ID | เทคนิค | metric หลัก | คาดการณ์ Δcomposite | ต้นทุน | ลำดับ |
|---|---|---|---|---|---|
| E0 | retrieval eval harness | (เครื่องมือ) | — | ต่ำ | **ทำก่อน** |
| E1 | cross-encoder rerank | recall@1 → IoU+SS+RougeL | +0.04 – 0.08 | กลาง | ⭐ 1 |
| E2 | bge-m3 colbert rerank | recall@1 | +0.02 – 0.04 | ต่ำ | 2 |
| E5 | แยก context/refs + cite | IoU + SS + RougeL | +0.02 – 0.05 | ต่ำ | ⭐ 3 |
| E4 | จูน hybrid fusion | recall@1 | +0.005 – 0.02 | ต่ำมาก | 4 |
| E6 | adaptive K | IoU | +0.01 – 0.03 | ต่ำมาก | 5 |
| E10 | prompt abstractive QA | RougeL + SS | +0.01 – 0.03 | ต่ำมาก | 6 |
| E7 | few-shot | RougeL | +0.01 – 0.03 | ต่ำ | 7 |
| E3 | HyDE / query expansion | recall@1 | +0.01 – 0.03 | กลาง | 8 |
| E8 | length calibration | RougeL | +0.005 – 0.02 | ต่ำมาก | 9 |
| E9 | model & sampling sweep | RougeL + SS | ? | กลาง | 10 |

> **delta บวกกันตรง ๆ ไม่ได้** — การแก้ retrieval (E1/E2) ช่วยทั้ง 3 metric แล้วทำให้
> ห้องว่างของ generation (E7–E10) เปลี่ยนไป; ต้องวัดสะสมจริงทีละขั้น

---

## 5. โปรโตคอลการทดลอง

1. **ทำ E0 ก่อน** — ไม่มี harness นี้ การจูน Tier A จะช้าจนทำไม่ไหว
2. **เปลี่ยนทีละอย่าง** — 1 การทดลอง = 1 ตัวแปร เทียบกับ baseline เดิมเสมอ
3. **โครงโฟลเดอร์** — ทำตามแบบ exp02: 1 การทดลอง = 1 โฟลเดอร์ (`exp03/`, `exp04/`, …)
   หรือ 1 git branch — เก็บ `run.py` + ผล eval ไว้ด้วยกัน
4. **แยก config knob** — เพิ่มตัวแปรแยก `RETRIEVE_K` (ดึงมา), `GEN_K` (ป้อน LLM),
   `REF_K` (รายงาน) แทน `TOP_K` ตัวเดียว (run.py:15) — ปลดล็อก E5/E6
5. **วัดผลด้วย** `score.py` บน train set เสมอ; retrieval ล้วนใช้ E0 harness
6. **บันทึกทุก run** — ต่อแถวในตารางผล (RougeL / SS / IoU / composite + config ที่เปลี่ยน)
   จะได้เห็นว่าเทคนิคไหนได้ผลจริงและได้เท่าไร
7. **ระวัง overfit train** — train เป็นชุดเดียวที่มี label; อย่าจูน threshold (E6) หรือ
   เลือกตัวอย่าง few-shot (E7) แนบ train แน่นเกินไป

---

## 6. อ้างอิง SOTA (พ.ค. 2026)

- **Reranker**: pipeline สองชั้น (hybrid retrieve → neural rerank) = SOTA ปัจจุบัน
  สำหรับ recall — `bge-reranker-v2-m3` เป็น baseline ที่ใช้งานได้จริง (เร็ว, multilingual),
  `jina-reranker-v3` (listwise, 0.6B) นำ benchmark BEIR, `Qwen3-Reranker` แม่นแต่ inference ช้า
- **Embedding**: BGE-M3 ยังเป็น SOTA multilingual (รองรับไทย), จุดเด่นคือรวม
  dense + sparse + colbert ในโมเดลเดียว — ใช้ทั้ง 3 mode ได้โดยไม่ต้องโหลดอะไรเพิ่ม (ดู E2)
- **HyDE / query expansion**: ช่วย recall เมื่อ query กับเอกสารใช้คำต่างกัน
  แต่ช่วยน้อยกับ query เชิงตัวเลข/เฉพาะเจาะจง — ใช้คู่ sparse retrieval เสมอ
