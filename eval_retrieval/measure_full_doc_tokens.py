"""Quick token-count measurement for exp35 planning (full-doc context, no retrieval)."""
import os
os.environ.setdefault("HF_HOME", "/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import json
from pathlib import Path
from transformers import AutoTokenizer

PROJECT = Path("/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047")

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B-AWQ")
data = json.load(open(PROJECT / "textsum/eval_train/test.json", encoding="utf-8"))

# Match exp34's prompt build exactly
SYSTEM_MSG = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)
SHOT1_QUERY = "ในการประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้นที่ใด"
SHOT1_PARAS = [
    "ครั้งที่ ๔๙",
    "วันพุธที่ ๑๙ มีนาคม ๒๕๖๘",
    "ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา",
    "_________________________",
    "กรรมาธิการผู้มาประชุม",
]
SHOT1_ANSWER = "การประชุมสถาบันการเงินครั้งที่ 49 มีการจัดประชุมขึ้น ณ ห้องประชุมกรรมาธิการ N 406 ชั้น ๔ อาคารรัฐสภา [อ้างอิง: 3]"
SHOT2_QUERY = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมาธิการจัดขึ้นเพื่ออะไร"
SHOT2_PARAS = [
    "เริ่มประชุมเวลา ๐๙.๔๖ นาฬิกา",
    "เมื่อกรรมาธิการมาครบองค์ประชุมแล้ว ประธานคณะกรรมาธิการได้กล่าวเปิดประชุม และดำเนินการประชุมตามระเบียบวาระการประชุม สรุปสาระสำคัญได้ ดังนี้",
    "ระเบียบวาระที่ ๑ เรื่องที่ประธานแจ้งต่อที่ประชุม",
    "สำนักงานเลขาธิการสภาผู้แทนราษฎรขอความอนุเคราะห์ตอบแบบสำรวจความพึงพอใจและความไม่พึงพอใจของคณะกรรมาธิการต่อการบริหารจัดการด้านการประชุม การศึกษาดูงาน และการจัดสัมมนา เพื่อนำผลการประเมินความพึงพอใจและความไม่พึงพอใจมาเป็นข้อมูลในการทบทวน ปรับปรุง และพัฒนาการปฏิบัติงานให้มีประสิทธิภาพต่อไป",
    "ที่ประชุมรับทราบ",
]
SHOT2_ANSWER = "การจัดทำแบบสำรวจความพึงพอใจและไม่พึงพอใจของคณะกรรมการในครั้งนี้ มีการจัดทำขึ้นเพื่อนำข้อมูลที่ได้มาทบทวน ปรับปรุง รวมถึงนำไปพัฒนาการปฏิบัติงานให้มีประสิทธิภาพยิ่งขึ้น [อ้างอิง: 4]"


def build_prompt(query, paras):
    ctx_lines = [f"[{i + 1}] {t}" for i, t in enumerate(paras)]
    context = "\n".join(ctx_lines)
    return (
        f"คำถาม: {query}\n\n"
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )


# Build full-doc message list, count chat-template tokens
prompt_lens = []
doc_index = {doc["doc_id"]: doc["paragraphs"] for doc in data["docs"]}
sample_q_text = "ในการประชุมครั้งนี้มีเรื่องอะไรบ้างที่ประธานแจ้งต่อที่ประชุม"  # placeholder

for doc_id, paragraphs in doc_index.items():
    valid = [p for p in paragraphs
             if p["text"].strip() and not set(p["text"].strip()) <= set("_-=. \t\n")]
    para_texts = [p["text"] for p in valid]
    msgs = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(SHOT1_QUERY, SHOT1_PARAS)},
        {"role": "assistant", "content": SHOT1_ANSWER},
        {"role": "user", "content": build_prompt(SHOT2_QUERY, SHOT2_PARAS)},
        {"role": "assistant", "content": SHOT2_ANSWER},
        {"role": "user", "content": build_prompt(sample_q_text, para_texts)},
    ]
    rendered = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    n = len(tok.encode(rendered))
    prompt_lens.append((doc_id, len(valid), n))

prompt_lens.sort(key=lambda x: -x[2])
print(f"Total prompt tokens (full doc + system + 2-shot + query, chat template):")
all_tok = sorted(x[2] for x in prompt_lens)
print(f"  min={all_tok[0]}  median={all_tok[len(all_tok)//2]}  mean={sum(all_tok)//len(all_tok)}")
print(f"  p90={all_tok[int(len(all_tok)*0.9)]}  p95={all_tok[int(len(all_tok)*0.95)]}  max={all_tok[-1]}")

print("\nTop-5 largest docs:")
for d, npara, ntok in prompt_lens[:5]:
    print(f"  {d}: {npara} paras -> {ntok} prompt tokens")

print("\nFit at various max_model_len:")
for budget in (8192, 16384, 24576, 32768, 49152, 65536):
    over = sum(1 for _,_,n in prompt_lens if n > budget)
    print(f"  budget {budget}: {over}/{len(prompt_lens)} docs OVER  ({100*over/len(prompt_lens):.0f}%)")
