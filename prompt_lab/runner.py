"""
Prompt-tuning lab — fast variant comparison on 100-query dev sample.
Loads Qwen3-30B-A3B-Instruct-2507-FP8 once, runs N prompt variants.
Outputs RougeL + IoU table (no SS-score for speed).

To add a new variant, append to PROMPT_VARIANTS list at bottom.
"""
import os, re, json, csv
from pathlib import Path
from collections import defaultdict
from vllm import LLM, SamplingParams

PROJECT = Path('/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047')
DEV_FILE = PROJECT / 'prompt_lab/dev_sample.json'
MODEL_NAME = os.environ.get('LLM_MODEL', 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8')
MODEL_TAG = MODEL_NAME.split('/')[-1].replace('.', '_')
RESULT_SUFFIX = os.environ.get('RESULT_SUFFIX', '').strip()
RESULT_DIR = PROJECT / f'prompt_lab/results_{MODEL_TAG}{("_" + RESULT_SUFFIX) if RESULT_SUFFIX else ""}'
RESULT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Model: {MODEL_NAME}  tag={MODEL_TAG}', flush=True)

# ============ shared shots (exp42's pair) ============
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

# ============ R5 shot variants ============
# Synthetic 2-ref shot — covers mid-cardinality between _SHOT1 (1 ref) and _SHOT2 (4 refs)
_SHOT3_QUERY = "ในการประชุมครั้งที่ 49 มีการพิจารณาเรื่องใดเป็นวาระสำคัญ"
_SHOT3_PARAS = [
    "วาระที่ ๑ เรื่องที่ประธานแจ้งต่อที่ประชุม",
    "วาระที่ ๒ รับรองรายงานการประชุม ครั้งที่ ๔๘",
    "วาระที่ ๓ พิจารณาแผนยุทธศาสตร์การเงินการคลังปี ๒๕๖๘",
    "วาระที่ ๔ พิจารณาความก้าวหน้าโครงการสินเชื่อ SME",
    "วาระที่ ๕ เรื่องอื่นๆ",
]
_SHOT3_ANSWER = "ในการประชุมครั้งที่ 49 มีการพิจารณาวาระสำคัญสองเรื่องคือ แผนยุทธศาสตร์การเงินการคลังปี ๒๕๖๘ และความก้าวหน้าโครงการสินเชื่อ SME [อ้างอิง: 3, 4]"

_DEFAULT_SHOTS = [
    (_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER),
    (_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER),
]
_SHOTS_NONE = []
_SHOTS_ONLY_SINGLE = [(_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER)]
_SHOTS_ONLY_MULTI = [(_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER)]
_SHOTS_SWAPPED = [
    (_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER),
    (_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER),
]
_SHOTS_THREE = [
    (_SHOT1_QUERY, _SHOT1_PARAS, _SHOT1_ANSWER),
    (_SHOT2_QUERY, _SHOT2_PARAS, _SHOT2_ANSWER),
    (_SHOT3_QUERY, _SHOT3_PARAS, _SHOT3_ANSWER),
]

_CITE_SHOTS = [
    (_SHOT1_QUERY, _SHOT1_PARAS, "[อ้างอิง: 3]"),
    (_SHOT2_QUERY, _SHOT2_PARAS, "[อ้างอิง: 2, 3, 4, 5]"),
]
_CITE_SHOTS_THREE = [
    (_SHOT1_QUERY, _SHOT1_PARAS, "[อ้างอิง: 3]"),
    (_SHOT2_QUERY, _SHOT2_PARAS, "[อ้างอิง: 2, 3, 4, 5]"),
    (_SHOT3_QUERY, _SHOT3_PARAS, "[อ้างอิง: 3, 4]"),
]


def filter_valid_paragraphs(paragraphs):
    def is_valid(p):
        text = p['text'].strip()
        if not text:
            return False
        if set(text) <= set('_-=. \t\n'):
            return False
        return True
    return [p for p in paragraphs if is_valid(p)]


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
    return re.sub(r'\s*\[อ้างอิง[^\]]*\]', '', text).strip()


def final_response_text(text, require_think_marker=False):
    """For thinking models, score only the final answer after </think>."""
    marker = '</think>'
    idx = text.rfind(marker)
    if idx == -1:
        if require_think_marker:
            return ''
        return text.strip()
    return text[idx + len(marker):].strip()


# ============ scoring (fast, no SS-score) ============
from pythainlp.tokenize import word_tokenize

def rouge_l(pred, gold):
    """ROUGE-L on Thai (newmm tokenizer)"""
    pred_tokens = word_tokenize(pred, engine='newmm')
    gold_tokens = word_tokenize(gold, engine='newmm')
    if not pred_tokens or not gold_tokens:
        return 0.0
    # LCS
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if pred_tokens[i-1] == gold_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def iou(pred_refs, gold_refs):
    p, g = set(pred_refs), set(gold_refs)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return len(p & g) / len(p | g)


# ============ PROMPT VARIANTS ============
# Each variant returns (system_msg, build_messages_fn) where build_messages_fn(query, paras) returns list of messages

SYSTEM_BASELINE = (
    "คุณเป็นผู้ช่วยสรุปเอกสารภาษาไทย "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

def _build_prompt_baseline(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_brevity(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทย**สั้นที่สุดเท่าที่ครอบคลุมคำถาม** "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_all(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"**ระบุเลขย่อหน้าที่ใช้เป็นข้อมูลทุกย่อหน้า**ในรูปแบบ [อ้างอิง: X, Y, Z, ...]\n"
        f"คำตอบ:"
    )

def _build_prompt_minimal_cite(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทยอย่างกระชับและครอบคลุม "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"**ระบุเฉพาะเลขย่อหน้าที่จำเป็นต่อคำตอบ** (ไม่ต้องอ้างย่อหน้าซ้ำซ้อน) "
        f"ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_extractive(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามโดย**คัดลอกข้อความจากย่อหน้าโดยตรงให้มากที่สุด** "
        f"หลีกเลี่ยงการเรียบเรียงใหม่ "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

# ---- Round 2 variants (combine/refine V1_brevity winner) ----
def _build_prompt_brevity_minimal(query, paras):
    """V1 (brevity) + V3 (minimal cite) combined"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทย**สั้นที่สุดเท่าที่ครอบคลุมคำถาม** "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"**ระบุเฉพาะเลขย่อหน้าที่จำเป็นต่อคำตอบ** ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_brevity_one_sentence(query, paras):
    """Stronger brevity: one sentence"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบเป็นภาษาไทย**ในประโยคเดียว**โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_brevity_length_hint(query, paras):
    """Brevity with explicit char-count hint"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดตอบคำถามเป็นภาษาไทย**กระชับและไม่เกินประมาณ 200 ตัวอักษร** "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_brevity_no_redundant_cite(query, paras):
    """V1 brevity + explicit 'no redundant' citation"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: โปรดสรุปคำตอบเป็นภาษาไทย**สั้นที่สุดเท่าที่ครอบคลุมคำถาม** "
        f"โดยอ้างอิงจากข้อมูลที่ให้มาเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] "
        f"**ห้ามอ้างย่อหน้าที่ไม่ได้นำเนื้อหามาใช้ในคำตอบ**\n"
        f"คำตอบ:"
    )

def _build_prompt_brevity_factual(query, paras):
    """V1 + emphasize factual / no inference"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

# ---- Round 3 variants: stack V10_factual (winner on A3B, 2nd on AWQ) with other winners / tricks ----
def _build_prompt_v11_factual_one_sentence(query, paras):
    """V10 + V7: factual + one sentence (best for 27B-FP8 which liked V7)"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**ในประโยคเดียวที่สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v12_factual_minimal_cite(query, paras):
    """V10 + V6: factual + minimal cite (best for AWQ which liked V6)"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"**ระบุเฉพาะเลขย่อหน้าที่จำเป็นต่อคำตอบ**ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v13_factual_extract(query, paras):
    """V10 + 'use original wording' — addresses 27B-FP8's RougeL crash by forcing extractive surface form"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"**ใช้ถ้อยคำเดิมจากย่อหน้าให้มากที่สุด** ห้ามเรียบเรียงใหม่ "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v14_factual_named_entities(query, paras):
    """V10 + force named entities (names, numbers, dates) — boost RougeL via key term coverage"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น **โดยต้องระบุชื่อ ตัวเลข วันที่ และตำแหน่งที่เกี่ยวข้องให้ครบ** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v15_factual_no_redundant(query, paras):
    """V10 + V9: factual + ห้ามอ้างย่อหน้าที่ไม่ได้ใช้"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] "
        f"**ห้ามอ้างย่อหน้าที่ไม่ได้นำเนื้อหามาใช้ในคำตอบ**\n"
        f"คำตอบ:"
    )

def _build_prompt_v16_factual_complete(query, paras):
    """V10 but loosen brevity — give the model permission to be slightly longer if coverage demands"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**ตรงประเด็น ครอบคลุมข้อมูลที่เกี่ยวข้องทุกประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

# ---- Round 4: stack R3 winners + system-prompt experiment + V13 recovery ----
def _build_prompt_v17_entities_no_redundant(query, paras):
    """V14 entities + V15 no_redundant guard — combine RougeL champ + AWQ champ"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น **โดยต้องระบุชื่อ ตัวเลข วันที่ และตำแหน่งที่เกี่ยวข้องให้ครบ** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] "
        f"**ห้ามอ้างย่อหน้าที่ไม่ได้นำเนื้อหามาใช้ในคำตอบ**\n"
        f"คำตอบ:"
    )

def _build_prompt_v18_complete_entities(query, paras):
    """V16 complete + V14 entities — coverage breadth + entity precision (27B-FP8 candidate)"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**ตรงประเด็น ครอบคลุมข้อมูลที่เกี่ยวข้องทุกประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น **โดยต้องระบุชื่อ ตัวเลข วันที่ และตำแหน่งที่เกี่ยวข้องให้ครบ** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v19_one_sentence_entities(query, paras):
    """V11 one-sentence + V14 entities — terse but fact-heavy"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**ในประโยคเดียวที่สั้นและตรงประเด็น** "
        f"**โดยต้องระบุชื่อ ตัวเลข วันที่ และตำแหน่งที่เกี่ยวข้องให้ครบ** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v20_role_factual(query, paras):
    """V10 user prompt — paired with SYSTEM_ROLE_EXPERT (persona swap, see registry)"""
    return _build_prompt_brevity_factual(query, paras)

SYSTEM_ROLE_EXPERT = (
    "คุณเป็นผู้เชี่ยวชาญสรุปบันทึกการประชุมรัฐสภาไทย "
    "เน้นข้อเท็จจริง สั้น กระชับ "
    "ตอบคำถามโดยอ้างอิงจากย่อหน้าที่ให้มาเท่านั้น ห้ามแต่งเติม"
)

def _build_prompt_v21_extract_soft(query, paras):
    """V13 weakened — 'cite key phrases' not 'use original wording entirely' (recover RougeL on 27B-FP8)"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"**อ้างถ้อยคำสำคัญจากย่อหน้า**โดยเรียบเรียงให้เป็นประโยคที่สมบูรณ์ "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]\n"
        f"คำตอบ:"
    )

def _build_prompt_v22_triple_stack(query, paras):
    """V10 factual + V14 entities + V15 no_redundant — kitchen sink"""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"**โดยต้องระบุชื่อ ตัวเลข วันที่ และตำแหน่งที่เกี่ยวข้องให้ครบ** "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] "
        f"**ห้ามอ้างย่อหน้าที่ไม่ได้นำเนื้อหามาใช้ในคำตอบ**\n"
        f"คำตอบ:"
    )

# ---- Round 6: citation-only prompts optimized for reference IoU ----
SYSTEM_CITATION_ONLY = (
    "คุณเป็นผู้คัดเลือกย่อหน้าอ้างอิงภาษาไทย "
    "ตอบด้วยรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุปหรือคำอธิบาย"
)

SYSTEM_CITATION_RECALL = (
    "คุณเป็นผู้ตรวจหลักฐานจากเอกสารภาษาไทย "
    "เลือกเฉพาะย่อหน้าที่สนับสนุนคำตอบโดยตรง "
    "ตอบด้วยรูปแบบ [อ้างอิง: X, Y] เท่านั้น"
)

def _build_prompt_cite_minimal(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเลขย่อหน้าที่จำเป็นที่สุดสำหรับตอบคำถาม "
        f"อ้างเฉพาะย่อหน้าที่มีข้อมูลคำตอบโดยตรง ไม่อ้างย่อหน้าซ้ำซ้อน "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_direct_all(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกทุกย่อหน้าที่เป็นหลักฐานโดยตรงของคำตอบ "
        f"ถ้าคำตอบประกอบด้วยชื่อ ตัวเลข วันที่ สถานที่ ตำแหน่ง หรือรายการหลายบรรทัด "
        f"ให้รวมย่อหน้าที่มีข้อมูลเหล่านั้นทุกย่อหน้า "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_with_heading(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกย่อหน้าที่เป็นหลักฐานคำตอบโดยตรง "
        f"ถ้ารายการย่อยต้องอาศัยหัวข้อหรือป้ายกำกับก่อนหน้าเพื่อเข้าใจความหมาย "
        f"ให้อ้างทั้งหัวข้อและรายการย่อยที่เกี่ยวข้อง "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_no_heading(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเฉพาะย่อหน้าที่มีเนื้อหาคำตอบจริง "
        f"หลีกเลี่ยงการอ้างหัวข้อ ป้ายกำกับ บรรทัดคั่น หรือบริบททั่วไป "
        f"ยกเว้นเมื่อหัวข้อนั้นเป็นคำตอบโดยตรง "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_key_phrases(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เทียบคำสำคัญในคำถามกับย่อหน้า เช่น ชื่อบุคคล หน่วยงาน ครั้งที่ วันที่ จำนวนเงิน จำนวนคน "
        f"และเลือกย่อหน้าที่มีคำสำคัญหรือค่าคำตอบตรงกับคำถาม "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_silent_answer(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: คิดเงียบ ๆ ว่าคำตอบคืออะไรและมาจากย่อหน้าใด "
        f"จากนั้นส่งออกเฉพาะเลขย่อหน้าที่ใช้ตอบคำถามจริง "
        f"ในรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนคำตอบหรือเหตุผล\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_balanced(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกชุดย่อหน้าที่ทำให้ผู้อ่านตรวจสอบคำตอบได้ครบถ้วนแต่ไม่กว้างเกินไป "
        f"รวมทุกย่อหน้าที่ให้ข้อเท็จจริงเฉพาะของคำตอบ และตัดย่อหน้าที่เป็นเพียงบริบททั่วไป "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_cite_conservative(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเฉพาะย่อหน้าที่มั่นใจว่าเป็นหลักฐานของคำตอบ "
        f"ถ้ามีหลายย่อหน้าต่อเนื่องที่ร่วมกันตอบคำถาม ให้ใส่เลขทุกย่อหน้านั้น "
        f"ถ้าย่อหน้าใดไม่จำเป็นต่อการตรวจคำตอบ ให้ตัดออก "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

# ---- Round 7: R3-style prompt variants, citation-only output ----
def _build_prompt_r3cite_factual_one_sentence(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เหมือนการตอบคำถามแบบสั้นในประโยคเดียวที่ตรงประเด็น "
        f"ให้เลือกเลขย่อหน้าที่มีข้อเท็จจริงซึ่งจำเป็นต่อคำตอบนั้นเท่านั้น "
        f"ห้ามตีความหรือสรุปเกินขอบเขต "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_r3cite_factual_minimal(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเฉพาะเลขย่อหน้าที่จำเป็นต่อคำตอบแบบสั้นและตรงประเด็น "
        f"ใช้ข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น "
        f"ไม่อ้างย่อหน้าที่เป็นเพียงบริบทหรือซ้ำซ้อน "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_r3cite_factual_extract(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ถ้าต้องตอบด้วยการใช้ถ้อยคำเดิมจากเอกสารให้มากที่สุด "
        f"ให้เลือกเลขย่อหน้าที่ควรคัดข้อความมาใช้เป็นคำตอบโดยตรง "
        f"ตัดย่อหน้าที่ไม่มีถ้อยคำคำตอบจริงออก "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_r3cite_named_entities(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเลขย่อหน้าที่มีชื่อ ตัวเลข วันที่ ตำแหน่ง สถานที่ "
        f"หรือหน่วยงานที่เกี่ยวข้องและจำเป็นต่อคำตอบให้ครบ "
        f"ไม่อ้างย่อหน้าที่ไม่มี entity หรือข้อเท็จจริงของคำตอบ "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_r3cite_no_redundant(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเลขย่อหน้าที่มีข้อเท็จจริงของคำตอบโดยตรง "
        f"ห้ามอ้างย่อหน้าที่ไม่ได้นำเนื้อหามาใช้ ห้ามอ้างซ้ำซ้อน "
        f"ถ้าหลายย่อหน้าร่วมกันตอบคำถาม ให้เลือกเฉพาะชุดที่จำเป็นที่สุด "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_r3cite_complete(query, paras):
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเลขย่อหน้าที่ครอบคลุมข้อมูลที่เกี่ยวข้องทุกประเด็นของคำตอบ "
        f"ใช้เฉพาะข้อเท็จจริงที่ปรากฏในย่อหน้า ห้ามตีความเกินเอกสาร "
        f"รวมย่อหน้าที่จำเป็นต่อการตรวจคำตอบให้ครบ แต่ไม่รวมบริบททั่วไปที่ไม่จำเป็น "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

# ---- Round 8: exp77 IoU failure-mode fixes (gemma NVFP4 ref-picker) ----
# Two levers from the exp77 IoU error analysis (train, IoU 0.8156):
#   FM1 block under-citation — gold is often a contiguous block (roster / list /
#        multi-item resolution); the model cites only the lead/heading paragraph
#        (292 block queries @ meanIoU 0.66, covers only 69% of the block).
#   FM2 heading-grab — for single-ref answers the model cites the agenda-title
#        line (ระเบียบวาระ / "๕.๑ พิจารณา…") instead of the actual outcome/มติ
#        paragraph (45% of single-gold misses include a header line).
# _BLOCK / _OUTCOME / _DEDUP are the reusable clause strings.
_CLAUSE_BLOCK = (
    "**หากคำตอบอ้างถึงรายการ รายชื่อ หรือหลายข้อย่อยที่เรียงต่อเนื่องกัน "
    "(เช่น รายชื่อกรรมการ มาตรการหลายข้อ หรือมติหลายข้อ) "
    "ให้อ้างอิงย่อหน้าทุกย่อหน้าในช่วงนั้นให้ครบ ไม่ใช่เฉพาะย่อหน้าหัวเรื่อง**"
)
_CLAUSE_OUTCOME = (
    "**ให้อ้างอิงย่อหน้าที่มีเนื้อหาคำตอบหรือมติ/ผลสรุปจริง "
    "ไม่ใช่บรรทัดหัวข้อระเบียบวาระหรือชื่อหัวข้อ (เช่น 'ระเบียบวาระที่ ๔' หรือ '๕.๑ พิจารณา…') "
    "เว้นแต่หัวข้อนั้นเป็นคำตอบโดยตรง**"
)
_CLAUSE_DEDUP = (
    "หากข้อความเดียวกันปรากฏซ้ำในหลายย่อหน้า (เช่น ในสารบัญและในเนื้อการประชุม) "
    "ให้เลือกย่อหน้าที่อยู่ในเนื้อการอภิปราย/มติจริง"
)


def _factual_with_clauses(query, paras, clauses):
    """V10_factual (exp77's deployed answer+cite prompt) + extra citation clauses."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    extra = (" " + " ".join(clauses)) if clauses else ""
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: ตอบคำถามเป็นภาษาไทย**สั้นและตรงประเด็น** "
        f"ระบุข้อเท็จจริงที่ปรากฏในย่อหน้าเท่านั้น ห้ามตีความหรือสรุปเกินขอบเขต "
        f"จากนั้นระบุเลขย่อหน้าที่ใช้ในรูปแบบ [อ้างอิง: X] หรือ [อ้างอิง: X, Y]{extra}\n"
        f"คำตอบ:"
    )

def _build_prompt_g1_factual_block(query, paras):
    return _factual_with_clauses(query, paras, [_CLAUSE_BLOCK])

def _build_prompt_g2_factual_outcome(query, paras):
    return _factual_with_clauses(query, paras, [_CLAUSE_OUTCOME])

def _build_prompt_g3_factual_block_outcome(query, paras):
    return _factual_with_clauses(query, paras, [_CLAUSE_OUTCOME, _CLAUSE_BLOCK])


def _cite_only_with_rules(query, paras, rules):
    """Citation-only output (refs IoU only); `rules` is a list of bullet strings."""
    context = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(paras))
    body = " ".join(rules)
    return (
        f"ข้อมูลอ้างอิงจากเอกสาร:\n{context}\n\n"
        f"คำถาม: {query}\n\n"
        f"คำสั่ง: เลือกเลขย่อหน้าที่เป็นหลักฐานโดยตรงของคำตอบ {body} "
        f"ตอบเฉพาะรูปแบบ [อ้างอิง: X, Y] เท่านั้น ห้ามเขียนสรุป\n"
        f"คำตอบ:"
    )

def _build_prompt_g4_cite_block(query, paras):
    return _cite_only_with_rules(query, paras, [
        "หากคำตอบเป็นรายการ รายชื่อ หรือหลายข้อย่อยที่เรียงต่อเนื่องกัน "
        "ให้รวมย่อหน้าทุกย่อหน้าในช่วงนั้นให้ครบ ไม่ใช่เฉพาะย่อหน้าหัวเรื่อง"])

def _build_prompt_g5_cite_outcome(query, paras):
    return _cite_only_with_rules(query, paras, [
        "อ้างย่อหน้าที่มีเนื้อหาคำตอบหรือมติ/ผลสรุปจริง "
        "หลีกเลี่ยงบรรทัดหัวข้อระเบียบวาระหรือชื่อหัวข้อ (เช่น 'ระเบียบวาระที่ ๔', '๕.๑ พิจารณา…') "
        "เว้นแต่หัวข้อนั้นเป็นคำตอบโดยตรง"])

def _build_prompt_g6_cite_block_outcome(query, paras):
    return _cite_only_with_rules(query, paras, [
        "ตามหลักต่อไปนี้:",
        "(1) อ้างย่อหน้าที่มีเนื้อหาคำตอบหรือมติ/ผลสรุปจริง ไม่ใช่บรรทัดหัวข้อระเบียบวาระหรือชื่อหัวข้อ เว้นแต่หัวข้อนั้นเป็นคำตอบโดยตรง",
        "(2) หากคำตอบเป็นรายการ รายชื่อ หรือหลายข้อย่อยที่เรียงต่อเนื่องกัน ให้รวมย่อหน้าทุกย่อหน้าในช่วงนั้นให้ครบ",
        "(3) " + _CLAUSE_DEDUP])

# variants registry
PROMPT_VARIANTS = [
    ('V1_brevity', SYSTEM_BASELINE, _build_prompt_brevity),               # winner baseline
    ('V6_brevity_minimal', SYSTEM_BASELINE, _build_prompt_brevity_minimal),     # V1+V3
    ('V7_one_sentence', SYSTEM_BASELINE, _build_prompt_brevity_one_sentence),
    ('V8_length_hint', SYSTEM_BASELINE, _build_prompt_brevity_length_hint),
    ('V9_no_redundant', SYSTEM_BASELINE, _build_prompt_brevity_no_redundant_cite),
    ('V10_factual', SYSTEM_BASELINE, _build_prompt_brevity_factual),
    # Round 3 — V10 stacks
    ('V11_factual_one_sentence', SYSTEM_BASELINE, _build_prompt_v11_factual_one_sentence),
    ('V12_factual_minimal_cite', SYSTEM_BASELINE, _build_prompt_v12_factual_minimal_cite),
    ('V13_factual_extract', SYSTEM_BASELINE, _build_prompt_v13_factual_extract),
    ('V14_factual_named_entities', SYSTEM_BASELINE, _build_prompt_v14_factual_named_entities),
    ('V15_factual_no_redundant', SYSTEM_BASELINE, _build_prompt_v15_factual_no_redundant),
    ('V16_factual_complete', SYSTEM_BASELINE, _build_prompt_v16_factual_complete),
    # Round 4 — multi-constraint stacks + system swap + V13 recovery
    ('V17_entities_no_redundant', SYSTEM_BASELINE, _build_prompt_v17_entities_no_redundant),
    ('V18_complete_entities', SYSTEM_BASELINE, _build_prompt_v18_complete_entities),
    ('V19_one_sentence_entities', SYSTEM_BASELINE, _build_prompt_v19_one_sentence_entities),
    ('V20_role_factual', SYSTEM_ROLE_EXPERT, _build_prompt_v20_role_factual),
    ('V21_extract_soft', SYSTEM_BASELINE, _build_prompt_v21_extract_soft),
    ('V22_triple_stack', SYSTEM_BASELINE, _build_prompt_v22_triple_stack),
    # Round 5 — few-shot ablation on V10_factual base prompt
    ('F1_zero_shot', SYSTEM_BASELINE, _build_prompt_brevity_factual, _SHOTS_NONE),
    ('F2_only_single', SYSTEM_BASELINE, _build_prompt_brevity_factual, _SHOTS_ONLY_SINGLE),
    ('F3_only_multi', SYSTEM_BASELINE, _build_prompt_brevity_factual, _SHOTS_ONLY_MULTI),
    ('F4_swap_order', SYSTEM_BASELINE, _build_prompt_brevity_factual, _SHOTS_SWAPPED),
    ('F5_three_shot', SYSTEM_BASELINE, _build_prompt_brevity_factual, _SHOTS_THREE),
    ('F6_baseline', SYSTEM_BASELINE, _build_prompt_brevity_factual, _DEFAULT_SHOTS),
    # Round 6 — citation-only, rank by IoU
    ('C1_cite_minimal', SYSTEM_CITATION_ONLY, _build_prompt_cite_minimal, _CITE_SHOTS),
    ('C2_cite_direct_all', SYSTEM_CITATION_ONLY, _build_prompt_cite_direct_all, _CITE_SHOTS),
    ('C3_cite_with_heading', SYSTEM_CITATION_RECALL, _build_prompt_cite_with_heading, _CITE_SHOTS),
    ('C4_cite_no_heading', SYSTEM_CITATION_ONLY, _build_prompt_cite_no_heading, _CITE_SHOTS),
    ('C5_cite_key_phrases', SYSTEM_CITATION_ONLY, _build_prompt_cite_key_phrases, _CITE_SHOTS),
    ('C6_cite_silent_answer', SYSTEM_CITATION_ONLY, _build_prompt_cite_silent_answer, _CITE_SHOTS),
    ('C7_cite_balanced_3shot', SYSTEM_CITATION_RECALL, _build_prompt_cite_balanced, _CITE_SHOTS_THREE),
    ('C8_cite_conservative_3shot', SYSTEM_CITATION_ONLY, _build_prompt_cite_conservative, _CITE_SHOTS_THREE),
    # Round 7 — R3 prompt shapes, citation-only output
    ('R3C11_factual_one_sentence', SYSTEM_CITATION_ONLY, _build_prompt_r3cite_factual_one_sentence, _CITE_SHOTS),
    ('R3C12_factual_minimal', SYSTEM_CITATION_ONLY, _build_prompt_r3cite_factual_minimal, _CITE_SHOTS),
    ('R3C13_factual_extract', SYSTEM_CITATION_ONLY, _build_prompt_r3cite_factual_extract, _CITE_SHOTS),
    ('R3C14_named_entities', SYSTEM_CITATION_ONLY, _build_prompt_r3cite_named_entities, _CITE_SHOTS),
    ('R3C15_no_redundant', SYSTEM_CITATION_ONLY, _build_prompt_r3cite_no_redundant, _CITE_SHOTS),
    ('R3C16_complete', SYSTEM_CITATION_RECALL, _build_prompt_r3cite_complete, _CITE_SHOTS),
    # Round 8 — exp77 IoU failure-mode fixes; answer+cite (G1-3, base V10_factual) + citation-only (G4-6)
    ('G1_factual_block', SYSTEM_BASELINE, _build_prompt_g1_factual_block, _DEFAULT_SHOTS),
    ('G2_factual_outcome', SYSTEM_BASELINE, _build_prompt_g2_factual_outcome, _DEFAULT_SHOTS),
    ('G3_factual_block_outcome', SYSTEM_BASELINE, _build_prompt_g3_factual_block_outcome, _DEFAULT_SHOTS),
    ('G4_cite_block', SYSTEM_CITATION_ONLY, _build_prompt_g4_cite_block, _CITE_SHOTS),
    ('G5_cite_outcome', SYSTEM_CITATION_ONLY, _build_prompt_g5_cite_outcome, _CITE_SHOTS),
    ('G6_cite_block_outcome', SYSTEM_CITATION_RECALL, _build_prompt_g6_cite_block_outcome, _CITE_SHOTS),
]

# Filter via env var (e.g. VARIANTS=V10_factual,V6_brevity_minimal)
_only = os.environ.get('VARIANTS', '').strip()
if _only:
    _keep = {v.strip() for v in _only.split(',') if v.strip()}
    PROMPT_VARIANTS = [v for v in PROMPT_VARIANTS if v[0] in _keep]
    print(f'Filtered to variants: {[v[0] for v in PROMPT_VARIANTS]}', flush=True)


def build_messages(system, build_fn, query, paras, shots=None):
    if shots is None:
        shots = _DEFAULT_SHOTS
    msgs = []
    if system:
        msgs.append({'role': 'system', 'content': system})
    for shot_q, shot_p, shot_a in shots:
        msgs.append({'role': 'user', 'content': build_fn(shot_q, shot_p)})
        msgs.append({'role': 'assistant', 'content': shot_a})
    msgs.append({'role': 'user', 'content': build_fn(query, paras)})
    return msgs


def main():
    data = json.load(open(DEV_FILE, encoding='utf-8'))
    doc_index = {d['doc_id']: d['paragraphs'] for d in data['docs']}
    queries = data['queries']
    print(f"Dev sample: {len(queries)} queries, {len(doc_index)} docs", flush=True)

    # Build per-query items: gen_pids, gen_texts
    items = []
    for q in queries:
        paras = filter_valid_paragraphs(doc_index[q['doc_id']])
        gen_pids = [p['para_id'] for p in paras]
        gen_texts = [p['text'] for p in paras]
        items.append({
            'ID': q['ID'],
            'query': q['query'],
            'gen_pids': gen_pids,
            'gen_texts': gen_texts,
            'gold_abs': q.get('abstractive', ''),
            'gold_refs': q.get('refs') or [],
        })

    print(f"avg pool size: {sum(len(it['gen_pids']) for it in items)/len(items):.1f}", flush=True)

    llm = LLM(
        model=MODEL_NAME, max_model_len=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(os.environ.get('GPU_MEM_UTIL', '0.95')),
        enable_prefix_caching=True,  # output-neutral; reuses the ~14K doc prefix
        dtype='bfloat16', enforce_eager=True, trust_remote_code=True,
        limit_mm_per_prompt={'image': 0, 'video': 0},
    )
    tokenizer = llm.get_tokenizer()
    enable_thinking = os.environ.get('ENABLE_THINKING', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
    max_tokens = int(os.environ.get('MAX_TOKENS', '1024'))
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.05)
    print(f"Sampling: temperature=0.0 max_tokens={max_tokens} enable_thinking={enable_thinking}", flush=True)

    results = []
    for entry in PROMPT_VARIANTS:
        if len(entry) == 4:
            name, system, build_fn, shots = entry
        else:
            name, system, build_fn = entry
            shots = None
        print(f"\n=== {name} ===", flush=True)
        prompts = []
        for it in items:
            msgs = build_messages(system, build_fn, it['query'], it['gen_texts'], shots)
            prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking))
        outputs = llm.generate(prompts, sampling)

        rouges, ious = [], []
        n_explicit = 0
        n_think_stripped = 0
        n_think_missing = 0
        ref_counts = []
        for it, out in zip(items, outputs):
            raw_full = out.outputs[0].text.strip()
            has_think_marker = raw_full.rfind('</think>') != -1
            raw = final_response_text(raw_full, require_think_marker=enable_thinking)
            if has_think_marker:
                n_think_stripped += 1
            elif enable_thinking:
                n_think_missing += 1
            ans = split_answer_citation(raw)
            cited_idx = parse_citation(raw, len(it['gen_pids'])) if raw else []
            pred_refs = [it['gen_pids'][j] for j in cited_idx if j < len(it['gen_pids'])]
            if re.search(r'\[อ้างอิง[:\s]+[0-9,\s]+\]', raw):
                n_explicit += 1
            ref_counts.append(len(pred_refs))
            rouges.append(rouge_l(ans, it['gold_abs']))
            ious.append(iou(pred_refs, it['gold_refs']))

        r_avg = sum(rouges)/len(rouges)
        i_avg = sum(ious)/len(ious)
        # composite proxy (no SS): 0.55×RougeL + 0.45×IoU (rebalanced w/o SS)
        # For ranking, also report what composite WOULD be assuming SS≈0.85 (typical)
        proxy = 0.55 * r_avg + 0.45 * i_avg
        full_proxy = 0.45 * 0.85 + 0.35 * r_avg + 0.20 * i_avg  # assume SS=0.85
        print(f"RougeL={r_avg:.4f}  IoU={i_avg:.4f}  citations={n_explicit}/{len(items)}  avg_refs={sum(ref_counts)/len(ref_counts):.2f}  think_stripped={n_think_stripped}/{len(items)}  think_missing={n_think_missing}/{len(items)}", flush=True)
        print(f"proxy (no SS) = {proxy:.4f}  proxy(SS=0.85) = {full_proxy:.4f}", flush=True)
        results.append({'name': name, 'rougeL': r_avg, 'IoU': i_avg, 'cites': n_explicit, 'avg_refs': sum(ref_counts)/len(ref_counts), 'proxy': proxy, 'think_stripped': n_think_stripped, 'think_missing': n_think_missing})

    # final summary table
    rank_by = os.environ.get('RANK_BY', 'proxy').strip()
    rank_key = {'iou': 'IoU', 'rougel': 'rougeL', 'proxy': 'proxy'}.get(rank_by.lower(), 'proxy')
    print("\n\n=== SUMMARY ===", flush=True)
    print(f"Ranked by: {rank_key}", flush=True)
    print(f"{'variant':<28} {'RougeL':>8} {'IoU':>8} {'cites':>8} {'refs':>6} {'proxy':>8}", flush=True)
    for r in sorted(results, key=lambda x: -x[rank_key]):
        print(f"{r['name']:<28} {r['rougeL']:>8.4f} {r['IoU']:>8.4f} {r['cites']:>6}/{len(items)} {r['avg_refs']:>6.2f} {r['proxy']:>8.4f}", flush=True)
    if results:
        best_iou = max(results, key=lambda x: x['IoU'])
        print(f"\nBest IoU: {best_iou['name']} = {best_iou['IoU']:.4f} (avg_refs={best_iou['avg_refs']:.2f})", flush=True)

    with open(RESULT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
