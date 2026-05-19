"""
รัน eval เปรียบเทียบ submission.csv (predictions) กับ ground truth จาก train_set.json
"""
import sys
import json
import csv
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

TRAIN_JSON = Path(__file__).parent / "test.json"   # symlink → train_set.json
SUBMISSION  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "result/submission.csv"
HF_HOME = None  # will use env var


class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split(" ")


def tokenize_thai(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    return " ".join(word_tokenize(text, engine="newmm", keep_whitespace=False))


def load_ground_truth(json_path):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for q in data["queries"]:
        refs = q.get("refs", [])
        rows.append({
            "ID": q["ID"],
            "abstractive": q["abstractive"],
            "refs": refs if isinstance(refs, list) else [refs],
        })
    return pd.DataFrame(rows)


def load_submission(csv_path):
    df = pd.read_csv(csv_path)

    def parse_refs(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        return [i.strip() for i in str(x).split(",")]

    df["refs"] = df["refs"].apply(parse_refs)
    return df


def calculate_iou(pred, sol):
    s, p = set(sol), set(pred)
    if not s:
        return 0.0
    return len(s & p) / len(s | p)


def run_evaluation(gt: pd.DataFrame, pred: pd.DataFrame):
    df = pd.merge(gt, pred, on="ID", suffixes=("_sol", "_pred"))
    if len(df) != len(gt):
        print(f"Warning: matched {len(df)}/{len(gt)} rows", file=sys.stderr)

    df["IoU"] = df.apply(lambda r: calculate_iou(r["refs_pred"], r["refs_sol"]), axis=1)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False,
                                      tokenizer=ThaiSpaceTokenizer())
    sol_toks  = df["abstractive_sol"].apply(tokenize_thai)
    pred_toks = df["abstractive_pred"].apply(tokenize_thai)
    df["rougeL"] = [scorer.score(g, p)["rougeL"].fmeasure
                    for g, p in zip(sol_toks, pred_toks)]

    model = SentenceTransformer("BAAI/bge-m3")
    texts = df["abstractive_sol"].tolist() + df["abstractive_pred"].tolist()
    embs = model.encode(texts, batch_size=32, convert_to_tensor=True, normalize_embeddings=True)
    ref_emb  = embs[:len(texts)//2]
    pred_emb = embs[len(texts)//2:]
    df["SS-score"] = F.cosine_similarity(pred_emb, ref_emb, dim=1).cpu().numpy()

    metrics = df[["rougeL", "SS-score", "IoU"]].mean().to_dict()
    wss, wrl, wj = 0.45, 0.35, 0.2
    metrics["score"] = wss*metrics["SS-score"] + wrl*metrics["rougeL"] + wj*metrics["IoU"]
    return metrics, df


if __name__ == "__main__":
    print(f"Ground truth: {TRAIN_JSON}")
    print(f"Submission:   {SUBMISSION}")

    gt   = load_ground_truth(TRAIN_JSON)
    pred = load_submission(SUBMISSION)

    print(f"GT rows: {len(gt)}, Pred rows: {len(pred)}")

    metrics, detail_df = run_evaluation(gt, pred)

    print("\n=== Train-set evaluation ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    out = SUBMISSION.parent / "train_eval_detail.csv"
    detail_df[["ID", "rougeL", "SS-score", "IoU"]].to_csv(out, index=False)
    print(f"\nPer-query detail → {out}")

    score_out = SUBMISSION.parent / "train_eval_score.json"
    import json as _json
    score_out.write_text(_json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Summary score   → {score_out}")
