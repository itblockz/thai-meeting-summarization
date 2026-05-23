"""
Binary classifier: can we tell |gold|=1 vs |gold|>1 from features?

Tests query-text features, qw3 score-distribution features, and the LLM's
own cite count as predictors. Reports per-feature AUC, simple threshold
classifiers, and a 3-fold logistic regression on all features combined.
Then applies the best classifier to predict K∈{1,2} for qw3 top-K refs.
"""
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

P = Path("/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047")
with open(P / "textsum/eval_train/test.json", encoding="utf-8") as f:
    data = json.load(f)
queries = {q["ID"]: q for q in data["queries"]}
gold = {q["ID"]: ([q["refs"]] if isinstance(q.get("refs"), str)
                  else (q.get("refs") or [])) for q in data["queries"]}
qdoc = {q["ID"]: q["doc_id"] for q in data["queries"]}


def load_scored(p):
    raw = json.loads(Path(p).read_text())
    return {q: sorted(s, key=lambda x: -x[1]) for q, s in raw.items()}


def load_csv(p):
    out = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["ID"]] = [x.strip() for x in r["refs"].split(",") if x.strip()]
    return out


qw3 = load_scored(P / "eval_retrieval/cache/rerank_qwen3_train.json")
bge = load_scored(P / "eval_retrieval/cache/rerank_train.json")
e23 = load_csv(P / "exp23/eval_result/submission.csv")
e22 = load_csv(P / "exp22/eval_result/submission.csv")

leakfree = [q for q in gold if qdoc[q] != "doc_050"]
n = len(leakfree)

y = np.array([1 if len(gold[q]) > 1 else 0 for q in leakfree])
print(f"class balance: |gold|=1 = {(y==0).sum()} ({(y==0).mean():.1%})  "
      f"|gold|>1 = {(y==1).sum()} ({(y==1).mean():.1%})")
print(f"baseline accuracy (always predict |gold|=1): {(y==0).mean():.4f}\n")


def feats(q):
    qtext = queries[q]["query"]
    qw3_s = qw3.get(q, [])[:5]
    bge_s = bge.get(q, [])[:5]
    s = np.array([x[1] for x in qw3_s]) if qw3_s else np.array([0.0])
    bs = np.array([x[1] for x in bge_s]) if bge_s else np.array([0.0])
    return {
        "qlen":          len(qtext),
        "qw3_top1":      float(s[0]) if len(s) else 0.0,
        "qw3_gap12":     float(s[0] - s[1]) if len(s) > 1 else float(s[0]),
        "qw3_ratio12":   float(s[1] / max(s[0], 1e-9)) if len(s) > 1 else 0.0,
        "qw3_n_ge_half": int((s >= 0.5 * s[0]).sum()) if len(s) else 0,
        "qw3_n_ge_90":   int((s >= 0.9 * s[0]).sum()) if len(s) else 0,
        "qw3_mean5":     float(s.mean()) if len(s) else 0.0,
        "qw3_std5":      float(s.std()) if len(s) else 0.0,
        "bge_top1":      float(bs[0]) if len(bs) else 0.0,
        "bge_gap12":     float(bs[0] - bs[1]) if len(bs) > 1 else float(bs[0]),
        "has_LIST":      int(any(k in qtext for k in [
                            "ประกอบด้วย", "มีอะไรบ้าง", "ได้แก่"])),
        "has_SUMMARY":   int(any(k in qtext for k in [
                            "สรุป", "สาระสำคัญ", "ประเด็น"])),
        "has_WHO":       int(any(k in qtext for k in ["ใคร", "ผู้ใด"])),
        "has_WHAT":      int(any(k in qtext for k in ["อะไร"])),
        "has_HOW":       int(any(k in qtext for k in ["อย่างไร"])),
        "e22_cited":     len(e22.get(q, [])),
        "e23_cited":     len(e23.get(q, [])),
    }


feat_list = [feats(q) for q in leakfree]
feat_names = list(feat_list[0].keys())
X = np.array([[f[k] for k in feat_names] for f in feat_list], dtype=float)


def auc(x, y):
    pos = x[y == 1]; neg = x[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    all_v = np.concatenate([pos, neg])
    ranks = all_v.argsort().argsort() + 1
    U = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return U / (len(pos) * len(neg))


print("=" * 80)
print("FEATURE COMPARISON: mean per class + univariate AUC")
print("=" * 80)
print(f"  {'feature':<18}{'|gold|=1':>12}{'|gold|>1':>12}{'diff':>10}{'AUC':>8}")
for i, name in enumerate(feat_names):
    x = X[:, i]
    m1 = x[y == 0].mean(); m2 = x[y == 1].mean()
    a = auc(x, y)
    print(f"  {name:<18}{m1:>12.3f}{m2:>12.3f}{m2 - m1:>10.3f}{max(a, 1 - a):>8.3f}")


def evaluate(pred_y, name):
    tp = ((pred_y == 1) & (y == 1)).sum()
    fp = ((pred_y == 1) & (y == 0)).sum()
    fn = ((pred_y == 0) & (y == 1)).sum()
    tn = ((pred_y == 0) & (y == 0)).sum()
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  {name:<55}{acc:>9.4f}{f1:>9.4f}{prec:>11.4f}{rec:>9.4f}")
    return acc, f1


print("\n" + "=" * 80)
print("SIMPLE PREDICTORS (full leak-free set)")
print("=" * 80)
print(f"  {'predictor':<55}{'acc':>9}{'F1(>1)':>9}{'precision':>11}{'recall':>9}")

evaluate(np.zeros_like(y), "always predict |gold|=1 (baseline)")
evaluate(np.ones_like(y), "always predict |gold|>1")
evaluate((np.array([f["e22_cited"] for f in feat_list]) > 1).astype(int),
         "exp22 LLM cited >1")
evaluate((np.array([f["e23_cited"] for f in feat_list]) > 1).astype(int),
         "exp23 LLM cited >1")


def best_threshold(feat_idx, name):
    x = X[:, feat_idx]
    best_acc = 0; best_t = None; best_dir = None
    for t in np.percentile(x, np.linspace(5, 95, 19)):
        for d in [1, -1]:
            pred = ((x * d) > (t * d)).astype(int)
            acc = (pred == y).mean()
            if acc > best_acc:
                best_acc = acc; best_t = t; best_dir = d
    pred = ((x * best_dir) > (best_t * best_dir)).astype(int)
    op = ">" if best_dir == 1 else "<"
    return pred, f"{name} {op} {best_t:.3f} (best threshold)"


for name in ["qlen", "qw3_top1", "qw3_gap12", "qw3_ratio12",
             "qw3_n_ge_half", "qw3_n_ge_90", "qw3_std5"]:
    pred, label = best_threshold(feat_names.index(name), name)
    evaluate(pred, label)

has_list = X[:, feat_names.index("has_LIST")] > 0
e23_gt1  = X[:, feat_names.index("e23_cited")] > 1
qw3_n9   = X[:, feat_names.index("qw3_n_ge_90")] >= 2
evaluate((has_list | e23_gt1 | qw3_n9).astype(int),
         "has_LIST OR e23_cited>1 OR qw3_n_ge_90>=2")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def logreg_fit(Xtr, ytr, lr=0.5, iters=5000, l2=0.01):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xs = np.concatenate([(Xtr - mu) / sd, np.ones((len(Xtr), 1))], axis=1)
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        p = sigmoid(Xs @ w)
        g = Xs.T @ (p - ytr) / len(ytr) + l2 * np.r_[w[:-1], 0]
        w -= lr * g
    return w, mu, sd


def logreg_pred(Xte, w, mu, sd):
    Xs = np.concatenate([(Xte - mu) / (sd + 1e-9), np.ones((len(Xte), 1))], axis=1)
    return sigmoid(Xs @ w)


print("\n  --- LOGISTIC REGRESSION on all features (3-fold CV) ---")
np.random.seed(0)
idx = np.random.permutation(len(y))
folds = np.array_split(idx, 3)
oof_p = np.zeros(len(y))
for k in range(3):
    test_idx = folds[k]
    train_idx = np.concatenate([folds[j] for j in range(3) if j != k])
    w, mu, sd = logreg_fit(X[train_idx], y[train_idx])
    oof_p[test_idx] = logreg_pred(X[test_idx], w, mu, sd)

evaluate((oof_p > 0.5).astype(int), "logreg 3-fold CV (threshold=0.5)")
for t in [0.3, 0.4, 0.6]:
    evaluate((oof_p > t).astype(int), f"logreg 3-fold CV (threshold={t})")
print(f"\n  logreg full-feature AUC (3-fold OOF): {auc(oof_p, y):.4f}")

# Apply to IoU
print("\n" + "=" * 80)
print("APPLY: predict K then take qw3 top-K (NO oracle pick — realistic)")
print("=" * 80)


def jaccard(a, b):
    a, b = set(a), set(b); u = a | b
    return len(a & b) / len(u) if u else 0.0


def measure_iou(pred_k_for_qid):
    total = 0
    for qid in leakfree:
        k = pred_k_for_qid(qid)
        pred = set(p for p, _ in qw3.get(qid, [])[:k])
        total += jaccard(pred, gold[qid])
    return total / n


qid_to_idx = {q: i for i, q in enumerate(leakfree)}
print(f"  always K=1                                IoU = {measure_iou(lambda q: 1):.4f}")
print(f"  always K=2                                IoU = {measure_iou(lambda q: 2):.4f}")
print(f"  K=2 if exp23 cited>1 else K=1             IoU = {measure_iou(lambda q: 2 if len(e23[q]) > 1 else 1):.4f}")
print(f"  K=2 if logreg pred>0.5 else K=1           IoU = {measure_iou(lambda q: 2 if oof_p[qid_to_idx[q]] > 0.5 else 1):.4f}")
print(f"  K=2 if logreg pred>0.3 else K=1           IoU = {measure_iou(lambda q: 2 if oof_p[qid_to_idx[q]] > 0.3 else 1):.4f}")
print(f"  K=2 if |gold|>1 else K=1 (oracle)         IoU = {measure_iou(lambda q: 2 if len(gold[q]) > 1 else 1):.4f}")
