"""Build hybrid submission: exp38 abstractive + exp50 refs."""
import csv
from pathlib import Path

PROJECT = Path('/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047')
EXP38_CSV = PROJECT / 'exp38/eval_result/submission.csv'
EXP50_CSV = PROJECT / 'exp50/eval_result/submission.csv'
OUT_DIR   = PROJECT / 'exp_hybrid_38ans_50ref'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV   = OUT_DIR / 'submission.csv'

def load(path):
    rows = {}
    with open(path, encoding='utf-8', newline='') as f:
        for r in csv.DictReader(f):
            rows[r['ID']] = r
    return rows

a = load(EXP38_CSV)
r = load(EXP50_CSV)
assert set(a) == set(r), f"ID mismatch: {set(a) ^ set(r)}"
print(f'loaded {len(a)} queries from each')

with open(OUT_CSV, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ID', 'abstractive', 'refs'])
    w.writeheader()
    for qid in sorted(a):
        w.writerow({
            'ID': qid,
            'abstractive': a[qid]['abstractive'],   # FROM EXP38
            'refs': r[qid]['refs'],                  # FROM EXP50
        })
print(f'wrote {OUT_CSV}')
