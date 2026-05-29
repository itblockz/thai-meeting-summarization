import csv, os
PROJECT = '/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047'
def load(p):
    rows = {}
    with open(p, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows[r['ID']] = r
    return rows
e38 = load(f'{PROJECT}/exp38/eval_result/submission.csv')
e51 = load(f'{PROJECT}/exp51/eval_result/submission.csv')
print(f'exp38 rows: {len(e38)}, exp51 rows: {len(e51)}')
ids = sorted(set(e38) & set(e51))
print(f'common IDs: {len(ids)}')
out = f'{PROJECT}/hybrid_38ans_51ref/eval_result/submission.csv'
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['ID','abstractive','refs'])
    w.writeheader()
    for i in ids:
        w.writerow({'ID': i, 'abstractive': e38[i]['abstractive'], 'refs': e51[i]['refs']})
print(f'wrote {out}')
