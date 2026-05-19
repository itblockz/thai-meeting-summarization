# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an NLP competition submission for Thai-language document summarization (`2026-textsum`). Given a JSON dataset of Thai parliamentary meeting documents and queries, the system retrieves relevant paragraphs and generates abstractive summaries. Output is a `submission.csv` with columns `ID`, `abstractive`, `refs`.

The score is a weighted composite: `0.45 × SS-score + 0.35 × RougeL + 0.20 × IoU`.

## Environment

This runs on LANTA HPC (SLURM + Lustre).

```
PROJECT = /lustrefs/disk/project/zz991000-zdeva/zz991021/ua047   ← this repo
SHARED  = /lustrefs/disk/project/zz991000-zdeva/zz991021          ← venv, .hf_cache (shared with ua048)
```

Activate the shared venv before running anything:
```bash
module load cray-python/3.11.7
source /lustrefs/disk/project/zz991000-zdeva/zz991021/venv/bin/activate
```

All models are pre-downloaded to `SHARED/.hf_cache/`. Always set offline flags on compute nodes:
```bash
export HF_HOME=/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

SLURM logs go to `PROJECT/logs/`.

## Key commands

**Submit inference jobs** (SLURM, run from PROJECT root):
```bash
sbatch textsum/submit_lanta.sh        # baseline: Qwen2.5-7B + dense retrieval → textsum/result/
sbatch textsum/submit_eval_train.sh   # baseline inference + score on train set
sbatch exp02/submit_lanta.sh          # exp02: Qwen3-32B-AWQ + BM25/bge-m3 RRF → exp02/result/
sbatch exp02/submit_eval_train.sh     # exp02 inference + score on train set
```

**Evaluate a submission locally** (requires GPU for bge-m3 SS-score):
```bash
python3 textsum/eval_train/score.py textsum/eval_train/result/submission.csv
```

**Build and push Docker image** (via GitHub Actions — manual trigger `workflow_dispatch`):
```
.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v3
```

## Architecture

### Pipeline (two phases)

1. **Retrieval**: For each query, find the top-K relevant paragraphs from the document referenced by `doc_id`.
2. **Generation**: Pass retrieved paragraphs as context to an LLM; generate a Thai-language abstractive summary.

### Experiments

| Path | Retrieval | LLM | Train score | Notes |
|------|-----------|-----|-------------|-------|
| `textsum/model/run.py` | Dense-only (bge-m3), TOP_K=1 | Qwen2.5-7B + vLLM | 0.5584 | Honest TOP_K=1; embed CPU |
| `exp02/run.py` | BM25 + bge-m3 → RRF, TOP_K=1 | Qwen3-32B-AWQ + vLLM | 0.5487 | Extractive prompt; honest TOP_K=1 |

**Score breakdown (train set):**

| | textsum (TOP_K=1) | exp02 (TOP_K=1) |
|--|--|--|
| RougeL | 0.3387 | 0.3336 |
| SS-score | 0.7667 | 0.7623 |
| IoU | 0.4744 | 0.4445 |
| **Composite** | **0.5584** | **0.5487** |

**Key design decisions:**
- `TOP_K=1`: retrieve, generate from, and report exactly 1 paragraph (honest IoU reporting)
- Extractive prompt in exp02: removes "สรุป" to prevent paraphrasing; instructs model to copy verbatim
- Embedding on CPU before loading vLLM (avoids CUDA fork/OOM issues)

### Data format

`textsum/eval_train/test.json` (symlink → train_set.json) and `textsum/model/test/test.json` share this schema:
```json
{
  "docs":    [{ "doc_id": "...", "paragraphs": [{ "para_id": "...", "text": "..." }] }],
  "queries": [{ "ID": "...", "doc_id": "...", "query": "..." }]
}
```
Train set additionally has `"abstractive"` and `"refs"` fields on each query (ground truth).

### Evaluation metrics

`textsum/eval_train/score.py` implements:
- **RougeL**: Thai word-tokenized with `pythainlp` (`newmm` engine), then space-split for ROUGE
- **SS-score**: Cosine similarity of bge-m3 embeddings between prediction and reference
- **IoU**: Jaccard overlap between predicted `refs` para IDs and ground-truth `refs`

### Environment variables

| Variable | Default (Docker) | LANTA override |
|----------|-----------------|----------------|
| `TEST_DIR` | `/model/test` | `$PROJECT/textsum/model/test` |
| `RESULT_DIR` | `/result` | `$PROJECT/textsum/result` |
| `PROGRESS_LIB` | `/benchmark_lib/progress` | `$PROJECT/textsum/benchmark_lib/progress` |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | set to override |

### Progress reporting

`textsum/benchmark_lib/progress` is a binary that the competition benchmark calls to track inference progress. It must be called once per query processed, and once more at the end with the total count.
