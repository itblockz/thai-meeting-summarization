# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an NLP competition submission for Thai-language document summarization (`2026-textsum`). Given a JSON dataset of Thai parliamentary meeting documents and queries, the system retrieves relevant paragraphs and generates abstractive summaries. Output is a `submission.csv` with columns `ID`, `abstractive`, `refs`.

The score is a weighted composite: `0.45 × SS-score + 0.35 × RougeL + 0.20 × IoU`.

## Environment

This runs on LANTA HPC (SLURM + Lustre). The project root is:
```
/lustrefs/disk/project/zz991000-zdeva/zz991021
```

Activate the shared venv before running anything:
```bash
module load cray-python/3.11.7
source /lustrefs/disk/project/zz991000-zdeva/zz991021/venv/bin/activate
```

All models are pre-downloaded to `.hf_cache/`. Always set offline flags when running on compute nodes:
```bash
export HF_HOME=$PROJECT/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Key commands

**Set up venv from scratch** (run on login node, needs internet):
```bash
bash textsum/setup_env.sh
```

**Download models** (needs internet):
```bash
bash textsum/download_models.sh          # bge-m3 + Qwen2.5-7B-Instruct
bash toey/exp01/download_models.sh       # bge-m3 + Qwen3-32B-AWQ
```

**Submit inference jobs** (SLURM):
```bash
sbatch textsum/submit_lanta.sh           # baseline: Qwen2.5-7B + dense retrieval
sbatch toey/exp01/submit_lanta.sh        # exp01: Qwen3-32B-AWQ + BM25/bge-m3 RRF (BROKEN: cuda/12.6 module bug)
sbatch exp02/submit_lanta.sh             # exp02: same as exp01 + extractive prompt + GEN_K=5/REF_K=1
sbatch textsum/submit_eval_train.sh      # run baseline inference + score against train set
sbatch exp02/submit_eval_train.sh        # run exp02 inference + score against train set
```

**Evaluate a submission locally** (requires GPU for bge-m3 SS-score):
```bash
python3 textsum/eval_train/score.py textsum/eval_train/result/submission.csv
```

**Build and push Docker image** (via GitHub Actions — manual trigger `workflow_dispatch`):
```
.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v2
```

**Run via Apptainer** (uses pre-built `.sif` image):
```bash
sbatch textsum/submit_apptainer.sh
```

## Architecture

### Pipeline (two phases)

1. **Retrieval**: For each query, find the top-K relevant paragraphs from the document referenced by `doc_id`.
2. **Generation**: Pass retrieved paragraphs as context to an LLM; generate a Thai-language abstractive summary.

### Experiments

| Path | Retrieval | LLM | Notes |
|------|-----------|-----|-------|
| `textsum/model/run.py` | Dense-only (bge-m3 cosine) | Qwen2.5-7B-Instruct | Baseline; score=0.530; HF `pipeline`, batch=4 |
| `toey/exp01/run.py` | BM25 + bge-m3 → RRF, TOP_K=3 | Qwen3-32B-AWQ | BROKEN: `module load cuda/12.6` causes driver mismatch |
| `exp02/run.py` | BM25 + bge-m3 → RRF, GEN_K=5/REF_K=1 | Qwen3-32B-AWQ | Extractive prompt; IoU-optimised refs; target score ~0.630 |

**exp02 key changes vs exp01:**
- `GEN_K=5`: passes 5 paragraphs to LLM (wider context for better RougeL)
- `REF_K=1`: reports only top-1 as refs (58.7% of train queries have 1 gold ref → IoU improvement)
- Extractive system prompt: removes "สรุป" which triggers paraphrasing; instructs model to copy words directly from source
- CUDA fix: no `module load cuda/12.6` in SLURM script

### Data format

`dataset/train_set.json` and `model/test/test.json` share this schema:
```json
{
  "docs":    [{ "doc_id": "...", "paragraphs": [{ "para_id": "...", "text": "..." }] }],
  "queries": [{ "ID": "...", "doc_id": "...", "query": "..." }]
}
```
Train set additionally has `"abstractive"` and `"refs"` fields on each query (ground truth).

### Evaluation metrics

`textsum/eval_train/score.py` and `evaluate_sample/eval.py` implement:
- **RougeL**: Thai word-tokenized with `pythainlp` (`newmm` engine), then space-split for ROUGE
- **SS-score**: Cosine similarity of bge-m3 embeddings between prediction and reference
- **IoU**: Jaccard overlap between predicted `refs` para IDs and ground-truth `refs`

The `textsum/eval_train/test.json` is a symlink to `train_set.json` so the same inference script works unchanged on the training set.

### Environment variables (configure paths per run context)

| Variable | Default (Docker) | LANTA override |
|----------|-----------------|----------------|
| `TEST_DIR` | `/model/test` | `$PROJECT/textsum/model/test` |
| `RESULT_DIR` | `/result` | `$PROJECT/textsum/result` |
| `PROGRESS_LIB` | `/benchmark_lib/progress` | `$PROJECT/textsum/benchmark_lib/progress` |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | set to override |

### Progress reporting

`textsum/benchmark_lib/progress` is a binary that the competition benchmark calls to track inference progress. It must be called once per query processed, and once more at the end with the total count.
