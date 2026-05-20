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

All models are pre-downloaded to `SHARED/.hf_cache/` (bge-m3, bge-reranker-v2-m3, Qwen2.5-7B). Always set offline flags on compute nodes:
```bash
export HF_HOME=/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

SLURM logs go to `PROJECT/logs/`.

## Key commands

**Submit inference jobs** (SLURM, run from PROJECT root):
```bash
sbatch textsum/submit_lanta.sh        # main pipeline: inference on test set → textsum/result/
sbatch textsum/submit_eval_train.sh   # main pipeline: inference + score on train set
```
Experiments live in `expNN/` directories with the same submit-script layout; see `IDEAS.md` for the roadmap.

**Evaluate a submission locally** (requires GPU for bge-m3 SS-score):
```bash
python3 textsum/eval_train/score.py textsum/eval_train/result/submission.csv
```

**Fast retrieval-only eval** (no LLM; run `sbatch eval_retrieval/submit_embed.sh` once first):
```bash
python3 eval_retrieval/eval.py
```

**Build and push Docker image** (via GitHub Actions — manual trigger `workflow_dispatch`):
```
.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v6
```

**Test a container image locally** (debug `Exit StatusCode 1` failures from benchmark):
```bash
# pull image to local SIF (interactive prompt for registry password)
module load Apptainer/1.1.6
APPTAINER_TMPDIR=$PROJECT/apptainer_tmp APPTAINER_CACHEDIR=$PROJECT/apptainer_tmp/cache \
  apptainer pull --docker-login $PROJECT/textsum_v6.sif \
  docker://registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v6

# run via SLURM with GPU; --containall blocks default mounts so env matches Docker
apptainer exec --nv --containall \
  --bind $PROJECT/textsum/model/test:/model/test \
  --bind /tmp/result:/result \
  $PROJECT/textsum_v6.sif python3 /model/run.py
```
Both `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` MUST point at Lustre — `/tmp` on the login node is too small to unpack the image (~15 GB).

## Architecture

### Pipeline (two phases)

1. **Retrieval**: For each query, build a candidate pool from the `doc_id` document (dense bge-m3 + BM25, top-20 each), rerank it with a cross-encoder (bge-reranker-v2-m3), and keep the top-K paragraph(s).
2. **Generation**: Pass the retrieved paragraph(s) as context to an LLM, which produces a Thai-language abstractive summary. The LANTA experiments use vLLM (Qwen3-32B-AWQ); the Docker pipeline at `textsum/model/run.py` uses `transformers.pipeline` (see "Docker container issues" below).

### Experiments

LANTA experiment history (train-set composite, ↑ better):

| Experiment | Change vs previous | Composite |
|------------|--------------------|-----------|
| baseline (textsum) | dense-only retrieval + Qwen2.5-7B | 0.5584 |
| `exp01/` — E1 rerank | + BM25 candidate pool + cross-encoder rerank | 0.6148 |
| `exp02/` — E5 self-cite | feed LLM top-5, self-citation drives `refs` | 0.5833 *(failed)* |
| `exp03/` — E9 model swap | exp01 retrieval + Qwen3-32B-AWQ (no-think) | **0.6256** ⭐ |
| `exp04/` — E9 model swap | exp03 retrieval + Typhoon2.1-Gemma3-12B | 0.6248 |
| `exp05/` — E10 prompt | exp03 + direct-QA prompt rewrite | 0.6216 *(failed)* |

**Current best**: `exp03/` (Qwen3-32B-AWQ + rerank). The Docker submission pipeline (`textsum/model/run.py`) is NOT exp03 — it still runs the exp01 setup (Qwen2.5-7B + rerank) because the Docker image is currently broken; see "Docker container issues".

See `IDEAS.md` for the full experiment roadmap and `eval_retrieval/` for the fast retrieval harness.

**Score breakdown (train set):**

|          | baseline | exp01    | exp03 (best) |
|----------|----------|----------|--------------|
| RougeL   | 0.3387   | 0.3723   | 0.3928       |
| SS-score | 0.7667   | 0.8016   | 0.8096       |
| IoU      | 0.4744   | 0.6190   | 0.6190       |
| **Composite** | **0.5584** | **0.6148** | **0.6256** |

**Key design decisions:**
- Two-stage retrieval: dense (bge-m3) + BM25 build a candidate pool (top-20 each), a cross-encoder (bge-reranker-v2-m3) reranks it.
- `TOP_K=1`: retrieve, generate from, and report exactly 1 paragraph (honest IoU reporting).
- bge-m3 embeds on CPU in LANTA jobs (so CUDA isn't initialized in the parent process before vLLM spawns workers). Cross-encoder runs on GPU and is freed (`del` + `empty_cache`) before the LLM loads.
- LANTA SLURM scripts set `VLLM_WORKER_MULTIPROC_METHOD=spawn` — required when CrossEncoder touches CUDA before vLLM init, otherwise the forked worker crashes.

### What does NOT help (verified, don't retry without new evidence)

- **Switching the reranker** to Qwen3-Reranker-0.6B / jina-reranker-v3 / Qwen3-Reranker-4B. On MIRACL Thai nDCG@10 our current `bge-reranker-v2-m3` (82.29) already beats them all at 0.6B and matches Qwen3-Reranker-4B (82.00). Reranker swap is a dead end for Thai.
- **Expanding the candidate pool** (`POOL_N=20 → 40`). hit@1 dropped 0.7401 → 0.7393; the rerank misranks the extra candidates and adds noise.
- **Rewriting the system prompt** to "direct QA, you may rephrase" (exp05). Composite dropped −0.0040; Qwen3-32B-AWQ already interprets the original "summarize concisely" prompt correctly, and granting explicit rephrase permission pushed outputs further from gold.
- **E5 self-citation with Qwen2.5-7B** (exp02). The model didn't follow the citation format reliably and IoU collapsed 0.6190 → 0.4870.

### Failure analysis (exp03 train set, 322 queries with IoU=0)

| Sub-pattern | Share of failures |
|-------------|-------------------|
| Gold IS in the top-20 pool but rerank picked something else | **83.5%** |
| Gold beyond rank 20 in rerank (in pool but pushed down) | 7.2% |
| Gold not in the pool at all | 9.9% |

→ The **rerank quality**, not pool size or retrieval recall, is the dominant bottleneck. Queries with digits in the question (meeting numbers, agenda numbers, dates) fail at 35.7% vs 20.6% on hits — bge-reranker-v2-m3 is weak at lexical/numeric disambiguation.

### Docker container issues

`textsum/model/run.py` runs end-to-end in LANTA SLURM but the corresponding Docker images fail in the competition benchmark backend with `Exit StatusCode 1` (no logs returned). Status:

| Tag | Changes vs previous working version | Benchmark result |
|-----|-------------------------------------|------------------|
| v3 (in parent repo, predates this dir) | dense-only, `transformers.pipeline` LLM | ✅ ran |
| v4 | + rerank pipeline, switched LLM to vLLM | ❌ exit 1 |
| v5 | v4 + `VLLM_WORKER_MULTIPROC_METHOD=spawn` env | ❌ exit 1 |
| v6 | reverted LLM back to `transformers.pipeline`; rerank pipeline kept | ❌ exit 1 |

v6 reproduces the v3 LLM pattern but still fails, so the regression is in something the rerank pipeline adds: `rank_bm25`, `pythainlp`, `CrossEncoder` + `bge-reranker-v2-m3`, or the larger image footprint. Debugging is now via Apptainer locally (see "Test a container image locally" above) so we can see the actual error instead of blindly bumping tags.

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
| `VLLM_WORKER_MULTIPROC_METHOD` | (unset) | `spawn` — required when CrossEncoder touches CUDA before vLLM |

### Progress reporting

`textsum/benchmark_lib/progress` is a binary that the competition benchmark calls to track inference progress. It must be called once per query processed, and once more at the end with the total count.
