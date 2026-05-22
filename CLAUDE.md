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

All models are pre-downloaded to `SHARED/.hf_cache/` (bge-m3, bge-reranker-v2-m3, Qwen2.5-7B, Qwen3-32B-AWQ). Always set offline flags on compute nodes:
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
.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v10
```

**Test a container image locally** (verify before pushing, or debug `Exit StatusCode 1`):
```bash
# A. build from .def on the login node (no internet on compute partitions)
sbatch -t 1:00:00 -p compute --wrap "false"   # (placeholder; use the script below)
# Actual local-build flow:
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
export APPTAINER_TMPDIR=$PROJECT/apptainer_tmp APPTAINER_CACHEDIR=$PROJECT/apptainer_tmp/cache
module load Apptainer/1.1.6
cd $PROJECT/textsum
nohup apptainer build --force $PROJECT/textsum_v8_local.sif textsum.def \
  > $PROJECT/logs/v8_build.log 2>&1 & disown
# ~30–40 min; pip downloads + HF snapshot_download (~25 GB embedded)

# B. or pull a tag already pushed by CI
apptainer pull --docker-login $PROJECT/textsum_v9.sif \
  docker://registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v9

# Run via SLURM with GPU; --containall mirrors what the benchmark backend gives:
sbatch textsum/submit_apptainer_test_v8.sh   # only test-data, benchmark_lib, result binds
```
Both `APPTAINER_TMPDIR` and `APPTAINER_CACHEDIR` MUST point at Lustre — `/tmp` on the login node is too small to unpack the image (~30 GB). Apptainer build must run on the **login node** (compute partitions have no internet to pull the base image / wheels). The `%post` section of `textsum.def` additionally exports `TMPDIR=/buildtmp` (a path inside the sandbox rootfs): Apptainer bind-mounts the host `/tmp` into `%post`, so when the login-node root filesystem is full pip's wheel-unpack hits `ENOSPC` without this.

## Architecture

### Pipeline (two phases)

1. **Retrieval**: For each query, build a candidate pool from the `doc_id` document (dense bge-m3 + BM25, top-20 each), rerank it with a cross-encoder (bge-reranker-v2-m3), and keep the top-K paragraph(s).
2. **Generation**: Pass the retrieved paragraph(s) as context to vLLM running Qwen3-32B-AWQ; it produces a Thai-language abstractive summary. The Docker pipeline (`textsum/model/run.py`) matches exp08 — exp03 retrieval + LLM plus 2-shot multi-turn-chat few-shot prompting — and additionally passes `enforce_eager=True` to skip the V1-engine torch.compile path, which crashes silently inside Apptainer/Docker containers on vllm 0.9.2 (see "Docker container issues").

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
| `exp06/` — E7 few-shot | exp03 + 2-shot from held-out doc_050 | 0.6329 † |
| `exp07/` — E7 few-shot | exp03 + 2-shot from held-out doc_047 | 0.6278 † |
| `exp08/` — E7 few-shot | exp06 + multi-turn chat-template (fixes "คำตอบ:" leak) | 0.6361 † |

† Held-out evaluation: exp06 scored on 1218 queries excluding doc_050, exp07 on 1211 excluding doc_047, exp08 on the same 1218 as exp06. Apples-to-apples exp03 baselines on those subsets are 0.6270 (doc_050) and 0.6237 (doc_047), so the few-shot deltas are **+0.0059 (exp06)**, **+0.0041 (exp07)** and **+0.0091 (exp08)**. All show statistically significant per-query RougeL improvement (paired t-test p<0.0002), confirming the few-shot signal is real and not a doc-choice artifact. IoU is identical to exp03 because the retrieval pipeline is unchanged.

**E7 sweep (exp09–exp16)**: a #-shots ablation (exp09/exp11 = 1/4-shot, exp10 = 3-shot) confirmed 2-shot is the inverted-U peak; exp12–exp13 reproduced the gain on a different held-out doc (doc_002); dynamic per-query k-NN few-shot (`exp14/`) gives the most *reproducible* gain, **+0.0052** on the full 1239 — exp08's +0.0091 is a high outlier. Criteria-based example selection (exp15–exp16) underperformed and is a dead end (see "What does NOT help"). exp08's hand-picked pair is the chosen production few-shot.

**Current best**: `exp03/` remains the canonical full-1239 baseline at 0.6256. Few-shot wins are measured on held-out subsets and are not directly comparable on the full set; exp08's **+0.0091** over exp03 is the largest, but it is a high outlier — the reproducible figure is **~+0.0052** (exp14 dynamic k-NN). The Docker submission pipeline (`textsum/model/run.py`) matches **exp08** — exp03 retrieval + Qwen3-32B-AWQ + 2-shot multi-turn few-shot — as of image tag **v10**. Intentional drift from the exp08 experiment: `enforce_eager=True` (see "Docker container issues") and `max_model_len=8192` (the few-shot prompt no longer fits 4096). The two worked examples are the `FEW_SHOT` constant in `textsum/model/run.py`.

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
- **E6 adaptive K from rerank score gap**. bge-reranker scores don't correlate with correctness; three strategies (abs gap, threshold, top1 ratio) all lose to K=1, and oracle K=|gold| only +0.0045 composite — not worth chasing.
- **E8 length calibration / truncation**. Pred is ~1.18x gold at the median, and corr(RougeL, |pred − gold|) = −0.414, but post-hoc truncation of exp03 predictions LOSES RougeL at every cap, including the oracle cap = |gold tokens| (−0.0025). The verbose preds carry recall-matching content; truncating drops recall faster than precision rises. Length is correlated with low RougeL but not causal — likely confounded by query difficulty.
- **E2 bge-m3 ColBERT (multi-vector) reranking**. Tested 3 fusion strategies on exp03's pool: pure ColBERT (hit@1 −0.127 vs CE baseline), CE+ColBERT z-score fusion (peak at α=0.9 only +0.0036 iou@1, well within 1σ noise of 0.014), and pool expansion with ColBERT top-20 (+0.0033 pool recall, negligible). bge-m3 ColBERT and bge-reranker-v2-m3 share too much of the underlying signal to be complementary on Thai. Not worth a full LLM run.
- **Criteria-based few-shot example selection** (exp15–exp16). Scoring candidate (query, paragraph, answer) examples by restate-pattern strength, length proximity, centrality and query-type frequency, then picking the top pair, underperforms exp08's hand-picked pair (+0.0049 / +0.0034 vs +0.0091). Maximising the "restate" proxy (answer echoes the question's opening tokens) selects mechanically templated examples that lift RougeL but leave SS-score flat. The proxy does not capture what makes a good teaching example; both exp08's pair and exp14's dynamic k-NN retrieval beat it.

### Failure analysis (exp03 train set, 322 queries with IoU=0)

| Sub-pattern | Share of failures |
|-------------|-------------------|
| Gold IS in the top-20 pool but rerank picked something else | **83.5%** |
| Gold beyond rank 20 in rerank (in pool but pushed down) | 7.2% |
| Gold not in the pool at all | 9.9% |

→ The **rerank quality**, not pool size or retrieval recall, is the dominant bottleneck. Queries with digits in the question (meeting numbers, agenda numbers, dates) fail at 35.7% vs 20.6% on hits — bge-reranker-v2-m3 is weak at lexical/numeric disambiguation.

### Docker container issues

History of `Exit StatusCode 1` in the benchmark backend (resolved at v8; v9 = perf; v10 = few-shot):

| Tag | Changes vs previous | Benchmark / local Apptainer |
|-----|---------------------|----------------------------|
| v3 (parent repo) | dense-only, `transformers.pipeline` LLM | ✅ ran |
| v4 | + rerank pipeline, switched LLM to vLLM | ❌ exit 1 |
| v5 | v4 + `VLLM_WORKER_MULTIPROC_METHOD=spawn` env | ❌ exit 1 |
| v6 | reverted LLM to `transformers.pipeline`; rerank kept | ❌ exit 1 |
| v7 | v6 + `tzdata` in requirements (pythainlp needed it) | ✅ local Apptainer |
| v8 | back to vLLM + Qwen3-32B-AWQ; five gotchas pinned | ✅ local Apptainer (4:48) |
| v9 | bge-m3 encoding moved from CPU to GPU | ✅ local Apptainer (2:57, −39%) |
| v10 | v9 + 2-shot few-shot prompting (exp08); `max_model_len` 4096→8192 | ✅ local Apptainer (test exit 0) |

**v8 = the five gotchas** (each one alone leaves the container in `Engine core initialization failed. Failed core proc(s): {}` — an empty dict because the worker dies before its first heartbeat):

1. `vllm==0.9.2` pinned in `requirements.txt`. Loose `vllm>=0.6.0` made pip backtrack through ~7 wheels (each ~500 MB) to satisfy `torch==2.6.0`, eventually timing out on PyPI.
2. `transformers==4.52.4` pinned. transformers ≥ 4.55 has `aimv2` as a built-in config; vllm 0.9.2's `OvisConfig` re-registers it and raises `ValueError: 'aimv2' is already used`.
3. `python3.11-dev` added to `apt-get install`. Triton runtime-compiles a C extension and gcc errors `fatal error: Python.h: No such file or directory` because the base image is `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` (runtime, no headers).
4. `enforce_eager=True` in the `LLM(...)` call. vllm 0.9.2's V1 engine torch.compile / Inductor path SIGKILLs the worker mid-compile inside Apptainer (Dynamo bytecode transform completes, FixFunctionalizationPass starts, then silent exit). vllm 0.19.1 on the LANTA shared venv has no such issue, but that needs torch 2.10 + cu128 — out of scope for the current base image. Eager mode costs ~0 latency on this workload (50 short prompts, ~15 s total).
5. `--timeout 300 --retries 5` on pip install. Login-node PyPI fetches are flaky enough that a default-timeout retry will fail the whole build at the very end.

Build locally via `nohup apptainer build … textsum.def` on the **login node** (compute partitions have no internet). Test with `sbatch textsum/submit_apptainer_test_v8.sh` which uses only the binds the benchmark provides (test-data, benchmark_lib, result) — no overlay tricks, so a pass confirms the SIF is self-contained.

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
| `LLM_MODEL` | `Qwen/Qwen3-32B-AWQ` | set to override |
| `VLLM_WORKER_MULTIPROC_METHOD` | (unset) | `spawn` — required when CrossEncoder touches CUDA before vLLM |

### Progress reporting

`textsum/benchmark_lib/progress` is a binary that the competition benchmark calls to track inference progress. It must be called once per query processed, and once more at the end with the total count.
