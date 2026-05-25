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
.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v11
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

1. **Retrieval**: For each query, build a candidate pool from the `doc_id` document (dense bge-m3 + BM25, top-20 each), rerank it with a cross-encoder (bge-reranker-v2-m3). The current best (`exp30/`) keeps the **full reranked union pool** (mean ~28 paragraphs); earlier configs truncated to `GEN_K` (5 or 20).
2. **Generation**: Pass the reranked paragraphs as a numbered `[1..K]` context to vLLM running Qwen3-32B-AWQ; it answers in Thai and cites which paragraphs it used as `[อ้างอิง: X]` (E5 self-citation — the cited paragraphs become `refs`). The Docker pipeline (`textsum/model/run.py`) still matches **exp22** (GEN_K=5) — exp03 retrieval + E5 self-citation + 2-shot multi-turn-chat few-shot — at image tag **v11**. exp30 (NEW BEST) is not yet ported to the container. Both pass `enforce_eager=True` to skip the V1-engine torch.compile path, which crashes silently inside Apptainer/Docker containers on vllm 0.9.2 (see "Docker container issues").

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
| `exp22/` — E5 + few-shot (**v11 prod**) | exp03 + E5 self-cite (GEN_K=5) + exp08 2-shot | 0.6647 † |
| `exp23/` — E1' reranker swap | exp22 + Qwen3-Reranker-8B (in place of bge) | 0.6663 † *(+0.0016)* |
| `exp25/` — GEN_K sweep | exp22 + GEN_K=10 | 0.6717 † |
| `exp26/` — GEN_K sweep | exp22 + GEN_K=15 | 0.6734 † |
| `exp27/` — GEN_K sweep | exp22 + GEN_K=20 | 0.6777 † |
| `exp28/` — fusion ablation | exp27 minus cross-encoder, RRF order only | 0.6743 † |
| `exp29/` — full pool RRF | exp28 + no GEN_K cap (full pool, RRF order) | 0.6762 † |
| `exp30/` — full pool rerank (**BEST**) | exp27 + no GEN_K cap (full pool, rerank order) | **0.6783** † ⭐ |
| `exp32/` — E3 HyDE | exp30 + HyDE-blended dense (α=0.5) | 0.6779 † *(−0.0004)* |

† Held-out evaluation: exp06 scored on 1218 queries excluding doc_050, exp07 on 1211 excluding doc_047, exp08 on the same 1218 as exp06. Apples-to-apples exp03 baselines on those subsets are 0.6270 (doc_050) and 0.6237 (doc_047), so the few-shot deltas are **+0.0059 (exp06)**, **+0.0041 (exp07)** and **+0.0091 (exp08)**. All show statistically significant per-query RougeL improvement (paired t-test p<0.0002), confirming the few-shot signal is real and not a doc-choice artifact. IoU is identical to exp03 because the retrieval pipeline is unchanged.

**E7 sweep (exp09–exp16)**: a #-shots ablation (exp09/exp11 = 1/4-shot, exp10 = 3-shot) confirmed 2-shot is the inverted-U peak; exp12–exp13 reproduced the gain on a different held-out doc (doc_002); dynamic per-query k-NN few-shot (`exp14/`) gives the most *reproducible* gain, **+0.0052** on the full 1239 — exp08's +0.0091 is a high outlier. Criteria-based example selection (exp15–exp16) underperformed and is a dead end (see "What does NOT help"). exp08's hand-picked pair is the chosen production few-shot.

**E5 sweep (exp17–exp22)**: revisited E5 self-citation — feed the LLM the top-`GEN_K=5` reranked paragraphs, it cites which it used → adaptive `refs` — the idea `exp02/` abandoned. exp02 failed only because Qwen2.5-7B could not follow the citation format; Qwen3-32B-AWQ follows it 99%+ of the time. Leak-free results (1218 queries, doc_050 held out): E5 alone is roughly break-even with exp03 (exp17/exp18 ≈ 0.625/0.630), but E5 + 2-shot few-shot lifts it sharply. exp19→exp21 is a single-variable ladder (few-shot **+0.0076**, system-prompt wording **+0.0025**, removing a stray per-paragraph length cap **+0.0169** — the cap, not over-citation, was the dominant IoU drag). `exp22/` swaps in exp08's few-shot pair (both single-ref, which keeps the model's avg cited refs near the 71.8%-single-ref dataset prior) — the **production pipeline at image v11**.

**Reranker swap (exp23, leak-free)**: replacing bge-reranker-v2-m3 with **Qwen3-Reranker-8B** lifts retrieval metrics dramatically (hit@1 0.7401→0.7579 +0.018, hit@5 0.9112→0.9314 +0.020, MRR 0.8155→0.8345 +0.019, iou@1 0.6190→0.6344 +0.015) but the composite barely moves: **0.6647 → 0.6663 (+0.0016)**. At GEN_K=5 both rerankers put gold in top-5 ≥91% of the time; the reordering within top-5 doesn't change what E5 self-cite picks. IoU actually drops −0.0062 — Qwen3-8B's rank distribution differs enough that the LLM cites a slightly different set. Not the headline gain the retrieval numbers suggested. *(Whether the gain transfers to exp30's full-pool setup is open — see exp33 below.)*

**GEN_K sweep (exp25–exp27, leak-free)**: lift GEN_K from 5 → 10 → 15 → 20 on top of exp22; each step monotonically improves IoU, RougeL, SS:
| GEN_K | RougeL | SS | IoU | composite |
|-------|--------|------|------|-----------|
| 5 (exp22) | 0.4454 | 0.8384 | 0.6575 | 0.6647 |
| 10 (exp25) | 0.4516 | 0.8445 | 0.6680 | 0.6717 |
| 15 (exp26) | 0.4530 | 0.8453 | 0.6725 | 0.6734 |
| 20 (exp27) | 0.4585 | 0.8460 | 0.6824 | **0.6777** |
The avg dense∪BM25 union is ~28.6 — GEN_K=20 already captures the bulk; further expansion has diminishing returns. The earlier exp02 verdict that "feeding more context hurts" was a Qwen2.5-7B artefact; Qwen3-32B-AWQ exploits the extra context cleanly.

**Order-vs-selection ablation (exp28–exp29, leak-free)**: at GEN_K=20, drop the cross-encoder and order by RRF instead — exp28 = 0.6743 (loses to exp27 by −0.0034). At full pool (no GEN_K cap) with RRF order — exp29 = 0.6762 (still loses to rerank). The cross-encoder's value at K≥20 is ordering quality (RougeL/SS lift from gold at rank 1), not selection — selection at K≥20 is solved by union recall.

**Full-pool + rerank order — exp30 (NEW BEST, leak-free)**: feed the full dense∪BM25 union (mean 28.57 paragraphs) ordered by bge-reranker score, no GEN_K cap. Combines exp27's order quality with exp29's pool inclusion → **0.6783** (+0.0006 over exp27, +0.0136 over exp22 production). RougeL 0.4584 / SS 0.8467 / IoU 0.6844. This is the current repo best.

**HyDE (exp31 retrieval-only, exp32 LLM, leak-free)**: Qwen3-32B-AWQ writes a hypothetical paragraph per query (preserving numeric/named entities), embed with bge-m3, blend with original query embedding at α=0.5 before dense retrieval. Retrieval-only: hit@1 +0.035, hit@20 +0.006 (`eval_retrieval/hyde_eval.py`). But exp32 = exp30 + HyDE blend gives **0.6779 — basically flat (−0.0004)**. Rerank uses the original query, so HyDE's dense-ranking lift is mostly washed out; pool composition barely changes (BM25 catches what HyDE adds). E3 closed.

**Current best**: `exp30/` — exp22 retrieval pool + bge-reranker order on the full union + E5 self-cite + exp08 few-shot — at **0.6783** leak-free, **+0.0136** over the v11 container (exp22 = 0.6647 leak-free). `exp03/` remains the canonical no-few-shot reference (0.6256 full-1239). The Docker submission pipeline (`textsum/model/run.py`) still matches **exp22** as of image tag **v11**; porting exp30 to the container needs `GEN_K` removed and the per-query LLM context to grow from ~1.5K to ~5K tokens (already inside `max_model_len=16384`). Intentional container settings: `enforce_eager=True` (see "Docker container issues") and `max_model_len=16384` (the few-shot E5 prompt needs it). The worked examples are the `_SHOT1_*`/`_SHOT2_*` constants — exp08's hand-picked pair rendered in E5 form.

See `IDEAS.md` for the full experiment roadmap and `eval_retrieval/` for the fast retrieval harness.

**Score breakdown:**

|          | baseline | exp01  | exp03  | exp22 (prod) | exp27  | **exp30 (best)** |
|----------|----------|--------|--------|--------------|--------|------------------|
| RougeL   | 0.3387   | 0.3723 | 0.3928 | 0.4454       | 0.4585 | **0.4584**       |
| SS-score | 0.7667   | 0.8016 | 0.8096 | 0.8384       | 0.8460 | **0.8467**       |
| IoU      | 0.4744   | 0.6190 | 0.6190 | 0.6575       | 0.6824 | **0.6844**       |
| **Composite** | **0.5584** | **0.6148** | **0.6256** | **0.6647** | **0.6777** | **0.6783** |

baseline/exp01/exp03 are full-1239; exp22/exp27/exp30 are leak-free (1218, doc_050 held out — the few-shot examples come from it). exp03 on that same 1218 subset = 0.6270, so exp30 is **+0.0513** over exp03 and **+0.0136** over the v11 container (exp22).

**Key design decisions:**
- Two-stage retrieval: dense (bge-m3) + BM25 build a candidate pool (top-20 each), a cross-encoder (bge-reranker-v2-m3) reranks it.
- **exp30 (current best)** feeds the LLM the *full reranked union pool* (mean 28.57 paras) — no GEN_K cap; E5 self-citation reports the subset actually cited as `refs`. The GEN_K sweep (exp22→exp25→exp26→exp27→exp30 = 5→10→15→20→full) showed monotonic gain; the v11 container still uses **GEN_K=5** (exp22).
- `GEN_K=5` (production v11) was the elbow of the rerank recall curve — hit@5 = 0.912 vs hit@1 = 0.739; the gain past K=5 only materialised once we switched to Qwen3-32B-AWQ (the 7B model in exp02 couldn't exploit the extra context).
- `exp03/` and the `exp04`–`exp16` line instead use `TOP_K=1` — retrieve, generate from, and report exactly one paragraph.
- bge-m3 and the cross-encoder both run on GPU; the cross-encoder is freed (`del` + `empty_cache`) before the LLM loads. (`exp03/` and earlier `expNN` embed bge-m3 on CPU — a pre-v9 convention to keep CUDA uninitialised until vLLM spawns its workers.)
- LANTA SLURM scripts set `VLLM_WORKER_MULTIPROC_METHOD=spawn` — required when CrossEncoder touches CUDA before vLLM init, otherwise the forked worker crashes.

### What does NOT help (verified, don't retry without new evidence)

- **Switching the reranker** to Qwen3-Reranker-0.6B / jina-reranker-v3 / Qwen3-Reranker-4B. On MIRACL Thai nDCG@10 our current `bge-reranker-v2-m3` (82.29) already beats them all at 0.6B and matches Qwen3-Reranker-4B (82.00). Reranker swap is a dead end for Thai at those sizes. **Qwen3-Reranker-8B is a separate story**: it lifts every retrieval metric (hit@1 +0.018, iou@1 +0.015, MRR +0.019) but at GEN_K=5 only adds +0.0016 composite (exp23) because both rerankers already put gold in top-5 ≥91% of the time. Whether the gain transfers to exp30's full-pool setup is the only open question on the reranker axis.
- **Expanding the candidate pool** (`POOL_N=20 → 40`). hit@1 dropped 0.7401 → 0.7393; the rerank misranks the extra candidates and adds noise. *Pool-recall sweep on `eval_retrieval/pool_recall_eval.py` confirms only **pool size** moves the needle on pool inclusion (`union_40` +0.0105 pool_recall, +0.0413 ref_recall vs `union_20`) — **fusion strategy** at matched pool size (RRF k=10/30/60/100, weighted score fusion α=0–1) is dead end (all within ±0.005 of `union_20`, mostly worse). E4 fusion-tuning closed.*
- **Rewriting the system prompt** to "direct QA, you may rephrase" (exp05). Composite dropped −0.0040; Qwen3-32B-AWQ already interprets the original "summarize concisely" prompt correctly, and granting explicit rephrase permission pushed outputs further from gold.
- **E5 self-citation fed to a small model** (exp02, Qwen2.5-7B). The 7B model didn't follow the citation format reliably and IoU collapsed 0.6190 → 0.4870. This is model-specific and **not** a verdict on E5 itself — with Qwen3-32B-AWQ the format is followed 99%+ of the time and E5 + few-shot (`exp22/`) is the current best (see "E5 sweep"). The lesson: don't feed E5 self-citation to a <10B model.
- **E6 adaptive K from rerank score gap**. bge-reranker scores don't correlate with correctness; three strategies (abs gap, threshold, top1 ratio) all lose to K=1, and oracle K=|gold| only +0.0045 composite — not worth chasing.
- **E8 length calibration / truncation**. Pred is ~1.18x gold at the median, and corr(RougeL, |pred − gold|) = −0.414, but post-hoc truncation of exp03 predictions LOSES RougeL at every cap, including the oracle cap = |gold tokens| (−0.0025). The verbose preds carry recall-matching content; truncating drops recall faster than precision rises. Length is correlated with low RougeL but not causal — likely confounded by query difficulty.
- **E2 bge-m3 ColBERT (multi-vector) reranking**. Tested 3 fusion strategies on exp03's pool: pure ColBERT (hit@1 −0.127 vs CE baseline), CE+ColBERT z-score fusion (peak at α=0.9 only +0.0036 iou@1, well within 1σ noise of 0.014), and pool expansion with ColBERT top-20 (+0.0033 pool recall, negligible). bge-m3 ColBERT and bge-reranker-v2-m3 share too much of the underlying signal to be complementary on Thai. Not worth a full LLM run.
- **E3 HyDE / query expansion** (exp31 retrieval-only, exp32 LLM). Qwen3-32B-AWQ writes a hypothetical paragraph per query, embed with bge-m3, blend at α=0.5 with the original query before dense retrieval. Retrieval-only: hit@1 +0.0348, hit@20 +0.0057. **exp32 = exp30 + HyDE-blend dropped to 0.6779 (−0.0004)** — the lift washes out because (a) cross-encoder rerank still uses the *raw* query so it ignores HyDE entirely, and (b) BM25 already covers what HyDE adds to the pool. Cache at `eval_retrieval/cache/hyde_train.json` is preserved for any future ablation.
- **RRF replacing the cross-encoder** at high K (exp28/exp29). At GEN_K=20 RRF (k=60) loses to bge-reranker by −0.0034; at full pool RRF loses by −0.0021. The cross-encoder's value at K≥20 is *ordering* quality (gold at rank 1 → higher LLM attention → RougeL/SS lift), not pool selection.
- **Criteria-based few-shot example selection** (exp15–exp16). Scoring candidate (query, paragraph, answer) examples by restate-pattern strength, length proximity, centrality and query-type frequency, then picking the top pair, underperforms exp08's hand-picked pair (+0.0049 / +0.0034 vs +0.0091). Maximising the "restate" proxy (answer echoes the question's opening tokens) selects mechanically templated examples that lift RougeL but leave SS-score flat. The proxy does not capture what makes a good teaching example; both exp08's pair and exp14's dynamic k-NN retrieval beat it.

### Failure analysis (exp03 train set, 322 queries with IoU=0)

| Sub-pattern | Share of failures |
|-------------|-------------------|
| Gold IS in the top-20 pool but rerank picked something else | **83.5%** |
| Gold beyond rank 20 in rerank (in pool but pushed down) | 7.2% |
| Gold not in the pool at all | 9.9% |

→ The **rerank quality**, not pool size or retrieval recall, is the dominant bottleneck. Queries with digits in the question (meeting numbers, agenda numbers, dates) fail at 35.7% vs 20.6% on hits — bge-reranker-v2-m3 is weak at lexical/numeric disambiguation.

### Docker container issues

History of `Exit StatusCode 1` in the benchmark backend (resolved at v8; v9 = perf; v10 = few-shot; v11 = E5):

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
| v11 | v10 + E5 self-citation (exp22): top-5 numbered context, `[อ้างอิง:]`-driven `refs`; `max_model_len` 8192→16384 | ✅ local Apptainer (test exit 0) |

**v11 → vllm 0.19.1: attempted, reverted.** To remove the venv (vllm 0.19.1) vs container (0.9.2) greedy-decode drift, v11's stack was bumped to match the venv — `cuda:12.8.1-cudnn` base, `torch 2.10.0+cu128`, `vllm 0.19.1`, `transformers 5.8.1`. It builds, but the vLLM EngineCore worker **segfaults** inside Apptainer right after model load (engine warmup) — the same `Engine core initialization failed. Failed core proc(s): {}` class as v4–v8. Forcing `FLASHINFER` / `TRITON_ATTN` / `FLEX_ATTENTION` backends all segfault too (job 5787048), so it is not the attention kernel; the root cause is deeper. Reverted — v11 stays on vllm 0.9.2. The venv↔container drift (~17/50 sample-test answers diverge under greedy decoding) is accepted, as for v8–v10.

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
