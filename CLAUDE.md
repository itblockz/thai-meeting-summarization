# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

NLP competition submission for Thai-language document summarization (`2026-textsum`). Given a JSON dataset of Thai parliamentary meeting documents and queries, the system retrieves relevant paragraphs and generates abstractive summaries. Output is `submission.csv` with columns `ID`, `abstractive`, `refs`.

Score = `0.45 × SS-score + 0.35 × RougeL + 0.20 × IoU`.

## Environment

Runs on LANTA HPC (SLURM + Lustre).

```
PROJECT = /lustrefs/disk/project/zz991000-zdeva/zz991021/ua047   ← this repo
SHARED  = /lustrefs/disk/project/zz991000-zdeva/zz991021          ← venv, .hf_cache (shared with ua048)
```

Activate the shared venv first:
```bash
module load cray-python/3.11.7
source /lustrefs/disk/project/zz991000-zdeva/zz991021/venv/bin/activate
```

Models pre-downloaded to `SHARED/.hf_cache/` (bge-m3, bge-reranker-v2-m3, Qwen2.5-7B, Qwen3-32B-AWQ, etc.). Always set offline flags on compute nodes:
```bash
export HF_HOME=/lustrefs/disk/project/zz991000-zdeva/zz991021/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```
SLURM logs go to `PROJECT/logs/`.

## Key commands

**Submit inference jobs** (from PROJECT root):
```bash
sbatch textsum/submit_lanta.sh        # inference on test set → textsum/result/
sbatch textsum/submit_eval_train.sh   # inference + score on train set
```
Experiments live in `expNN/` dirs with the same submit-script layout; see `IDEAS.md` for the roadmap.

**Evaluate a submission locally** (needs GPU for bge-m3 SS-score):
```bash
python3 textsum/eval_train/score.py textsum/eval_train/result/submission.csv
```

**Fast retrieval-only eval** (no LLM; run `sbatch eval_retrieval/submit_embed.sh` once first):
```bash
python3 eval_retrieval/eval.py
```

**Build/push Docker image** via GitHub Actions (`workflow_dispatch`):
`.github/workflows/build-push.yml → registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:<tag>`

**Build/test a container image locally** (verify before pushing, or debug `Exit StatusCode 1`):
```bash
PROJECT=/lustrefs/disk/project/zz991000-zdeva/zz991021/ua047
export APPTAINER_TMPDIR=$PROJECT/apptainer_tmp APPTAINER_CACHEDIR=$PROJECT/apptainer_tmp/cache
module load Apptainer/1.1.6
cd $PROJECT/textsum
nohup apptainer build --force $PROJECT/textsum_v8_local.sif textsum.def \
  > $PROJECT/logs/v8_build.log 2>&1 & disown          # ~30–40 min, ~25 GB embedded
# OR pull a CI-pushed tag:
apptainer pull --docker-login $PROJECT/textsum_v9.sif \
  docker://registry.ai.in.th/2026-textsum/47b13a1c/nontapat.jf0n:v9
sbatch textsum/submit_apptainer_test_v8.sh            # binds mirror benchmark backend
```
- `APPTAINER_TMPDIR`/`APPTAINER_CACHEDIR` MUST point at Lustre — login-node `/tmp` is too small (~30 GB).
- Build must run on the **login node** (compute partitions have no internet).
- `textsum.def`'s `%post` exports `TMPDIR=/buildtmp` (path inside sandbox rootfs) to avoid `ENOSPC` when the host `/tmp` (bind-mounted into `%post`) is full.

## Architecture

### Pipeline

**Retrieval is closed.** exp35/exp37 showed feeding the *whole doc* to a strong LLM beats every retrieval config (exp37 0.6944 vs best-retrieval exp30 0.6783). All work from exp35 onward has no retrieval step. The legacy hybrid retrieval (bge-m3 dense + BM25 top-20 → bge-reranker-v2-m3 cross-encoder) survives only in the `exp22`/`exp30` lineage and the v11 container.

**Generation (E5 self-citation):** paragraphs are passed as numbered `[1..K]` context to vLLM; the model answers in Thai and cites paragraphs as `[อ้างอิง: X]` → cited paras become `refs`. Context-first prompt order lets vLLM `enable_prefix_caching` reuse the ~14K-token doc prefix across queries of the same doc.

### Current best — `exp56` (0.7215 leak-free), two-stage LLM hybrid
- **Stage A** (ref-picker): Qwen3.6-27B-FP8 + **V10_factual** prompt + exp38 multi-ref shots, on the full doc. Produces sharp refs (IoU 0.8006). Refs ONLY.
- **Stage B** (answer-writer): Qwen3-32B-AWQ + exp38 E5 prompt + hint `**โดยเน้นย่อหน้าหมายเลข [X, Y, Z] เป็นข้อมูลหลัก**` pointing at Stage A's selection, on the full doc.
- Final **refs FIXED to Stage A**; abstractive from Stage B.
- Both models can't coexist on one A100-40GB → `del llm; gc.collect(); torch.cuda.empty_cache()` between stages. Not yet containerized (needs ~50 GB SIF with both weights).

### Production — image **v16** (`textsum/model/run.py`), single-model port of exp42 (0.7087)
Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, ~3B active) + exp38 E5 prompt + 2-shot few-shot (single-ref shot1 + multi-ref shot2 = Q0746) + `enforce_eager=True`. A3B follows the citation format reliably without exp38's fallback queue.

### Key design decisions / gotchas
- **`enforce_eager=True`** required inside Apptainer/Docker on vllm 0.9.2 — the V1-engine torch.compile path SIGKILLs the worker mid-compile. ~0 latency cost. (LANTA venv vllm 0.19.1 tolerates eager off; the container does not.)
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** in every LANTA SLURM script — required when any code touches CUDA before vLLM init (CrossEncoder, sentence-transformers, even `import torch`), else the forked worker crashes on model load.
- **27B-FP8 is slow because of prefix caching, not just FP8**: it resolves to a multimodal arch (`Qwen3_5ForConditionalGeneration`) so vLLM auto-disables `enable_prefix_caching` → re-prefills the whole doc every query (~25× prefill work). Forcing the flag on (exp63) gave ~2.9× speedup, output-neutral. Still loses to AWQ/A3B on speed+quality → dead-end as a single model.
- A100 has no native FP8 cores → vLLM uses `fp8_marlin`/`fp8_w8a16` software dequant (~1.5–2× slower than AWQ).

### Experiment history (train-set composite, ↑ better; † = leak-free 1218 q, doc_050 held out)

| Exp | Change | Composite |
|-----|--------|-----------|
| baseline | dense-only retrieval + Qwen2.5-7B | 0.5584 |
| exp01 | + BM25 pool + cross-encoder rerank | 0.6148 |
| exp03 ⭐ | exp01 + Qwen3-32B-AWQ (no-think) — no-few-shot baseline | 0.6256 |
| exp08 † | + 2-shot multi-turn-chat few-shot (hand-picked pair) | 0.6361 |
| exp22 (v11) † | exp03 + E5 self-cite (GEN_K=5) + exp08 shots | 0.6647 |
| exp27 † | GEN_K sweep → 20 | 0.6777 |
| exp30 † | full reranked union pool, no GEN_K cap — best retrieval-line | 0.6783 |
| exp35 † | NO RETRIEVAL — feed full doc | 0.6929 |
| exp37 † | + context-first prompt (prefix-cache) | 0.6944 |
| exp38 † | + multi-ref shot2 (Q0746) — fixes missing-citation fallbacks | 0.6987 |
| exp39 † | MAX_NEW_TOKENS 512→1024 | 0.6991 |
| exp42 (v16) ⭐ | model → Qwen3-30B-A3B-Instruct-2507-FP8 | 0.7087 |
| exp50/51 † | V10_factual on 27B-FP8 / A3B (record single-stage IoU 0.7998 on exp50) | — |
| **exp56** ⭐⭐ | hybrid: 27B-FP8 picks refs → 32B-AWQ writes answer, refs fixed | **0.7215** |
| exp59 † | hybrid pipeline 2 with A3B Stage B | 0.7196 |

**Score breakdown (leak-free except baseline/exp03):**

|          | exp03  | exp22  | exp30  | exp37  | exp38  | exp59  | **exp56** |
|----------|--------|--------|--------|--------|--------|--------|-----------|
| RougeL   | 0.3928 | 0.4454 | 0.4584 | 0.4939 | 0.4935 | 0.4899 | —         |
| SS-score | 0.8096 | 0.8384 | 0.8467 | 0.8626 | 0.8619 | 0.8624 | —         |
| IoU      | 0.6190 | 0.6575 | 0.6844 | 0.6669 | 0.6906 | 0.8006 | 0.8006    |
| **Composite** | **0.6256** | **0.6647** | **0.6783** | **0.6944** | **0.6987** | **0.7196** | **0.7215** |

exp03 on the leak-free 1218 subset = 0.6270, so exp56 is **+0.0945** over exp03 and **+0.0228** over the best single-stage (exp38). exp56 IoU = exp59 (both fix refs to the same Stage A); the +0.0019 is 32B-AWQ Stage B's slightly better answer quality.

### Key experiment findings (verdicts)
- **Few-shot (E7, exp06–exp16):** 2-shot is the inverted-U peak; dynamic k-NN few-shot is the most *reproducible* (+0.0052 on full 1239). exp08's hand-picked pair is the chosen production few-shot.
- **GEN_K sweep (exp25–27):** more context monotonically helps (5→20 = +0.013); avg union ~28.6. The old "more context hurts" was a Qwen2.5-7B artefact.
- **Order vs selection (exp28–30):** at K≥20, the cross-encoder's value is *ordering* (gold at rank 1 → RougeL/SS lift), not selection; RRF loses to rerank.
- **exp38 fixed the citation-fallback bug:** 153/1239 queries omitted `[อ้างอิง:]` (→ IoU=0). A multi-ref shot2 (cites 4 paras) taught the model to cite ≥3 paras; IoU 0.6669→0.6906.
- **Hybrid pipelines (exp55–66):** decoupling refs (sharp via V10_factual+27B-FP8) from answers (strong via AWQ/A3B) is the win. **Pipeline 2 (full doc + hint, refs fixed) beats filtered-context (pipeline 1) by +0.012.** 32B-AWQ Stage B narrowly beats A3B (+0.0019). IoU breaks the single-stage citation ceiling: 0.6906 → 0.8006.

### What does NOT help (verified — don't retry without new evidence)
- **Reranker swaps** Qwen3-Reranker-0.6B/4B, jina-reranker-v3: bge-reranker-v2-m3 already ≥ them on MIRACL Thai. Qwen3-Reranker-8B lifts every retrieval metric but only +0.0016 composite (exp23) — gold already in top-5 ≥91%.
- **Expanding candidate pool** 20→40: hit@1 dropped; rerank misranks extras. Only pool *size* moves recall; *fusion strategy* (RRF k, weighted α) is a dead end (E4 closed).
- **Direct-QA / rephrase prompt** (exp05): −0.0040.
- **E5 self-cite on a <10B model** (exp02, 7B): IoU collapsed. Model-specific — works 99%+ on Qwen3-32B-AWQ.
- **E6 adaptive K** from rerank score gap: scores don't correlate with correctness.
- **E8 length calibration / truncation**: verbose preds carry recall; truncating loses RougeL at every cap.
- **E2 bge-m3 ColBERT reranking**: shares too much signal with bge-reranker on Thai.
- **E3 HyDE** (exp31/32): retrieval-only lift (+0.035 hit@1) washes out (−0.0004) — rerank uses raw query, BM25 covers what HyDE adds.
- **RRF replacing cross-encoder** at high K (exp28/29): loses on ordering quality.
- **Criteria-based few-shot selection** (exp15/16): the "restate" proxy picks templated examples that lift RougeL but leave SS flat; loses to exp08's hand-picked pair.
- **V10_factual on single-stage AWQ** (exp67): −0.0039 — V10's terser answer costs RougeL/SS more than its IoU gain; only pays off isolated to ref-picking (exp56 Stage A).
- **Full-doc few-shot** (exp68–70): showing the demo over the whole doc_050 vs a 5-para snippet is +0.0004 (noise) at ~2.3× prompt tokens. Keep the snippet.
- **A3B as Stage-A ref-picker** (exp64–66): −0.0097 — A3B's ~3B-active MoE picks wrong paragraphs more often (IoU drops ~0.04). Keep 27B-FP8 dense extraction for refs.

### Failure analysis (exp03, 322 IoU=0 queries)
83.5% = gold was in the top-20 pool but rerank picked something else; 7.2% pushed past rank 20; 9.9% not in pool. **Rerank quality, not pool recall, was the bottleneck** — digit-bearing queries (meeting/agenda numbers, dates) failed at 35.7% vs 20.6%. (Moot since retrieval is closed, but explains the exp35 motivation.)

## Docker container issues

Tag history (resolved exit-1 at v8; v9 perf; v10 few-shot; v11 E5; v12 no-retrieval; v14 context-first; v15-K exp38; **v16 = exp42 A3B, production**).

**v8 = the five gotchas** (each alone leaves `Engine core initialization failed. Failed core proc(s): {}` — worker dies before first heartbeat):
1. `vllm==0.9.2` pinned. Loose `vllm>=0.6.0` makes pip backtrack ~7 wheels and time out on PyPI.
2. `transformers==4.52.4` pinned. ≥4.55 has built-in `aimv2` config; vllm 0.9.2's `OvisConfig` re-registers it → `ValueError: 'aimv2' is already used`.
3. `python3.11-dev` in apt (base image is `runtime`, no headers) — Triton compiles a C ext, else `fatal error: Python.h`.
4. `enforce_eager=True` — see gotchas above.
5. `--timeout 300 --retries 5` on pip install — login-node PyPI is flaky.

**v11 → vllm 0.19.1 bump: attempted, reverted.** Matching the LANTA venv stack (cuda 12.8.1, torch 2.10+cu128, vllm 0.19.1, transformers 5.8.1) builds but the EngineCore worker segfaults inside Apptainer after model load (all attention backends segfault → not the kernel). venv↔container greedy-decode drift (~17/50 sample answers diverge) is accepted, as for v8–v10.

Build on the **login node** (`nohup apptainer build … textsum.def`); test with `sbatch textsum/submit_apptainer_test_v8.sh` (only the benchmark binds → a pass confirms the SIF is self-contained).

## Data format

`textsum/eval_train/test.json` (symlink → train_set.json) and `textsum/model/test/test.json`:
```json
{
  "docs":    [{ "doc_id": "...", "paragraphs": [{ "para_id": "...", "text": "..." }] }],
  "queries": [{ "ID": "...", "doc_id": "...", "query": "..." }]
}
```
Train set additionally has `"abstractive"` and `"refs"` (ground truth) on each query.

## Evaluation metrics (`textsum/eval_train/score.py`)
- **RougeL**: Thai word-tokenized with pythainlp (`newmm`), space-split for ROUGE.
- **SS-score**: cosine similarity of bge-m3 embeddings (pred vs ref).
- **IoU**: Jaccard overlap of predicted vs ground-truth `refs` para IDs.

## Environment variables

| Variable | Default (Docker) | LANTA override |
|----------|-----------------|----------------|
| `TEST_DIR` | `/model/test` | `$PROJECT/textsum/model/test` |
| `RESULT_DIR` | `/result` | `$PROJECT/textsum/result` |
| `PROGRESS_LIB` | `/benchmark_lib/progress` | `$PROJECT/textsum/benchmark_lib/progress` |
| `LLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (v16; was `Qwen/Qwen3-32B-AWQ` ≤v15) | set to override |
| `MAX_MODEL_LEN` | `32768` (v12+) | set to override |
| `VLLM_WORKER_MULTIPROC_METHOD` | (unset) | `spawn` |

## Progress reporting

`textsum/benchmark_lib/progress` is a binary the benchmark calls to track progress — once per query processed, and once more at the end with the total count.
