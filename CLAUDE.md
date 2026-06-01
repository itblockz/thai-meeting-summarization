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

1. **Retrieval (legacy)**: For each query, build a candidate pool from the `doc_id` document (dense bge-m3 + BM25, top-20 each), rerank it with a cross-encoder (bge-reranker-v2-m3). `exp22` (v11 prod) truncated to `GEN_K=5`; `exp30` (previous retrieval-line best, 0.6783) keeps the **full reranked union pool** (mean ~28 paragraphs). **exp35 closed this phase**: skipping retrieval and feeding the whole doc to a strong LLM beats every retrieval config (exp35 0.6929 vs exp30 0.6783). All experiments from exp35 onward (including v16 production and exp56 repo-best) have no retrieval step — both the legacy hybrid retrieval and exp30's full pool are unused outside their own lineage.
2. **Generation**: Pass the paragraphs as a numbered `[1..K]` context to vLLM; the model answers in Thai and cites which paragraphs it used as `[อ้างอิง: X]` (E5 self-citation — the cited paragraphs become `refs`). The Docker pipeline (`textsum/model/run.py`) is now at image tag **v16** — full-doc / no-retrieval / context-first / E5 + exp38's 2-shot multi-turn-chat few-shot, generation model **Qwen3-30B-A3B-Instruct-2507-FP8** (port of exp42, composite **0.7087** leak-free). exp56 (NEW repo best, two-stage hybrid at 0.7215) is not yet containerized — would need a ~50 GB SIF holding both 27B-FP8 + 32B-AWQ weights. All container runs pass `enforce_eager=True` to skip the V1-engine torch.compile path, which crashes silently inside Apptainer/Docker containers on vllm 0.9.2 (see "Docker container issues").

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
| `exp30/` — full pool rerank | exp27 + no GEN_K cap (full pool, rerank order) | 0.6783 † |
| `exp32/` — E3 HyDE | exp30 + HyDE-blended dense (α=0.5) | 0.6779 † *(−0.0004)* |
| `exp35/` — ceiling test | NO RETRIEVAL — feed full doc to LLM | 0.6929 † |
| `exp37/` — context-first prompt | exp35 + context BEFORE query (prefix-cache hit) | 0.6944 † |
| `exp38/` — multi-ref shot2 | exp37 + replace shot2 with 4-ref example (Q0746) | 0.6987 † |
| `exp39/` — token budget | exp38 + MAX_NEW_TOKENS 512→1024 | 0.6991 † |
| `exp40/` — E9 model swap | exp39 + 32B-AWQ → Qwen3.6-27B-FP8 | 0.6969 † |
| `exp42/` — E9 model swap (**v16 prod**) | exp39 + 32B-AWQ → Qwen3-30B-A3B-Instruct-2507-FP8 | **0.7087** † ⭐ |
| `exp50–54/` — prompt sweep | port prompt_lab R3 winners (V10/V13/V14/V15) to 27B-FP8 / AWQ / A3B | see prompt_lab |
| `exp55/` — hybrid pipeline 1 (AWQ) | 27B-FP8 picks refs → 32B-AWQ writes answer on filtered context | (see narrative) |
| `exp56/` — hybrid pipeline 2 (**NEW BEST**) | 27B-FP8 picks refs → 32B-AWQ writes answer on full ctx + "เน้นย่อหน้า" hint, refs FIXED | **0.7215** † ⭐⭐ |
| `exp57/` — hybrid pipeline 3 (AWQ, refs free) | same as exp56 but Stage B can override refs | (see narrative) |
| `exp58/` — hybrid pipeline 1 (A3B) | exp55 with Stage B = A3B-Instruct | 0.7072 † |
| `exp59/` — hybrid pipeline 2 (A3B) | exp56 with Stage B = A3B-Instruct, refs fixed | 0.7196 † |
| `exp60/` — hybrid pipeline 3 (A3B) | exp57 with Stage B = A3B-Instruct, refs free | 0.7199 † |

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

**No-retrieval ceiling (exp35/exp37, leak-free)**: skip retrieval entirely — feed the full doc (mean 180 paragraphs, ~14K tokens) to Qwen3-32B-AWQ via E5 self-cite. exp35 (query-first prompt) lands at **0.6929** (+0.0146 over exp30 0.6783). exp37 swaps to **context-first prompt** so vLLM's `enable_prefix_caching` reuses the ~14K-token doc prefix across all queries from the same doc (~5% → ~90% cache hit ceiling). Score equivalent: **0.6944** (+0.0015 from prompt order — negligible) but throughput ~50% faster on doc-grouped batches. The win is that retrieval is no longer the IoU bottleneck — IoU 0.6669 here is bounded by *citation* quality, not pool recall.

**Multi-ref few-shot (exp38, leak-free, NEW BEST)**: exp37's v15 container deploy revealed 153/1239 queries (12.4%) failed to emit `[อ้างอิง: …]` → fell back to `gen_pids[0]` (= doc header para), all IoU=0. Analysis: model produces correct comprehensive answers spanning multiple paragraphs but omits the citation tag entirely. Root cause: both shot1 and shot2 (exp08 pair, carried over) cite a single paragraph, so the model learned "answer + [อ้างอิง: ONE_NUMBER]" and stalls when the answer genuinely needs 3+ paragraphs. exp38 replaces shot2 with **Q0746 from doc_050** — 5-paragraph context where 4 are gold (header + 3 absentees) and 1 is a same-section distractor, answer cites `[อ้างอิง: 2, 3, 4, 5]`. Shot 1 (single-ref) preserved so the 71.8% single-ref dataset prior is still represented. Result: **0.6987** (+0.0043 vs exp37), driven entirely by IoU (0.6669 → 0.6906, +0.0237); RougeL/SS dip slightly (−0.0005/−0.0008) from mild over-citation (avg refs/query 1.20 → 1.54). 47 more queries now emit tags (1086 → 1133); 105 of the 109 fallback queries are *stubborn* (still fail after exp38) and skew toward >5 gold refs.

**Token & model exploration (exp39–exp42, leak-free)**: with retrieval closed at exp37 and citation closed at exp38, focus shifted to inference-side levers. **exp39** lifts `MAX_NEW_TOKENS` 512→1024 — rescues queries whose answer was cut mid-sentence (~50/1218 affected) → **0.6991** (+0.0004, within noise). **exp40** swaps Qwen3-32B-AWQ → Qwen3.6-27B-FP8 (dense 30.87 GB, fits A100-40GB with `limit_mm_per_prompt`) → **0.6969** (−0.0022); A100 has no native FP8 cores so vLLM falls back to `fp8_marlin`/`fp8_w8a16` software dequant — ~1.5–2× slower than AWQ on identical hardware, and slightly behind on quality at this prompt. **(Wall-clock verified from LANTA SLURM: exp40 took 2:01:38 vs exp38/AWQ 27:00 vs exp42/A3B 16:20 — far more than the 1.5–2× dequant penalty alone. Root cause is NOT just FP8: 27B-FP8 resolves to `Qwen3_5ForConditionalGeneration` (a multimodal arch), so vLLM auto-disabled `enable_prefix_caching` (the other two are text-only → caching ON). With context-first prompts the ~14K-token doc prefix is shared across all queries of a doc; caching turns ~1239 full-doc prefills into ~50. With it OFF, exp40 re-prefills the whole doc every query — ~25× more prefill work — and FP8-Marlin's "degrades on compute-heavy workloads" warning bites hardest exactly there. The two compound. The tqdm `44 it/s` bar in the .out is tokenize/score, NOT generation; `disable_log_stats=True` means only SLURM Elapsed reflects true speed.)** **exp42** swaps to **Qwen3-30B-A3B-Instruct-2507-FP8** (MoE, ~3B active of 30B, same FP8 weights) → **0.7087** (+0.0096 vs exp38) — best single-model so far. Now production at image tag **v16**. The win comes from A3B's instruction-following — emits `[อ้างอิง: …]` reliably without the exp38 fallback queues that hurt exp22/exp37 generations. (exp41/exp43–exp49 were OOM/quality-loss variants in the same swap exploration; abandoned.)

**Prompt-lab cross-model sweep (exp50–exp54, leak-free)**: built `prompt_lab/` to compare 15 prompt variants × 3 finalist models (32B-AWQ / 27B-FP8 / A3B-Instruct) on 100 held-out dev queries before committing a full LANTA job per variant. R3 winners ported to full eval (1218 leak-free):
- `exp50/`: 27B-FP8 + **V10_factual** ("สั้น+ตรงประเด็น, ระบุข้อเท็จจริงเท่านั้น, ห้ามตีความ") → **IoU 0.7998** (record across all single-stage variants × models).
- `exp51/`: A3B-Instruct + V10_factual.
- `exp52/`: AWQ + **V14_named_entities** — highest RougeL on dev (0.4963), forces entity coverage.
- `exp53/`: AWQ + **V15_no_redundant** — best AWQ proxy on dev (0.6955), adds "ห้ามอ้างย่อหน้าที่ไม่ได้ใช้" guard.
- `exp54/`: 27B-FP8 + **V13_extract** — IoU **0.8225** on dev (highest IoU ever), at cost of RougeL (model becomes ultra-extractive). Ceiling test.

R4/R5 (R5 = few-shot composition ablation) confirmed model-specific shot preferences: A3B and AWQ both prefer the exp38 2-shot baseline; 27B-FP8 was more sensitive to shot wording (`F1_zero_shot` and `F2_only_single` competitive). See `prompt_lab/results_*/summary.json` for the full grid.

**Two-stage hybrid pipelines (exp55–exp60, leak-free) — NEW BEST**: insight from the prompt sweep — V10_factual + 27B-FP8 produces sharper *refs* (IoU 0.7998 vs exp38's 0.6906) but worse *answers* (RougeL/SS drop); exp38's E5 + AWQ produces stronger *answers* but weaker *refs*. Decoupling the two — Stage A picks refs, Stage B writes the answer — captures both halves. **Stage A is always Qwen3.6-27B-FP8 + V10_factual + exp38 shots** running on the full doc. All values are FRESH model output — no precomputed CSVs are read (the constraint that drove this design: refs and answer must both be generated, not joined from prior runs).

Three Stage-B variants × two Stage-B models (32B-AWQ vs A3B-Instruct, same A100-40GB):

| Pipeline | Stage-B context | Stage-B refs | 32B-AWQ Stage B | A3B Stage B |
|---|---|---|--:|--:|
| 1 (filter) | only Stage A's cited paras (1..K renumbered) | fixed to Stage A | `exp55/` | `exp58/` = 0.7072 |
| 2 (hint, fixed) | full doc + "เน้นย่อหน้า [X,Y,Z] เป็นข้อมูลหลัก" line | fixed to Stage A | `exp56/` = **0.7215** ⭐ | `exp59/` = 0.7196 |
| 3 (hint, free) | full doc + hint | Stage B can override | `exp57/` | `exp60/` = 0.7199 |

Key findings:
- **Pipeline 2 wins both model families** — full context + hint beats filtered context decisively (exp58 0.7072 < exp59 0.7196 = +0.0124). Filtering away non-cited paras throws away signal Stage B can still use to phrase the answer.
- **32B-AWQ Stage B narrowly beats A3B** (exp56 0.7215 vs exp59 0.7196 = +0.0019). A3B is ~4× faster wall-clock but the AWQ answer quality edge is the headline.
- **Letting Stage B override refs (pipeline 3) is flat** — exp60 (0.7199) only +0.0003 over exp59. Stage A's V10_factual refs are already near-optimal; A3B mostly agrees and occasionally adds noise.
- **IoU breaks the citation ceiling**: exp38's IoU 0.6906 → exp59's **0.8006** (+0.0100). The "model answers spanning multi-paras but forgets the citation tag" failure mode that bottlenecked single-stage exp38 (153/1239 → IoU=0 fallbacks) is solved by giving citation its own dedicated model+prompt that *only* picks refs.

**Current best**: `exp56/` — Stage A = Qwen3.6-27B-FP8 + V10_factual prompt + exp38 multi-ref shots (refs only); Stage B = Qwen3-32B-AWQ + exp38 E5 prompt with `**โดยเน้นย่อหน้าหมายเลข [X, Y, Z] เป็นข้อมูลหลัก**` hint pointing at Stage A's selection; **final refs FIXED to Stage A**, abstractive from Stage B. Composite **0.7215** leak-free (+0.0228 over exp38). The Docker pipeline still runs single-model v16 (port of exp42, composite 0.7087); exp56 is not yet containerized — would need a hybrid SIF (~50 GB, both model weights) and `del llm; gc.collect(); torch.cuda.empty_cache()` between stages. exp38 remains the canonical single-AWQ reference (0.6987); exp03 the no-few-shot baseline (0.6256 full-1239); exp30 (0.6783) the previous-best retrieval-line score before exp35 closed the retrieval phase.

See `IDEAS.md` for the full experiment roadmap and `eval_retrieval/` for the fast retrieval harness.

**Score breakdown:**

|          | baseline | exp03  | exp22 (v11)  | exp30  | exp37  | exp38  | exp42 (v16) | exp59 (A3B hybrid) | **exp56 (best)** |
|----------|----------|--------|--------------|--------|--------|--------|-------------|--------------------|------------------|
| RougeL   | 0.3387   | 0.3928 | 0.4454       | 0.4584 | 0.4939 | 0.4935 | —           | 0.4899             | —                |
| SS-score | 0.7667   | 0.8096 | 0.8384       | 0.8467 | 0.8626 | 0.8619 | —           | 0.8624             | —                |
| IoU      | 0.4744   | 0.6190 | 0.6575       | 0.6844 | 0.6669 | 0.6906 | —           | 0.8006             | 0.8006*          |
| **Composite** | **0.5584** | **0.6256** | **0.6647** | **0.6783** | **0.6944** | **0.6987** | **0.7087** | **0.7196** | **0.7215** |

baseline/exp03 are full-1239; exp22/exp30/exp37/exp38/exp42/exp56/exp59 are leak-free (1218, doc_050 held out — the few-shot examples come from it). exp03 on the same 1218 subset = 0.6270, so exp56 is **+0.0945** over exp03 and **+0.0228** over the previous single-stage best (exp38). \*exp56 IoU is identical to exp59 because both fix refs to Stage A (same 27B-FP8 + V10_factual model). The +0.0019 composite over exp59 comes from 32B-AWQ Stage B's marginally better RougeL/SS at writing the answer.

**Key design decisions:**
- **Retrieval is closed**: exp35 / exp37 showed feeding the full doc to a strong LLM beats any retrieval config we tried (exp30 = 0.6783 → exp37 = 0.6944 by removing retrieval). The current best (exp56) inherits this — both Stage A and Stage B see the full doc. The old two-stage *retrieval* (bge-m3 dense + BM25 → bge-reranker-v2-m3) survives only in legacy paths (`exp22`/`exp30` lineage and the v11 container) and is unused by exp35+.
- **exp56 (current best)** is a two-stage *LLM* pipeline: Stage A (Qwen3.6-27B-FP8 + V10_factual prompt) picks refs; Stage B (Qwen3-32B-AWQ + exp38 E5 prompt + hint "เน้นย่อหน้า [X,Y,Z]") writes the answer on the full doc. Refs are FIXED to Stage A. Free Stage A's LLM (`del llm; gc.collect(); torch.cuda.empty_cache()`) before loading Stage B — both 27B-FP8 (30 GB) and 32B-AWQ (~18 GB) cannot coexist on a single A100-40GB.
- **Production v16** (`textsum/model/run.py`) still runs the single-model port of exp42: Qwen3-30B-A3B-Instruct-2507-FP8 + exp38 E5 prompt + (single-ref shot1 + multi-ref shot2 = Q0746) + `enforce_eager=True`. Composite 0.7087 leak-free — the hybrid exp56 (0.7215) is not yet containerized (would need ~50 GB SIF holding both 27B-FP8 + 32B-AWQ weights).
- **`enforce_eager=True`** is required inside Apptainer/Docker on vllm 0.9.2's V1-engine torch.compile path — without it the worker SIGKILLs mid-compile. Eager mode costs ~0 latency on this workload. The LANTA venv (vllm 0.19.1) tolerates `enforce_eager=False` but the container does not.
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** is set in every LANTA SLURM script. Required when any code (CrossEncoder, sentence-transformers, even just `import torch`) touches CUDA before vLLM init — otherwise the forked worker crashes during model load.
- **Legacy retrieval path** (kept for the `exp22`/`exp30` lineage and v11 container): bge-m3 dense + BM25 build top-20 each, cross-encoder reranks. bge-m3 and the cross-encoder run on GPU; the cross-encoder is freed before the LLM loads. `exp03/` and the `exp04`–`exp16` line use `TOP_K=1` (retrieve one para and report it as the only ref); `exp30` keeps the full reranked union pool with no GEN_K cap (mean 28.6 paras).

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

History of `Exit StatusCode 1` in the benchmark backend (resolved at v8; v9 = perf; v10 = few-shot; v11 = E5; v12 = no retrieval; v14 = context-first; v15-K = exp38 multi-ref shot2; **v16 = exp42 A3B-Instruct, current production**):

| Tag | Changes vs previous | Score (leak-free) | Benchmark / local Apptainer |
|-----|---------------------|--:|----------------------------|
| v3 (parent repo) | dense-only, `transformers.pipeline` LLM | 0.5584 (baseline) | ✅ ran |
| v4 | + rerank pipeline, switched LLM to vLLM | — | ❌ exit 1 |
| v5 | v4 + `VLLM_WORKER_MULTIPROC_METHOD=spawn` env | — | ❌ exit 1 |
| v6 | reverted LLM to `transformers.pipeline`; rerank kept | — | ❌ exit 1 |
| v7 | v6 + `tzdata` in requirements (pythainlp needed it) | — | ✅ local Apptainer |
| v8 | back to vLLM + Qwen3-32B-AWQ; five gotchas pinned | (exp03 0.6256) | ✅ local Apptainer (4:48) |
| v9 | bge-m3 encoding moved from CPU to GPU | — | ✅ local Apptainer (2:57, −39%) |
| v10 | v9 + 2-shot few-shot prompting (exp08); `max_model_len` 4096→8192 | (exp08 0.6361) | ✅ local Apptainer (test exit 0) |
| v11 | v10 + E5 self-citation (exp22): top-5 numbered context, `[อ้างอิง:]`-driven `refs`; `max_model_len` 8192→16384 | 0.6647 (exp22) | ✅ local Apptainer (test exit 0) |
| v12 | port exp35: drop retrieval entirely — feed full doc; `max_model_len` 16384→32768 | 0.6929 (exp35) | ✅ |
| v14 | port exp37: context-first prompt (context BEFORE query) — enables prefix-cache reuse | 0.6944 (exp37) | ✅ |
| v15* | bump vllm 0.9.2 → 0.19.1 to match LANTA venv. Run of v15-A through v15-J chased an Apptainer SIGSEGV in Triton's pybind11 IRBuilder traced to Ubuntu's `python3.11.0~rc1`. v15-I fixed the root cause by installing **python3.11.x stable** via `deadsnakes` PPA. v15-G/H were Triton-bypass workarounds preserved as escape hatches. | — | ✅ (after v15-I) |
| v15-K | port exp38: replace shot2 with multi-ref Q0746 example | 0.6987 (exp38) | ✅ |
| v16 (**prod**) | port exp42: generation model 32B-AWQ → Qwen3-30B-A3B-Instruct-2507-FP8 (FP8 weights, ~3B active MoE) | **0.7087** (exp42) | ✅ |

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
| `LLM_MODEL` | `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` (v16; was `Qwen/Qwen3-32B-AWQ` ≤v15) | set to override |
| `MAX_MODEL_LEN` | `32768` (v12+) | set to override |
| `VLLM_WORKER_MULTIPROC_METHOD` | (unset) | `spawn` — required when CrossEncoder touches CUDA before vLLM |

### Progress reporting

`textsum/benchmark_lib/progress` is a binary that the competition benchmark calls to track inference progress. It must be called once per query processed, and once more at the end with the total count.
