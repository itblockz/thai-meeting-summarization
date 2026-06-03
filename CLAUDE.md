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

### Current best — `exp86` (0.7235 leak-free), two-stage LLM hybrid (base exp81)
- **Stage A** (ref-picker): `nvidia/Gemma-4-26B-A4B-NVFP4` (exp77) + **V10_factual** prompt + exp38 multi-ref shots, full doc. Sharp refs (IoU **0.8155**) → refs ONLY, and the ref-INDEX hint for Stage B.
- **Stage B** (answer-writer): Qwen3-32B-AWQ + **exp37 E5 prompt + exp37 single-ref shots** + exp81 hint line `ย่อหน้าที่เกี่ยวข้องเบื้องต้น: [X, Y]`, full doc. RougeL 0.4913 / SS 0.8631.
- Final **refs FIXED to Stage A** (combo `ansB_refA`); abstractive from Stage B. The grid emits all 4 `ans{A,B}×ref{A,B}` combos; `ansA_refA` reproduces exp77 (IoU 0.8155 ≈ 0.8165) as a sanity check.
- Beats exp56 (prev best 0.7215) **+0.0020** and exp81 (0.7207) +0.0028 — the win is Stage A's NVFP4 refs (IoU 0.8155 vs exp56's 27B-FP8 0.7982), only slightly offset by ansB RougeL (0.4913 vs exp56's 0.4940). Realized 0.7235 vs the invalid column-merge ceiling 0.7243 → ref-hinting costs only −0.0008. Both models load one at a time on a single A100-40GB (`del llm; gc.collect(); torch.cuda.empty_cache()`). Not yet containerized (~40 GB SIF, both weights).
- **Previous best `exp56` (0.7215):** Stage A Qwen3.6-27B-FP8 + V10_factual (refs IoU 0.8006) → Stage B 32B-AWQ + exp38 E5 + hint `**โดยเน้นย่อหน้าหมายเลข [X, Y, Z] เป็นข้อมูลหลัก**`, refs fixed to Stage A.

### Production — image **v16** (`textsum/model/run.py`), single-model port of exp42 (0.7087)
Qwen3-30B-A3B-Instruct-2507-FP8 (MoE, ~3B active) + exp38 E5 prompt + 2-shot few-shot (single-ref shot1 + multi-ref shot2 = Q0746) + `enforce_eager=True`. A3B follows the citation format reliably without exp38's fallback queue.

### Key design decisions / gotchas
- **`enforce_eager=True`** required inside Apptainer/Docker on vllm 0.9.2 — the V1-engine torch.compile path SIGKILLs the worker mid-compile. ~0 latency cost. (LANTA venv vllm 0.19.1 tolerates eager off; the container does not.)
- **`VLLM_WORKER_MULTIPROC_METHOD=spawn`** in every LANTA SLURM script — required when any code touches CUDA before vLLM init (CrossEncoder, sentence-transformers, even `import torch`), else the forked worker crashes on model load.
- **vllm 0.19.1 is V1-ONLY; the hybrid handoff DEPENDS on V1 multiprocessing.** V0 is gone — `VLLM_USE_V1` is a dead env var (just warns; the v22 image's old `VLLM_USE_V1=0` did nothing, it always ran V1). The two-stage `del engine; gc; empty_cache()` only frees GPU between stages because each stage's EngineCore is a **child process** that the OS reaps on teardown. Forcing in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) leaves Stage A's ~22 GB resident → Stage B OOMs at init (`Free memory 3.27/39.5 GiB`, job 5824350). Dockerfile now pins `VLLM_ENABLE_V1_MULTIPROCESSING=1`. v22 verified end-to-end on A100 under apptainer `--containall` (job 5824281; `MAX_MODEL_LEN=20480` to fit fp16 KV — H100 uses fp8 KV → 32768 fits).
- **27B-FP8 is slow because of prefix caching, not just FP8**: it resolves to a multimodal arch (`Qwen3_5ForConditionalGeneration`) so vLLM auto-disables `enable_prefix_caching` → re-prefills the whole doc every query (~25× prefill work). Forcing the flag on (exp63) gave ~2.9× speedup, output-neutral. Still loses to AWQ/A3B on speed+quality → dead-end as a single model.
- A100 has no native FP8 cores → vLLM uses `fp8_marlin`/`fp8_w8a16` software dequant (~1.5–2× slower than AWQ).
- **A 31B FP8 model can't do full-doc on one A100-40GB** (exp71/72): ~30 GB of bf16-loaded weights leave only ~6.49 GiB for KV, and **FP8 KV cache is impossible on A100 with an FP8 checkpoint** — `kv_cache_dtype="fp8"` (=e4m3) has no sm80 reshape_and_cache Triton kernel (`fp8e4nv not supported`), and `fp8_e5m2` is rejected outright (`not supported with fp8 checkpoints`). With bf16 KV forced, vLLM caps context at ~7.7K tokens (gemma-4's attention layout makes KV expensive) → 89% of queries truncated. Fix is a smaller quant: **NVFP4** (exp73) loads at ~18.5 GB via vLLM's `NvFp4LinearBackend.MARLIN` (weight-only FP4 dequant — warns it degrades compute-heavy/prefill, but runs fine), leaving ~16 GiB KV → full 32768 context with no truncation. For FP8 specifically, use TP=2.

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
| exp73 † | exp51 + model → gemma-4-31B-it-**NVFP4** (single-stage; IoU 0.8091, beats hybrid's citation) | 0.7140 |
| exp74 † | exp73 pipeline + model → gemma-4-**26B-A4B-FP8** MoE (best-ever IoU 0.8139 but answer quality sinks it) | 0.6970 |
| exp75 | exp73 pipeline + model → **nvidia/Gemma-4-31B-IT-NVFP4** (ModelOpt) — INFEASIBLE on 1×A100-40GB (keeps attention bf16 → 29.96 GiB weights) | — |
| exp76 † | exp74 pipeline + model → **RedHatAI/gemma-4-26B-A4B-it-NVFP4** (4-bit attn) | 0.6882 |
| exp77 † | exp74 pipeline + model → **nvidia/Gemma-4-26B-A4B-NVFP4** (ModelOpt, bf16 attn) — best of the NVFP4 MoE | 0.6982 |
| exp78 | exp77 pipeline + model → **nvidia/Qwen3.6-35B-A3B-NVFP4** (hybrid linear-attn MoE) — INFEASIBLE on 1×A100-40GB (`modelopt_mixed` quant hard-gated to sm89+) | — |
| exp79 † | exp78 pipeline + model → **unsloth/Qwen3.6-35B-A3B-NVFP4** (compressed-tensors pure NVFP4 — runs on A100; RougeL 0.4430 / SS 0.8163 / IoU 0.8042) | 0.6832 |
| exp80–85 † | **A3B↔gemma combo-matrix grid** (hint-type × direction × 4 answer×ref combos): best = exp81 `s2ans` (gemma cite cold → A3B answer+cite hinted by gemma's REF indices) = **0.7207**, ties exp56. *(exp80/81 here REPLACE the earlier gemma-Stage-A run, 0.7191/0.7188, preserved at commit 30cae9c.)* | **0.7207** |
| **exp86** ⭐⭐⭐ | best-of-both grid (base exp81): **exp77 NVFP4 gemma-26B refs → exp37 32B-AWQ answer**, refs fixed (`ansB_refA`) — breaks the ~0.721 two-model ceiling | **0.7235** |

**Score breakdown (leak-free except baseline/exp03):**

|          | exp03  | exp22  | exp30  | exp37  | exp38  | exp59  | exp56  | **exp86** |
|----------|--------|--------|--------|--------|--------|--------|--------|-----------|
| RougeL   | 0.3928 | 0.4454 | 0.4584 | 0.4939 | 0.4935 | 0.4899 | 0.4940 | 0.4913    |
| SS-score | 0.8096 | 0.8384 | 0.8467 | 0.8626 | 0.8619 | 0.8624 | 0.8642 | 0.8631    |
| IoU      | 0.6190 | 0.6575 | 0.6844 | 0.6669 | 0.6906 | 0.8006 | 0.8006 | 0.8155    |
| **Composite** | **0.6256** | **0.6647** | **0.6783** | **0.6944** | **0.6987** | **0.7196** | **0.7215** | **0.7235** |

exp03 on the leak-free 1218 subset = 0.6270, so exp56 is **+0.0945** over exp03 and **+0.0228** over the best single-stage (exp38). exp56 IoU = exp59 (both fix refs to the same Stage A); the +0.0019 is 32B-AWQ Stage B's slightly better answer quality. **exp86** (NEW best, 0.7235) swaps exp56's Stage-A ref-picker to exp77's NVFP4 gemma-26B → IoU 0.8006→**0.8155** (+0.0149), and uses exp37's E5 prompt for the 32B-AWQ Stage B — net **+0.0020** over exp56 (the +0.0149 IoU × 0.20 outweighs ansB's RougeL/SS dip).

### Key experiment findings (verdicts)
- **Few-shot (E7, exp06–exp16):** 2-shot is the inverted-U peak; dynamic k-NN few-shot is the most *reproducible* (+0.0052 on full 1239). exp08's hand-picked pair is the chosen production few-shot.
- **GEN_K sweep (exp25–27):** more context monotonically helps (5→20 = +0.013); avg union ~28.6. The old "more context hurts" was a Qwen2.5-7B artefact.
- **Order vs selection (exp28–30):** at K≥20, the cross-encoder's value is *ordering* (gold at rank 1 → RougeL/SS lift), not selection; RRF loses to rerank.
- **exp38 fixed the citation-fallback bug:** 153/1239 queries omitted `[อ้างอิง:]` (→ IoU=0). A multi-ref shot2 (cites 4 paras) taught the model to cite ≥3 paras; IoU 0.6669→0.6906.
- **Hybrid pipelines (exp55–66):** decoupling refs (sharp via V10_factual+27B-FP8) from answers (strong via AWQ/A3B) is the win. **Pipeline 2 (full doc + hint, refs fixed) beats filtered-context (pipeline 1) by +0.012.** 32B-AWQ Stage B narrowly beats A3B (+0.0019). IoU breaks the single-stage citation ceiling: 0.6906 → 0.8006.
- **gemma-4-31B on this task (exp71–73):** the model swap (exp51 pipeline, V10_factual + exp38 shots, single A100-40GB) lands at **0.7140 leak-free with the NVFP4 quant (exp73, +0.0030 over exp51, +0.0053 over v16/exp42)** — RougeL 0.4790 / SS 0.8546 / **IoU 0.8091**. The standout is IoU: a *single* model matching the exp56/exp59 *hybrid* citation quality (0.8006), because gemma-4 follows V10_factual's citation instruction near-perfectly (99.7% tag rate, 4/1239 fallbacks vs exp38's 153). Answer quality (RougeL/SS) is a touch below the A3B line, but the 0.20-weighted IoU more than compensates. **The two FP8 checkpoints (exp71 FP8-Dynamic, exp72 FP8-block) are infeasible on a single A100-40GB** for this full-doc task — see gotchas. exp71/72 code is committed but unrun; a real Dynamic-vs-block comparison needs TP=2. The **MoE** sibling gemma-4-26B-A4B-FP8 *does* fit a single A100 (exp74) — FP8 is only infeasible for the *dense* 31B — but its answer quality loses (see "What does NOT help"). The **NVFP4 *publisher* matters**: exp73's working build is **RedHatAI** (compressed-tensors `nvfp4-pack-quantized`, `config_groups.targets:['Linear']` → quantizes the decoder's self_attn too → 18.54 GiB load). The **nvidia/NVIDIA ModelOpt** build (exp75) `exclude_modules` *all 60* `language_model…self_attn*` → attention stays bf16 → 29.96 GiB load, +11.4 GiB, which **doesn't fit a single A100-40GB** (see "What does NOT help"). For the *dense* 31B, use the RedHatAI checkpoint, not NVIDIA's.
- **NVFP4 publisher REVERSES on the MoE (exp76/exp77):** both NVFP4 26B-A4B builds fit a single A100 (quantized experts dominate the footprint), and here `nvidia/Gemma-4-26B-A4B-NVFP4` (**0.6982**) *beats* `RedHatAI/gemma-4-26B-A4B-it-NVFP4` (**0.6882**) by **+0.0100** — the exact thing that doomed the dense build, ModelOpt keeping `self_attn` in **bf16**, is a *quality advantage* on the MoE (RougeL/SS/IoU all higher: 0.4524/0.8367/0.8165 vs 0.4436/0.8257/0.8070). So: **dense ≥30B → RedHatAI (ModelOpt won't fit); MoE → ModelOpt is both feasible and slightly better.** exp77 (0.6982) ≈ the FP8 MoE exp74 (0.6970); neither beats dense exp73 (0.7140) — the ~4B-active answer-quality ceiling holds (cf. exp74). The NVFP4 *fused-MoE* runs on sm80 via the `MARLIN NvFp4 MoE backend` (FlashInfer/CUTLASS options need Blackwell). Two MoE-specific traps fixed (see "What does NOT help"): the `TRITON_ATTN` MoE path *rejects* the exp75 `kv_cache_dtype="bfloat16"` dense fix (`AssertionError: …got bfloat16`; assert ∈ {"auto","fp8*"}) → must strip the FP8-KV directive from the config so `"auto"` resolves to bf16; and util 0.95 OOM'd the batched frequency-penalty buffer at first decode → use **util 0.90**.
- **exp86 — NVFP4 gemma as the hybrid Stage-A ref-picker BREAKS the ceiling (NEW BEST 0.7235).** exp80–85 concluded the two-model hybrid caps at ~0.721, but that verdict assumed an **A3B Stage-B answer-writer** (~4B-active ceiling). exp86 = exp81's 4-combo grid with both stages upgraded: Stage A = exp77 `nvidia/Gemma-4-26B-A4B-NVFP4` (the NVFP4 *publisher*, IoU 0.8155 — reproduces exp77's 0.8165) picks refs; Stage B = exp37 `Qwen3-32B-AWQ` + E5 prompt + exp37 single-ref shots writes the answer hinted by Stage A's ref *indices* (exp81 hint line). Best cell **`ansB_refA` = 0.7235** (RougeL 0.4913 / SS 0.8631 / IoU 0.8155), +0.0020 over exp56. The win: NVFP4 refs (0.8155) clear 27B-FP8's 0.7982 by +0.0173, and a strong 32B-AWQ (not A3B) Stage-B keeps the answer near exp37's. Confirms `refA`>`refB` (fixed-refs/pipeline-2 still beats free-refs: 0.7235>0.7215). So the hybrid lever is **both** a higher-IoU Stage-A *and* a 32B-AWQ Stage-B — gemma-Stage-A was never the problem; A3B-Stage-B was. The earlier "gemma Stage-A is a wash" (exp80/81, A3B answer) is superseded.

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
- **gemma-4-26B-A4B-FP8 single-stage** (exp74): −0.0170 vs exp73 (0.6970 vs 0.7140) — RougeL 0.4528 / SS 0.8350 / **IoU 0.8139**. The MoE *citation* is the best single-model number on record (99.9% tag rate, 1/1239 fallback, IoU even beats dense gemma-4-31B's 0.8091), but the ~4B-active answers lose RougeL −0.0262 and SS −0.0196 vs the dense 31B, and that 0.45+0.35-weighted answer-quality loss swamps the +0.0048 IoU gain. Same lesson as the A3B ref-picker and the <10B self-cite: **active-param count caps abstractive quality even when citation is flawless.** *Infra all worked* (FP8 fit confirmed — MoE's ~26 GB weights leave 11.05 GiB KV → full 32768 ctx, no truncation, prefix caching honored; ~9 min generation, near-A3B speed). Keep dense gemma-4-31B-NVFP4 (exp73) as the best single model.
- **nvidia/Gemma-4-31B-IT-NVFP4 (ModelOpt) as a drop-in for exp73's RedHatAI build** (exp75): **infeasible on 1×A100-40GB, never scored.** Two checkpoints both labelled "NVFP4 gemma-4-31B" are NOT interchangeable — they quantize different layers. RedHatAI (`config_groups.targets:['Linear']`, compressed-tensors) quantizes the decoder's self_attn to 4-bit → **18.54 GiB** load → ~19 GiB free for KV → full ctx fits (exp73 = 0.7140). NVIDIA's ModelOpt build `exclude_modules` *all 60* `language_model…self_attn*` → attention stays bf16 → **29.96 GiB** load (+11.4 GiB). At util 0.97 that leaves only ~6.9 GiB KV, **below gemma-4's ~7 GiB sliding-window KV floor**, so engine init OOMs (est. max len 8224 < p50 prompt 9162) and *trimming max_model_len barely helps* (KV need 32768→20480 only drops 9.54→8.61 GiB — the floor is fixed, only the global-attention layers scale). Same A100-doesn't-fit wall as exp71/72, but the cause is unquantized attention, not FP8. Two other ModelOpt traps surfaced+fixed before the OOM: the loader path is fine on sm80 (`NvFp4LinearBackend.MARLIN`, weight-only dequant), but (a) the build bakes `kv_cache_quant_algo:"FP8"` into hf_quant_config → default `kv_cache_dtype="auto"` resolves to fp8_e4m3 → sm80 has no `fp8e4nv` reshape_and_cache kernel → pass `kv_cache_dtype="bfloat16"` (any non-"auto" value wins via `resolve_kv_cache_dtype_string`); RedHatAI carries no such directive. **Use the RedHatAI checkpoint.** A real nvidia-vs-redhat answer-quality comparison needs TP=2, which can't deploy to the single-GPU benchmark anyway → not worth it.
- **NVFP4 quant on the MoE (exp76/exp77) doesn't beat FP8 or the dense 31B.** Swapping exp74's FP8 MoE for NVFP4: RedHatAI = **0.6882** (−0.0088 vs exp74 FP8 0.6970, all three metrics down — 4-bit attn costs answer quality); nvidia/ModelOpt = **0.6982** (≈ exp74 FP8, +0.0100 over RedHatAI because its bf16 attn is higher-precision). Both still −0.0158/−0.0158 below dense exp73 (0.7140) — the ~4B-active ceiling (cf. exp74) is the wall, not the quant. *Useful sub-findings, all infra worked:* (1) NVFP4 fused-MoE **does** run on A100/sm80 (`MARLIN NvFp4 MoE backend`); (2) the nvidia/ModelOpt MoE build **fits** a single A100 (KV 82,096 tok / 7.88x at 32768) where its dense 31B sibling didn't (exp75) — quantized experts dominate, bf16 attn is a smaller slice; (3) **publisher choice flips by arch** — RedHatAI for dense, ModelOpt for MoE. Two MoE gotchas: the `TRITON_ATTN` reshape_and_cache kernel asserts `kv_cache_dtype ∈ {"auto","fp8*"}` so exp75's `"bfloat16"` override is rejected → strip `kv_cache_quant_algo`/`kv_cache_scheme` from a local config-override dir so `"auto"`→bf16; and the heavier ~18 GiB nvidia weights need **util 0.90** (0.95 OOM'd the sampling frequency-penalty buffer at first decode). Keep dense gemma-4-31B-NVFP4 (exp73) as the best single model.
- **nvidia/Qwen3.6-35B-A3B-NVFP4 as a single-model swap** (exp78): **infeasible on 1×A100-40GB, never scored — a THIRD A100 wall.** This checkpoint is NOT pure NVFP4 — `hf_quant_config.json` has `quant_algo:"MIXED_PRECISION"` (FP8 attention + W4A16_NVFP4 experts). vLLM resolves it to quant method `modelopt_mixed`, which raises at `VllmConfig` validation: *"The quantization method modelopt_mixed is not supported for the current GPU. Minimum capability: 89. Current capability: 80."* — a hard pydantic gate with NO sm80 fallback (unlike pure NVFP4's MARLIN sm80 path, exp76/77; and unlike exp75's memory-OOM cause). **Everything else worked**, confirming the surrounding plumbing for future Qwen3.5-MoE runs: vLLM DID support the arch (`Qwen3_5MoeForConditionalGeneration`, a **hybrid linear-attention MoE** — recognized `linear_attn` as Mamba layers, 3 of every 4 layers Mamba + every 4th full `self_attn`, auto-enabled prefix caching in Mamba `'align'` experimental mode, set attention block 1056 to match mamba page size); the exp77-style override-dir strip of `kv_cache_quant_algo` succeeded (log: `stripped: ['hf_quant_config.kv_cache_quant_algo']`); and `limit_mm_per_prompt` was required (it's a multimodal `ConditionalGeneration` arch, same as the `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` predecessor that exp41/exp48 ran). **For a Qwen3.x-35B-A3B single model on A100, the sm80-feasible quant is GPTQ-Int4 (GPTQ-Marlin, exp41/48) — not the nvidia mixed-precision NVFP4.** (The 22 GiB NVFP4 download was deleted from `.hf_cache` after exp78.)
- **unsloth/Qwen3.6-35B-A3B-NVFP4 (the sm80-feasible publisher of exp78's model)** (exp79): **runs, but loses — 0.6832 leak-free, the LOWEST recent MoE.** Unsloth's checkpoint is compressed-tensors `nvfp4-pack-quantized` (no `hf_quant_config.json`, `config_groups targets:['Linear']`, `kv_cache_scheme:None`) — the RedHatAI-style PURE-NVFP4 layout, NOT nvidia's sm89-gated `modelopt_mixed` — so it cleared **both** exp78 walls and ran end-to-end on A100 (`NvFp4LinearBackend.MARLIN` GEMM + `MARLIN NvFp4 MoE` backend, 20.7 GiB load, full 32768 ctx, prefix caching in Mamba `'align'` mode; NO override dir needed since no baked KV directive). But the score is **−0.0255 under the Qwen A3B production single model (exp42/v16 0.7087)** and below every gemma NVFP4/FP8 MoE (exp76 0.6882 / exp77 0.6982 / exp74 0.6970). Breakdown: RougeL 0.4430 / SS 0.8163 / **IoU 0.8042** — citation (IoU) is competitive with the gemma MoEs, but **answer quality (RougeL −0.0094, SS −0.0204 vs exp77) is the lowest**, i.e. the *newer* Qwen3.6 A3B base does NOT lift the ~4B-active answer-quality ceiling — it sits below gemma-4's MoE here despite equal active params. Caveat (unconfirmed): the engine log emitted a linear-attention `fla` UserWarning (`seq_len (16) < num_heads (32) … potential format mismatch [head-first?]`) — likely benign (SS 0.816 is coherent, not garbage) but may shave answer quality. Verdict: the active-param ceiling holds across Qwen generations (cf. exp74/77); dense gemma-4-31B-NVFP4 (exp73 0.7140) stays the best single model. enable_thinking was OFF (exp41-style); a thinking-on retry (exp48-style) is the only untried lever on this base.
- **gemma-4-26B-A4B-FP8 as the hybrid Stage-A ref-picker** (original exp80/exp81 design, now superseded — code at commit 30cae9c): **a wash, doesn't beat 27B-FP8.** Swapping exp59/exp60's Stage A (Qwen3.6-27B-FP8) for the exp74 gemma MoE nudged refs up (refs-fixed IoU 0.8074 vs exp59's 0.8006; refs-free IoU 0.8135) but composite stayed flat (0.7191/0.7188). The single-stage gemma IoU edge (0.8139) doesn't transfer to Stage A — it competes with the *already-strong* 27B-FP8 (0.8006), not the weaker A3B self-cite; both dense extractors hit the same ~0.80–0.81 Stage-A ceiling. *(The exp80/81 numbers were later repurposed for the combo-matrix grid below.)*
- **A3B↔gemma combo-matrix grid (exp80–85): hint as REF indices helps, hint as ANSWER text HURTS; best 0.7207 only ties exp56.** A 3×2 grid — hint type {ref-only (exp80/81) / answer-only (exp82/83) / answer+ref (exp84/85)} × direction {A3B→gemma / gemma→A3B} — where each cell runs two staged V10 passes (normal, then the 2nd model hinted by the 1st's output) and scores all 4 answer×ref combos (24 submissions total). **Top = exp81 `s2ans` = 0.7207** (gemma cites cold → A3B writes answer+cite hinted by gemma's ref *indices* → take A3B's answer + A3B-or-gemma refs): RougeL **0.4879** / SS **0.8607** (the grid's best answer quality) / IoU 0.8132. That's −0.0008 under exp56 (0.7215) — **ties, never beats.** Three findings: **(1) ref-index hint helps, answer-text hint contaminates** — A3B-final composite by hint type: ref 0.7207 > answer+ref 0.7103 > answer-only 0.7057; feeding gemma's *weak answer text* as a hint drags A3B's strong answer down −0.018 RougeL / −0.018 SS, while feeding only *ref numbers* leaves A3B's answer intact (actually +0.002/+0.004 vs cold). **(2) direction gemma→A3B beats A3B→gemma** — every A3B-final cell (~0.72) tops every gemma-final cell (~0.716); the 0.80-weighted answer quality dominates and gemma's ~4B-active answer is the ceiling (cf. exp74). **(3) "best of both" (A3B answer + gemma cold refs, IoU ~0.812) confirmed** across cells (0.7153–0.7205) but caps at the exp56 line. Verdict: the two-model hybrid ceiling is ~0.721 regardless of hint plumbing; **pass refs as indices, never as answer text**; gemma-cold-refs → A3B-answer is the clean recipe but doesn't unseat exp56.

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

**v11 → vllm 0.19.1 bump: was reverted at v11; now SHIPPING in v22.** The earlier revert hit an EngineCore segfault inside Apptainer after model load — but that was with the torch.compile path active. v22 runs vllm 0.19.1 (cuda 12.8.1, torch 2.10+cu128, transformers 5.8.1) successfully end-to-end with `enforce_eager=True` (compile off) + V1 multiprocessing on (see the V1-only gotcha above). venv↔container greedy-decode drift (~17/50 sample answers diverge) is accepted, as for v8–v10.

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
