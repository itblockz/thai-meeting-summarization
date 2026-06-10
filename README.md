# Thai Meeting-Minutes Summarization

**🥈 2nd place — AI Benchmark 2026**, a national Thai-language NLP competition organized by NECTEC / NSTDA.

A retrieval-augmented summarization system for **Thai parliamentary meeting documents**.
Given a long meeting document and a query, it produces an abstractive Thai answer
together with the supporting paragraph references — output as `submission.csv`
(`ID`, `abstractive`, `refs`).

## Scoring

Submissions are ranked by a composite metric:

```
score = 0.45 × SS-score (semantic similarity, bge-m3)
      + 0.35 × RougeL   (answer overlap)
      + 0.20 × IoU       (reference set match)
```

## Approach

The winning configuration is a **two-stage LLM pipeline** over the *whole document*
(an early finding — `exp35`/`exp37` — was that feeding the full doc to a strong LLM
beats every retrieval setup, so the final system has no retrieval step):

- **Stage A — reference picker:** an **NVFP4-quantized Gemma 26B** model reads the
  full document and selects the supporting paragraphs (reference IoU ≈ **0.82**).
  Its chosen indices also become a hint passed to Stage B.
- **Stage B — answer writer:** an **AWQ-quantized Qwen3-32B** model writes the final
  Thai abstractive answer, citing paragraphs inline as `[อ้างอิง: X]` and using
  Stage A's hint. Context-first prompting lets vLLM reuse the ~14K-token document
  prefix across queries via prefix caching.

Final references are fixed to Stage A; the abstractive answer comes from Stage B.
A lighter single-model production variant (Qwen3-30B-A3B-FP8) is also containerized
for deployment.

## Infrastructure

- **Serving:** [vLLM](https://github.com/vllm-project/vllm) with FP8 / AWQ / NVFP4 quantization — both models fit one at a time on a single **A100-40GB**.
- **Compute:** the **LANTA** HPC cluster (SLURM, Lustre); inference dispatched as `sbatch` jobs.
- **Packaging:** reproducible **Apptainer** (Singularity) images for the benchmark's container backend.
- **CI/CD:** **GitHub Actions** + **Google Cloud Build** build and push images to a private registry.

## Repository layout

| Path | What it is |
| --- | --- |
| `expNN/` | 90+ logged experiments, each a self-contained run + submit script |
| `textsum/` | Production pipeline and container definition (`textsum.def`, `model/run.py`) |
| `eval_retrieval/` | Fast retrieval-only evaluation (no LLM) |
| `tools/` | Scoring / score-collection utilities |
| `SCORES.md` | Auto-generated leaderboard of every experiment's leak-free score |
| `CLAUDE.md` | Detailed engineering notes — architecture, environment, gotchas |

The whole project is **experiment-driven**: every idea is a numbered `expNN/`
directory, scored on a held-out leak-free split, and tracked in `SCORES.md`.

## Team

Built by a two-person team for the competition; I was the primary author of the
pipeline, experiments, and serving infrastructure.
