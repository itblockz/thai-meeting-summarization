#!/usr/bin/env python3
"""
Collect leak-free experiment scores DIRECTLY from the per-query numbers
score.py already wrote — no model run, no metric recompute, no GPU.

For each expNN/eval_result/:
  * detail-mean  = average of train_eval_detail.csv over the leak-free set
                   (every query NOT in the held-out doc, default doc_050).
                   These per-query rougeL / SS-score / IoU were computed by
                   score.py on the actual submission, so the mean is a
                   direct pull, not a re-derivation.
  * stored       = the composite inside train_eval_score_heldout.json.

These two CAN disagree, and that disagreement is the audit signal:
  - column-merge overwrite (e.g. exp40/exp42): score_heldout.py was re-run
    on a hybrid CSV AFTER the detail was written, so the stored JSON is
    inflated while the detail stays pure  -> trust detail-mean.
  - raw-thinking submission (e.g. exp45/exp48): score.py ran on a
    submission still full of <think> tokens, so the detail is garbage; a
    submission_fixed.csv was made later and the JSON re-scored on it
    -> trust the stored JSON.

Tiebreak rule (no recompute needed): if a *_fixed.csv exists in the dir
the raw detail was superseded -> authoritative = stored JSON; otherwise
authoritative = detail-mean. Any disagreement is flagged regardless.

Run on LANTA (data lives there); pure stdlib:
    PROJECT_ROOT=/project/zz991000-zdeva/zz991021/ua047 python3 tools/collect_scores.py
Writes SCORES.json and SCORES.md next to PROJECT_ROOT and prints a sorted table.
"""
import csv
import glob
import json
import os
import sys
from pathlib import Path

WRL, WSS, WJ = 0.35, 0.45, 0.20  # composite weights
HELDOUT_DOC = "doc_050"

PROJECT = Path(os.environ.get("PROJECT_ROOT", "/project/zz991000-zdeva/zz991021/ua047"))
TEST_JSON = PROJECT / "textsum" / "eval_train" / "test.json"
SCORES_JSON = PROJECT / "SCORES.json"
SCORES_MD = PROJECT / "SCORES.md"

# Curated model / type labels (numbers are computed, labels are maintained).
LABELS = {
    "exp01": ("7B", "single"), "exp02": ("7B", "single"),
    "exp03": ("32B-AWQ", "single"), "exp04": ("Typhoon-12B", "single"),
    "exp05": ("32B-AWQ", "single"), "exp06": ("32B-AWQ", "single"),
    "exp07": ("32B-AWQ", "single"), "exp08": ("32B-AWQ", "single"),
    "exp09": ("32B-AWQ", "single"), "exp10": ("32B-AWQ", "single"),
    "exp11": ("32B-AWQ", "single"), "exp12": ("32B-AWQ", "single"),
    "exp13": ("32B-AWQ", "single"), "exp14": ("32B-AWQ", "single"),
    "exp15": ("32B-AWQ", "single"), "exp16": ("32B-AWQ", "single"),
    "exp17": ("32B-AWQ", "single"), "exp18": ("32B-AWQ", "single"),
    "exp19": ("32B-AWQ", "single"), "exp20": ("32B-AWQ", "single"),
    "exp21": ("32B-AWQ", "single"), "exp22": ("32B-AWQ (v11 prod)", "single"),
    "exp23": ("32B-AWQ", "single"), "exp24": ("32B-AWQ", "single"),
    "exp25": ("32B-AWQ", "single"), "exp26": ("32B-AWQ", "single"),
    "exp27": ("32B-AWQ", "single"), "exp28": ("32B-AWQ", "single"),
    "exp29": ("32B-AWQ", "single"), "exp30": ("32B-AWQ", "single"),
    "exp32": ("32B-AWQ", "single"), "exp33": ("32B-AWQ", "single"),
    "exp34": ("32B-AWQ", "single"), "exp35": ("32B-AWQ", "single"),
    "exp36": ("32B-AWQ", "single"), "exp37": ("32B-AWQ", "single"),
    "exp38": ("32B-AWQ", "single"), "exp39": ("32B-AWQ", "single"),
    "exp40": ("27B-FP8", "single"), "exp41": ("35B-GPTQ", "single"),
    "exp42": ("A3B (v16 prod)", "single"), "exp44": ("27B-FP8.3.5", "single"),
    "exp45": ("A3B-Think", "single"), "exp46": ("32B-AWQ", "single"),
    "exp48": ("35B-GPTQ", "single"), "exp50": ("27B-FP8 + V10", "single"),
    "exp51": ("A3B + V10", "single"), "exp52": ("32B-AWQ + V14", "single"),
    "exp53": ("32B-AWQ + V15", "single"), "exp54": ("27B-FP8 + V13", "single"),
    "exp55": ("27B-FP8 -> 32B-AWQ", "hybrid"), "exp56": ("27B-FP8 -> 32B-AWQ", "hybrid"),
    "exp57": ("27B-FP8 -> 32B-AWQ", "hybrid"), "exp58": ("27B-FP8 -> A3B", "hybrid"),
    "exp59": ("27B-FP8 -> A3B", "hybrid"), "exp60": ("27B-FP8 -> A3B", "hybrid"),
    "exp61": ("A3B", "single"), "exp62": ("A3B", "single"),
    "exp63": ("27B-FP8", "single"), "exp64": ("A3B -> 32B-AWQ", "hybrid"),
    "exp65": ("A3B -> 32B-AWQ", "hybrid"), "exp66": ("A3B -> 32B-AWQ", "hybrid"),
    "exp67": ("32B-AWQ + V10", "single"), "exp68": ("A3B", "single"),
    "exp69": ("A3B", "single"), "exp70": ("A3B", "single"),
    "exp73": ("gemma-31B-NVFP4", "single"), "exp74": ("gemma-26B-FP8", "single"),
    "exp76": ("gemma-26B-NVFP4.RH", "single"), "exp77": ("gemma-26B-NVFP4.nv", "single"),
    "exp79": ("35B-NVFP4", "single"), "exp80": ("gemma-26B-FP8 -> A3B", "hybrid"),
    "exp81": ("gemma-26B-FP8 -> A3B", "hybrid"), "exp82": ("gemma-26B -> A3B", "hybrid"),
    "exp83": ("gemma-26B -> A3B", "hybrid"),
    "exp84": ("gemma-26B <-> A3B (grid)", "hybrid"),
    "exp85": ("gemma-26B <-> A3B (grid)", "hybrid"),
    "exp86": ("gemma-26B-NVFP4.nv -> 32B-AWQ", "hybrid"),
    "exp87": ("27B-FP8", "single"), "exp88": ("A3B", "single"),
}


def heldout_ids():
    data = json.load(open(TEST_JSON, encoding="utf-8"))
    return {q["ID"] for q in data["queries"] if q["doc_id"] == HELDOUT_DOC}


def detail_mean(detail_csv, held):
    rows = [r for r in csv.DictReader(open(detail_csv, encoding="utf-8"))
            if r["ID"] not in held]
    if not rows:
        return None
    n = len(rows)
    rl = sum(float(r["rougeL"]) for r in rows) / n
    ss = sum(float(r["SS-score"]) for r in rows) / n
    iou = sum(float(r["IoU"]) for r in rows) / n
    return {"rougeL": rl, "SS-score": ss, "IoU": iou,
            "composite": WRL * rl + WSS * ss + WJ * iou, "n": n}


def score_dirs():
    """Every dir holding a score artifact, at eval_result/ (standard) or
    eval_result/<combo>/ (the exp84/85/86 answer x ref grid)."""
    dirs = set()
    for depth in ("", "*/"):
        for pat in ("train_eval_detail.csv", "train_eval_score_heldout.json"):
            for f in glob.glob(str(PROJECT / "exp*" / "eval_result" / depth / pat)):
                dirs.add(str(Path(f).parent))
    return sorted(dirs)


def label_of(d):
    """(exp, display) from a score dir path .../expNN/eval_result[/combo]."""
    parts = Path(d).parts
    i = parts.index("eval_result")
    exp = parts[i - 1]
    combo = parts[i + 1] if len(parts) > i + 1 else None
    return exp, (exp if combo is None else "{}/{}".format(exp, combo))


def collect():
    held = heldout_ids()
    out, missing = [], []
    for d in score_dirs():
        exp, display = label_of(d)
        det = Path(d) / "train_eval_detail.csv"
        js = Path(d) / "train_eval_score_heldout.json"
        has_fixed = bool(glob.glob(str(Path(d) / "*_fixed.csv")))
        dm = detail_mean(det, held) if det.exists() else None
        stored = json.load(open(js, encoding="utf-8")) if js.exists() else None

        if dm is None and stored is None:
            missing.append((display, "no detail.csv and no heldout.json"))
            continue

        # Authoritative source (no recompute): fixed.csv present => the raw
        # detail was superseded, trust the re-scored JSON; else trust detail.
        if has_fixed and stored is not None:
            src = "json(fixed)"
            rl, ss, iou, comp = (stored["rougeL"], stored["SS-score"],
                                 stored["IoU"], stored["score"])
        elif dm is not None:
            src, comp = "detail", dm["composite"]
            rl, ss, iou = dm["rougeL"], dm["SS-score"], dm["IoU"]
        else:  # only a stored JSON, no detail
            src, comp = "json", stored["score"]
            rl, ss, iou = stored["rougeL"], stored["SS-score"], stored["IoU"]

        stored_comp = stored["score"] if stored else None
        flag = ""
        if dm is not None and stored is not None:
            if abs(dm["composite"] - stored["score"]) > 0.0005:
                flag = "DISAGREE detail={:.4f} json={:.4f}".format(
                    dm["composite"], stored["score"])
        model, typ = LABELS.get(exp, ("?", "?"))
        out.append({"exp": display, "model": model, "type": typ,
                    "rougeL": round(rl, 4), "SS": round(ss, 4),
                    "IoU": round(iou, 4), "composite": round(comp, 4),
                    "source": src, "stored_composite": round(stored_comp, 4) if stored_comp else None,
                    "has_fixed_csv": has_fixed, "flag": flag})
    out.sort(key=lambda r: r["composite"], reverse=True)
    return out, missing


INTRO = """# Experiment scores

Auto-generated by `tools/collect_scores.py` (re-run to refresh) — DO NOT edit
the tables by hand. Numbers are pulled DIRECTLY from each
`eval_result[/combo]/train_eval_detail.csv` (the per-query scores `score.py`
already wrote), averaged over the leak-free set. No model run, no recompute.

Composite = **0.35 × RougeL + 0.45 × SS-score + 0.20 × IoU**.

- **Every row is leak-free** (1218 q, `doc_050` held out) — including exp01-16,
  which earlier appeared as full-1239 (so a few old composites shifted slightly).
- **`src`**: `detail` = mean of the per-query detail CSV; `json(fixed)` = stored
  heldout JSON (the raw submission had `<think>` tokens and was superseded by a
  `submission_fixed.csv`, so the detail CSV is stale).
- **⚠ rows**: the detail-mean and the stored `train_eval_score_heldout.json`
  DISAGREE. The authoritative value is shown; the stale stored composite and the
  cause are in the audit footnote below each table. (exp40/exp42 = the stored
  JSON was overwritten by a column-merge `Xans_Yrefs.csv`; see
  `score-decomposes-linearly` memory.)
- `0.35RL+0.45SS` = answer-quality portion only (IoU excluded).
- `expNN/<combo>` rows = the exp84/85/86 answer×ref grid (each combo a real
  re-scored submission). Bare `expNN` = standard single/hybrid run.

## Model legend

| Short | Full model |
|-------|-----------|
| 7B | Qwen2.5-7B-Instruct |
| Typhoon-12B | Typhoon2.1-Gemma3-12B |
| 32B-AWQ | Qwen3-32B-AWQ |
| 27B-FP8 | Qwen3.6-27B-FP8 |
| 27B-FP8.3.5 | Qwen3.5-27B-FP8 |
| A3B | Qwen3-30B-A3B-Instruct-2507-FP8 |
| A3B-Think | Qwen3-30B-A3B-Thinking-2507-FP8 |
| 35B-GPTQ | Qwen3.5-35B-A3B-GPTQ-Int4 |
| 35B-NVFP4 | unsloth/Qwen3.6-35B-A3B-NVFP4 |
| gemma-31B-NVFP4 | RedHatAI gemma-4-31B-it-NVFP4 |
| gemma-26B-FP8 | gemma-4-26B-A4B-it-FP8-Dynamic |
| gemma-26B-NVFP4.RH | RedHatAI gemma-4-26B-A4B-NVFP4 |
| gemma-26B-NVFP4.nv | nvidia/ModelOpt gemma-4-26B-A4B-NVFP4 |
"""


def _table(rows, key, title):
    lines = [title, "",
             "| Exp | Model | Type | RougeL | SS | IoU | Composite | 0.35RL+0.45SS | src |",
             "|-----|-------|------|--------|------|------|-----------|---------------|-----|"]
    foot = []
    for r in sorted(rows, key=key, reverse=True):
        aq = WRL * r["rougeL"] + WSS * r["SS"]
        comp = "{:.4f}".format(r["composite"])
        if r["flag"]:
            comp += " ⚠"
            foot.append((r["exp"], r["stored_composite"], r["source"], r["flag"]))
        lines.append("| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {} | {:.4f} | {} |".format(
            r["exp"], r["model"], r["type"], r["rougeL"], r["SS"], r["IoU"],
            comp, aq, r["source"]))
    if foot:
        lines.append("")
        lines.append("> ⚠ audit (authoritative shown above; stored JSON differs):")
        for exp, stored, src, flag in foot:
            lines.append("> - **{}**: stored heldout JSON = {:.4f}; using `{}` — {}".format(
                exp, stored if stored else 0.0, src, flag))
    return "\n".join(lines)


def load_scores_json():
    if not SCORES_JSON.exists():
        raise FileNotFoundError(
            "{} does not exist; run without --render-only where score artifacts exist".format(
                SCORES_JSON))
    return json.loads(SCORES_JSON.read_text(encoding="utf-8"))


def render_md(rows=None):
    if rows is None:
        rows = load_scores_json()
    tail = ""
    if SCORES_MD.exists():
        old = SCORES_MD.read_text(encoding="utf-8")
        if "## Notes" in old:
            tail = "\n" + old[old.index("## Notes"):].rstrip() + "\n"
    t1 = _table(rows, lambda r: r["composite"], "## Sorted by composite")
    t2 = _table(rows, lambda r: WRL * r["rougeL"] + WSS * r["SS"],
                "## Sorted by 0.35·RougeL + 0.45·SS (answer-quality only, IoU excluded)")
    t3 = _table(rows, lambda r: r["IoU"], "## Sorted by IoU")
    SCORES_MD.write_text(
        INTRO + "\n" + t1 + "\n\n" + t2 + "\n\n" + t3 + "\n" + tail,
        encoding="utf-8")
    print("Rendered {} ({} rows)".format(SCORES_MD, len(rows)))


def main():
    render_only = "--render-only" in sys.argv[1:]
    rows = None
    dirs = [] if render_only else score_dirs()
    if dirs:
        rows, missing = collect()
        SCORES_JSON.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote {} ({} rows)".format(SCORES_JSON, len(rows)))
        disagree = [r for r in rows if r["flag"]]
        print("{} DISAGREEMENT(s) flagged:".format(len(disagree)))
        for r in disagree:
            print("  {:>16}  auth={:.4f} ({})  | {}".format(
                r["exp"], r["composite"], r["source"], r["flag"]))
        if missing:
            print("{} dirs skipped (no score files): {}".format(
                len(missing), ", ".join(e for e, _ in missing)))
    elif not render_only:
        print("No score artifacts found under {}; rendering markdown from {}".format(
            PROJECT, SCORES_JSON))
    render_md(rows)


if __name__ == "__main__":
    main()
