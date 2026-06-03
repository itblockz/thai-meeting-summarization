# Experiment scores

Pulled from LANTA `eval_result/train_eval_score*.json` on 2026-06-03. Model column reconstructed from each `run.py` (exp01–66, plus exp87/88) and from git-log commit messages (exp67–86, whose run.py files were removed — only eval_result remains).

Composite = **0.35 × RougeL + 0.45 × SS-score + 0.20 × IoU**.

- **Eval**: `heldout` = 1218 queries, doc_050 held out (few-shot examples come from it). `full` = full 1239. Not directly comparable; exp03 = 0.6270 (heldout) / 0.6256 (full) anchors the gap.
- **Type**: `single` = one LLM; `hybrid` = two-stage, written `A → B` = Stage A (ref-picker) → Stage B (answer writer).
- `0.35RL+0.45SS` = answer-quality portion only (IoU excluded).

## Model legend

| Short | Full model |
|-------|-----------|
| 7B | Qwen2.5-7B-Instruct |
| Typhoon-12B | Typhoon2.1-Gemma3-12B |
| 32B-AWQ | Qwen3-32B-AWQ |
| 27B-FP8 | Qwen3.6-27B-FP8 |
| 27B-FP8·3.5 | Qwen3.5-27B-FP8 |
| A3B | Qwen3-30B-A3B-Instruct-2507-FP8 |
| A3B-Think | Qwen3-30B-A3B-Thinking-2507-FP8 |
| 35B-GPTQ | Qwen3.5-35B-A3B-GPTQ-Int4 |
| 35B-NVFP4 | unsloth/Qwen3.6-35B-A3B-NVFP4 |
| gemma-31B-NVFP4 | RedHatAI gemma-4-31B-it-NVFP4 |
| gemma-26B-FP8 | gemma-4-26B-A4B-it-FP8-Dynamic |
| gemma-26B-NVFP4·RH | RedHatAI gemma-4-26B-A4B-NVFP4 |
| gemma-26B-NVFP4·nv | nvidia/ModelOpt gemma-4-26B-A4B-NVFP4 |

## Sorted by composite

| Exp | Model | Type | Eval | RougeL | SS | IoU | **Composite** | 0.35RL+0.45SS |
|-----|-------|------|------|--------|------|------|--------------|---------------|
| exp86 | gemma-26B-NVFP4·nv → 32B-AWQ | hybrid | heldout | 0.4913 | 0.8631 | 0.8155 | **0.7235** ⭐ | 0.5604 |
| exp56 | 27B-FP8 → 32B-AWQ | hybrid | heldout | 0.4940 | 0.8642 | 0.7982 | 0.7215 | 0.5618 |
| exp57 | 27B-FP8 → 32B-AWQ | hybrid | heldout | 0.4970 | 0.8629 | 0.7957 | 0.7214 | 0.5623 |
| exp60 | 27B-FP8 → A3B | hybrid | heldout | 0.4891 | 0.8589 | 0.8109 | 0.7199 | 0.5577 |
| exp59 | 27B-FP8 → A3B | hybrid | heldout | 0.4899 | 0.8624 | 0.8006 | 0.7196 | 0.5595 |
| exp80 | gemma-26B-FP8 → A3B | hybrid | heldout | 0.4858 | 0.8614 | 0.8074 | 0.7191 | 0.5576 |
| exp81 | gemma-26B-FP8 → A3B | hybrid | heldout | 0.4861 | 0.8577 | 0.8135 | 0.7188 | 0.5560 |
| exp82 | gemma-26B → A3B | hybrid | heldout | 0.4876 | 0.8576 | 0.8016 | 0.7169 | 0.5566 |
| exp40 | 27B-FP8 | single | heldout | 0.4935 | 0.8619 | 0.7719 | 0.7150 | 0.5606 |
| exp66 | A3B → 32B-AWQ | hybrid | heldout | 0.4944 | 0.8619 | 0.7696 | 0.7148 | 0.5609 |
| exp73 | gemma-31B-NVFP4 | single | heldout | 0.4790 | 0.8546 | 0.8091 | 0.7140 | 0.5522 |
| exp55 | 27B-FP8 → 32B-AWQ | hybrid | heldout | 0.4873 | 0.8523 | 0.7982 | 0.7138 | 0.5541 |
| exp83 | gemma-26B → A3B | hybrid | heldout | 0.4816 | 0.8539 | 0.8022 | 0.7133 | 0.5528 |
| exp65 | A3B → 32B-AWQ | hybrid | heldout | 0.4895 | 0.8622 | 0.7627 | 0.7118 | 0.5593 |
| exp70 | A3B (full-doc few-shot) | single | heldout | 0.4864 | 0.8577 | 0.7758 | 0.7114 | 0.5562 |
| exp51 | A3B + V10 | single | heldout | 0.4858 | 0.8574 | 0.7754 | 0.7110 | 0.5559 |
| exp42 | A3B (v16 prod) | single | heldout | 0.4934 | 0.8618 | 0.7410 | 0.7087 | 0.5606 |
| exp69 | A3B (full-doc few-shot) | single | heldout | 0.4802 | 0.8574 | 0.7740 | 0.7087 | 0.5539 |
| exp58 | 27B-FP8 → A3B | hybrid | heldout | 0.4777 | 0.8442 | 0.8006 | 0.7072 | 0.5471 |
| exp68 | A3B (full-doc few-shot) | single | heldout | 0.4818 | 0.8564 | 0.7637 | 0.7067 | 0.5540 |
| exp64 | A3B → 32B-AWQ | hybrid | heldout | 0.4832 | 0.8471 | 0.7644 | 0.7032 | 0.5503 |
| exp88 | A3B | single | heldout | 0.4711 | 0.8607 | 0.7512 | 0.7025 | 0.5522 |
| exp61 | A3B | single | heldout | 0.4671 | 0.8513 | 0.7761 | 0.7018 | 0.5466 |
| exp44 | 27B-FP8·3.5 | single | heldout | 0.4643 | 0.8581 | 0.7612 | 0.7009 | 0.5486 |
| exp39 | 32B-AWQ | single | heldout | 0.4934 | 0.8618 | 0.6930 | 0.6991 | 0.5606 |
| exp38 | 32B-AWQ | single | heldout | 0.4935 | 0.8619 | 0.6906 | 0.6987 | 0.5606 |
| exp77 | gemma-26B-NVFP4·nv | single | heldout | 0.4524 | 0.8367 | 0.8165 | 0.6982 | 0.5348 |
| exp62 | A3B | single | heldout | 0.4660 | 0.8529 | 0.7554 | 0.6980 | 0.5469 |
| exp53 | 32B-AWQ + V15 | single | heldout | 0.4886 | 0.8522 | 0.7141 | 0.6973 | 0.5545 |
| exp63 | 27B-FP8 (prefix-cache probe) | single | heldout | 0.4616 | 0.8480 | 0.7709 | 0.6973 | 0.5431 |
| exp74 | gemma-26B-FP8 | single | heldout | 0.4528 | 0.8350 | 0.8139 | 0.6970 | 0.5343 |
| exp87 | 27B-FP8 | single | heldout | 0.4558 | 0.8480 | 0.7771 | 0.6966 | 0.5411 |
| exp67 | 32B-AWQ + V10 | single | heldout | 0.4892 | 0.8534 | 0.6980 | 0.6948 | 0.5552 |
| exp37 | 32B-AWQ | single | heldout | 0.4939 | 0.8626 | 0.6669 | 0.6944 | 0.5611 |
| exp35 | 32B-AWQ | single | heldout | 0.4853 | 0.8592 | 0.6820 | 0.6929 | 0.5565 |
| exp52 | 32B-AWQ + V14 | single | heldout | 0.4860 | 0.8558 | 0.6871 | 0.6926 | 0.5552 |
| exp76 | gemma-26B-NVFP4·RH | single | heldout | 0.4436 | 0.8257 | 0.8070 | 0.6882 | 0.5268 |
| exp79 | 35B-NVFP4 | single | heldout | 0.4430 | 0.8163 | 0.8042 | 0.6832 | 0.5224 |
| exp45 | A3B-Think | single | heldout | 0.4575 | 0.8523 | 0.6943 | 0.6825 | 0.5436 |
| exp36 | 32B-AWQ | single | heldout | 0.4608 | 0.8468 | 0.6980 | 0.6819 | 0.5424 |
| exp41 | 35B-GPTQ | single | heldout | 0.4315 | 0.8532 | 0.7333 | 0.6816 | 0.5350 |
| exp34 | 32B-AWQ | single | heldout | 0.4586 | 0.8454 | 0.6905 | 0.6790 | 0.5409 |
| exp30 | 32B-AWQ | single | heldout | 0.4584 | 0.8467 | 0.6844 | 0.6783 | 0.5414 |
| exp33 | 32B-AWQ | single | heldout | 0.4573 | 0.8466 | 0.6850 | 0.6780 | 0.5410 |
| exp32 | 32B-AWQ | single | heldout | 0.4571 | 0.8454 | 0.6874 | 0.6779 | 0.5404 |
| exp27 | 32B-AWQ | single | heldout | 0.4585 | 0.8460 | 0.6824 | 0.6777 | 0.5412 |
| exp29 | 32B-AWQ | single | heldout | 0.4552 | 0.8445 | 0.6844 | 0.6762 | 0.5393 |
| exp28 | 32B-AWQ | single | heldout | 0.4530 | 0.8441 | 0.6794 | 0.6743 | 0.5384 |
| exp26 | 32B-AWQ | single | heldout | 0.4530 | 0.8453 | 0.6725 | 0.6734 | 0.5389 |
| exp25 | 32B-AWQ | single | heldout | 0.4516 | 0.8445 | 0.6680 | 0.6717 | 0.5381 |
| exp23 | 32B-AWQ | single | heldout | 0.4491 | 0.8419 | 0.6513 | 0.6663 | 0.5361 |
| exp22 | 32B-AWQ (v11 prod) | single | heldout | 0.4454 | 0.8384 | 0.6575 | 0.6647 | 0.5332 |
| exp24 | 32B-AWQ | single | heldout | 0.4456 | 0.8386 | 0.6558 | 0.6645 | 0.5333 |
| exp50 | 27B-FP8 + V10 | single | heldout | 0.4117 | 0.7977 | 0.7998 | 0.6630 | 0.5031 |
| exp46 | 32B-AWQ | single | heldout | 0.4457 | 0.8550 | 0.5949 | 0.6597 | 0.5408 |
| exp54 | 27B-FP8 + V13 | single | heldout | 0.3881 | 0.7817 | 0.7925 | 0.6461 | 0.4876 |
| exp48 | 35B-GPTQ | single | heldout | 0.4058 | 0.8279 | 0.6333 | 0.6413 | 0.5146 |
| exp20 | 32B-AWQ | single | heldout | 0.4290 | 0.8313 | 0.5819 | 0.6406 | 0.5242 |
| exp19 | 32B-AWQ | single | heldout | 0.4268 | 0.8314 | 0.5729 | 0.6381 | 0.5235 |
| exp08 | 32B-AWQ | single | full | 0.4161 | 0.8130 | 0.6233 | 0.6361 | 0.5115 |
| exp09 | 32B-AWQ | single | full | 0.4126 | 0.8134 | 0.6233 | 0.6351 | 0.5104 |
| exp10 | 32B-AWQ | single | full | 0.4113 | 0.8129 | 0.6233 | 0.6344 | 0.5098 |
| exp06 | 32B-AWQ | single | full | 0.4070 | 0.8129 | 0.6233 | 0.6329 | 0.5083 |
| exp15 | 32B-AWQ | single | full | 0.4065 | 0.8110 | 0.6233 | 0.6319 | 0.5072 |
| exp11 | 32B-AWQ | single | full | 0.4036 | 0.8109 | 0.6233 | 0.6308 | 0.5061 |
| exp14 | 32B-AWQ | single | full | 0.4033 | 0.8129 | 0.6190 | 0.6308 | 0.5070 |
| exp13 | 32B-AWQ | single | full | 0.4035 | 0.8092 | 0.6186 | 0.6291 | 0.5054 |
| exp16 | 32B-AWQ | single | full | 0.4019 | 0.8094 | 0.6184 | 0.6286 | 0.5049 |
| exp07 | 32B-AWQ | single | full | 0.3998 | 0.8101 | 0.6166 | 0.6278 | 0.5045 |
| exp03 | 32B-AWQ | single | heldout | 0.3944 | 0.8095 | 0.6233 | 0.6270 | 0.5024 |
| exp12 | 32B-AWQ | single | full | 0.4066 | 0.8012 | 0.6186 | 0.6265 | 0.5028 |
| exp17 | 32B-AWQ | single | heldout | 0.4105 | 0.8140 | 0.5776 | 0.6255 | 0.5100 |
| exp04 | Typhoon-12B | single | full | 0.3936 | 0.8073 | 0.6190 | 0.6248 | 0.5010 |
| exp18 | 32B-AWQ | single | heldout | 0.4068 | 0.8273 | 0.5766 | 0.6300 | 0.5146 |
| exp05 | 32B-AWQ | single | full | 0.3843 | 0.8073 | 0.6190 | 0.6216 | 0.4978 |
| exp01 | 7B | single | full | 0.3723 | 0.8016 | 0.6190 | 0.6148 | 0.4910 |
| exp02 | 7B | single | full | 0.3739 | 0.7891 | 0.4870 | 0.5833 | 0.4860 |

## Sorted by 0.35·RougeL + 0.45·SS (answer-quality only, IoU excluded)

| Rank | Exp | Model | Type | RougeL | SS | **0.35RL+0.45SS** | Composite |
|------|-----|-------|------|--------|------|-------------------|-----------|
| 1 | exp57 | 27B-FP8 → 32B-AWQ | hybrid | 0.4970 | 0.8629 | **0.5623** | 0.7214 |
| 2 | exp56 | 27B-FP8 → 32B-AWQ | hybrid | 0.4940 | 0.8642 | 0.5618 | 0.7215 |
| 3 | exp37 | 32B-AWQ | single | 0.4939 | 0.8626 | 0.5611 | 0.6944 |
| 4 | exp66 | A3B → 32B-AWQ | hybrid | 0.4944 | 0.8619 | 0.5609 | 0.7148 |
| 5 | exp38 | 32B-AWQ | single | 0.4935 | 0.8619 | 0.5606 | 0.6987 |
| 5 | exp40 | 27B-FP8 | single | 0.4935 | 0.8619 | 0.5606 | 0.7150 |
| 7 | exp39 | 32B-AWQ | single | 0.4934 | 0.8618 | 0.5605 | 0.6991 |
| 7 | exp42 | A3B | single | 0.4934 | 0.8618 | 0.5605 | 0.7087 |
| 9 | exp86 | gemma-26B-NVFP4·nv → 32B-AWQ | hybrid | 0.4913 | 0.8631 | 0.5604 | 0.7235 |
| 10 | exp59 | 27B-FP8 → A3B | hybrid | 0.4899 | 0.8624 | 0.5595 | 0.7196 |
| 11 | exp65 | A3B → 32B-AWQ | hybrid | 0.4895 | 0.8622 | 0.5593 | 0.7118 |
| 12 | exp60 | 27B-FP8 → A3B | hybrid | 0.4891 | 0.8589 | 0.5577 | 0.7199 |
| 13 | exp80 | gemma-26B-FP8 → A3B | hybrid | 0.4858 | 0.8614 | 0.5576 | 0.7191 |
| 14 | exp82 | gemma-26B → A3B | hybrid | 0.4876 | 0.8576 | 0.5566 | 0.7169 |
| 15 | exp35 | 32B-AWQ | single | 0.4853 | 0.8592 | 0.5565 | 0.6929 |
| 16 | exp70 | A3B | single | 0.4864 | 0.8577 | 0.5562 | 0.7114 |
| 17 | exp81 | gemma-26B-FP8 → A3B | hybrid | 0.4861 | 0.8577 | 0.5561 | 0.7188 |
| 18 | exp51 | A3B + V10 | single | 0.4858 | 0.8574 | 0.5559 | 0.7110 |
| 19 | exp67 | 32B-AWQ + V10 | single | 0.4892 | 0.8534 | 0.5552 | 0.6948 |
| 20 | exp52 | 32B-AWQ + V14 | single | 0.4860 | 0.8558 | 0.5552 | 0.6926 |
| 21 | exp53 | 32B-AWQ + V15 | single | 0.4886 | 0.8522 | 0.5545 | 0.6973 |
| 22 | exp55 | 27B-FP8 → 32B-AWQ | hybrid | 0.4873 | 0.8523 | 0.5541 | 0.7138 |
| 23 | exp68 | A3B | single | 0.4818 | 0.8564 | 0.5540 | 0.7067 |
| 24 | exp69 | A3B | single | 0.4802 | 0.8574 | 0.5539 | 0.7087 |
| 25 | exp83 | gemma-26B → A3B | hybrid | 0.4816 | 0.8539 | 0.5528 | 0.7133 |
| 26 | exp73 | gemma-31B-NVFP4 | single | 0.4790 | 0.8546 | 0.5522 | 0.7140 |
| 26 | exp88 | A3B | single | 0.4711 | 0.8607 | 0.5522 | 0.7025 |
| 27 | exp64 | A3B → 32B-AWQ | hybrid | 0.4832 | 0.8471 | 0.5503 | 0.7032 |
| 28 | exp44 | 27B-FP8·3.5 | single | 0.4643 | 0.8581 | 0.5486 | 0.7009 |
| 29 | exp58 | 27B-FP8 → A3B | hybrid | 0.4777 | 0.8442 | 0.5471 | 0.7072 |
| 30 | exp62 | A3B | single | 0.4660 | 0.8529 | 0.5469 | 0.6980 |
| 31 | exp61 | A3B | single | 0.4671 | 0.8513 | 0.5466 | 0.7018 |
| 32 | exp45 | A3B-Think | single | 0.4575 | 0.8523 | 0.5437 | 0.6825 |
| 33 | exp63 | 27B-FP8 | single | 0.4616 | 0.8480 | 0.5431 | 0.6973 |
| 34 | exp36 | 32B-AWQ | single | 0.4608 | 0.8468 | 0.5424 | 0.6819 |
| 35 | exp30 | 32B-AWQ | single | 0.4584 | 0.8467 | 0.5414 | 0.6783 |
| 36 | exp27 | 32B-AWQ | single | 0.4585 | 0.8460 | 0.5412 | 0.6777 |
| 37 | exp87 | 27B-FP8 | single | 0.4558 | 0.8480 | 0.5411 | 0.6966 |
| 37 | exp33 | 32B-AWQ | single | 0.4573 | 0.8466 | 0.5410 | 0.6780 |
| 38 | exp34 | 32B-AWQ | single | 0.4586 | 0.8454 | 0.5409 | 0.6790 |
| 39 | exp46 | 32B-AWQ | single | 0.4457 | 0.8550 | 0.5407 | 0.6597 |
| 40 | exp32 | 32B-AWQ | single | 0.4571 | 0.8454 | 0.5404 | 0.6779 |
| 41 | exp29 | 32B-AWQ | single | 0.4552 | 0.8445 | 0.5394 | 0.6762 |
| 42 | exp26 | 32B-AWQ | single | 0.4530 | 0.8453 | 0.5389 | 0.6734 |
| 43 | exp28 | 32B-AWQ | single | 0.4530 | 0.8441 | 0.5384 | 0.6743 |
| 44 | exp25 | 32B-AWQ | single | 0.4516 | 0.8445 | 0.5381 | 0.6717 |
| 45 | exp23 | 32B-AWQ | single | 0.4491 | 0.8419 | 0.5361 | 0.6663 |
| 46 | exp41 | 35B-GPTQ | single | 0.4315 | 0.8532 | 0.5350 | 0.6816 |
| 47 | exp77 | gemma-26B-NVFP4·nv | single | 0.4524 | 0.8367 | 0.5349 | 0.6982 |
| 48 | exp74 | gemma-26B-FP8 | single | 0.4528 | 0.8350 | 0.5343 | 0.6970 |
| 49 | exp24 | 32B-AWQ | single | 0.4456 | 0.8386 | 0.5333 | 0.6645 |
| 50 | exp22 | 32B-AWQ | single | 0.4454 | 0.8384 | 0.5332 | 0.6647 |
| 51 | exp76 | gemma-26B-NVFP4·RH | single | 0.4436 | 0.8257 | 0.5268 | 0.6882 |
| 52 | exp20 | 32B-AWQ | single | 0.4290 | 0.8313 | 0.5242 | 0.6406 |
| 53 | exp19 | 32B-AWQ | single | 0.4268 | 0.8314 | 0.5235 | 0.6381 |
| 54 | exp79 | 35B-NVFP4 | single | 0.4430 | 0.8163 | 0.5224 | 0.6832 |
| 55 | exp18 | 32B-AWQ | single | 0.4068 | 0.8273 | 0.5146 | 0.6300 |
| 56 | exp48 | 35B-GPTQ | single | 0.4058 | 0.8279 | 0.5146 | 0.6413 |
| 57 | exp08 | 32B-AWQ | single | 0.4161 | 0.8130 | 0.5115 | 0.6361 |
| 58 | exp09 | 32B-AWQ | single | 0.4126 | 0.8134 | 0.5104 | 0.6351 |
| 59 | exp17 | 32B-AWQ | single | 0.4105 | 0.8140 | 0.5100 | 0.6255 |
| 60 | exp10 | 32B-AWQ | single | 0.4113 | 0.8129 | 0.5098 | 0.6344 |
| 61 | exp06 | 32B-AWQ | single | 0.4070 | 0.8129 | 0.5082 | 0.6329 |
| 62 | exp15 | 32B-AWQ | single | 0.4065 | 0.8110 | 0.5072 | 0.6319 |
| 63 | exp14 | 32B-AWQ | single | 0.4033 | 0.8129 | 0.5070 | 0.6308 |
| 64 | exp11 | 32B-AWQ | single | 0.4036 | 0.8109 | 0.5061 | 0.6308 |
| 65 | exp13 | 32B-AWQ | single | 0.4035 | 0.8092 | 0.5054 | 0.6291 |
| 66 | exp16 | 32B-AWQ | single | 0.4019 | 0.8094 | 0.5049 | 0.6286 |
| 67 | exp07 | 32B-AWQ | single | 0.3998 | 0.8101 | 0.5045 | 0.6278 |
| 68 | exp50 | 27B-FP8 + V10 | single | 0.4117 | 0.7977 | 0.5030 | 0.6630 |
| 69 | exp12 | 32B-AWQ | single | 0.4066 | 0.8012 | 0.5028 | 0.6265 |
| 70 | exp03 | 32B-AWQ | single | 0.3944 | 0.8095 | 0.5024 | 0.6270 |
| 71 | exp04 | Typhoon-12B | single | 0.3936 | 0.8073 | 0.5010 | 0.6248 |
| 72 | exp05 | 32B-AWQ | single | 0.3843 | 0.8073 | 0.4978 | 0.6216 |
| 73 | exp01 | 7B | single | 0.3723 | 0.8016 | 0.4910 | 0.6148 |
| 74 | exp54 | 27B-FP8 + V13 | single | 0.3881 | 0.7817 | 0.4876 | 0.6461 |
| 75 | exp02 | 7B | single | 0.3739 | 0.7891 | 0.4860 | 0.5833 |

## Notes

- **Best overall**: exp86 (hybrid, **0.7235**) — gemma-26B-NVFP4·nv Stage-A refs + 32B-AWQ Stage-B answer; ansB_refA combo (exp86 also has ansA/ansB × refA/refB variants).
- **Best single model**: exp40 (27B-FP8, 0.7150) by composite; exp42 (A3B, 0.7087 = v16 production) and exp51 (A3B+V10, 0.7110) lead on answer quality. exp73 (gemma-31B-NVFP4, 0.7140) is the best single by IoU among A100-feasible models.
- **V10_factual single models** (answer-quality order): exp51 (A3B) > exp67 (32B-AWQ) ≈ exp52 (32B-AWQ/V14) > exp53 (32B-AWQ/V15) > exp50 (27B-FP8) > exp54 (27B-FP8/V13). The 27B-FP8 + V10/V13 pair (exp50/54) trades answer quality for record IoU → reused as Stage-A ref-pickers, not single models.
- **Stage assignment** (`A → B`): exp55/56/57 = 27B-FP8 → 32B-AWQ; exp58/59/60 = 27B-FP8 → A3B; exp64/65/66 = A3B → 32B-AWQ; exp80/81/82/83 = gemma-26B → A3B; exp86 = gemma-26B-NVFP4·nv → 32B-AWQ. exp86 confirmed the rule: gemma Stage-A refs only convert under a 32B-AWQ Stage-B (under A3B Stage-B they're a wash at ~0.719, exp80/81).
- **No score JSON on LANTA** (omitted): exp21, 31, 43, 47, 49, 71, 72, 75, 78, 84, 85 — retrieval-only, OOM/infeasible (gemma FP8/NVFP4 that didn't fit A100), or scored but not under eval_result. exp21 (0.6574, 32B-AWQ) appears in CLAUDE.md but its JSON wasn't collected here.
- exp80–83 Stage models reconstructed from commit messages; the exact answer-vs-ref split for exp82/83 (A3B answers / gemma cites) is per git log, run.py removed.
- **exp37-recipe model sweep (exp37/87/88)**: same recipe (context-first, single-ref shots, 512 tok), model only. A3B (exp88, 0.7025) > 27B-FP8 (exp87, 0.6966) > 32B-AWQ (exp37, 0.6944). exp87 has the sharpest single-stage citation (IoU 0.7771) but the weakest answers (0.35RL+0.45SS = 0.5411) → reinforces 27B-FP8 as a ref-picker, not an answer-writer. exp88 < exp42 (0.7087) shows the exp38 recipe (1024 tok + multi-ref shot2) adds the remaining lift on A3B.
