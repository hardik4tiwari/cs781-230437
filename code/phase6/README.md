# Phase 6 · Abstractive summarization

Attach the Phase 1 foundations and tokenizer, plus the official `Phase_6_Summ` data. Set `PHASE1_ROOT` and `DATA_ROOT` in the notebook. Use two T4 GPUs. The notebooks create a development split from labelled training data and keep test references out of training.

| Notebook | Candidates and workflow |
| --- | --- |
| `summarization_suite.ipynb` | Six candidates with a train-only salience selector: full encoder–decoder (01), decoder LoRMA+ (02), local GQA adapters (03), hybrid LoRA (04), decoder soft prompts (05), and a smaller decoder with sequence-level distillation (06). Includes the nested-summary loader repair. |
| `summarization_fast_04_to_07.ipynb` | Fast hybrid LoRA (04), decoder soft prompts (05), half-depth decoder distilled from 04 with teacher logits (06), and encoder–decoder cross-attention tuning (07). Uses a shared token cache and a shorter default source/target budget. |

Fast candidate 06 trains or reuses 04 as its teacher. The default 20-minute cap is checked between training updates. Data preparation, validation, and test generation have separate runtimes. Use separate output directories for the two variants because candidate IDs overlap and the implementations differ.

The reference chat records synthetic CPU smoke checks. A completed GPU run, local evaluation, official score, and submission count have yet to be verified. When artifacts arrive, record the notebook variant, candidate IDs, and scores before identifying the best model.
