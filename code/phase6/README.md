# Phase 6 · Abstractive summarization

Attach the Phase 1 foundations and tokenizer plus the official `Phase_6_Summ` data. Set `PHASE1_ROOT` and `DATA_ROOT` in the notebook. Use two T4 GPUs. Both notebooks create a development split from labelled training data and exclude test references.

## Three featured models

The completed submissions came from `summarization_fast_04_to_07.ipynb`. Select the candidate IDs below in `RUN_MODELS` to export these three models. Candidate 06 uses candidate 04 as its teacher. The fourth comparison, 07 encoder–decoder cross-attention tuning, scored 0.8050 on Codabench.

| Rank among Phase 6 candidates | Candidate ID | Codabench BERTScore F1 | Local probe BERTScore F1 |
| --- | --- | --- | --- |
| 1 | `04_hybrid_lora` | 0.8253 | 0.8221 |
| 2 | `06_decoder_student_kd` | 0.8218 | 0.8240 |
| 3 | `05_decoder_prompt` | 0.8070 | 0.8029 |

The local probe contains 24 development documents. All four saved archives cover the 100 official test IDs, contain distinct prediction sets, and match the saved run receipts. `src_fast/` matches the executed source, and `configs/` contains the configurations from that run. The notebook regenerates paths for a new Kaggle session.

## Notebook variants

- `summarization_fast_04_to_07.ipynb` uses a shared train-only sentence selector, a 1,024-token source budget, and a 640-token target budget. Each completed run used 384 optimizer updates. Training took 1.54–4.30 minutes per model, with a separate 1.02–1.92 minutes for export.
- `summarization_suite.ipynb` preserves the earlier full suite for comparison. It covers full encoder–decoder tuning, decoder LoRMA+, local GQA adapters, hybrid LoRA, decoder soft prompts, and sequence-level distillation. Use separate output directories because some candidate numbers overlap with those of the fast suite.
