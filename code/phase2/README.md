# Phase 2 · Legal named entity recognition

Open `legal_ner.ipynb`. Attach the Phase 1 tokenizer and all four foundation checkpoints, plus the official `Phase_2_LNER` inputs. Set `PHASE1_ROOT`, `FOUNDATION_OVERRIDES`, `TOKENIZER_OVERRIDE`, and `DATA_ROOT` for the attached layout. The notebook uses overlapping 512-token windows and keeps exact offsets for span reconstruction. Selection uses a document-level holdout and strict macro-F1 over the official labels. Refit uses the labelled folds after selection.

| Candidate | Method |
| --- | --- |
| 01 decoder | Full tuning |
| 02 encoder–decoder | Full tuning |
| 03 local GQA | LoRA |
| 04 hybrid | Partial tuning |
| 05 decoder | LoRMA+ with span prediction |
| 06 decoder | Bottleneck adapters |
| 07 decoder student | Knowledge distillation |
| 08 decoder | Bidirectional span boost with token and boundary predictions |

The notebook logs eight exported candidate ZIPs, including the separate boost export. Keep compatible Phase 2 backbones and tokenizer for Phase 3. The local evaluator is `src/evaluate_phase2_official.py`.

## Three featured models

Use the candidate IDs in `legal_ner.ipynb`. Candidate 08 runs in the separate span-boost cell after the main suite. Its dev score uses the holdout for model and threshold selection.

| Slot | Candidate ID | Selection evidence |
| --- | --- | --- |
| 1 | `08_bidir_span_boost` | Codabench strict macro-F1 0.5061, prior report mapping. Validation-tuned strict macro-F1 0.47304. |
| 2 | `01_decoder_full` | Holdout strict macro-F1 0.47203 |
| 3 | `03_local_gqa_lora` | Holdout strict macro-F1 0.43654 |
