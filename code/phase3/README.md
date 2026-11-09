# Phase 3 · Rhetorical role prediction

Open `rhetorical_roles.ipynb` with the Phase 2 output tree and official `Phase_3_RR` data. Configure `PHASE2_ROOT` and `DATA_ROOT`. The notebook preserves official sentence boundaries, pools token chunks into sentence vectors, contextualizes sequences of sentences, and emits one label per sentence. The development evaluator combines CL and IT sentences using the official role set.

The seven candidates are full decoder (01), full encoder–decoder (02), local GQA LoRA (03), hybrid rsLoRA (04), decoder LoRMA+ (05), encoder–decoder IA³ (06), and a distilled decoder student (07). The prior report associates the top visible upload with the full encoder–decoder candidate. The saved local comparison ranks the full decoder highest on development macro-F1. Preserve compatible Phase 3 checkpoints for Phase 4.

## Three featured models

Use the candidate IDs in `rhetorical_roles.ipynb`. Candidate 02 is linked to the top Codabench result by the prior report. Candidates 01 and 07 have the highest local development scores. Retain the decoder teacher stages for candidate 07.

| Slot | Candidate ID | Selection evidence |
| --- | --- | --- |
| 1 | `02_encoder_decoder_full` | Codabench macro-F1 0.4542, prior report mapping |
| 2 | `01_decoder_full` | Development macro-F1 0.4359 |
| 3 | `07_decoder_student_kd` | Development macro-F1 0.4173 |
