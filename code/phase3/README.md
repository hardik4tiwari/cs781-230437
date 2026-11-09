# Phase 3 · Rhetorical role prediction

Open `rhetorical_roles.ipynb` with the Phase 2 output tree and official `Phase_3_RR` data. Configure `PHASE2_ROOT` and `DATA_ROOT`. The notebook preserves official sentence boundaries, pools token chunks into sentence vectors, contextualizes sequences of sentences, and emits one label per sentence. The development evaluator combines CL and IT sentences using the official role set.

The seven candidates are full decoder (01), full encoder–decoder (02), local GQA LoRA (03), hybrid rsLoRA (04), decoder LoRMA+ (05), encoder–decoder IA³ (06), and a distilled decoder student (07). The prior report associates the top visible upload with the full encoder–decoder candidate. The saved local comparison ranks the full decoder highest on development macro-F1. Preserve compatible Phase 3 checkpoints for Phase 4.
