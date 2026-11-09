# Phase 4 · Court judgment prediction

Open `judgment_prediction.ipynb` with the official `Phase_4_CJPE` data and compatible Phase 3 models. Configure `PHASE3_ROOT`, `DATA_ROOT`, and any parent overrides. The workflow uses overlapping 512-token chunks and a hierarchical document head. The decoder adapter candidate uses width-48 adapters and a bidirectional GRU over chunks. The notebook fits its lexical branch on training data and selects on development macro-F1.

The seven candidates are full decoder (01), partial encoder–decoder (02), local GQA LoRMA+ (03), decoder adapters (04), hybrid BitFit (05), gradual encoder–decoder tuning (06), and a distilled decoder student (07). The notebook also supports an optional joint-refined run and score ensemble. The prior report associates the top visible upload with candidate 04. Supplied test-label metrics in the execution log come from a separate reporting step.

The executed notebook export ended inside the optional full-workspace backup traceback after a disk-space error. The packaged notebook uses the intact companion source, whose first 28 cell sources match the executed export. The optional full backup defaults to off. The individual prediction ZIPs and the combined candidate collection still use the original export code.
