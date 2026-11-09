# Phase 5 · Legal statute identification

Use `statute_suite.ipynb` for the seven-candidate comparison. Use `statute_full_document.ipynb` for the separately trained decoder with statute-wise token attention, a document BiGRU, and a train-only sparse residual. Each notebook takes official `Phase_5_LSI` data and the corresponding Phase 1 parents and tokenizer. Configure the input paths before running.

The suite contains full decoder (01), partial encoder–decoder (02), local GQA LoRMA+ (03), hybrid adapters (04), decoder soft prompts (05), gradual encoder–decoder tuning (06), and a distilled decoder student (07). It combines neural outputs with a train-only lexical branch and selects decision thresholds on development data.

The separate notebook selects its checkpoint on one development half after calibrating on the other. Its all-development evaluation reuses those development examples for threshold calibration. The local `230437.zip` has exactly the test-case and predicted-label totals logged by the separate notebook, which supports the association with the top visible upload. The execution log does not contain an exported prediction hash.

## Three featured models

These are the three highest visible individual-model Codabench scores. The mapping of the generic `230437.zip` upload to the extra notebook is supported by the matching case and predicted-label totals.

| Rank | Candidate or upload | Codabench macro-F1 | Notebook |
| --- | --- | --- | --- |
| 1 | Extra full-document decoder (`230437.zip`) | 0.3982 | `statute_full_document.ipynb` |
| 2 | `02_encoder_decoder_partial` | 0.3835 | `statute_suite.ipynb` |
| 3 | `01_decoder_full` | 0.3807 | `statute_suite.ipynb` |
