# Evidence index

| Evidence | Source and scope |
| --- | --- |
| `competition_summary.csv` | Prior project report draft for Phases 1–4. Directly inspected Codabench My Submissions screenshots for Phases 5 and 7. The student confirmed that Phase 6 had not been submitted. The saved records do not show ranks. |
| `phase1_foundations.csv`, `phase1_heldout_generation.csv`, `phase1_decoder_curve.csv` | Saved Phase 1 artifacts in `Phase 1/all_kaggle_outputs/capstone_phase1`. The curve uses the recorded decoder training NLL. |
| `phase2_comparison.csv` | Executed Phase 2 notebook cells 21, 22 and 24 (zero-based). Candidate 08 is validation-tuned. |
| `phase3_comparison.csv`, `phase3_annotation_audit.json` | Saved Phase 3 comparison and annotation audit in `phase 3/capstone_phase3`. |
| `phase4_comparison.csv` | Executed Phase 4 notebook cells 23 and 25 (zero-based). Local supplied-label test results and dev results use separate columns. The optional complete workspace backup failed after these outputs. |
| `phase5_suite_comparison.csv`, `phase5_data_audit.json`, `phase5_suite_label_analysis.csv` | Saved Phase 5 suite artifacts. The audit records split-overlap counts in place of ID lists. |
| `phase5_extra_curve.csv`, `phase5_extra_archive_match.json` | Extra-model notebook and local `230437.zip`. The archive has the logged 13,019 cases and 57,733 predicted labels, for the same mean of 4.434518780244258. |
| `phase7_comparison.csv`, `phase7_ensemble_selection.json`, `phase7_evaluator_dev_score.json` | Executed Phase 7 development outputs and Codabench screenshot. The bundled evaluator was rerun on the saved development predictions of candidates 03 and 07. The file named `official_dev_score` in the original outputs is a local evaluator result. |

Additional report diagnostics come from the executed Phase 2 notebook's per-label table and error examples, the Phase 3 run's domain metrics, the Phase 4 supplied-label test confusion matrix, and the Phase 7 shortlist audit and recorded fusion sweep. The Phase 1 tokenizer sample sizes come from its tokenizer provenance record. Original case text, predictions, and run logs stay outside this repository.

The Phase 7 document lengths use whitespace-delimited word counts from the official development query and candidate files. The Phase 5 label-count summary uses the saved development gold file.
