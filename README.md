# CS781 Capstone · In-House Legal Intelligence

Code deliverables for Hardik Tiwari (230437), IIT Kanpur. Each phase directory contains its Kaggle entry notebook, readable Python source, and a run guide. The notebooks write their embedded runtime into the phase's Kaggle working directory. The source copies make it easier to inspect the implementation and invoke the local evaluators. For Phase 7, the source and configurations come from the completed run. The notebook is the preflight-repair version kept in the project chat.

| Phase | Task | Inputs from earlier work | Code |
| --- | --- | --- | --- |
| 1 | Legal language model pretraining | Official course corpus | [Foundation](code/phase1/) |
| 2 | Legal named entity recognition | Phase 1 checkpoints and tokenizer | [NER](code/phase2/) |
| 3 | Rhetorical role prediction | Compatible Phase 2 models and tokenizer | [Rhetorical roles](code/phase3/) |
| 4 | Court judgment prediction | Compatible Phase 3 models and tokenizer | [Judgment prediction](code/phase4/) |
| 5 | Legal statute identification | Phase 1 checkpoints and tokenizer | [Statute identification](code/phase5/) |
| 6 | Abstractive summarization | Phase 1 checkpoints and tokenizer | [Summarization](code/phase6/) |
| 7 | Prior case retrieval | Phase 1 checkpoints and tokenizer | [Retrieval](code/phase7/) |

## Run a phase

1. Import the notebook listed in the phase guide into Kaggle and select **GPU T4 ×2**. Keep each phase in a separate working directory.
2. Attach the official course data and the earlier-phase artifacts named in the guide. Edit the input paths in the configuration cell to match your attachments. For notebooks with a downloader, use authorized dataset access via the Kaggle secret named `HF_TOKEN`.
3. Run the setup, data audit, and preflight before training. Preserve the tokenizer, parent checkpoints, manifest, configurations, and data revision for a resumed run.
4. Use the phase's evaluation and export cells to create candidate ZIPs. Codabench uploads and rankings are recorded separately. Some notebooks include optional full-workspace backups, which can require several gigabytes.

The repository contains source and small report evidence files. The datasets, tokenizer artifacts, model weights, predictions, and caches stay with the local phase outputs or Kaggle attachments.

## Report and packaging

- [Report source and checklist](report/README.md)
- [Local and Codabench evidence](report/evidence/README.md)
- [Archive builder](build_submission.py)

A report labeled **Draft** still needs the items in the checklist. Use the exact course filename after completing the report and confirming the results.
