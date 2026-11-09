# Phase 7 · Prior case retrieval

Open `prior_case_retrieval.ipynb` with the original Phase 1 tokenizer and all four foundations, plus official `Phase_7_PCR` data. Set `PHASE1_ROOT` and `DATA_ROOT` for the attached inputs. BM25 builds split-specific pools and supplies hard negatives and a lexical baseline. Learned rankers emit exactly seven distinct candidate IDs per query. The code supports development Micro-F1@7 selection and reciprocal rank fusion (RRF).

| Candidate | Method |
| --- | --- |
| 01 decoder | Full dense bi-encoder tuning |
| 02 encoder–decoder | Partial tuning |
| 03 local GQA | LoRMA+ with late interaction |
| 04 hybrid | Soft prompts |
| 05 decoder | IA³ |
| 06 decoder | Cross-encoder reranking |
| 07 decoder student | Knowledge distillation and lexical/neural fusion |
| RRF | Development-selected optional ensemble of 03, 05 and 01 |

Candidate 07 led the visible Codabench submissions. The local development ensemble and the locally selected individual model (03) are separate records. `src/` and `configs/` capture the completed run. The configurations retain the original Kaggle paths as provenance. Generate fresh configurations through the notebook for new runs. For a saved dev run, use `src/evaluate_phase7_official.py --truth dev_truth.json --candidate-ids dev_candidate_ids.json --prediction dev_prediction.json`.
