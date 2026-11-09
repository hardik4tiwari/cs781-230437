# Phase 1 · Legal language model pretraining

Open `foundation.ipynb` with the official Phase 1 corpus and test prompts. The notebook trains a byte-level BPE tokenizer on the training corpus, packs 1,024-token sequences, and runs four scratch-trained foundations using two T4s. The data builder separates validation by document hash. The common-suffix perplexity table compares the four runs over the same continuation evaluation.

| Candidate | Foundation | Generation |
| --- | --- | --- |
| `decoder_balanced` | Causal Transformer | Balanced sampling |
| `decoder_focused` | Causal Transformer | Focused sampling |
| `encoder_decoder_balanced` | Encoder–decoder Transformer | Balanced sampling |
| `local_gqa_balanced` | Local grouped-query attention decoder | Balanced sampling |
| `hybrid_balanced` | LSTM–Transformer hybrid | Balanced sampling |

The prior report records the balanced decoder as the top visible Phase 1 submission. The notebook's local five-candidate manifest is a separate record of generated archives. Keep the original `capstone_phase1/tokenizer.json` and all four `runs/main_<architecture>/best_model.pt` files for the downstream phases. See [research references](research_references.md) for the foundation design provenance.

## Three featured models

Use the listed generation candidates in `foundation.ipynb`. The balanced decoder is the top submission reported in the prior project report. The other two have the lowest local common-suffix perplexities among the remaining foundations.

| Slot | Candidate ID | Selection evidence |
| --- | --- | --- |
| 1 | `decoder_balanced` | Codabench BERTScore F1 0.8302, prior report |
| 2 | `hybrid_balanced` | Local common-suffix perplexity 4.6401 |
| 3 | `local_gqa_balanced` | Local common-suffix perplexity 4.9667 |
