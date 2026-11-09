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
