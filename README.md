# G-Decide

A small English decision model, trained from scratch on free Kaggle GPUs. You give it a
state and a typed question (yes or no, one option from a list, or a point on a scale) and
it answers with calibrated probabilities instead of text, in a single forward pass. Same
idea as the new decision models such as Jev, at a hobby budget. Part of the G family of
models.

**Status:** pretraining.

## How it works

- **Model:** a ~110M-parameter encoder (12 layers, 768 wide, pre-norm, RoPE, GeGLU),
  pretrained ELECTRA-style: a small generator fills in masked tokens and the
  discriminator learns to spot which tokens were replaced. Embeddings are shared the
  DeBERTa-v3 way (gradient-disentangled). Only the discriminator is kept.
- **Pretraining data:** ~5B tokens of English: FineWeb-Edu, FineWeb, English Wikipedia,
  and permissively licensed HTML from The Stack v2. 32k byte-level BPE, digits split.
- **Training:** Kaggle T4 x2 sessions chained by `tools/chain.py`, each resuming the
  previous checkpoint, 512 tokens of context switching to 2048 near the end.
- **Next:** decision fine-tuning (open classification datasets plus synthetic decisions
  from an open-weight model), calibration, and evaluation on the public JevBench set.

## Layout

| Path | What |
|---|---|
| `model/electra.py` | encoder, generator, discriminator, ELECTRA loss |
| `train/pretrain.py` | resumable pretraining loop (also the throughput probe) |
| `data/` | corpus fetchers, tokenizer, token packing |
| `kaggle/` | kernels: `00-probe`, `01-prep`, `02-pretrain` |
| `tools/chain.py` | unattended session chain (LaunchAgent) |

## License

MIT. Training data keeps its own licenses (ODC-By, CC BY-SA 4.0, Blue Oak permissive).
