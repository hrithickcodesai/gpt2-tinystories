# GPT-2 on TinyStories

Clean implementation of GPT-2 on TinyStories (~474M tokens). 6-layer, 384-dim decoder-only transformer with causal self-attention, mixed precision training (bfloat16), and torch.compile optimization.


**Config**:
- `model/config.py`: BabyGPTConfig (6L/384D)
- `model/gpt2.py`: Decoder-only with tied embeddings, pre-norm residual blocks
- `training/dataset.py`: Memmap-based efficient streaming
- `scripts/prepare_data.py`: Tokenize [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) → binary files

## Setup & Run

```bash
uv sync
python scripts/prepare_data.py  # one-time
python train.py                 # train and checkpoint
make format                     # lint
```

Logs to W&B if `WANDB_ENABLED = True`.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) (GPT-2)
- [TinyStories Paper](https://huggingface.co/datasets/roneneldan/TinyStories)
