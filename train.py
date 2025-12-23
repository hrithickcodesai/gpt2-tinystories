import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import BabyGPTConfig
from model.gpt2 import GPT2Model
from training.dataset import TinyStoriesDataset

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
MAX_TOKENS = 1000000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

config = BabyGPTConfig()

gpt2 = GPT2Model(
    vocab_size=config.vocab_size,
    block_size=config.block_size,
    n_layer=config.n_layer,
    n_head=config.n_head,
    n_embd=config.n_embd,
    dropout=config.dropout,
)
params = sum([p.numel() for p in gpt2.parameters()])
print(f"Model parameters: {params / 1e6:.2f}M")

gpt2.to(DEVICE)
optimizer = torch.optim.AdamW(gpt2.parameters(), lr=LEARNING_RATE)


tiny_dataset = TinyStoriesDataset(
    data_path="data/train_473992236tkns.bin", block_size=config.block_size
)
train_dataloader = DataLoader(tiny_dataset, batch_size=BATCH_SIZE, shuffle=True)


gpt2.train()

step = 0
tokens_seen = 0
while tokens_seen < MAX_TOKENS:
    for batch_idx, (x, y) in enumerate(train_dataloader):
        if tokens_seen >= MAX_TOKENS:
            break

        start_time = time.time()

        x, y = x.to(DEVICE), y.to(DEVICE)

        logits = gpt2(x)
        step += 1

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        perplexity = torch.exp(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tokens_in_batch = B * T
        tokens_seen += tokens_in_batch
        end_time = time.time()
        dt = end_time - start_time
        tok_per_sec = tokens_in_batch / dt
        print(
            f"Step {step} | "
            f"Tokens: {tokens_seen} | "
            f"Loss: {loss.item():.4f} | "
            f"Perplexity: {perplexity.item():.4f} | "
            f"Speed: {tok_per_sec:.2f} tok/sec"
        )
