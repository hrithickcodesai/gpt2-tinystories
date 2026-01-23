import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import BabyGPTConfig
from model.gpt2 import GPT2Model
from training.dataset import TinyStoriesDataset

BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
MAX_TOKENS = 10000000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PT_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"Using device: {DEVICE} | Dtype: {PT_DTYPE}")

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

print("Moving model to device and compiling...")
gpt2.to(DEVICE)
gpt2 = torch.compile(gpt2)

optimizer = torch.optim.AdamW(
    gpt2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)


scaler = torch.cuda.amp.GradScaler(enabled=(PT_DTYPE == torch.float16))

tiny_dataset = TinyStoriesDataset(
    data_path="data/train_473992236tkns.bin", block_size=config.block_size
)
train_dataloader = DataLoader(tiny_dataset, batch_size=BATCH_SIZE, shuffle=True)


val_tiny_dataset = TinyStoriesDataset(
    data_path="data/val_4765918tkns.bin", block_size=config.block_size
)
val_dataloader = DataLoader(val_tiny_dataset, batch_size=BATCH_SIZE, shuffle=False)

gpt2.train()

step = 0
tokens_seen = 0
while tokens_seen < MAX_TOKENS:
    for batch_idx, (x, y) in enumerate(train_dataloader):
        if tokens_seen >= MAX_TOKENS:
            break

        start_time = time.time()

        x, y = x.to(DEVICE), y.to(DEVICE)

        with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE):
            logits = gpt2(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        perplexity = torch.exp(loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        torch.nn.utils.clip_grad_norm_(gpt2.parameters(), max_norm=1.0)
        scaler.update()

        tokens_in_batch = B * T
        tokens_seen += tokens_in_batch
        end_time = time.time()
        dt = end_time - start_time
        tok_per_sec = tokens_in_batch / dt
        step += 1

        print(
            f"Step {step} | "
            f"Tokens: {tokens_seen} | "
            f"Loss: {loss.item():.4f} | "
            f"Perplexity: {perplexity.item():.4f} | "
            f"Speed: {tok_per_sec:.2f} tok/sec"
        )
        torch.cuda.synchronize()


torch.save(gpt2.state_dict(), f"checkpoints/baby_gpt2_{tokens_seen}.pth")
print(f"Model checkpoint saved as checkpoints/baby_gpt2_{tokens_seen}.pth")
