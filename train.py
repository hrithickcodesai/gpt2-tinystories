import math
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.config import GPT2Config
from model.gpt2 import GPT2Model
from training.dataset import TinyStoriesDataset

BATCH_SIZE = 256
LEARNING_RATE = 6e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 2000
MAX_STEPS = 175000

torch.set_float32_matmul_precision("high")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
PT_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"Using device: {DEVICE} | Dtype: {PT_DTYPE}")


config = GPT2Config()

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


scaler = torch.amp.GradScaler(enabled=(PT_DTYPE == torch.float16))

print("Preparing datasets and dataloaders...")
tiny_dataset = TinyStoriesDataset(
    data_path="data/train_473992236tkns.bin", block_size=config.block_size
)
train_dataloader = DataLoader(
    tiny_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)


val_tiny_dataset = TinyStoriesDataset(
    data_path="data/val_4765918tkns.bin",
    block_size=config.block_size,
)
val_dataloader = DataLoader(
    val_tiny_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)


@torch.no_grad()
def estimate_val_loss():
    gpt2.eval()
    total_loss = 0
    num_batches = 0
    for i, (x, y) in enumerate(val_dataloader):
        if i >= 100:
            break
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE):
            logits = gpt2(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T))
        total_loss += loss.item()
        num_batches += 1
    gpt2.train()
    return total_loss / num_batches


def get_lr(it, min_lr, max_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    if it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


train_iter = iter(train_dataloader)
gpt2.train()

step = 0
tokens_seen = 0
for step in range(1, MAX_STEPS + 1):
    start_time = time.time()
    # detects the gradients memory and sets it to None
    optimizer.zero_grad(set_to_none=True)

    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        x, y = next(train_iter)

    # Move to device
    x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

    # Adjust learning rate
    lr = get_lr(
        step,
        min_lr=LEARNING_RATE * 0.1,
        max_lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
    )
    for params in optimizer.param_groups:
        params["lr"] = lr

    # forward pass with autocasting
    with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE):
        logits = gpt2(x)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

    # Backward pass and update
    # compute gradients
    scaler.scale(loss).backward()
    # unscale and perform clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(gpt2.parameters(), max_norm=1.0)
    # update parameters
    scaler.step(optimizer)
    scaler.update()

    tokens_in_batch = B * T
    tokens_seen += tokens_in_batch
    end_time = time.time()
    torch.cuda.synchronize()
    dt = end_time - start_time
    tok_per_sec = tokens_in_batch / dt

    print(
        f"Step {step} | "
        f"Tokens: {tokens_seen} | "
        f"Loss: {loss.item():.4f} | "
        f"Speed: {tok_per_sec:.2f} tok/sec"
    )

    if step % 1000 == 0:
        print("Evaluating on validation set...")
        val_loss = estimate_val_loss()
        val_perplexity = torch.exp(torch.tensor(val_loss))
        print(
            f"Validation Loss: {val_loss:.4f} | Validation Perplexity: {val_perplexity:.4f}"
        )

torch.save(gpt2.state_dict(), f"checkpoints/gpt2_{tokens_seen}.pth")
print(f"Model checkpoint saved as checkpoints/gpt2_{tokens_seen}.pth")
