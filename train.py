import math
import time

import tiktoken
import torch
import torch.nn.functional as F
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from model.config import BabyGPTConfig
from model.gpt2 import GPT2Model
from training.dataset import TinyStoriesDataset

load_dotenv()

BATCH_SIZE = 16
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 1000
MAX_STEPS = 15000
NUM_WORKERS = 4

# set up tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

# weights and biases
WANDB_ENABLED = True
WANDB_PROJECT = "tinystories-gpt2"

# speed up performance on Ampere GPUs by using TF32 for matrix multiplications.
torch.set_float32_matmul_precision("high")

# check device if GPU is available, otherwise use CPU/MPS
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# use bfloat16 if supported, otherwise fall back to float16 for mixed precision training
USE_BFLOAT16 = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
PT_DTYPE = torch.bfloat16 if USE_BFLOAT16 else torch.float16
print(f"Using device: {DEVICE} | Dtype: {PT_DTYPE}")


def main():
    # init wandb for experiment tracking and logging
    wandb.init(
        project=WANDB_PROJECT,
        config={
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_steps": MAX_STEPS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "architecture": "GPT2",
            "dataset": "TinyStories",
            "device": DEVICE,
            "precision": str(PT_DTYPE),
        },
        mode="disabled" if not WANDB_ENABLED else "online",
    )

    # init the config and load the model
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

    # move the model to device and kernal fusion using torch.compile for faster training
    print("Moving model to device and compiling...")
    gpt2.to(DEVICE)
    gpt2 = torch.compile(gpt2)

    # AdamW optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(gpt2.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # turn on grad scaler for mixed precision training
    # should only be enabled if using fp16 otherwise for bf16 and fp32 it is not needed
    scaler = torch.amp.GradScaler(enabled=(PT_DTYPE == torch.float16))

    print("Preparing datasets and dataloaders...")
    tiny_dataset = TinyStoriesDataset(
        data_path="data/train_473992236tkns.bin",
        block_size=config.block_size,
    )
    train_dataloader = DataLoader(
        tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=2,
    )

    val_tiny_dataset = TinyStoriesDataset(
        data_path="data/val_4765918tkns.bin",
        block_size=config.block_size,
    )
    val_dataloader = DataLoader(
        val_tiny_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        prefetch_factor=2,
    )

    @torch.no_grad()
    def estimate_val_loss():
        gpt2.eval()
        total_loss = 0
        num_batches = 0
        for i, (x, y) in enumerate(tqdm(val_dataloader)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE):
                logits = gpt2(x)
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, C), y.reshape(B * T))
            total_loss += loss.item()
            num_batches += 1
        gpt2.train()
        return total_loss / num_batches

    @torch.no_grad()
    def inference(prompt, max_length=100, temperature=1.0, top_k=50):
        gpt2.eval()
        tokens = enc.encode_ordinary(prompt)
        generated = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(DEVICE)

        for _ in range(max_length - len(tokens)):
            idx_cond = generated[:, -config.block_size :]

            with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE):
                logits = gpt2(idx_cond)
                logits = logits[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if next_token.item() == eot:
                break

        gpt2.train()
        return enc.decode(generated[0].tolist())

    # we need to check this to make sure weight decay is not broken
    @torch.no_grad()
    def get_model_param_norm(model):
        norm = 0.0
        for p in model.parameters():
            norm += p.detach().data.norm(2).item() ** 2
        return norm**0.5

    def get_lr(it, min_lr, max_lr, warmup_steps, max_steps):
        if it < warmup_steps:
            return max_lr * it / warmup_steps
        if it > max_steps:
            return min_lr

        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

    train_iter = iter(train_dataloader)

    sample_prompts = [
        "Once upon a time, there was a little",
        "Lily and her dog went to the",
        "The robot wanted to learn how to",
    ]
    gpt2.train()

    print(f"Starting training for {MAX_STEPS} steps with batch size: {BATCH_SIZE}")
    print(f"Total training tokens: {MAX_STEPS * BATCH_SIZE * config.block_size / 1e6:.2f}M")
    step = 0
    tokens_seen = 0
    best_val_loss = float("inf")

    pbar = tqdm(range(1, MAX_STEPS + 1), desc="Training", position=0)
    stats = tqdm(total=0, bar_format="{desc}", position=1)
    for step in pbar:
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

        # enable autocast only if in fp16 mode, for bf16 and fp32 it is not needed
        with torch.autocast(device_type=DEVICE, dtype=PT_DTYPE, enabled=(PT_DTYPE == torch.float16)):
            logits = gpt2(x)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))

        # Backward pass and update
        # compute gradients
        # for docs: https://docs.pytorch.org/docs/stable/notes/amp_examples.html#gradient-clipping
        scaler.scale(loss).backward()
        # unscale and perform clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(gpt2.parameters(), max_norm=1.0)
        # update parameters
        scaler.step(optimizer)
        scaler.update()

        tokens_in_batch = B * T
        tokens_seen += tokens_in_batch
        end_time = time.time()
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        dt = end_time - start_time
        tok_per_sec = tokens_in_batch / dt

        wandb.log(
            {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/grad_norm": grad_norm.item(),
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen": tokens_seen,
                "train/param_norm": get_model_param_norm(gpt2),
                "system/dt": dt * 1000,
                "system/gpu_mem_allocated(gb)": torch.cuda.max_memory_allocated() / 1e9,  # GB
            },
            step=step,
        )

        logs = f"Step: {step} | Loss: {loss.item():.4f} | LR: {lr:.2e} | Grad Norm: {grad_norm:.2f} | Tokens/sec: {tok_per_sec:.2f} | Tokens Seen: {tokens_seen}"
        stats.set_description_str(logs)

        if step % 1000 == 0:
            pbar.write("Evaluating on validation set...")
            val_loss = estimate_val_loss()
            val_perplexity = torch.exp(torch.tensor(val_loss))
            pbar.write(f"Validation Loss: {val_loss:.4f} | Validation Perplexity: {val_perplexity:.4f}")
            wandb.log(
                {
                    "val/loss": val_loss,
                    "val/perplexity": val_perplexity.item(),
                },
                step=step,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = f"checkpoints/best_tinystories_babygpt2{tokens_seen}.pth"
                pbar.write(f"Found new best val_loss at step {step}: {val_loss:.4f} | Saving: {checkpoint_path}")
                torch.save(gpt2.state_dict(), checkpoint_path)

            pbar.write(f"Running inference on sample prompts at step {step}...")
            for prompt in sample_prompts:
                completion = inference(prompt, max_length=100)
                output_text = f"PROMPT: {prompt}\nOUTPUT: {completion}\n{'-' * 20}"
                pbar.write(output_text)
                pbar.write("=" * 30 + "\n")

    torch.save(gpt2.state_dict(), f"checkpoints/tinystories_babygpt2{tokens_seen}.pth")
    pbar.write(f"Model checkpoint saved as checkpoints/tinystories_babygpt2{tokens_seen}.pth")


if __name__ == "__main__":
    main()
