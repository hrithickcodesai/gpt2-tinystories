import os
import random

import tiktoken
import torch
import torch.nn.functional as F

from model.config import BabyGPTConfig
from model.gpt2 import GPT2Model

CHECKPOINT_PATH = "checkpoints/best_tinystories_babygpt2122880000.pth"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
seed = 1337
torch.manual_seed(seed)

PROMPTS = [
    "Once upon a time, there was a little",
    "Lily and her dog went to the",
    "The robot wanted to learn how to",
    "Timmy found a shiny blue",
    "In the deep dark forest, a",
    "The sun was shining so bright that",
    "One day, a cat met a",
    "Sarah wanted to bake a big",
    "The little bird could not",
    "Mommy said, 'It is time to",
]


def load_model(checkpoint_path, device):
    config = BabyGPTConfig()
    model = GPT2Model(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            new_key = k[len(unwanted_prefix) :]
            state_dict[new_key] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


@torch.no_grad()
def generate_stream(
    model,
    enc,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
    top_k=50,
    config=BabyGPTConfig(),
):
    tokens = enc.encode_ordinary(prompt)
    x = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    print(f"\nPROMPT: {prompt}")
    print("RESPONSE: ", end="", flush=True)

    eot_token = enc._special_tokens["<|endoftext|>"]

    for _ in range(max_new_tokens):
        if x.shape[1] > config.block_size:
            x_cond = x[:, -config.block_size :]
        else:
            x_cond = x

        logits = model(x_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, next_token), dim=1)
        token_id = next_token.item()
        if token_id == eot_token:
            break
        decoded_token = enc.decode([token_id])
        print(decoded_token, end="", flush=True)


def main():
    enc = tiktoken.get_encoding("gpt2")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return
    model = load_model(CHECKPOINT_PATH, DEVICE)

    prompt = random.choice(PROMPTS)
    generate_stream(
        model,
        enc,
        prompt,
        max_new_tokens=256,
        temperature=0.8,
        top_k=50,
        config=BabyGPTConfig(),
    )


if __name__ == "__main__":
    main()
