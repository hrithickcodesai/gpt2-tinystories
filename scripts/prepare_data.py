import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

num_proc = os.cpu_count()
print(num_proc)

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens["<|endoftext|>"]

dataset = load_dataset("roneneldan/TinyStories")


def tokenize(row):
    tokens = enc.encode_ordinary(row["text"]) + [eot]
    length = len(tokens)
    return {"tokens": tokens, "length": length}


train_ds = dataset["train"]
val_ds = dataset["validation"]


train_tokenized_dataset = train_ds.map(
    tokenize,
    remove_columns=["text"],
    num_proc=num_proc,
    desc="Tokenizing training dataset",
)

val_tokenized_dataset = val_ds.map(
    tokenize,
    remove_columns=["text"],
    num_proc=num_proc,
    desc="Tokenizing validation dataset",
)


print("Total number of tokens in train set:", np.sum(train_tokenized_dataset["length"]))
print("Total number of tokens in val set:", np.sum(val_tokenized_dataset["length"]))


def save_to_binary(dataset, folder_path, split_name):
    arr_len = np.sum(dataset["length"], dtype=np.uint64)
    os.makedirs(folder_path, exist_ok=True)
    filename = f"data/{split_name}_{arr_len}tkns.bin"

    dtype = np.uint16
    print(f"Writing {filename} ({arr_len / 1e6:.2f}M tokens)...")
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

    idx = 0
    total_batches = 1024
    for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
        batch = dataset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")

        arr_batch = np.concatenate(batch["tokens"]).astype(dtype)

        arr[idx : idx + len(arr_batch)] = arr_batch  # noqa: E203
        idx += len(arr_batch)

    arr.flush()
    print(f"Saved {filename}")


save_to_binary(train_tokenized_dataset, "data", "train")
save_to_binary(val_tokenized_dataset, "data", "val")
