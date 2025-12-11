# import os
# import numpy as np
# import tiktoken
# from tqdm import tqdm
# from datasets import load_dataset


# num_proc = os.cpu_count()

# dataset = load_dataset("roneneldan/TinyStories", split="train")

# split_datasets = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)

# enc = tiktoken.get_encoding("gpt2")
# eot = enc._special_tokens["<|endoftext|>"]


# def tokenize_function(example):
#     tokens = enc.encode_ordinary(example["text"]) + [eot]
#     return {"tokens": tokens, "length": len(tokens)}


# tokenized_datasets = split_datasets.map(
#     tokenize_function,
#     remove_columns=["text"],
#     num_proc=num_proc,
#     desc="Tokenizing dataset",
# )

# for split, ds in tokenized_datasets.items():
#     arr_len = np.sum(ds["length"], dtype=np.uint64)
#     folder = "data"
#     os.makedirs(folder, exist_ok=True)
#     filename = f"folder/{split}.bin"

#     dtype = np.uint16

#     print(f"Writing {filename} ({arr_len / 1e6:.2f}M tokens)...")
#     arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

#     idx = 0
#     total_batches = 1024
#     for batch_idx in tqdm(range(total_batches), desc=f"Writing {filename}"):
#         batch = ds.shard(
#             num_shards=total_batches, index=batch_idx, contiguous=True
#         ).with_format("numpy")

#         arr_batch = np.concatenate(batch["ids"])

#         arr[idx : idx + len(arr_batch)] = arr_batch  # noqa: E203
#         idx += len(arr_batch)

#     arr.flush()
#     print(f"Saved {filename}")

# print("Done. Ready for training.")
