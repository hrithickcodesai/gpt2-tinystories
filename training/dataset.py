import numpy as np
import torch
from torch.utils.data import Dataset


class TinyStoriesDataset(Dataset):
    def __init__(self, data_path: str, block_size: int):
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")
        self.length = len(self.data)
        self.block_size = block_size

    def __len__(self):
        return (self.length - 1) // self.block_size

    def __getitem__(self, idx: int):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        chunk = self.data[start_idx:end_idx]
        chunk = torch.from_numpy(chunk.astype(np.int64))
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
