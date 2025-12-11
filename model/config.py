from dataclasses import dataclass


@dataclass
class GPTConfig:
    # -------------------------------------------------------------------------
    # GPT-2 Small Config
    # -------------------------------------------------------------------------
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    bias: bool = True

    # -------------------------------------------------------------------------
    # TinyStories Debug Config
    # -------------------------------------------------------------------------
    # block_size: int = 256
    # n_layer: int = 4
    # n_head: int = 4
    # n_embd: int = 128
