import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, context_window: int, dropout: float):
        super().__init__()
        self.emb_layer = nn.Embedding(vocab_size, embedding_dim)
        self.pe_layer = nn.Embedding(context_window, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        _, T = x.shape
        embedding = self.emb_layer(x)

        indices = torch.arange(T, device=x.device)
        pe_embedding = self.pe_layer(indices)
        x = embedding + pe_embedding
        return self.drop(x)


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim, elementwise_affine=True)

    def forward(self, x: torch.tensor):
        return self.layer_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, n_heads: int, context_window: int, dropout: float):
        super().__init__()
        assert embedding_dim % n_heads == 0, "embedding_dim must be divisible by n_heads"
        self.head_dim = embedding_dim // n_heads
        self.n_heads = n_heads

        # projection layers for query, key, and value
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)

        # causal mask to stop attention to future tokens
        # Need to broadcast later during attention score computation
        # so shape is (1, 1, context_window, context_window)
        mask = torch.tril(torch.ones(context_window, context_window)).view(1, 1, context_window, context_window)
        self.register_buffer("mask", mask)

        # output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        # dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.tensor):
        B, T, C = x.size()

        # each of them has shape (B, n_heads, T, head_dim)
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # compute attention scores
        attn_weights = torch.matmul(q, k.transpose(3, 2)) / (self.head_dim**0.5)

        # apply causal mask
        attn_weights = attn_weights.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # softmax to get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # weigh the values
        attn_output = torch.matmul(attn_weights, v)

        # transpose and reshape back to (B, T, C)
        y = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # final output projection
        output = self.out_proj(y)
        return self.resid_dropout(output)


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.fc2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x: torch.tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        context_window: int,
        dropout: float,
    ):
        super().__init__()
        self.ln1 = LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, n_heads, context_window, dropout)
        self.ln2 = LayerNorm(embedding_dim)
        self.ff = FeedForward(embedding_dim, dropout)

    def forward(self, x: torch.tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int,
        n_head: int,
        n_embd: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, n_embd, block_size, dropout)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weights are shared between embedding and head
        self.head.weight = self.embedding.emb_layer.weight

    def forward(self, x: torch.tensor):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
