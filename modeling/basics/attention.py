"""Interview-friendly attention implementations.

The goal of this file is not to be feature-complete. It is to provide a few
small, conventional implementations that are easy to study and easy to rewrite
on a whiteboard:

1. single-head causal self-attention
2. multi-head attention (MHA)
3. MHA with KV cache for decoding
4. grouped-query attention (GQA)
5. multi-query attention (MQA) as a special case of GQA
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len, device=None):
    """Return a lower-triangular mask with shape [1, 1, T, T]."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.view(1, 1, seq_len, seq_len)


class SingleHeadAttention(nn.Module):
    """Smallest useful causal self-attention implementation."""

    def __init__(self, d_model, dropout=0.0, bias=True):
        super().__init__()
        self.d_model = d_model
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_model)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            scores = scores.masked_fill(~mask.squeeze(1), torch.finfo(scores.dtype).min)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v
        out = self.wo(out)

        if return_attn:
            return out, attn
        return out


class MHA(nn.Module):
    """Conventional multi-head self-attention.

    Shapes:
    - input x: [B, T, D]
    - q/k/v after reshape: [B, H, T, Dh]
    - output: [B, T, D]
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.wo(out)

        if return_attn:
            return out, attn
        return out


class MHAWithKVCache(MHA):
    """MHA for autoregressive decoding with an explicit KV cache.

    Usage:
    - Step 1: out, k_cache, v_cache = attn(x[:, :1], kv_cache=None)
    - Step 2: out, k_cache, v_cache = attn(x[:, 1:2], kv_cache=(k_cache, v_cache))

    The cache update is simply:
    - new_k_cache = concat(old_k_cache, k_new)
    - new_v_cache = concat(old_v_cache, v_new)
    """

    def forward(
        self,
        x,
        kv_cache=None,
        *,
        k_cache=None,
        v_cache=None,
        return_attn=False,
    ):
        if kv_cache is not None:
            if k_cache is not None or v_cache is not None:
                raise ValueError("Pass either kv_cache or k_cache/v_cache, not both.")
            k_cache, v_cache = kv_cache

        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_new = self.wk(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v_new = self.wv(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        k = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=2)
        v = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.wo(out)

        if return_attn:
            return out, k, v, attn
        return out, k, v


class GroupedQueryAttention(nn.Module):
    """GQA: many query heads, fewer key/value heads.

    Examples:
    - num_heads=8, num_kv_heads=8 -> standard MHA-style head count
    - num_heads=8, num_kv_heads=2 -> GQA
    - num_heads=8, num_kv_heads=1 -> MQA
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.group_size = num_heads // num_kv_heads

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.wo(out)

        if return_attn:
            return out, attn
        return out


class MQA(GroupedQueryAttention):
    """MQA is just GQA with one KV head."""

    def __init__(self, d_model, num_heads, dropout=0.0, bias=True):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=1,
            dropout=dropout,
            bias=bias,
        )


class PositionalEncoding(nn.Module):
    """Batch-first sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Minimal pre-norm transformer block."""

    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MHA(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """Small stack of transformer blocks for playground experiments."""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_len=2048,
        dropout=0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


MultiHeadAttention = MHA
TransformerEncoderBlock = TransformerBlock
