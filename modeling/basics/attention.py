"""Interview-friendly attention implementations plus small playground helpers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(seq_len, device=None):
    """Lower-triangular mask shaped for attention broadcasting."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask.view(1, 1, seq_len, seq_len)


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention on pre-projected heads."""

    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, return_attn=False):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask != 0
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = attn @ v
        if return_attn:
            return out, attn
        return out


class MHA(nn.Module):
    """Concise multi-head self-attention for interview-style implementations."""

    def __init__(self, d_model, n_heads, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.h = n_heads
        self.d = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(dropout=dropout)

    def _split_heads(self, x):
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.h, self.d).transpose(1, 2)

    def _merge_heads(self, x):
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

    def _project_qkv(self, x):
        return (
            self._split_heads(self.wq(x)),
            self._split_heads(self.wk(x)),
            self._split_heads(self.wv(x)),
        )

    def forward(self, x, mask=None, return_attn=False):
        q, k, v = self._project_qkv(x)
        out = self.attn(q, k, v, mask=mask, return_attn=return_attn)
        if return_attn:
            out, attn = out
        out = self.wo(self._merge_heads(out))
        if return_attn:
            return out, attn
        return out


class MHAWithKVCache(MHA):
    """Incremental decoding MHA that reuses past K/V states."""

    def forward(self, x, k_cache=None, v_cache=None, return_attn=False):
        q, k_new, v_new = self._project_qkv(x)
        k = k_new if k_cache is None else torch.cat([k_cache, k_new], dim=2)
        v = v_new if v_cache is None else torch.cat([v_cache, v_new], dim=2)
        out = self.attn(q, k, v, return_attn=return_attn)
        if return_attn:
            out, attn = out
        out = self.wo(self._merge_heads(out))
        if return_attn:
            return out, k, v, attn
        return out, k, v


class GroupedQueryAttention(nn.Module):
    """GQA/MQA variant with fewer KV heads than query heads."""

    def __init__(self, d_model, num_heads, num_kv_heads=1, dropout=0.0, bias=True):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert (
            num_heads % num_kv_heads == 0
        ), "num_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d = d_model // num_heads
        self.head_ratio = num_heads // num_kv_heads
        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, num_kv_heads * self.d, bias=bias)
        self.wv = nn.Linear(d_model, num_kv_heads * self.d, bias=bias)
        self.wo = nn.Linear(d_model, d_model, bias=bias)
        self.attn = ScaledDotProductAttention(dropout=dropout)

    def forward(self, q, k, v, mask=None):
        bsz, seq_len, _ = q.shape
        q = self.wq(q).view(bsz, seq_len, self.num_heads, self.d).transpose(1, 2)
        k = self.wk(k).view(bsz, seq_len, self.num_kv_heads, self.d).transpose(1, 2)
        v = self.wv(v).view(bsz, seq_len, self.num_kv_heads, self.d).transpose(1, 2)
        k = k.repeat_interleave(self.head_ratio, dim=1)
        v = v.repeat_interleave(self.head_ratio, dim=1)
        out = self.attn(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.wo(out)


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
    """Minimal pre-norm transformer block built on the hand-written MHA."""

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
