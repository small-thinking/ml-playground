"""
Minimal implementation of attention mechanisms for interview purposes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, num_heads, seq_len, d_k)
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        batch_size = q.size(0)

        # Linear projections
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output = self.attention(q, k, v, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.w_o(attn_output)


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention mechanism.

    In GQA, query heads are grouped and each group shares key and value heads,
    reducing memory usage and computational cost compared to MHA.
    Supports both GQA (num_kv_heads > 1) and MQA (num_kv_heads = 1) variants.
    """

    def __init__(self, d_model, num_heads, num_kv_heads=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        # If num_kv_heads is not specified, use 1 (grouped-query attention)
        if num_kv_heads is None:
            num_kv_heads = 1
        assert d_model % num_kv_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads  # Same dimension as query heads
        self.head_ratio = num_heads // num_kv_heads

        # Query projection for all heads
        self.w_q = nn.Linear(d_model, d_model)
        # Key and value projections for fewer heads
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_v)
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)

    def forward(self, q, k, v, mask=None):
        # q, k, v: (batch_size, seq_len, d_model)
        batch_size, seq_len = q.size(0), q.size(1)

        # Linear projections
        q = self.w_q(q)  # (batch_size, seq_len, d_model)
        k = self.w_k(k)  # (batch_size, seq_len, num_kv_heads * d_k)
        v = self.w_v(v)  # (batch_size, seq_len, num_kv_heads * d_v)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.d_v).transpose(1, 2)

        # Repeat k and v for each query head group
        k = k.repeat_interleave(self.head_ratio, dim=1)
        v = v.repeat_interleave(self.head_ratio, dim=1)

        # Apply attention
        attn_output = self.attention(q, k, v, mask)

        # Concatenate heads
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        return self.w_o(attn_output)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerEncoderBlock(nn.Module):
    """Single transformer encoder block."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class SimpleTransformer(nn.Module):
    """Minimal transformer model for interview purposes."""

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_len=5000,
        dropout=0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (batch_size, seq_len)
        # Input embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x


# Example usage and test
if __name__ == "__main__":
    # Model parameters
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    seq_len = 100
    batch_size = 32

    # Create model
    model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)

    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
