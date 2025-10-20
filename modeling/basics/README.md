# Attention & Transformer Basics

Minimal implementations for interview purposes.

## Components

- **ScaledDotProductAttention**: Basic attention mechanism
- **MultiHeadAttention**: Multi-head attention with Q, K, V projections
- **GroupedQueryAttention**: Grouped-query attention (GQA) with configurable KV heads
- **PositionalEncoding**: Sinusoidal positional encoding
- **TransformerEncoderBlock**: Self-attention + FFN + layer norm
- **SimpleTransformer**: Complete transformer model

## Usage

```python
from attention import SimpleTransformer, GroupedQueryAttention

# Basic transformer
model = SimpleTransformer(vocab_size=1000, d_model=512, num_heads=8, num_layers=6, d_ff=2048)

# Grouped-query attention (it is also MQA when num_kv_heads=1)
gqa = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=1)  # MQA
gqa_balanced = GroupedQueryAttention(d_model=512, num_heads=8, num_kv_heads=2)  # GQA
```

## Test

```bash
python test_attention.py
```

## Key Concepts

1. **Attention**: Q, K, V matrices and scaled dot-product
2. **Multi-Head**: Parallel attention heads
3. **Grouped-Query**: Query heads grouped, sharing K, V heads (reduces memory/compute)
4. **Positional Encoding**: Sinusoidal encoding for sequence position
5. **Residual Connections**: Skip connections for gradient flow
6. **Layer Normalization**: Normalization for training stability

## Attention Comparison

| Method  | Query Heads | KV Heads | Memory Usage |
| ------- | ----------- | -------- | ------------ |
| **MHA** | 8           | 8        | 100%         |
| **GQA** | 8           | 2        | ~50%         |
| **MQA** | 8           | 1        | ~33%         |

**Benefits**: Memory efficiency, computational efficiency, scalability, maintains quality.
