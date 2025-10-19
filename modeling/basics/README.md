# Attention & Transformer Basics

Minimal implementations for interview purposes.

## Components

- **ScaledDotProductAttention**: Basic attention mechanism
- **MultiHeadAttention**: Multi-head attention with Q, K, V projections
- **PositionalEncoding**: Sinusoidal positional encoding
- **TransformerEncoderBlock**: Self-attention + FFN + layer norm
- **SimpleTransformer**: Complete transformer model

## Usage

```python
from attention import SimpleTransformer

model = SimpleTransformer(vocab_size=1000, d_model=512, num_heads=8, num_layers=6, d_ff=2048)
x = torch.randint(0, 1000, (batch_size, seq_len))
output = model(x)
```

## Test

```bash
python test_attention.py
```

## Key Concepts

1. **Attention**: Q, K, V matrices and scaled dot-product
2. **Multi-Head**: Parallel attention heads
3. **Positional Encoding**: Sinusoidal encoding for sequence position
4. **Residual Connections**: Skip connections for gradient flow
5. **Layer Normalization**: Normalization for training stability
