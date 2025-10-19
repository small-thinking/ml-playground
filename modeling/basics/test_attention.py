"""
Test script for attention and transformer implementations.
"""

import torch
from attention import ScaledDotProductAttention, MultiHeadAttention, SimpleTransformer


def test_scaled_dot_product_attention():
    """Test basic attention mechanism."""
    print("Testing Scaled Dot-Product Attention...")

    batch_size, seq_len, d_k = 2, 10, 64
    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_k)

    attention = ScaledDotProductAttention()
    output = attention(q, k, v)

    print(f"Input shapes: Q{q.shape}, K{k.shape}, V{v.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Scaled dot-product attention works\n")


def test_multihead_attention():
    """Test multi-head attention."""
    print("Testing Multi-Head Attention...")

    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)
    output = mha(x, x, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✓ Multi-head attention works\n")


def test_transformer():
    """Test complete transformer model."""
    print("Testing Simple Transformer...")

    # Model parameters
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 1024
    seq_len = 50
    batch_size = 4

    # Create model
    model = SimpleTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)

    # Test forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Transformer model works\n")


def test_attention_visualization():
    """Test attention weights visualization."""
    print("Testing attention weights...")

    batch_size, seq_len, d_model, num_heads = 1, 8, 64, 4
    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model, num_heads)

    # Get attention weights by modifying the attention module
    with torch.no_grad():
        q = mha.w_q(x)
        k = mha.w_k(x)

        q = q.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model // num_heads) ** 0.5
        attn_weights = torch.softmax(scores, dim=-1)

        print(f"Attention weights shape: {attn_weights.shape}")
        print("Sample attention weights (head 0):")
        print(attn_weights[0, 0, :, :].numpy())
        print("✓ Attention weights computed\n")


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_multihead_attention()
    test_transformer()
    test_attention_visualization()

    print("All tests passed! 🎉")
