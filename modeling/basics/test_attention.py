"""
Test script for attention and transformer implementations.
"""

import torch
from attention import (
    MultiHeadAttention,
    GroupedQueryAttention,
    SimpleTransformer,
)


def test_attention_mechanisms():
    """Test all attention mechanisms."""
    print("Testing Attention Mechanisms...")

    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    x = torch.randn(batch_size, seq_len, d_model)

    # Test Multi-Head Attention
    mha = MultiHeadAttention(d_model, num_heads)
    mha_output = mha(x, x, x)
    print(f"MHA output shape: {mha_output.shape}")

    # Test Grouped-Query Attention (MQA case)
    gqa_mqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=1)
    gqa_mqa_output = gqa_mqa(x, x, x)
    print(f"GQA-MQA output shape: {gqa_mqa_output.shape}")

    # Test Grouped-Query Attention (GQA case)
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads=2)
    gqa_output = gqa(x, x, x)
    print(f"GQA output shape: {gqa_output.shape}")

    print("✓ All attention mechanisms work\n")


def test_transformer():
    """Test complete transformer model."""
    print("Testing Transformer...")

    model = SimpleTransformer(
        vocab_size=1000, d_model=256, num_heads=8, num_layers=4, d_ff=1024
    )

    x = torch.randint(0, 1000, (4, 50))
    output = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✓ Transformer works\n")


def test_attention_weights():
    """Test attention weight computation."""
    print("Testing Attention Weights...")

    batch_size, seq_len, d_model, num_heads = 1, 8, 64, 4
    x = torch.randn(batch_size, seq_len, d_model)
    mha = MultiHeadAttention(d_model, num_heads)

    with torch.no_grad():
        q = mha.w_q(x)
        k = mha.w_k(x)
        q = q.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, num_heads, d_model // num_heads).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_model // num_heads) ** 0.5
        attn_weights = torch.softmax(scores, dim=-1)

    print(f"Attention weights shape: {attn_weights.shape}")
    print("✓ Attention weights computed\n")


if __name__ == "__main__":
    test_attention_mechanisms()
    test_transformer()
    test_attention_weights()
    print("All tests passed! 🎉")
