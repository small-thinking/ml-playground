import torch

from modeling.basics.attention import (
    MHA,
    MHAWithKVCache,
    GroupedQueryAttention,
    MQA,
    PositionalEncoding,
    SingleHeadAttention,
    SimpleTransformer,
    TransformerBlock,
    causal_mask,
)


def test_single_head_attention_output_shape():
    x = torch.randn(2, 5, 32)
    attn = SingleHeadAttention(d_model=32)
    out = attn(x, mask=causal_mask(x.size(1)))
    assert out.shape == x.shape


def test_mha_output_shape():
    x = torch.randn(2, 7, 32)
    mha = MHA(d_model=32, n_heads=4)
    out = mha(x)
    assert out.shape == x.shape


def test_attention_weights_sum_to_one():
    x = torch.randn(2, 5, 32)
    mha = MHA(d_model=32, n_heads=4)
    _, attn = mha(x, return_attn=True)
    probs = attn.sum(dim=-1)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)


def test_kv_cache_matches_full_causal_attention():
    torch.manual_seed(0)
    x = torch.randn(2, 9, 64)

    full = MHA(d_model=64, n_heads=4, dropout=0.0).eval()
    cached = MHAWithKVCache(d_model=64, n_heads=4, dropout=0.0).eval()
    cached.load_state_dict(full.state_dict())

    expected = full(x, mask=causal_mask(x.size(1)))

    outputs = []
    k_cache = None
    v_cache = None
    for idx in range(x.size(1)):
        out, k_cache, v_cache = cached(
            x[:, idx : idx + 1],
            k_cache=k_cache,
            v_cache=v_cache,
        )
        outputs.append(out)

    actual = torch.cat(outputs, dim=1)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_kv_cache_tuple_api_grows_over_time():
    x = torch.randn(2, 3, 32)
    cached = MHAWithKVCache(d_model=32, n_heads=4, dropout=0.0).eval()

    out1, k1, v1 = cached(x[:, :1], kv_cache=None)
    out2, k2, v2 = cached(x[:, 1:2], kv_cache=(k1, v1))

    assert out1.shape == (2, 1, 32)
    assert out2.shape == (2, 1, 32)
    assert k1.shape == (2, 4, 1, 8)
    assert v1.shape == (2, 4, 1, 8)
    assert k2.shape == (2, 4, 2, 8)
    assert v2.shape == (2, 4, 2, 8)


def test_grouped_query_attention_output_shape():
    x = torch.randn(2, 6, 32)
    gqa = GroupedQueryAttention(d_model=32, num_heads=4, num_kv_heads=2)
    out = gqa(x, mask=causal_mask(x.size(1)))
    assert out.shape == x.shape


def test_mqa_is_gqa_with_one_kv_head():
    x = torch.randn(2, 6, 32)
    mqa = MQA(d_model=32, num_heads=4)
    out = mqa(x, mask=causal_mask(x.size(1)))
    assert out.shape == x.shape


def test_positional_encoding_is_batch_first():
    x = torch.zeros(3, 6, 16)
    pe = PositionalEncoding(d_model=16, max_len=32)
    out = pe(x)
    assert out.shape == x.shape
    assert torch.allclose(out[0, 0], out[1, 0])
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_transformer_block_preserves_shape():
    x = torch.randn(2, 8, 32)
    block = TransformerBlock(d_model=32, n_heads=4, hidden_dim=64)
    out = block(x, mask=causal_mask(x.size(1)))
    assert out.shape == x.shape


def test_simple_transformer_output_shape():
    model = SimpleTransformer(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ff=64,
    )
    tokens = torch.randint(0, 100, (2, 10))
    out = model(tokens, mask=causal_mask(tokens.size(1)))
    assert out.shape == (2, 10, 32)
