# Attention & Transformer Basics

This folder now has two layers:

- the interview core: `MHA`, `MHAWithKVCache`, `TransformerBlock`
- the playground layer: tests, a benchmark script, and a small transformer stack

## Interview Core

```python
from modeling.basics.attention import MHA, MHAWithKVCache, TransformerBlock, causal_mask

x = torch.randn(2, 16, 128)
mha = MHA(d_model=128, n_heads=8)
y = mha(x, mask=causal_mask(x.size(1)))

cached = MHAWithKVCache(d_model=128, n_heads=8)
step_out, k_cache, v_cache = cached(x[:, :1])

block = TransformerBlock(d_model=128, n_heads=8, hidden_dim=512)
z = block(x, mask=causal_mask(x.size(1)))
```

The idea is to keep the live-coding path short:

1. split `Q/K/V` into heads
2. compute `QK^T / sqrt(d)`
3. `softmax`
4. weight `V`
5. merge heads and project out

`MHAWithKVCache` keeps the same core, but appends new `K/V` to the cache during decoding so each step only projects the newest token.

## Playground Extras

- `GroupedQueryAttention`: a compact GQA/MQA variant
- `PositionalEncoding`: batch-first sinusoidal encoding
- `SimpleTransformer`: stack of hand-written transformer blocks
- `benchmark_kv_cache.py`: compares autoregressive decoding with and without KV cache

## Test

```bash
uv run pytest tests/modeling/basics/test_attention.py
```

The tests check:

- output shapes
- attention probability normalization
- cached decoding matches full causal attention
- positional encoding uses batch-first shapes correctly

## Benchmark

```bash
uv run python -m modeling.basics.benchmark_kv_cache --device cpu
```

Or sweep a few decode lengths in one run:

```bash
uv run python -m modeling.basics.benchmark_kv_cache --device cpu --seq-lens 32 64 128
```

This benchmark measures a realistic interview talking point:

- without cache: recompute attention over the whole prefix at every decode step
- with cache: reuse old `K/V` and only process the newest token

It also prints:

- `max_output_diff`: verifies the cached path stays numerically aligned with full causal attention
- `measured_speedup`: the end-to-end timing difference on your device
- `cache_hit_rate`: what fraction of attended `K/V` tokens came from cache instead of being newly projected
- `kv_projection_reduction`: how much `K/V` projection work drops in the cached path
- `attention_score_reduction`: how much attention-score work drops when we stop recomputing old query positions

That makes it easier to explain both the empirical timing result and the theoretical reason KV cache helps.
