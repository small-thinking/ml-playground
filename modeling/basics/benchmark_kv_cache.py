"""Benchmark cached vs uncached autoregressive decoding."""

import argparse
import time
from dataclasses import dataclass

import torch

from modeling.basics.attention import MHA, MHAWithKVCache, causal_mask


@dataclass(frozen=True)
class DecodeCost:
    batch_size: int
    seq_len: int
    uncached_q_tokens: int
    uncached_kv_tokens: int
    cached_q_tokens: int
    cached_kv_tokens: int
    cached_reused_kv_tokens: int
    uncached_attention_score_pairs: int
    cached_attention_score_pairs: int

    @property
    def cache_hit_rate(self):
        total_context_tokens = self.cached_kv_tokens + self.cached_reused_kv_tokens
        return self.cached_reused_kv_tokens / total_context_tokens

    @property
    def kv_projection_reduction(self):
        return self.uncached_kv_tokens / self.cached_kv_tokens

    @property
    def attention_score_reduction(self):
        return self.uncached_attention_score_pairs / self.cached_attention_score_pairs


def compute_decode_cost(batch_size, seq_len):
    prefix_sum = seq_len * (seq_len + 1) // 2
    prefix_square_sum = seq_len * (seq_len + 1) * (2 * seq_len + 1) // 6
    return DecodeCost(
        batch_size=batch_size,
        seq_len=seq_len,
        uncached_q_tokens=batch_size * prefix_sum,
        uncached_kv_tokens=batch_size * prefix_sum,
        cached_q_tokens=batch_size * seq_len,
        cached_kv_tokens=batch_size * seq_len,
        cached_reused_kv_tokens=batch_size * seq_len * (seq_len - 1) // 2,
        uncached_attention_score_pairs=batch_size * prefix_square_sum,
        cached_attention_score_pairs=batch_size * prefix_sum,
    )


def _sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def decode_without_cache(module, x):
    outputs = []
    for end in range(1, x.size(1) + 1):
        prefix = x[:, :end]
        mask = causal_mask(end, device=x.device)
        outputs.append(module(prefix, mask=mask)[:, -1:, :])
    return torch.cat(outputs, dim=1)


@torch.no_grad()
def decode_with_cache(module, x):
    outputs = []
    k_cache = None
    v_cache = None
    for start in range(x.size(1)):
        out, k_cache, v_cache = module(
            x[:, start : start + 1],
            k_cache=k_cache,
            v_cache=v_cache,
        )
        outputs.append(out)
    return torch.cat(outputs, dim=1)


def benchmark(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    _sync_if_needed(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync_if_needed(device)
    return (time.perf_counter() - start) / iters


def run_case(args, seq_len):
    device = torch.device(args.device)
    x = torch.randn(args.batch_size, seq_len, args.d_model, device=device)

    mha = MHA(args.d_model, args.n_heads).to(device).eval()
    cached = MHAWithKVCache(args.d_model, args.n_heads).to(device).eval()
    cached.load_state_dict(mha.state_dict())

    uncached_out = decode_without_cache(mha, x)
    cached_out = decode_with_cache(cached, x)
    max_diff = (uncached_out - cached_out).abs().max().item()

    uncached_s = benchmark(
        lambda: decode_without_cache(mha, x),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    cached_s = benchmark(
        lambda: decode_with_cache(cached, x),
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )
    return {
        "device": device.type,
        "max_diff": max_diff,
        "uncached_s": uncached_s,
        "cached_s": cached_s,
        "cost": compute_decode_cost(args.batch_size, seq_len),
    }


def print_case(metrics, d_model, n_heads):
    cost = metrics["cost"]
    uncached_ms = metrics["uncached_s"] * 1000
    cached_ms = metrics["cached_s"] * 1000
    print(
        f"device={metrics['device']} batch={cost.batch_size} seq_len={cost.seq_len} "
        f"d_model={d_model} n_heads={n_heads}"
    )
    print(f"max_output_diff={metrics['max_diff']:.3e}")
    print(f"without_cache={uncached_ms:.2f} ms")
    print(f"with_cache={cached_ms:.2f} ms")
    print(f"measured_speedup={metrics['uncached_s'] / metrics['cached_s']:.2f}x")
    print(f"cache_hit_rate={cost.cache_hit_rate:.2%}")
    print(f"kv_projection_reduction={cost.kv_projection_reduction:.2f}x")
    print(f"attention_score_reduction={cost.attention_score_reduction:.2f}x")
    print(f"uncached_kv_tokens={cost.uncached_kv_tokens}")
    print(f"cached_new_kv_tokens={cost.cached_kv_tokens}")
    print(f"cached_reused_kv_tokens={cost.cached_reused_kv_tokens}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seq-lens", type=int, nargs="+")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    seq_lens = args.seq_lens or [args.seq_len]

    for idx, seq_len in enumerate(seq_lens):
        metrics = run_case(args, seq_len)
        print_case(metrics, d_model=args.d_model, n_heads=args.n_heads)
        if idx != len(seq_lens) - 1:
            print()


if __name__ == "__main__":
    main()
