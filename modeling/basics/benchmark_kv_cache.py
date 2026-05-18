"""Benchmark cached vs uncached autoregressive decoding."""

import argparse
import time

import torch

from modeling.basics.attention import MHA, MHAWithKVCache, causal_mask


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model, device=device)

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

    print(
        f"device={device.type} batch={args.batch_size} seq_len={args.seq_len} "
        f"d_model={args.d_model} n_heads={args.n_heads}"
    )
    print(f"max_output_diff={max_diff:.3e}")
    print(f"without_cache={uncached_s * 1000:.2f} ms")
    print(f"with_cache={cached_s * 1000:.2f} ms")
    print(f"speedup={uncached_s / cached_s:.2f}x")


if __name__ == "__main__":
    main()
