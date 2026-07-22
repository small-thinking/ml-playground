# Tinker SFT + W&B MVP validation — 2026-07-21

Result: **PASS** for end-to-end plumbing. This run validates connectivity,
training control flow, sampling, and metric delivery. It does not measure model
quality.

## Run identity

- Model: `Qwen/Qwen3.5-4B`
- Adaptation: rank-16 LoRA
- Optimizer: Adam, learning rate `1e-4`
- Data: two repository-authored arithmetic smoke examples
- Training: exactly 3 steps, 2 examples per step
- Sampling: one bounded sample before training and one after training
- W&B run: [major-shadow-1](https://wandb.ai/techtao-small-thinking/ml-playground-tinker/runs/3zne613h)
- W&B entity/project: `techtao-small-thinking/ml-playground-tinker`
- W&B state after API readback: `finished`

## Observed metrics

The rows below were read back from the W&B API after the CLI completed, rather
than copied only from local console output.

| Step | Mean cross-entropy loss | Train tokens | Cumulative train tokens | Step time (s) | Estimated cumulative train cost (USD) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.0247530093 | 30 | 30 | 2.1606 | 0.00002211 |
| 2 | 0.0000180995 | 30 | 60 | 2.1246 | 0.00004422 |
| 3 | 0.0000019669 | 30 | 90 | 2.1797 | 0.00006633 |

Additional observations:

- `steps_completed`: 3
- before-sample output tokens: 32
- after-sample output tokens: 32
- estimated total token cost: `$0.00014649`
- preflight maximum: `$0.000714816`
- local hard stop: `$0.01`

The total is a local token-price estimate, not a provider billing record. The
Tinker console remains the source of truth for actual charges.

Both bounded samples reached the 32-token output limit and contained the same
truncated reasoning prefix. That is acceptable for this plumbing smoke test and
is not evidence of model improvement.

## Connectivity finding

The first remote attempt failed during Tinker JWT exchange, before sampling,
training, or W&B initialization, because the SDK's default `pyqwest` transport
reported `invalid peer certificate: UnknownIssuer` on macOS. A read-only
authenticated capabilities request succeeded with standard HTTPX. The MVP now
supplies that verified HTTPX transport explicitly; TLS verification remains
enabled. The successful run used this fix.

The successful run also surfaced two SDK warnings. This diff subsequently
switches the client constructors to their async equivalents and removes the
deprecated, no-op name passed to the ephemeral sampling checkpoint. A second
paid run was intentionally not made for those interface-only warning cleanups.

## Verification commands

```bash
# No-network readiness and budget check
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_mvp

# Explicitly approved remote run
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_mvp \
  --run \
  --allow-paid
```

The remote command exited successfully, and a separate W&B API readback
returned all three metric rows plus the final summary.
