# Experiment Log

Add one row after each calibration or experiment. A failed or inconclusive run
is evidence and must remain recorded.

| Experiment | W&B run | Parent checkpoint | Frozen config | Comparison figure | Decision | Actual cost |
| --- | --- | --- | --- | --- | --- | --- |
| E0a Base calibration (a02) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/3bld9jr8) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v1-4e2019c407d3`, 32 × G4, temp 1.0, 512/512 | [summary](figures/e0a-base-calibration-a02-wandb-summary.png), [charts](figures/e0a-base-calibration-a02-wandb.png) | Superseded by a04: W&B rejected tuple table data after aggregate metrics. | $0.07735 |
| E0a retry (a03) | — | Base model | Same E0a protocol, 32 × G4 | — | Superseded by a04: W&B rejected tuple table columns after sampling and metric logging. | not captured |
| E0a Base calibration (a04) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/kzhvpdh5) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v1-4e2019c407d3`, 32 × G4, temp 1.0, 512/512, commit `fdf6849` | [summary](figures/e0a-base-calibration-a04-wandb-summary.png) | Audit table uploaded. `group_mixed_frac=0`; do not start GRPO from this pool. | $0.08001 |
