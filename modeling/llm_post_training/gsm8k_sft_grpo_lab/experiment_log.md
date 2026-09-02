# Experiment Log

Add one row after each calibration or experiment. A failed or inconclusive run
is evidence and must remain recorded.

| Experiment | W&B run | Parent checkpoint | Frozen config | Comparison figure | Decision | Actual cost |
| --- | --- | --- | --- | --- | --- | --- |
| E0a Base calibration (a02) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/3bld9jr8) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v1-4e2019c407d3`, 32 × G4, temp 1.0, 512/512 | [summary](figures/e0a-base-calibration-a02-wandb-summary.png), [charts](figures/e0a-base-calibration-a02-wandb.png) | Superseded by a04: W&B rejected tuple table data after aggregate metrics. | $0.07735 |
| E0a retry (a03) | — | Base model | Same E0a protocol, 32 × G4 | — | Superseded by a04: W&B rejected tuple table columns after sampling and metric logging. | not captured |
| E0a Base calibration (a04) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/kzhvpdh5) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v1-4e2019c407d3`, 32 × G4, temp 1.0, 512/512, commit `fdf6849` | [summary](figures/e0a-base-calibration-a04-wandb-summary.png) | Audit table uploaded. `group_mixed_frac=0`; do not start GRPO from this pool. | $0.08001 |
| E0 Base formal (a01) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/26x556g9) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v2-b1922d7384a3`, 1,287 × G4, temp 1.0, 512/512, manifest `ca90287bd3633d7e3d36703362df9b22a4b3f8d7230ebce7f43fe955258a38d0` | screenshot pending | Formal baseline: pass@1=0.6603, pass@4=0.6884, format=0.7702, truncation=0.2514. Build E1 SFT next. | $3.24310 |
