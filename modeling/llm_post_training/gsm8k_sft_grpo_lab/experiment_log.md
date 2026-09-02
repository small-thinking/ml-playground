# Experiment Log

Add one row after each calibration or experiment. A failed or inconclusive run
is evidence and must remain recorded.

| Experiment | W&B run | Parent checkpoint | Frozen config | Comparison figure | Decision | Actual cost |
| --- | --- | --- | --- | --- | --- | --- |
| E0a Base calibration (a02) | [run](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/3bld9jr8) | Base model | [`base_eval.py`](base_eval.py): `gsm8k-eval-v1-4e2019c407d3`, 32 × G4, temp 1.0, 512/512 | [summary](figures/e0a-base-calibration-a02-wandb-summary.png), [charts](figures/e0a-base-calibration-a02-wandb.png) | Calibration only: `group_mixed_frac=0`, so do not start GRPO from this pool. The rollout-table upload failed after aggregate metrics; the client conversion is fixed for the next run. | $0.07735 |
