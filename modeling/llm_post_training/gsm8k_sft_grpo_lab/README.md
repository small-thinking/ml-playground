# GSM8K SFT → GRPO Lab

A diagnostics-first post-training study on `Qwen/Qwen3.5-9B-Base`. The question
is not simply which checkpoint scores highest: it is which learning-signal
change explains the result on a fixed GSM8K protocol.

![Formal GSM8K results: Pass@1 and Pass@4 across Base, SFT, and GRPO experiments](figures/gsm8k-posttraining-formal-results-v1.png)

**Current result.** E7's fixed-sign advantage is the formal leader: Pass@1
`0.7675`, Pass@4 `0.7879`. It exceeds E4 clean GRPO by `+4.78pp` and `+3.73pp`.
This is meaningful evidence for the gradient-starvation hypothesis, not yet a
general equal-compute or broad-generalization result.

## Formal scorecard

Every row uses the same `gsm8k-eval-v2-b1922d7384a3` protocol: 1,287 frozen
test prompts, four independent samples per prompt, temperature 1.0, the same
limits, parser, and answer scorer. W&B is the source of the numbers; the
[detailed ledger](experiment_log.md) retains costs, parent checkpoints, and
frozen configurations.

| Experiment | Change / question | Pass@1 | Pass@4 | Decision |
| --- | --- | ---: | ---: | --- |
| E0 Base | What does the untouched base model establish? | 0.6603 | 0.6884 | Baseline. |
| E1 SFT | Does NLL-selected SFT preserve generation quality? | 0.5488 | 0.5843 | No—format improved, but reasoning regressed. |
| E2 SFT | Does a frozen generation monitor select a better SFT state? | 0.6898 | 0.7195 | Yes—promote step 250 as the RL parent. |
| E4 GRPO | Does clean binary-reward group-mean GRPO help E2? | 0.7197 | 0.7506 | Yes—initial GRPO leader. |
| E5 GRPO | Does resampling to pack mixed groups improve E4? | 0.7038 | 0.7296 | No. |
| E6 GRPO | Does restoring E4's mixed-group budget rescue E5? | 0.7034 | 0.7343 | No. |
| E7 GRPO | Do fixed-sign advantages avoid zero-advantage groups? | **0.7675** | **0.7879** | **Yes for one seed; replicate next.** |

## What E7 changed

E4–E6 use group-relative advantages, `r - mean(r)`. With binary rewards, an
all-correct or all-wrong G4 group has zero advantage and is skipped. E7 keeps
the parent, prompts, rollout count, learning rate, monitor, and checkpoint
cadence fixed, but uses `2r - 1`: correct completions receive `+1`, incorrect
ones `-1`. Degenerate groups can therefore train.

| W&B training evidence | E4 | E7 |
| --- | ---: | ---: |
| Selected monitor step | 75 | 100 |
| Selected monitor Pass@1 / Pass@4 | 0.8516 / 0.8594 | 0.9219 / 0.9219 |
| Mixed groups | 52 / 800 | 38 / 800 |
| Groups with nonzero advantage | 52 / 800 | 800 / 800 |
| Optimized input tokens | 54,760 | 696,641 |

E7 preserves the rollout budget but not training compute: it optimized about
12.7× as many input tokens as E4. The direct paired formal comparison favors
E7 on 146 prompts, favors E4 on 98, and ties on 1,043 (`Pass@4` exact sign
test `p=0.00255`; 95% CI for the difference `[+1.36, +6.10]pp`). The next
decision-quality experiment is one independent seed with the same E7 config,
not a hyperparameter sweep.

## Metrics

For a prompt `x` and a sampled completion `y ~ pi(. | x)`, standard Pass@1 is
the probability that one sample is correct. This lab estimates it with the
mean correctness of four independent rollouts per prompt:

`Pass@1 = mean_i mean_j correct(y_ij)`.

For GSM8K's binary exact-answer scorer, this estimate equals the legacy
`eval/exact_match` value. `Pass@4` is the fraction of prompts for which at
least one of the four rollouts is correct. Format accuracy and truncation are
guardrails, not headline metrics: E1 is the counterexample where format
improved while task success fell.

## Reproduce or audit

- [E7 training in W&B](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/p0035t59) and [E7 formal evaluation](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/h3gmmogp)
- [E4 formal comparison](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/2p1o07v4)
- [Frozen split manifest](manifests/gsm8k_splits.json), [evaluation harness](evaluation.py), [training entry point](grpo_train.py), and [full experiment ledger](experiment_log.md)

Use the repository's uv environment for command help and local preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train --help
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.checkpoint_eval --help
```

Paid commands require an explicit `--run --allow-paid` authorization and a
bounded preflight; raw rollouts, local checkpoints, and exports stay in ignored
`outputs/`.
