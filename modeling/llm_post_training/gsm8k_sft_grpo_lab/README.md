# GSM8K SFT → GRPO Lab

A diagnostics-first post-training study on `Qwen/Qwen3.5-9B-Base`. The question
is not simply which checkpoint scores highest: it is which learning-signal
change explains the result on a fixed GSM8K protocol.

![Formal GSM8K Pass@1 and Pass@4 for the E2-to-E7 ablation path](figures/gsm8k-posttraining-formal-results-v1.png)

**Current result.** E7's fixed-sign advantage is the formal leader: Pass@1
`0.7675`, Pass@4 `0.7879`. It exceeds E4 clean GRPO by `+4.78pp` and `+3.73pp`.
This is meaningful evidence for the gradient-starvation hypothesis, not yet a
general equal-compute or broad-generalization result.

## Main results

Every row uses the same `gsm8k-eval-v2-b1922d7384a3` protocol: 1,287 frozen
test prompts, four independent samples per prompt, temperature 1.0, the same
limits, parser, and answer scorer. W&B is the source of the numbers; the
[detailed ledger](experiment_log.md) retains costs, parent checkpoints, and
frozen configurations.

| Experiment | Change / question | Pass@1 | Pass@4 | Decision |
| --- | --- | ---: | ---: | --- |
| E0 Base | What does the untouched base model establish? | 0.6603 | 0.6884 | Baseline. |
| E2 SFT | Does a frozen generation monitor select a better SFT state? | 0.6898 | 0.7195 | Yes—promote step 250 as the RL parent. |
| E4 GRPO | Does clean binary-reward group-mean GRPO help E2? | 0.7197 | 0.7506 | Yes—initial GRPO leader. |
| E5 GRPO | Does resampling to pack mixed groups improve E4? | 0.7038 | 0.7296 | No. |
| E6 GRPO | Does restoring E4's mixed-group budget rescue E5? | 0.7034 | 0.7343 | No. |
| E7 GRPO | Do fixed-sign advantages avoid zero-advantage groups? | **0.7675** | **0.7879** | **Yes for one seed; replicate next.** |

**Teaching note — E1, deliberately excluded from the main figure.** Selecting
SFT by NLL alone gave Pass@1/Pass@4 `0.5488/0.5843` despite format accuracy
`0.9905`. Keep it as the teaching counterexample: lower teacher-forcing loss
and cleaner formatting are not valid generation-quality selection criteria.

## Table 1 — controlled GRPO ablations after E2

All rows start from E2 step 250, use G4, LR `2e-5`, exact binary reward, the
same 64-prompt monitor, and the same formal evaluation. The changed component
and learning-signal budget—not the experiment ID—define the ablation.

| ID | Method name | Changed component | Training signal | Selected step | Formal Pass@1 / Pass@4 | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| E4 | **Clean GRPO** | Baseline group-mean advantage, `r - mean(r)` | 52 / 800 mixed groups | 75 | 0.7197 / 0.7506 | Initial leader. |
| E5 | **Signal-packed GRPO** | Up to four fresh batches; target two mixed groups per optimizer step | 37 / 688 mixed groups | 15 | 0.7038 / 0.7296 | More per-step signal did not help. |
| E6 | **Fixed-effective-budget GRPO** | E5 packing, continued to a 56-mixed-group target (1,200 candidate cap) | 56 / 1,128 mixed groups | 25 | 0.7034 / 0.7343 | Matching E4's total mixed signal did not help. |
| E7 | **Fixed-sign-advantage GRPO** | `2r - 1`; no mixed-group resampling | 38 / 800 mixed; **800 / 800 nonzero-advantage** | 100 | **0.7675 / 0.7879** | New leader; replicate. |

E4–E6 skip all-correct and all-wrong groups because their group-relative
advantage is zero. E7 lets those degenerate groups train: correct completions
receive `+1`, incorrect ones `-1`. It therefore preserves the rollout budget
but not training compute—E7 optimized about 12.7× as many input tokens as E4.
The direct paired formal comparison favors E7 on 146 prompts, favors E4 on 98,
and ties on 1,043 (`Pass@4` exact sign test `p=0.00255`; 95% CI
`[+1.36, +6.10]pp`). The next decision-quality experiment is one independent
seed with the same E7 config, not a hyperparameter sweep.

## E8 — pre-registered compute-matched fixed-sign control

E8 asks a narrower question than E7: does fixed-sign learning help when its
**optimized input-token** budget is approximately E4's? It retains E2 step 250,
G4, LR `2e-5`, temperature `1.0`, exact binary rewards, and fixed-sign
advantages, but runs exactly eight steps. This was fixed before the run from
the completed E7 accounting: `54,760 / (696,641 / 100) = 7.86`, so eight steps
target about `55,731` optimized input tokens versus E4's `54,760` (+1.8%).
The realized count is logged to W&B and the local report because completion
lengths vary.

E8 matches **optimization compute**, not the rollout count: it will sample 256
rollouts, versus E4's 3,200. It tests whether fixed-sign provides a better
learning signal per optimized token. A later rollout-matched sparse-update
control is needed to isolate that separate axis.

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

E8's no-network preflight and paid command are:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --experiment-id e8 --advantage-estimator fixed-sign --seed 20260901 \
  --hard-cap-usd 1.50

UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --run --allow-paid --experiment-id e8 --advantage-estimator fixed-sign \
  --seed 20260901 --hard-cap-usd 1.50
```

Paid commands require an explicit `--run --allow-paid` authorization and a
bounded preflight; raw rollouts, local checkpoints, and exports stay in ignored
`outputs/`.
