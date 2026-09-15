# First OPD800 comparison with traditional KD800

2026-09-14. The user authorized implementing and running OPD, with traditional
KD800 as the primary comparison. Base remains a reference; SFT is not the main
comparison for this experiment. Existing W&B results are retained.

## Fixed protocol

| Setting | Traditional KD800 | OPD800 |
| --- | --- | --- |
| Student initialization | Fresh Qwen3.5-4B LoRA | Same Base, fresh LoRA |
| Teacher | Frozen Qwen3.6-35B-A3B | Same |
| Training images | Fixed Train800 | Same 800, manifest and shuffled batch order |
| LoRA / epochs / batch / updates | rank8 / 1 / 8 / 100 | Same |
| Optimizer | Adam, LR1e-4, warmup10, constant thereafter | Same |
| Trajectories | Teacher greedy outputs, fixed | Current student, temperature1, refreshed every batch |
| Supervision | Renormalized teacher Top10, forward-KL equivalent CE | Sampled immediate reverse-KL, no TopK truncation |
| Updates per trajectory | One | One; no replay or extra mini-epochs |
| Context / output / pixels | 16384 / 8192 / 1048576 | Same |
| Dev gold NLL/PPL | Full100 at0/25/50/75/100 | Same |
| Intermediate weights | Only final saved | Save25/50/75 in addition to final |
| Primary checkpoint | Final step100 | Final step100, fixed before seeing results |
| Final generation evaluation | Full Dev100 and Test100, greedy | Same |

This compares two typical recipes. Trajectory source, training sampling temperature,
KL direction and probability estimator change together; this is not an ablation
isolating only on-policy versus off-policy data. Token counts and costs are measured,
not forced equal. A single seed does not establish a general method ranking.

## OPD update

For each batch, synchronously export current student weights and sample one full
trajectory per image, with temperature1, top_p1 and no TopK sampling cutoff. Do not
prefetch the next batch before the update. Ask the teacher to score those exact
multimodal prefixes, using the verified matching tokenizer and renderer.

For each completion token, freeze
`A_t = log p_teacher(token | prefix) - log p_old_student(token | prefix)`.
Pass the original sampled token and sampling log probability into Tinker's native
`importance_sampling` loss: `-sum(exp(log p_current - log p_old) * A_t / M)`,
where `M` is the batch completion-token count. Prompt/image positions have zero
advantage. Take one optimizer step. Feedback discount is0: this is the immediate
per-token recipe, not a complete sequence-return policy-gradient estimator.

The sampled reverse-KL statistic can be negative in finite samples and is not
perplexity. Log gold Dev NLL/PPL separately, along with sampling entropy estimate,
teacher NLL on student samples, importance-ratio drift, advantage range, rollout
length, format failures, truncation and token/cost totals. OPD and KD training KL
curves use different directions and states, so their numeric levels are not directly
comparable. Compare final generation with the same evaluator instead.

Sources: [Thinking Machines OPD recipe](https://thinkingmachines.ai/blog/on-policy-distillation/)
and [native importance-sampling contract](https://tinker-docs.thinkingmachines.ai/tinker/losses/importance-sampling/).

## Validation and paid execution

First run offline gradient, multimodal shift/masking, protocol and privacy tests.
Then run an8-image/2-update smoke from fresh weights, with full Dev100 likelihood
checks and W&B disabled. Inspect actual sampler/learner agreement, teacher logprobs,
updates, and budget ledger before starting the separate fresh800 run.

At1600 generated tokens/image, full training and Dev are estimated at$4.7824 when
reserving the entire8192-token cap for final Dev generation. If final Dev resembles
prior runs, actual cost should be lower. The all-trajectories-at-cap bound is$16.8168;
that is not an approved spending level. Runtime cumulative training/Dev budget is
$5.50 including10% accounting margin. A batch reserves its capped sampling, teacher
scoring and training before making requests; final Dev generation reserves each
request separately against that same cumulative ledger; uncertain calls stop without automatic
application retries or resuming an optimizer. The smoke is separately capped at
$0.40 and Test evaluation will have its own estimate and at most$1.00 allowance.
The combined configured allowance is$6.90, not a promise that all of it will be spent.
If required reservations cannot fit, stop and report the remaining work rather than
silently shortening outputs, skipping images or increasing the cap.

Pricing checked2026-09-14: [Tinker models](https://tinker-docs.thinkingmachines.ai/tinker/models/).
Estimates use uncached token prices; they are not final bills. The SDK teacher
`compute_logprobs` also samples one token; its cost is included. Checkpoints expire
in7days. Inter-batch synchronization adds latency; smoke timing will refine runtime.

## Entry points and privacy

`opd_targets.py` contains tensor construction, normalization and metrics.
`opd.py` contains synchronous collection, teacher scoring, update and evaluation.
`opd_sampling_budget.py` preserves raw Dev sampling receipts and settles each
request before proceeding. Formal within-batch concurrency is8; updates remain
synchronous. The successful smoke used concurrency4.
Only permitted aggregate metrics and configuration enter W&B. Image paths, prompts,
rollouts, raw probability arrays, sample identifiers and checkpoint addresses stay
local in ignored data/output directories. All actual paths come from CLI parameters.

```bash
PACKAGE=modeling.llm_post_training.vlm_table_extraction_lab
uv run --no-sync python -m "$PACKAGE.opd" \
  --train-manifest "$TRAIN_MANIFEST" --dev-manifest "$DEV_MANIFEST" \
  --data-root "$DATA_ROOT" --output-dir "$NEW_OUTPUT_DIR" \
  --tinker-cookbook-dir "$COOKBOOK_DIR" --official-repo "$RD_REPO" \
  --train-examples 800 --batch-size 8 --epochs 1 --rank 8 \
  --learning-rate 1e-4 --warmup-ratio 0.1 --eval-every 25 \
  --generate-dev --max-estimated-usd 5.50 \
  --dataset-label rd-opd-train800-v1 --wandb-mode online --inference-concurrency 8
# Default is preflight; execute with a separate fresh output directory, adding:
# --execute --env-file "$ENV_FILE"
```

The existing `checkpoint_eval` and `log_checkpoint_eval` commands evaluate the
final sampler and publish it with model_role=opd in the fixed Test100 group. Main
analysis compares OPD800 to KD800 using paired examples and existing KD predictions.
