# GSM8K SFT → GRPO Lab

This is a small, diagnostics-first post-training laboratory. Its purpose is to
learn why an experiment improved, regressed, or produced no useful learning
signal—not to maximize GSM8K accuracy.

Tinker is the remote training backend. W&B is the experiment record. Neither
tool defines the laboratory's identity or directory name.

## Execution ladder

1. Build the offline safety layer: configuration validation, immutable split
   manifests, a cost gate, and the W&B logging contract.
2. Build one GSM8K evaluation harness: exact-answer parsing, pass@4, format and
   truncation checks, group-difficulty diagnostics, and a conservative
   arithmetic-process diagnostic.
3. Run a small Base-model calibration. Stop if GSM8K is saturated or produces
   too few mixed rollout groups.
4. Run E0 Base, E1 clean SFT, then E2 high-learning-rate SFT and E3 noisy-data
   SFT. Each failure experiment changes one intended variable.
5. Classify the disjoint RL pool as easy, mixed, or hard from E1 rollouts.
6. Run E4 clean GRPO from E1, then E5 bad difficulty, E6 high learning rate,
   and E7 exploitable reward. E8 process-aware reward is optional.
7. Export only the promoted clean SFT and GRPO adapters, compare them locally,
   and publish the final experiment report.

Every paid phase requires an explicit approval, a frozen configuration, a
worst-case cost estimate, and a held-out evaluation. Training loss or proxy
reward alone never promotes a checkpoint.

## Frozen data protocol

[`manifests/gsm8k_splits.json`](manifests/gsm8k_splits.json) pins GSM8K revision
`740312add88f781978c0658806c59bc2815b9866` with 512 SFT IDs, 1,500 disjoint RL
IDs, and 256 held-out test IDs. It contains hashes and provenance only; raw
questions and answers remain outside Git.

[`manifests/gsm8k_profile.json`](manifests/gsm8k_profile.json) records only
split counts, answer-marker coverage, and question/answer length percentiles.

## W&B contract

The W&B project is `mini-posttraining-lab`; related runs share the group
`gsm8k-sft-grpo-v1`. Names begin with their experiment ID, stage, model, and
group size (for example, `e0a-base-calibration-qwen3.5-9b-base-g4`).

Each calibration or experiment is one W&B run with an experiment ID, parent
checkpoint, model identity, dataset revision and split hash, Git commit,
hyperparameters, reward version, hypothesis, expected failure mode, and
planned versus actual cost.

`E0a` is a 32-example calibration subset for a bounded first paid request. It
is not a stage-comparison result. E0 Base, E1 SFT, and E4 GRPO use all 256
held-out IDs with the same prompt version, `G=4`, decoding limits, parser, and
metric names; only the evaluated checkpoint changes.

The core metrics are:

- Outcome: `eval/exact_match`, `eval/pass_at_4`, `eval/format_accuracy`,
  `eval/truncation_rate`, and `eval/avg_output_tokens`.
- SFT: `train/nll` and `train/learning_rate`.
- GRPO signal: `train/reward_mean`, `train/group_mixed_frac`,
  `train/degenerate_group_frac`, `train/effective_group_count`, and
  `train/group_reward_std`.
- Guardrails: `eval/true_exact_match`, process coverage beside process
  validity, policy-drift metrics when the backend exposes them, and separated
  planned/actual token and USD metrics.

Run-level prediction and rollout tables preserve examples behind aggregate
metrics. Dataset manifests, evaluation protocols, promoted checkpoints, and
prediction tables are versioned artifacts.

## Evidence and storage

`figures/` contains the curated W&B comparison screenshots used in the
experiment record. `experiment_log.md` links every screenshot to its run,
configuration, parent checkpoint, and decision. Capture comparisons after each
experiment, including failed or inconclusive runs.

`outputs/` is intentionally ignored. It holds downloaded data, raw rollouts,
local checkpoints, and temporary exports; those may be large or contain more
examples than a curated report needs.

Keep normal screenshots in Git when they are small and useful for review. If a
single image exceeds 1 MB, or the figure history becomes materially large, add
a path-scoped Git LFS rule for the affected binary type before committing it.
Do not put raw W&B exports, model weights, tokens, or sensitive content in
`figures/`.
