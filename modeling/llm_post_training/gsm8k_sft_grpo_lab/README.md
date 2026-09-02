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
4. Run E0 Base formal evaluation on the complete frozen test partition.
5. Run E1 clean SFT while selecting a checkpoint with SFT validation NLL/PPL.
6. Run E1 formal evaluation with the exact E0 protocol and compare results.
7. Classify the disjoint RL pool as easy, mixed, or hard from E1 rollouts.
8. Run E4 clean GRPO from E1, then E5 bad difficulty, E6 high learning rate,
   and E7 exploitable reward. E8 process-aware reward is optional.
9. Export only the promoted clean SFT and GRPO adapters, compare them locally,
   and publish the final experiment report.

Every paid phase requires an explicit approval, a frozen configuration, a
worst-case cost estimate, and a held-out evaluation. Training loss or proxy
reward alone never promotes a checkpoint.

## Data strategy

GSM8K revision `740312add88f781978c0658806c59bc2815b9866` has 7,473 official
training examples and 1,319 official test examples. The next frozen manifest
will use the following disjoint partitions. It stores hashes and provenance
only; raw questions and answers remain outside Git.

| Partition | Examples | Purpose |
| --- | ---: | --- |
| `sft_train` | 5,000 | SFT gradient updates |
| `sft_validation` | 500 | SFT checkpoint selection by NLL/PPL |
| `rl_train` | 1,800 | GRPO prompts and reward scoring |
| `rl_monitor` | 173 | Fixed G4 RL health monitor; never updated on |
| `calibration_test` | 32 | Completed E0a audit; never a stage-comparison result |
| `formal_test` | 1,287 | Common, unseen Base/SFT/GRPO comparison set |

SFT and RL use different training prompts. GSM8K has enough examples to avoid
reusing SFT questions for the first RL experiment, so an RL gain is less likely
to be simple repetition of the SFT examples. The ground-truth answer remains
available to the RL reward scorer but never appears in the model prompt.

`sft_validation` is not an evaluation benchmark. It is observed while choosing
an SFT checkpoint, so it logs only training diagnostics and cannot be compared
directly with a Base-model generation result. Formal E0 Base, E1 SFT, and E4
GRPO results use exactly the same `formal_test` IDs and decoding protocol; only
the evaluated checkpoint changes.

[`manifests/gsm8k_splits.json`](manifests/gsm8k_splits.json) is the immutable
v2 manifest for these exact IDs. The first 32 deterministically ordered test
IDs are the completed calibration; all other official test IDs form the formal
partition. This retains calibration provenance without using those examples for
model selection or stage comparison.

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
is not a stage-comparison result. E0 Base, E1 SFT, and E4 GRPO use all 1,287
`formal_test` IDs with the same prompt version, `G=4`, decoding limits, parser,
and metric names; only the evaluated checkpoint changes. One G4 rollout group
produces both `pass@1` and `pass@4`; G8 or G16 is unnecessary unless a future
experiment explicitly studies higher-k sampling.

The core metrics are:

- Outcome: `eval/pass_at_1`, `eval/pass_at_4`, `eval/format_accuracy`,
  `eval/truncation_rate`, and `eval/avg_output_tokens`.
- SFT: `train/nll`, `train/perplexity`, `train/learning_rate`, throughput,
  step time, and planned/actual cost. `sft_validation/nll` and
  `sft_validation/perplexity` select checkpoints; the matching formal
  generation metrics are the only metrics compared with E0 Base.
- GRPO signal: `train/reward_mean`, `train/group_mixed_frac`,
  `train/degenerate_group_frac`, `train/effective_group_count`, and
  `train/group_reward_std`.
- Guardrails: `eval/true_exact_match`, process coverage beside process
  validity, policy-drift metrics when the backend exposes them, and separated
  planned/actual token and USD metrics.

Run-level prediction and rollout tables preserve examples behind aggregate
metrics. Dataset manifests, evaluation protocols, promoted checkpoints, and
prediction tables are versioned artifacts.

Long paid commands print a start record, W&B URL, completed prompts and
rollouts, elapsed time, observed token cost, and a final aggregate summary to
the terminal. These updates use stderr so stdout remains a parseable JSON
report. The SFT runner will additionally print train step, NLL, perplexity,
learning rate, and each validation result; it will not print raw prompts or
responses.

## Immediate execution plan

1. Run E0 Base formal evaluation: all 1,287 `formal_test` IDs × G4.
2. Run E1 SFT: one epoch over `sft_train`; validation NLL/PPL selects its
   checkpoint without generating against `formal_test`.
3. Run E1 formal evaluation: the selected checkpoint uses the same E0 IDs,
   prompt, G4, parser, limits, and metric names.

The E0 preflight is local-only:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval \
  --stage formal --hard-cap-usd 7
```

After reviewing its JSON report, run the paid evaluation explicitly:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval \
  --stage formal --run --allow-paid --hard-cap-usd 7
```

## Formal-evaluation cost boundary

The formal protocol requests 1,287 × G4 = 5,148 rollouts per evaluated
checkpoint. The completed 32 × G4 calibration cost $0.08001, so a linear
actual-cost extrapolation is about $3.22 for one full formal evaluation; this
is a planning estimate, not a price guarantee. A worst-case preflight and an
explicit paid-run approval are required before every E0, E1, or E4 evaluation.
The earlier 256 × G4 ($0.64 extrapolated) option is a smaller diagnostic, not
part of the formal protocol.

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
