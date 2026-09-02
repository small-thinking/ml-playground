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

### Dashboard hierarchy

The saved W&B view is [GSM8K Base → SFT → GRPO Comparison](https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/workspace?nw=1tk8jr6cvwm).
Its named sections encode this fixed priority:

1. `01 Outcome`: `eval/pass_at_1`, then `eval/pass_at_4`.
2. `02 Reliability`: `eval/format_accuracy`, then `eval/truncation_rate`.
3. Later: GRPO learning signal, then process diagnostics.
4. `run_stats/*` and `tables/rollouts` are supporting evidence, not headline
   panels. They deliberately do not occupy the `eval` section.

Use this saved view for Base/SFT/GRPO screenshots; keep the automated W&B
workspace for ad-hoc debugging. W&B's automatic workspace paginates large
sections, so it cannot guarantee a meaningful panel order.

### Terminal progress

Long paid commands print status to stderr and retain a parseable JSON report
on stdout. Evaluation prints its run URL, completed prompts and rollouts,
elapsed time, observed cost, scoring/upload boundaries, and final pass metrics.

SFT and GRPO use the same cadence. Their progress lines must include current
step or batch out of total, throughput, elapsed time, ETA after the first
measured interval, and cumulative estimated cost. SFT additionally prints
NLL, perplexity, learning rate, and validation boundaries/results. GRPO
additionally prints rollout groups, reward mean, mixed-group fraction, and
degenerate-group fraction. No progress line includes raw prompts or responses.

## Immediate execution plan

1. Run E0 Base formal evaluation: all 1,287 `formal_test` IDs × G4.
2. Run E1 SFT: one epoch over `sft_train`; validation NLL/PPL selects its
   checkpoint without generating against `formal_test`.
3. Run E1 formal evaluation: the selected checkpoint uses the same E0 IDs,
   prompt, G4, parser, limits, and metric names.

## E1 SFT command

E1 is one LoRA SFT epoch over all 5,000 frozen `sft_train` examples: rank 32,
batch size 8, learning rate `5e-4`, and a 1,024-token limit. It writes
`train/nll`, `train/perplexity`, learning rate, throughput, timing, and
estimated cost at every step. It evaluates all 500 `sft_validation` examples
at steps 0, 250, 500, and 625 with forward-only NLL/PPL; the lowest-NLL
checkpoint is exported as the selected sampler path.

The local-only preflight checks the frozen configuration and credentials. It
does not download data or call either remote service:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --hard-cap-usd 12
```

Its current worst-case bound is `$10.486784`: 5,120,000 training-token slots
plus four 500-example validation passes, all at the configured 1,024-token
limit. The paid run is explicit and prints step, ETA, batch-level validation
progress, and cumulative estimated cost to the terminal:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --run --allow-paid --hard-cap-usd 12
```

E1 validation is for checkpoint selection only. Do not compare its NLL/PPL
with E0 generation metrics. After recording the selected sampler path, run a
separate E1 formal evaluation against the unchanged 1,287-example test set.

## E1 formal-evaluation command

E1 formal evaluation accepts only a `sampler_weights` URI, never the larger
training-state URI. It runs the exact E0 formal protocol: all 1,287 frozen test
prompts, G4, temperature 1.0, 512 prompt/output token limits, and the same
parser. Thus its pass@1 and pass@4 are directly comparable with E0.

First run a local-only preflight using the selected checkpoint and its source
training W&B run:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_eval \
  --sampler-path 'tinker://.../sampler_weights/selected-checkpoint' \
  --source-training-run-url 'https://wandb.ai/.../runs/e1-sft-run' \
  --hard-cap-usd 7
```

Then run the explicit paid evaluation with the same arguments plus
`--run --allow-paid`. The separate W&B run records both the SFT source run and
the exact sampler path, so an evaluation cannot silently use a different
checkpoint.

For the first SFT run, retain both selected step-625 checkpoints: the sampler
weights support formal evaluation and inference; the training state supports a
future GRPO branch. Intermediate step-250 and step-500 pairs are diagnostic
only and may be deleted after E1 formal evaluation and checkpoint promotion.

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
