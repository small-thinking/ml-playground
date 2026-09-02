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
7. Treat E1 as a regression: NLL/PPL selection alone did not preserve generation
   quality. Add generation validation before attempting another SFT configuration.
8. Classify the disjoint RL pool only after an SFT checkpoint is selected by
   generation validation.
9. Run E4 clean GRPO from the promoted SFT checkpoint, then E5 signal-aware
   GRPO with bounded resampling and monitor-based early stopping, E6 high learning rate,
   and E7 exploitable reward. E8 process-aware reward is optional.
10. Export only the promoted clean SFT and GRPO adapters, compare them locally,
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
| `sft_validation` | 500 | SFT NLL/PPL diagnostics; E2 uses a fixed 128-prompt subset for generation-based checkpoint selection |
| `rl_train` | 1,800 | GRPO prompts and reward scoring |
| `rl_monitor` | 173 | Fixed G4 RL health monitor; never updated on |
| `calibration_test` | 32 | Completed E0a audit; never a stage-comparison result |
| `formal_test` | 1,287 | Common, unseen Base/SFT/GRPO comparison set |

SFT and RL use different training prompts. GSM8K has enough examples to avoid
reusing SFT questions for the first RL experiment, so an RL gain is less likely
to be simple repetition of the SFT examples. The ground-truth answer remains
available to the RL reward scorer but never appears in the model prompt.

`sft_validation` is not an evaluation benchmark. It is observed while choosing
an SFT checkpoint; E2 uses a fixed subset for generation-based selection, not
for a headline Base/SFT comparison. Formal E0 Base, E1 SFT, and E4 GRPO results
use exactly the same `formal_test` IDs and decoding protocol; only the evaluated
checkpoint changes.

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
  step time, and planned/actual cost. E1 selected checkpoints by
  `sft_validation/nll`/perplexity and regressed formally. E2 additionally logs
  `sft_generation_validation/pass_at_1`, `pass_at_4`, format, truncation, and
  output length on a frozen subset, and selects by pass@4, then pass@1, then
  NLL. E3 also logs its best-so-far generation score, deltas, regression
  streak, and explicit stop decision. Only a matching formal generation run is
  compared with E0 Base.
- GRPO signal: `train/reward_mean`, `train/group_mixed_frac`,
  `train/degenerate_group_frac`, `train/effective_group_count`,
  `train/candidate_group_count`, `train/resample_rounds`, and
  `train/group_reward_std`. E5 also logs its effective-group target and whether
  the bounded resampling budget reached it.
- GRPO selection: fixed `rl_monitor/pass_at_4`, then pass@1, records every
  checkpoint table including step 0, and logs best-so-far monitor scores,
  material-regression streak, and stop decision when early stopping is enabled.
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
2. E1 showed that NLL/PPL alone is insufficient for checkpoint selection:
   its SFT checkpoint regressed on the formal generation test despite lower NLL.
3. E2 selected step 250 by its frozen generation monitor and improved the
   shared formal protocol over Base. Treat that result as a useful scoreboard,
   not an endlessly reusable pristine test set.
4. Run E4 clean GRPO from E2 step 250 on `rl_train`; use the disjoint
   `rl_monitor` only for checkpoint selection, then formally evaluate the
   selected checkpoint on the shared `formal_test` partition.
5. E5 directly targets E4's sparse binary-reward signal: it resamples fresh
   RL prompts only until it obtains two mixed groups or exhausts a fixed budget,
   and stops only after repeated material monitor regressions.

## E4 clean-GRPO command

E4 defaults to restoring E2 step 250 from its Tinker **training-state** URI
with a fresh RL optimizer. This means its initial policy is the promoted SFT
adapter, not bare Qwen Base. It never puts a GSM8K answer in the model prompt.
For each prompt it samples G4 on-policy completions, gives an exact
final-answer reward of `1` or `0`, subtracts that group's mean reward, skips
all-correct and all-wrong groups, and uses the saved rollout log-probabilities
with `importance_sampling`.

The initial economical run is 100 updates of 8 prompts (800 deterministic
`rl_train` prompts) at LR `2e-5`. It uses the first 64 frozen `rl_monitor`
prompts at step 0 and every 25 steps; monitor pass@4, then pass@1, selects a
saved checkpoint. It logs reward, mixed/degenerate groups, effective-group
count, output/format diagnostics, rollout throughput, ETA, and cumulative
estimated token cost. The four persisted checkpoint pairs have the normal
30-day Tinker TTL.

Review the entirely local preflight first:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --steps 100 --batch-size 8 --group-size 4 --learning-rate 2e-5 \
  --monitor-examples 64 --checkpoint-every 25 --hard-cap-usd 12
```

Its worst-case token bound is `$10.8838912`; the bound assumes every prompt
and completion reaches 512 tokens, so it is deliberately higher than the
likely charge. After reviewing that JSON, run the same configuration with the
explicit paid-run gate:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --steps 100 --batch-size 8 --group-size 4 --learning-rate 2e-5 \
  --monitor-examples 64 --checkpoint-every 25 --hard-cap-usd 12 \
  --run --allow-paid
```

`--steps`, `--batch-size`, `--group-size` (minimum four),
`--learning-rate`, `--temperature`, `--max-prompt-tokens`,
`--max-output-tokens`, `--monitor-examples`,
`--checkpoint-every`, `--hard-cap-usd`, and all E2 parent paths are explicit
CLI parameters. Changing a setting creates a distinct W&B run name and config.

## E4 selected-checkpoint formal evaluation

E4's fixed monitor selected step 75, not the final step 100. Evaluate that
exact **sampler** checkpoint—not its 1.1 GB training-state checkpoint—with the
same 1,287-prompt G4 formal protocol used for Base and SFT. This is the only
comparison that can establish whether E4 improved the shared scoreboard.

Preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.checkpoint_eval \
  --sampler-path 'tinker://fe6861a7-c997-538b-807f-a1e2f8e2fa2c:train:0/sampler_weights/e4-grpo-qwen-qwen3-5-9b-base-r32-b8-g4-lr2e-5-s100-m64-a01-step75' \
  --source-training-run-url 'https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/uyou2i6z' \
  --experiment-id e4 --evaluation-stage grpo \
  --parent-checkpoint e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250 \
  --hard-cap-usd 7
```

Append `--run --allow-paid` only after the local report is approved. The
preflight maximum is `$6.99798528`; the prior complete formal evaluations cost
materially less because most completions were shorter than 512 tokens.

## E5 signal-aware GRPO command

E4 produced an update on only 46 of 100 steps because 93.5% of its G4 prompt
groups were all correct or all wrong. E5 keeps the parent, reward, LR, and G4
fixed. It requests up to four fresh 8-prompt batches per optimizer step, but
stops sampling that step once it has two mixed groups; all candidates remain
on-policy and only mixed groups become GRPO datums. It does not reuse the
`rl_monitor` for gradients.

At step 0 and every five steps, E5 evaluates the same 64 frozen monitor
prompts. It stops only after two consecutive checkpoints where both pass@1 and
pass@4 fall by more than `0.03125` (two of 64 prompts) from the best monitor
score. A stop reduces actual spend; the preflight bound still assumes all 25
steps and all four sampling rounds.

Preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --experiment-id e5 --steps 25 --batch-size 8 --group-size 4 \
  --learning-rate 2e-5 --monitor-examples 64 --checkpoint-every 5 \
  --min-effective-groups 2 --max-resample-rounds 3 \
  --early-stopping-patience 2 --early-stopping-max-regression 0.03125 \
  --hard-cap-usd 14
```

The verified worst-case bound is `$11.23188736` (3,200 possible training
rollouts plus six monitor passes). Paid run:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train \
  --experiment-id e5 --steps 25 --batch-size 8 --group-size 4 \
  --learning-rate 2e-5 --monitor-examples 64 --checkpoint-every 5 \
  --min-effective-groups 2 --max-resample-rounds 3 \
  --early-stopping-patience 2 --early-stopping-max-regression 0.03125 \
  --hard-cap-usd 14 --run --allow-paid
```

### Initialization ablations

`--init-source sft` is the default. It requires a matching training-state URI
and sampler URI, so any future SFT checkpoint can be substituted with
`--parent-state-path`, `--parent-sampler-path`, and `--init-label`.

`--init-source base` instead creates a fresh rank-32 LoRA on `--model-id` and
uses the bare model as the step-0 monitor. It ignores the SFT parent paths and
names the W&B run with `from-base` (or a supplied `--init-label`). This is the
direct Base→GRPO ablation; it must use the same `rl_train`, `rl_monitor`, and
eventual formal protocol as the SFT→GRPO condition.

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

## E2 SFT command

E2 changes the training control rather than the frozen data: rank 32, batch 8,
one 5,000-example epoch, and the worked-solution target remain fixed. Its peak
LR is `3e-4` with linear decay to 1% of the peak. This is a conservative trial
below Tinker's roughly `4.73e-4` Qwen3.5-9B LoRA starting-point formula, not a
claim that E1's `5e-4` was itself invalid.

At step 0 and steps 125/250/375/500/625, E2 samples a deterministic 128-prompt
prefix of `sft_validation` with G4, temperature 1.0, and the same 512-token
generation limits as formal evaluation. Its ID hash is stored in W&B config.
The training run selects the checkpoint by monitor pass@4, then pass@1, then
NLL. The complete 1,287-prompt `formal_test` remains untouched until one
checkpoint is selected.

Run a local-only preflight first:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --recipe e2
```

The current worst-case cap is `$16.16084992`: it includes the one epoch, six
500-example NLL passes, and six 128 × G4 monitor passes at maximum token
length. This is a bound, not an expected charge; it must remain under the
explicit `--hard-cap-usd` value. After you review that JSON, the paid command
is:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --recipe e2 --run --allow-paid --hard-cap-usd 18
```

E2 checkpoints retain the existing 30-day TTL. Do not merge its PR until this
manual paid run finishes successfully and its W&B monitor is visible.

## E2 formal-evaluation command

E2 selected step 250 by monitor pass@4, then pass@1, then NLL. The command
below evaluates precisely that sampler with the unchanged E0 formal protocol;
its resulting `eval/pass_at_1` and `eval/pass_at_4` are the only numbers that
answer whether E2 improved on Base.

First run the local-only preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_eval \
  --experiment-id e2 \
  --sampler-path 'tinker://5048e951-841f-53d9-9388-87cb865de0bb:train:0/sampler_weights/e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250' \
  --source-training-run-url 'https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/etl4870w' \
  --hard-cap-usd 7
```

Paid run:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_eval \
  --experiment-id e2 \
  --sampler-path 'tinker://5048e951-841f-53d9-9388-87cb865de0bb:train:0/sampler_weights/e2-sft-qwen-qwen3-5-9b-base-r32-b8-lr3e-4-linear-gm128-a01-step250' \
  --source-training-run-url 'https://wandb.ai/techtao-small-thinking/mini-posttraining-lab/runs/etl4870w' \
  --run --allow-paid --hard-cap-usd 7
```

The hard cap is a worst-case guardrail; the completed E0 formal run cost
$3.24310.

## E3 buffered-early-stop SFT command

E3 keeps E2's data, rank 32, batch 8, peak LR `3e-4`, linear decay, fixed
128-prompt G4 monitor, and selection rule. At each validation checkpoint it
compares the monitor to the best score observed so far, including Base. A
checkpoint counts as a regression only when both pass@1 and pass@4 are lower
by more than `0.03125` (four of 128 prompts); one consecutive material
regression stops later training. Both the tolerance and patience are recorded
in W&B config and can be overridden from the command line.

Preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --recipe e3
```

Paid run:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m \
  modeling.llm_post_training.gsm8k_sft_grpo_lab.sft_train \
  --recipe e3 --run --allow-paid --hard-cap-usd 18
```

For example, `--early-stopping-patience 2
--early-stopping-max-regression 0.046875` requires two consecutive drops of
more than six monitor prompts. The preflight retains E2's full-run worst-case
bound; an early stop only lowers the actual cost.

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
