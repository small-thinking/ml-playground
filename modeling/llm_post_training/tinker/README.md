# Tinker Post-Training Experiments

This directory is the home for post-training experiments that run on Tinker.
It is intentionally broader than any single method: supervised fine-tuning
(SFT), reinforcement learning (RL), and on-policy distillation (OPD) should
share evaluation, provenance, budget, and artifact conventions without being
forced into one experiment implementation.

Status: the local connectivity and three-step SFT MVPs are implemented and
tested. The first real three-step Tinker + W&B run passed on 2026-07-21; see the
[validation record](validation/2026-07-21-sft-wandb-mvp.md). A configurable,
real-data DeepMath SFT pilot is also implemented. Its first paid 10-step run
completed on 2026-07-22; the training path passed, while the quality result was
inconclusive because every evaluation response was truncated. See the
[10-step validation record](validation/2026-07-22-deepmath-sft-10-step.md).

## Proposed layout

The local connectivity MVP exists today. Add the remaining directories when
their code is needed.

```text
modeling/llm_post_training/tinker/
├── README.md
├── mvp.py                       # Local doctor and gated one-sample smoke
├── train_mvp.py                 # Gated 3-step SFT + W&B smoke
├── train_sft.py                 # Configurable DeepMath train/eval pilot
├── configs/
│   └── sft_deepmath.toml        # Pinned default pilot parameters
├── common/                      # Shared evaluation, data, cost, and I/O code
└── experiments/
    ├── sft_then_rl/             # First controlled checkpoint ladder
    └── opd/                     # Later on-policy distillation experiment
```

The first experiment should establish a trustworthy baseline and evaluation
loop before OPD is introduced.

## MVP: prove the smallest Tinker path first

Before benchmark evaluation or training, run a deliberately tiny connectivity
MVP implemented in `mvp.py`.

The default command is local-only:

```bash
uv run --extra tinker python -m modeling.llm_post_training.tinker.mvp
```

It checks the Python and SDK versions, reports whether `TINKER_API_KEY` is
configured without printing its value, freezes the model ID and request limits,
and prints a machine-readable maximum-cost estimate. It does not construct a
Tinker client or make a network request.

The remote mode is a later approval gate:

```bash
uv run --extra tinker python -m modeling.llm_post_training.tinker.mvp \
  --remote-sample \
  --allow-paid
```

That command makes one logical sampling request to `Qwen/Qwen3.5-4B` with at
most 512 prompt tokens and 64 output tokens. It refuses to run unless all of
the following are true:

- `--remote-sample` and `--allow-paid` are both present;
- `TINKER_API_KEY` is configured;
- the worst-case token estimate is no more than the `$0.01` hard cap;
- the actual tokenized prompt is no longer than 512 tokens.

Using the public rates inspected on 2026-07-21 (`$0.33` per million prefill
tokens and `$1.005` per million sampled tokens), the preflight maximum is
`$0.00023328`. This is a local estimate rather than a provider-enforced billing
cap; the result must record actual token counts, and the Tinker console remains
the billing source of truth.

This MVP intentionally does **not**:

- load or score LiveBench;
- create a LoRA training client;
- call `forward_backward` or `optim_step`;
- save a checkpoint;
- run SFT, RL, or OPD.

## Training MVP: three updates with W&B

The next gate is implemented separately in `train_mvp.py`. It proves this
minimal end-to-end sequence without attempting meaningful model improvement:

1. sample once from the unadapted model;
2. create a rank-16 LoRA training client;
3. repeat `forward_backward` plus `optim_step` exactly three times over two
   tiny, repository-authored arithmetic examples;
4. log basic loss, token, estimated-cost, and step-timing metrics to Weights &
   Biases;
5. save the disposable weights for sampling and sample the same prompt again.

The two examples are smoke fixtures, not a benchmark or proposed training
dataset. The before/after text only verifies data flow; three updates cannot
support a quality claim.

Copy the committed template to the ignored repository-root `.env` and fill the
values locally. Never commit `.env` or paste its values into an issue or PR:

```bash
cp .env.example .env
```

Required variables:

```dotenv
TINKER_API_KEY=...
WANDB_API_KEY=...
WANDB_PROJECT=ml-playground-tinker
# WANDB_ENTITY=...
```

The default command loads that file and runs a local-only preflight. It reports
only whether each key is present, never the values:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_mvp
```

The paid path remains double-gated and must not be run until its exact command
and budget have received an explicit `GO`:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_mvp \
  --run \
  --allow-paid
```

Run those commands from the repository root. The paid command prints live
progress to stderr and leaves the final machine-readable report on stdout. A
run looks like this (metric values and the W&B URL will vary):

```text
[tinker-mvp] authorized model=Qwen/Qwen3.5-4B steps=3 max_token_cost_usd=0.000714816
[tinker-mvp] connecting to Tinker with verified HTTPX transport
[tinker-mvp] clients ready examples_per_step=2 train_tokens_per_step=30
[tinker-mvp] initializing W&B project=ml-playground-tinker
[tinker-mvp] W&B run=https://wandb.ai/.../runs/...
[tinker-mvp] sampling unadapted model
[tinker-mvp] baseline sample complete prompt_tokens=... output_tokens=32
[tinker-mvp] step=1/3 loss=... cumulative_train_tokens=30 step_seconds=... estimated_train_cost_usd=0.00002211
[tinker-mvp] step=2/3 loss=... cumulative_train_tokens=60 step_seconds=... estimated_train_cost_usd=0.00004422
[tinker-mvp] step=3/3 loss=... cumulative_train_tokens=90 step_seconds=... estimated_train_cost_usd=0.00006633
[tinker-mvp] sampling trained ephemeral checkpoint
[tinker-mvp] trained sample complete prompt_tokens=... output_tokens=32
[tinker-mvp] complete estimated_total_token_cost_usd=...
{
  "mode": "remote-sft-wandb-mvp",
  "steps_completed": 3,
  "wandb_run_url": "https://wandb.ai/.../runs/..."
}
```

Each invocation of the paid command creates a new Tinker/W&B run. The local
preflight command is the right way to inspect readiness without another charge.

The frozen upper-bound estimate is `$0.000714816` in token charges, below the
local `$0.01` hard stop. This estimate covers six tiny SFT examples processed
across three steps plus the two bounded samples. It is not a provider-enforced
billing cap, and the Tinker console remains the billing source of truth.

The MVP explicitly supplies a standard verified HTTPX client to Tinker. This
avoids a macOS CA-store incompatibility observed with the SDK's default
`pyqwest` transport while keeping TLS certificate verification enabled.

The validated run completed all three updates and synced its metrics to
[W&B](https://wandb.ai/techtao-small-thinking/ml-playground-tinker/runs/3zne613h).
It processed 90 train tokens and used an estimated `$0.00014649` in total token
charges. Both samples reached the 32-token limit, so the run proves plumbing
only and does not support a model-quality conclusion.

The checked-in validation record includes a static chart generated from the
three W&B API history rows. Rebuild it with:

```bash
MPLCONFIGDIR=/tmp/ml-playground-matplotlib uv run python \
  modeling/llm_post_training/tinker/validation/plot_mvp_metrics.py
```

Only after this training-path smoke succeeds should the project run a pilot
benchmark baseline or a real training experiment.

## Real-data SFT pilot: configurable end-to-end training

`train_sft.py` is the next layer after the disposable three-step smoke. It
streams a pinned MIT-licensed revision of `zwhe99/DeepMath-103K`, builds
deterministic and disjoint train/evaluation subsets, applies the Qwen3.5 chat
template, runs a
matched baseline evaluation, trains a LoRA adapter, saves persistent state and
sampler checkpoints, evaluates the trained checkpoint, and writes aggregate
metrics to W&B. Raw questions, solutions, and model responses are not committed;
the checked output manifest contains only content-derived IDs and metadata.

The default command is still local-only and makes no network request:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_sft
```

It validates the checked-in TOML config and prints the maximum token-cost
estimate. To validate the real Hugging Face schema and tokenizer without using
Tinker credit, run:

```bash
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_sft \
  --prepare-data
```

That command streams only the configured 256-example candidate pool rather
than downloading the full multi-gigabyte dataset. On 2026-07-21 it prepared 64
training examples and 8 held-out evaluation examples, skipped 58 candidates
that exceeded the 4096-token training cap, and wrote the ignored manifest to
`outputs/tinker/sft/deepmath-mvp/dataset_manifest.json`.

The paid training path remains double-gated. Override `training.steps` at run
time to turn the same pipeline into a short integration test:

```bash
# Two optimizer updates: paid integration test, not a quality experiment.
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_sft \
  --run --allow-paid --steps 2

# The checked-in default: 100 optimizer updates.
uv run --extra tinker python -m \
  modeling.llm_post_training.tinker.train_sft \
  --run --allow-paid
```

`--iterations` is an alias for `--steps`. One step means one
`forward_backward` plus one optimizer update over `training.batch_size`
examples. The fixed training subset is cycled deterministically when the step
count consumes more examples than it contains.

The defaults live in `configs/sft_deepmath.toml`:

| Section | Important defaults |
| --- | --- |
| `model` | `Qwen/Qwen3.5-4B`, LoRA rank 32 |
| `dataset` | pinned DeepMath revision, streaming, 64 train / 8 eval |
| `training` | 100 steps, batch size 2, LR `1e-4`, 4096-token cap |
| `evaluation` | matched sampling, 2048 output tokens, 0.8 completion floor |
| `checkpoint` | persistent state and sampler weights with a 7-day TTL |
| `pricing` | verified public per-token rates and a `$1.00` local hard stop |

At the checked-in public rates, the frozen maximum token estimate is about
`$0.6421` for the default 100-step pilot and about `$0.0504` for `--steps 2`.
These bounds include baseline/final evaluation but exclude checkpoint storage,
which Tinker lists separately. The hard stop is a client-side preflight, not a
provider-enforced billing cap, so the exact command and budget still require an
explicit `GO` before execution.

Add an optional Hugging Face read token to the ignored repository-root `.env`
to avoid anonymous Hub rate limits:

```dotenv
HF_TOKEN=hf_...
```

The dataset revision, model, sample counts, batch size, learning rate, sequence
limits, checkpoint TTL, W&B project, pricing inputs, and dollar cap can all be
changed in a copied TOML config and selected with `--config path/to/config.toml`.

## First experiment: baseline -> SFT -> RL

### Question

How much does SFT improve a small Qwen model on a fixed mathematical reasoning
benchmark, and how much additional improvement does correctness-driven RL add
on top of that SFT checkpoint?

### Checkpoint lineage

| Checkpoint | Parent | Change |
| --- | --- | --- |
| `M0` | Tinker-supported small Qwen checkpoint | No adaptation performed by this repository |
| `M1` | `M0` | SFT on a versioned mathematical reasoning training set |
| `M2` | `M1` | RL with a mechanically checkable correctness reward |

`M0` is an *unadapted experiment baseline*, not necessarily a pretrained base
model. For example, `Qwen/Qwen3.5-4B` is already a vendor post-trained model.
Reports must not describe it as a model that has never received SFT or RL.

The comparisons of interest are:

- SFT gain: `score(M1) - score(M0)`
- incremental RL gain: `score(M2) - score(M1)`
- total gain: `score(M2) - score(M0)`

RL starts from `M1`, not independently from `M0`, so the second comparison
measures the incremental value of RL in the planned training sequence.

## Benchmark recommendation

### Primary candidate: LiveBench Mathematics

Use a pinned, fully public release of the
[LiveBench Mathematics dataset](https://huggingface.co/datasets/livebench/math)
with the corresponding
[official evaluation code](https://github.com/LiveBench/LiveBench).

Why it is the current first choice:

- It is small enough for repeated evaluation. The Hugging Face math snapshot
  inspected on 2026-07-21 contained 368 rows.
- Questions have objective ground-truth answers and official automatic
  scoring, so an LLM judge is not required.
- It is designed to reduce test contamination through dated releases and new
  questions.
- Its Apache-2.0 datasheet describes it as evaluation-only and includes the
  expected question, ground-truth, and metadata fields.
- It is harder and less static than GSM8K or MATH-500, while providing many
  more paired observations than a 30-question AIME set.

The official leaderboard showed `2026-06-25` as the latest release during the
initial research. Do not depend on a mutable `main` branch or silently assume
that release is downloadable. Before the first baseline run, record and freeze:

- the public LiveBench release date;
- the dataset repository revision or content hash;
- the exact included math tasks and example IDs;
- the official scorer revision;
- the prompt template, renderer, decoding parameters, and maximum response
  tokens.

LiveBench must remain evaluation-only. Its questions, answers, model outputs,
and close paraphrases must not enter SFT or RL training data.

### Difficulty calibration gate

The benchmark is frozen before any training, but its difficulty must first be
calibrated with `M0`:

1. Run the complete candidate benchmark once on `M0`.
2. Accept it if the score leaves meaningful room in both directions. The
   target baseline band is roughly 20-80; scores from 10-85 can still be used
   if the per-task distribution is informative.
3. If the baseline is above 85, replace or augment it with a harder, pinned
   public math set before training.
4. If the baseline is below 10, add an easier stratum such as MATH-500 before
   training so improvement is measurable rather than hidden by a floor effect.
5. Once accepted, freeze the benchmark manifest and evaluation config for
   `M0`, `M1`, and `M2`. Never change the test set after observing a trained
   checkpoint.

MATH-500 is a useful easier fallback or secondary sanity check, but not the
primary recommendation: it is a static 500-problem subset of an older public
benchmark and is more exposed to contamination and saturation. AIME can be a
small diagnostic panel, but 30 questions are too few to be the sole measure of
incremental SFT and RL gains.

## Controlled evaluation protocol

Evaluate `M0`, `M1`, and `M2` with exactly the same:

- benchmark manifest and scorer revision;
- system/user prompt and chat renderer;
- sampling parameters, seed policy, and maximum response tokens;
- completion parser and invalid/truncated-response policy;
- per-example artifact schema and summary code.

The primary metric is the official LiveBench Mathematics score. Also report:

- per-task scores and example counts;
- paired per-example score changes;
- wrong-to-right and right-to-wrong transitions for binary-scored items;
- paired bootstrap 95% confidence intervals for score deltas;
- completion, truncation, and parser-success rates;
- response-token length distribution;
- a small fixed qualitative panel shared by all checkpoints.

Do not infer improvement from training loss alone. A stage counts as an
improvement only when the frozen evaluation shows a positive, uncertainty-aware
change without a large increase in truncation or formatting failures.

## Training-data boundary

Benchmark selection and training-data selection are separate decisions.
Before either SFT or RL:

- pin the training dataset name, revision, license, split, and filtering rules;
- preserve deterministic example IDs and normalized content hashes;
- check exact and normalized overlap with every benchmark prompt;
- keep raw datasets and generated trajectories out of git;
- use the same underlying problem distribution for SFT and RL where practical,
  while keeping the benchmark disjoint.

The first SFT dataset and RL recipe remain open decisions. They should be
selected only after current Tinker model support, cookbook APIs, and pricing
are reverified.

## Execution phases

### Phase 0: no-cost experiment foundation

- Pin the benchmark release and official scorer.
- Implement the shared evaluator and artifact schema.
- Add deterministic manifests and leakage checks.
- Add a local cost estimator and hard budget stops.
- Exercise all commands with fake clients and zero-network dry runs.

### Phase 1: baseline (`M0`)

- Reverify the exact Tinker model ID and price.
- Submit the exact command and cost estimate for approval.
- Run the difficulty calibration gate.
- Freeze the accepted benchmark and save per-example baseline results.

### Phase 2: SFT (`M0` -> `M1`)

- Build a small, versioned, benchmark-disjoint SFT training manifest.
- Train under an explicit token and dollar cap.
- Evaluate `M1` with the frozen protocol.
- Report the paired SFT gain relative to `M0`.

### Phase 3: RL (`M1` -> `M2`)

- Use a mechanically checkable answer reward; do not use benchmark questions.
- Train from the exact `M1` checkpoint under a separate cap.
- Evaluate `M2` with the frozen protocol.
- Report incremental RL gain relative to `M1` and total gain relative to `M0`.

### Phase 4: later OPD experiment

Add OPD as a sibling under `experiments/opd/`. It should reuse the frozen
evaluation harness, cost ledger, manifests, and report format so OPD results
can be compared with the baseline/SFT/RL ladder without duplicating platform
code.

## Cost and approval gates

The user's Tinker credit is a resource limit, not authorization to spend it.
Before every paid phase, present:

- exact commands, model IDs, dataset revisions, and checkpoint parent;
- expected prompt, sample, train, and evaluator token counts;
- low/base/high dollar estimates and a phase hard cap;
- automatic stop behavior and remaining uncertainty.

Wait for an explicit `GO` before the first paid call. Approval for one phase
does not authorize a later phase or a higher cap.

## Definition of done for the first experiment

- `M0`, `M1`, and `M2` use one frozen evaluation protocol.
- Training and benchmark prompts have no detected exact or normalized overlap.
- All checkpoints have paired per-example results, not only aggregate scores.
- SFT, incremental RL, and total deltas include uncertainty estimates.
- Planned and actual tokens and dollars are recorded for every phase.
- Failures, regressions, and null results are preserved in the final report.

## Research sources

- [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- [Tinker models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [Official Tinker SDK on PyPI](https://pypi.org/project/tinker/)
- [LiveBench dataset](https://huggingface.co/datasets/livebench/math)
- [LiveBench official repository and evaluator](https://github.com/LiveBench/LiveBench)
- [LiveBench datasheet](https://github.com/LiveBench/LiveBench/blob/main/docs/DATASHEET.md)
- [LiveBench leaderboard and dated releases](https://livebench.ai/)
- [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
