# Tinker Post-Training Experiments

This directory is the home for post-training experiments that run on Tinker.
It is intentionally broader than any single method: supervised fine-tuning
(SFT), reinforcement learning (RL), and on-policy distillation (OPD) should
share evaluation, provenance, budget, and artifact conventions without being
forced into one experiment implementation.

Status: local MVP implemented and tested. No remote Tinker call or training run
has been made or authorized by this document.

## Proposed layout

The local connectivity MVP exists today. Add the remaining directories when
their code is needed.

```text
modeling/llm_post_training/tinker/
├── README.md
├── mvp.py                       # Local doctor and gated one-sample smoke
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

After the one-request sampling smoke succeeds, the next smallest gate is one
disposable LoRA batch (`forward_backward` plus `optim_step`). Only after that
training-path smoke succeeds should the project run a pilot benchmark baseline
or a real training experiment.

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
