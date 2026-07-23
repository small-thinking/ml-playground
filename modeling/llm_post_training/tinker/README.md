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
That pilot uses a static DeepSeek-R1 solution as its SFT target. It remains
useful as a plumbing validation, but it is not the proposed Qwen-to-Qwen
experiment and must not be scaled or used for a quality claim.

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

This implemented pilot trains on `r1_solution_1`. Treat it as a completed
cross-model R1-trace integration spike. The controlled experiment below instead
uses DeepMath questions and reference answers, generates new SFT trajectories
with the designated Qwen teacher, and keeps the R1 solution fields out of the
main training path.

## Controlled experiment: Qwen SFT, correctness RL, and OPD

### Question and model identity

How much capability can a designated Qwen teacher transfer to a smaller Qwen
student through:

- offline SFT on verified Qwen-teacher solutions;
- correctness RL on student rollouts; and
- on-policy distillation (OPD) on student rollouts?

The primary pair is:

- teacher: `Qwen/Qwen3.6-35B-A3B`;
- student: `Qwen/Qwen3.5-4B`.

Do not automatically select the largest available Qwen model. The designated
35B-A3B teacher is a cost-effective MoE model and must first demonstrate a
meaningful accuracy gap over the student. If it does not, stop and revise the
teacher choice before generating a training corpus.

### Data roles and immutable splits

DeepMath fields have separate roles:

| Field or generated artifact | Role |
| --- | --- |
| `question` | Prompt pool for SFT generation, RL, and OPD |
| `final_answer` | Correctness label, SFT filter, RL reward, and internal evaluation target |
| `difficulty`, `topic` | Stratification and distribution checks |
| `r1_solution_1/2/3` | Excluded from the controlled experiment |
| Verified Qwen-teacher solution | SFT completion target |
| Student on-policy rollout | RL and OPD training trajectory |

Create the following deterministic, content-hashed, non-overlapping prompt
sets before any generation:

- `D_sft`: Qwen-teacher trajectory generation and SFT;
- `D_post`: a shared prompt pool used in separate correctness-RL and OPD runs;
- `D_eval`: held-out DeepMath internal evaluation only.

MATH-500 is the external evaluation set. It is never used for screening,
generation, SFT, RL, or OPD. SFT and post-training prompts should come from the
same domain and have comparable topic/difficulty distributions, but they should
not be the same examples.

### Ground truth and Qwen SFT generation

Do not use an LLM judge for mathematical correctness. Extract the model's final
answer and compare it with the dataset reference using a pinned mathematical
equivalence verifier. Parser failures, ambiguous answers, and unverifiable
questions fail closed and are preserved in the artifacts.

Generate SFT data once and cache it locally:

1. Sample a complete solution from the Qwen teacher for each `D_sft` prompt.
2. Extract its final answer and compare it with DeepMath `final_answer`.
3. Keep only correct, completed, non-truncated trajectories.
4. Store the prompt, Qwen solution, token counts, sampling config, model ID,
   verifier result, and content hashes in an ignored versioned artifact.
5. Use the verified Qwen solution, not `final_answer`, as the SFT completion.

The teacher does not need perfect accuracy because incorrect generations are
filtered out. A small calibration batch must estimate the acceptance rate
before the project commits to a candidate-pool size.

### Checkpoint lineage and matched branches

| Checkpoint | Parent | Training signal |
| --- | --- | --- |
| `M0` | Tinker-supported `Qwen/Qwen3.5-4B` | No repository adaptation |
| `M1-SFT` | `M0` | Cross-entropy on verified Qwen-teacher solutions from `D_sft` |
| `M2-RL` | `M1-SFT` | Mechanical final-answer reward on `D_post` |
| `M2-OPD` | `M1-SFT` | Teacher KL signal on student rollouts from `D_post` |
| `T` | `Qwen/Qwen3.6-35B-A3B` | Evaluation-only teacher reference |

`M2-RL` and `M2-OPD` start from the exact same `M1-SFT` checkpoint and use the
same `D_post` question IDs in separate runs. This isolates the main difference
between correctness reward and teacher-distribution supervision. A later
`M0 -> OPD` ablation may isolate the effect of the SFT initialization, but it is
not required for the first proof of concept.

`M0` is an unadapted experiment baseline, not a raw pretrained model:
`Qwen/Qwen3.5-4B` is already vendor post-trained.

Before OPD, a compatibility gate must compare full token-ID mappings, special
tokens, chat rendering, stop behavior, and trajectory alignment for the exact
teacher/student pair. A minimal remote smoke must also produce finite teacher
log probabilities and finite student loss. Stop rather than silently changing
the teacher if any gate fails.

### Training and evaluation sequence

1. Run local compatibility, renderer, verifier, manifest, fake-client, and cost
   tests.
2. Evaluate `M0` and `T` on a small frozen `D_eval` calibration slice. Stop if
   the teacher is not meaningfully better or either model is mostly truncated.
3. Generate and verify the Qwen SFT corpus from `D_sft`.
4. Train `M1-SFT`, then run the same internal evaluation.
5. Train `M2-RL` and `M2-OPD` independently from `M1-SFT` on `D_post`.
6. Evaluate all checkpoints with identical decoding and scoring.
7. Run full MATH-500 only for `T`, `M0`, `M1-SFT`, and the promoted post-training
   checkpoint. Running every exploratory checkpoint on all 500 examples is an
   avoidable cost.

Every evaluation must report accuracy, completed-only accuracy, completion and
truncation rates, parser failures, response-token lengths, teacher-student gap,
paired wrong-to-right/right-to-wrong transitions, and bootstrap confidence
intervals. Training loss alone is not evidence of improvement.

## Cost estimate

Pricing was rechecked on 2026-07-22 from the
[Tinker models and pricing page](https://tinker-docs.thinkingmachines.ai/tinker/models/).
All prices below are USD per million tokens:

| Model | Prefill | Sample | Train |
| --- | ---: | ---: | ---: |
| `Qwen/Qwen3.6-35B-A3B` teacher | `$0.540` | `$1.335` | `$1.177` |
| `Qwen/Qwen3.5-4B` student | `$0.330` | `$1.005` | `$0.737` |

The estimate intentionally ignores the advertised cached-prefill discount, so
cache hits reduce actual cost rather than rescue an under-budgeted run.
Checkpoint storage (`$0.10/GB/month`) is excluded and checkpoints must retain a
short TTL.

### Planning assumptions

- 768 `D_sft` candidates produce up to 512 accepted training trajectories;
- average prompt length: 256 tokens;
- average generated solution/rollout: 2,048 tokens;
- SFT: 512 accepted examples, one epoch, about 2,304 sequence tokens each;
- internal evaluation: 64 prompts across `T`, `M0`, `M1-SFT`, `M2-RL`, and
  `M2-OPD`;
- final MATH-500: `T`, `M0`, `M1-SFT`, and one promoted post-training model;
- initial correctness RL and OPD: 1 million student rollout tokens each;
- OPD teacher scoring is budgeted as uncached teacher prefill over the prompt
  plus student trajectory;
- student training is budgeted over the prompt plus trajectory.

These are planning assumptions, not evidence that 768 candidates will yield
512 accepted examples. A paid 64-example calibration must measure teacher
accuracy, completion length, and filter yield before the main generation job.

### Base estimate

| Phase | Base calculation | Expected cost |
| --- | --- | ---: |
| Qwen generation and gap screening | 768 prompts sampled once by teacher and student | `$3.85` |
| One SFT epoch | 512 x 2,304 student train tokens | `$0.87` |
| 64-example matched internal evaluation | teacher plus four student checkpoints | `$0.73` |
| 1M-token correctness RL | student prefill + sample + train | `$1.88` |
| 1M-token OPD | RL-shaped student cost + teacher prefill/KL scoring | `$2.48` |
| Final MATH-500 | teacher plus three student checkpoints | `$4.65` |
| **Expected total** | excludes storage and excess retries | **`$14.46`** |

Teacher generation is therefore not free and is more expensive than reusing
the bundled R1 traces. The controlled plan saves money by generating only a
small verified corpus once, caching it, avoiding duplicate teacher calls, and
promoting only useful checkpoints to the full external evaluation.

If both post-training branches are promoted from 1M to 3M rollout tokens, their
estimated costs rise to about `$5.63` for correctness RL and `$7.45` for OPD.
The total expected program cost becomes about `$23.18`. Keep the promoted
program under a `$30` target by requiring the 1M-token results before approving
either extension.

The most important sensitivities are:

- each additional 100 teacher-plus-student screening prompts costs about
  `$0.50` at the assumed lengths;
- increasing average generated length from 2,048 to 4,096 tokens raises the
  768-candidate generation phase from about `$3.85` to `$7.53`;
- each additional full SFT epoch costs about `$0.87`;
- each additional 1M rollout tokens costs about `$1.88` for correctness RL or
  `$2.48` for OPD;
- each additional student checkpoint evaluated on full MATH-500 costs about
  `$1.07`.

### Recommended staged caps

| Paid gate | Expected | Local hard cap |
| --- | ---: | ---: |
| 64-example teacher/student generation calibration | about `$0.32` | `$0.50` |
| Remaining Qwen generation + one-epoch SFT + internal eval | about `$4.86` | `$6.50` |
| Initial 1M-token correctness RL + internal eval | about `$2.01` | `$2.50` |
| Initial 1M-token OPD + internal eval | about `$2.62` | `$3.25` |
| Promoted four-checkpoint MATH-500 evaluation | about `$4.65` | `$6.00` |
| **Initial POC total** | **about `$14.46`** | **`$18.75` summed caps** |

Each gate requires a new explicit approval. Unused headroom in one gate does
not authorize another phase. After the calibration smoke, replace assumed
lengths and acceptance rates with measured values and lower or raise the next
cap explicitly.

### Cost formulas

For reproducibility, the implementation should calculate:

```text
sampling =
  prompt_tokens * prefill_rate
  + generated_tokens * sample_rate

SFT =
  all_sequence_tokens * student_train_rate

correctness_RL =
  prompt_tokens * student_prefill_rate
  + rollout_tokens * student_sample_rate
  + (prompt_tokens + rollout_tokens) * student_train_rate

OPD =
  correctness_RL
  + (prompt_tokens + rollout_tokens) * teacher_prefill_rate
```

Divide the right-hand side by one million when using the rates in the table.
The cost ledger must record planned and actual prompt, sample, teacher-score,
and train tokens separately.

## Approval gates and definition of done

The user's Tinker credit is a resource limit, not authorization. Before every
paid phase, present its exact command, model IDs, dataset/checkpoint revisions,
measured low/base/high estimate, hard cap, and automatic stop behavior. Wait
for an explicit `GO`; approval never carries forward to a later phase.

The first experiment is complete only when:

- `T`, `M0`, `M1-SFT`, `M2-RL`, and `M2-OPD` use one frozen evaluation protocol;
- training and benchmark prompts have no detected exact or normalized overlap;
- all checkpoints have paired per-example results, not only aggregate scores;
- the SFT corpus contains only verifier-approved Qwen-teacher trajectories;
- SFT, correctness-RL, and OPD deltas include uncertainty estimates;
- planned and actual tokens and dollars are recorded for every phase;
- failures, regressions, and null results are preserved in the final report.

## Research sources

- [Tinker quickstart](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
- [Tinker models and pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/)
- [Tinker SamplingClient](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/)
- [Tinker model-distillation recipe](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)
- [Tinker evaluation framework](https://tinker-docs.thinkingmachines.ai/cookbook/eval/)
- [Official Tinker SDK on PyPI](https://pypi.org/project/tinker/)
- [DeepMath-103K dataset](https://huggingface.co/datasets/zwhe99/DeepMath-103K)
- [DeepMath-103K paper](https://arxiv.org/abs/2504.11456)
- [MATH-500 dataset](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
- [Qwen3.5-4B model card](https://huggingface.co/Qwen/Qwen3.5-4B)
