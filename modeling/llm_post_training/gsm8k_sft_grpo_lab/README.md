# GSM8K SFT → GRPO → KD Lab

A diagnostics-first post-training study on `Qwen/Qwen3.5-9B-Base`. The question
is not simply which checkpoint scores highest: it is which learning-signal
change explains the result on a fixed GSM8K protocol.

![Formal GSM8K Pass@1 and Pass@4 for the E2-to-E7 ablation path](figures/gsm8k-posttraining-formal-results-v1.png)

**Current decision baseline.** E4 clean GRPO remains the frozen controlled
`Base → SFT → GRPO` reference: Pass@1 `0.7197`, Pass@4 `0.7506`. E9's separate
`Base → KD` route reaches `0.9126 / 0.9308` on the reused formal protocol, but
it consumes verifier-filtered traces from all 6,800 allowed prompts while the
historical selected E4 state had seen only 600 RL prompt groups. It is a strong
parallel end-to-end result, not a same-coverage causal replacement for E4. E10
therefore asks the narrower question: can a small on-policy distribution-KD
phase improve the already-strong E9 checkpoint, or is it saturated?

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
| E7 GRPO | Do fixed-sign advantages avoid zero-advantage groups? | 0.7675 | 0.7879 | Higher point estimate, but 12.7× E4 optimization tokens. |
| E8 GRPO | Does fixed-sign help at approximately E4's token budget? | 0.7020 | 0.7296 | No—do not promote it over E4. |
| E9 hard KD | Can verifier-filtered teacher traces replace SFT + GRPO? | **0.9126** | **0.9308** | Strong parallel `Base → KD` result; audit passed, but coverage is not matched to historical E4. |

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

E8 matches **optimization compute**, not the rollout count: it samples 256
rollouts, versus E4's 3,200. Its realized formal Pass@1/Pass@4 was
`0.7020/0.7296`, below E4's `0.7197/0.7506`. It does not show a fixed-sign
advantage at this approximate student optimization-token budget; it also
changed update granularity and prompt coverage, so it is not a pure estimator
isolation.

## E9 — Base-to-full-corpus verifier-filtered teacher-response KD

E9 is the first knowledge-distillation baseline, not RLAIF. A frozen
`Qwen/Qwen3.5-397B-A17B` teacher writes one solution for each of the 6,800
allowed training prompts: the ordered union of frozen `sft_train` (5,000) and
`rl_train` (1,800). The exact GSM8K verifier uses the original answer only to
keep correct, non-truncated responses with a numeric `\boxed{...}` conclusion;
the student never receives that original answer. It creates a **fresh rank-32
LoRA on the untouched Base model**, with a fresh KD optimizer, then applies
ordinary cross-entropy only on every accepted teacher-response token, once.

This is intentionally `Base → KD`, not E4 continuation and not a token-matched
probe. Its purpose is to compare an end-to-end KD route against the existing
`Base → SFT → GRPO` route. Candidate IDs are checked against the exact ordered
manifest union before any paid call, and the report records its hash, accepted
trace digest, actual optimized/supervised tokens, and teacher/student/development
costs separately.

There is an important **coverage caveat**: historical E4 trained the full SFT
run but only 800 `rl_train` prompt groups during its 100 GRPO steps (the selected
step-75 state had seen 600). E9's 6,800-prompt full-corpus recipe is therefore
the correct parallel *method* alternative, but is not a same-data-coverage
causal control for that historical E4 checkpoint. A future full-corpus SFT+RL
control is required to isolate algorithmic differences at matched coverage.

KD development selection is deliberately separated from KD training: the
teacher may generate targets only from `sft_train + rl_train`, while
student-only G4 rollouts run on a fixed 64-prompt prefix of `sft_validation` at
initialization and every 500,000 optimized student input tokens (plus the
terminal checkpoint). Each event logs Pass@1, Pass@4, rollout count,
response-diversity diagnostics, and a W&B table of the four raw responses per
prompt. The selected checkpoint's Pass@1/Pass@4 are written explicitly to the
W&B summary. This development set is held out from KD training but has been
reused elsewhere in the study, so it selects checkpoints within a run and never
substitutes for the frozen formal evaluation.

E9's selected step 204 had development Pass@1/Pass@4 `0.8906/0.9062`. Its
formal G4 evaluation reached `0.9126/0.9308` on all 1,287 frozen formal prompts
(`+19.29pp/+18.03pp` versus E4). The raw formal table matches the current scorer
exactly, contains each frozen formal ID exactly four times, and has zero ID
overlap with the accepted E9 training traces. This eliminates a direct split
or scorer leak in this implementation; the usual public-benchmark
pretraining-contamination caveat still remains.

### E10 — small on-policy Top-K external-teacher KD probe

E10 is **not** a second full E9 run and is not parameter-sharing
"self-distillation." It restores E9 step 204's weights with a fresh optimizer,
then makes eight fixed updates over 64 `rl_train` prompts that also have an
accepted E9 teacher trace. At each update, the current student produces four
rollouts per prompt. The frozen 397B teacher receives its verified E9 trace as
privileged reference context plus the student rollout, returns Top-K next-token
log-probabilities at two student-visited positions per rollout, and the student
optimizes CE against the teacher distribution renormalized on that support.
The reference trace is never inserted into the student input.

The E10 preflight locks the parent state/sampler URI, exact trace digest,
selected training-ID hash, and verifies zero overlap with both the development
and formal splits. The preflight's conservative token bound is `$3.2357` and
the run refuses to exceed a `$5.00` cap. It logs actual teacher Top-K-query,
student-rollout, student-training, and development costs separately. It also
logs Top-K retained probability mass, normalized teacher entropy, Top-K CE,
the explicitly named truncated-target `KL(q_topk || p_student)`, prefix
coverage, G4 rollout correctness/format/truncation, and unique-response
diversity. Only the frozen development Pass@4 → Pass@1 rule selects the parent
or terminal checkpoint; none of those training diagnostics can trigger a
formal evaluation by themselves.

### A shared schema for the entire KD ladder

E9 is the first **implemented recipe**, not a bespoke metric universe. The
[`distillation_schema.py`](distillation_schema.py) registry gives every later
KD method the same core contract: initialization provenance, optimized and
weighted-token ledgers, teacher/student/development cost ledgers, fixed
behavioral development metrics, and the rule that `dev/pass_at_4` then
`dev/pass_at_1` can select a checkpoint. The metric dictionary written beside
each run records the definition, unit, decision role, and caveat for every
chart. The v4 schema intentionally does not claim a held-out teacher-trace NLL:
until the runner implements and verifies a non-mutating forward-only path,
behavioral student rollouts are the KD validation signal.

Each method then adds only its semantically meaningful diagnostics. E9 adds
teacher acceptance/rejection statistics and hard-KD NLL; an offline Top-K route
would add retained-probability mass and Top-K CE; E10's implemented on-policy Top-K
route adds an explicitly truncated-target forward KL, current-policy rollout
cost and diversity, and prefix coverage. A teacher judge/RLAIF route adds
judge-verifier agreement and reward/advantage diagnostics; offline Top-K,
preference, and teacher-judge routes remain registered but intentionally fail
before any paid call rather than silently treating them as E9.

This is deliberately not one numerical score for every method. Training loss,
teacher reward, judge score, and format are diagnostics or guardrails; they
must never choose a checkpoint or justify a formal claim. The formal evaluation
stays algorithm-independent inference on the frozen formal split.

### Actual-token cost ledger

The E9 terminal, W&B, and local report distinguish a conservative preflight
maximum from the run's **actual-token-priced ledger**. Teacher generation and
student development rollouts record observed input/output tokens separately and
apply their respective configured input/output prices. KD optimization records
actual optimized sequence tokens at Tinker's flat training-token rate; its
teacher-response target tokens are recorded as a diagnostic rather than charged
again as inference output. The local report preserves every token count, rate,
component cost, and total so the pricing calculation can be audited after a run.

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
- [Frozen split manifest](manifests/gsm8k_splits.json), [evaluation harness](evaluation.py), [GRPO training entry point](grpo_train.py), [KD training entry point](kd_train.py), and [full experiment ledger](experiment_log.md)

Use the repository's uv environment for command help and local preflight:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.grpo_train --help
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.checkpoint_eval --help
```

E9's no-network preflight and paid command are:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.kd_train \
  --hard-cap-usd 60.00

UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.kd_train \
  --run --allow-paid --hard-cap-usd 60.00
```

The first command is local-only. Add `--run --allow-paid` only after inspecting
its bounded cost; after training, first audit the selected checkpoint on the
separate frozen calibration split before deciding whether a formal evaluation is
warranted.

E10's local-only preflight and bounded paid command are:

```bash
UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.opd_train \
  --hard-cap-usd 5.00

UV_CACHE_DIR=.uv-cache uv run --extra tinker python -m modeling.llm_post_training.gsm8k_sft_grpo_lab.opd_train \
  --run --allow-paid --hard-cap-usd 5.00
```

The E10 command does not run a formal evaluation. Run the frozen formal
inference protocol only if the terminal checkpoint beats the E9 parent under
the declared development selector and the raw on-policy diagnostics are sane.

Paid commands require an explicit `--run --allow-paid` authorization and a
bounded preflight; raw rollouts, local checkpoints, and exports stay in ignored
`outputs/`.
