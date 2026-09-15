# OPD hypotheses: preregistered diagnostic experiments

2026-09-14, before new paid calls. The user authorized recording and testing the four
proposed hypotheses and improving telemetry, with total incremental spend below $10.
This is a continuation of OPD800; no PR merge is authorized.

## Common protocol and budget

One sequential campaign uses a durable shared $9.90 ceiling with reservations before
launching each paid operation and a 10% accounting margin. Each training job also
retains its per-request ledger. Uncertain failures retain their reservation; no
optimizer replay or silent budget increase. Estimated new compute is roughly $6–8;
this is a length-based scenario, not a billing guarantee. Expected execution is about
1.5–2.5 hours on Tinker, with service latency and debugging uncertainty.

H1 reuses saved OPD800 weights. H2–H4 share one NEW control and the same fixed Train160
prefix from Train800, fresh Base Qwen3.5-4B rank8 LoRA, frozen Qwen3.6-35B-A3B teacher,
1 epoch, batch8,20 updates, seed20260914,10% warmup, unchanged Adam/length/pixel limits.
Train160 runs are mechanism probes; never compare their training exposure with KD800
as if matched. All variants use complete Dev100. Full Dev gold NLL at0/10/20, complete
Dev generation at10/20; a fixed Train32 gold likelihood probe uses training images only.
Test100 uses the frozen evaluation protocol after the variants/configurations are fixed.
No Test-driven hyperparameter changes. All variants and unsuccessful results are kept.

## Hypotheses and falsifiable contrasts

| ID | Hypothesis | Treatment versus control | Evaluation |
| --- | --- | --- | --- |
| H1 | Final step100 missed a better checkpoint | Reuse step25/50/75 versus existing step100, same Dev generation seed20260914 | Full Dev100 generation per checkpoint; a Dev-selected earlier checkpoint may receive Test100, final Test already exists |
| H2 | LR1e-4 causes excessive updates or late instability | Train160 LR3e-5 versus NEW Train160 LR1e-4, all else fixed | Pre/post-update probability diagnostics, matched Train/Dev NLL, full Dev and Test generation |
| H3 | Single sampled-token reverse-KL feedback is limiting | Teacher Top10 CE on CURRENT STUDENT prefixes versus sampled reverse-KL | Same Train160/control; full Dev/Test, retained mass and soft-target diagnostics |
| H4 | Pure teacher matching is misaligned with correct tables | 0.75 OPD token-mean loss +0.25 gold token-mean CE, accumulated before ONE optimizer step, versus pure OPD | Same Train160/control; full Dev/Test quality, format, truncation and gold likelihood |

H3 changes both KL direction and estimator, so it tests a Top-K on-policy recipe,
not information density alone. H4 is explicitly a hybrid objective, not pure OPD.
We do not replace the teacher in this first H4 contrast; that would add another variable.

## Decision rules and statistical limits

Primary generation endpoint is macro cell F1; report numeric F1, raw/gated official RD,
exact table, structure exact, format success, truncation, NLL/PPL and actual token/cost
counts. Compare matched examples, 10,000 paired percentile bootstrap resamples,
seed20260914. A positive point estimate with a CI spanning zero is inconclusive,
not confirmation. Strong support requires positive paired cell-F1 CI plus no observed
format decline or truncation increase; consistent likelihood/mechanistic diagnostics
strengthen interpretation but cannot replace generation quality.

H1 chooses the highest Dev cell F1 among checkpoints with format/truncation no worse
than step100; ties use numeric F1 then earlier step. If no earlier checkpoint improves
cell F1, keep step100. Checkpoint selection is exploratory; Test is reported separately.
One training seed and small training subsets limit generalization. Reused Test100 is
not a pristine new holdout, and several hypotheses/endpoints introduce multiplicity.
Do not declare a unique root cause from this campaign alone.

## Added logging

Log advantage standard deviation/quantiles/sign/near-zero fractions; concentration of
absolute importance-weighted feedback; long/invalid trajectory signal shares; importance
ESS and tail rate; token-category diagnostics with alignment failures explicitly unknown.
These are signal proxies, not parameter-gradient attribution.
The advantage/importance-weighted signal proxy always uses sampled reverse-KL:
for H3 it is a diagnostic probe, not the Top10 CE training signal; for H4 it covers
only the OPD branch, not the combined objective's gradient.

On diagnostic updates, run a fresh forward pass after optimizer.step on the SAME
original tokens and compare pre/post learner probabilities. Report sampled old-policy
KL and a nonnegative k3 proxy with sampling-distribution caveats. Pre-update importance
ratio remains a sampler/learner agreement check, not update magnitude.

Capture returned finite scalar optimizer metrics using an explicit safe allowlist;
log availability. If hosted gradients or clipping statistics are not returned, record
unavailable rather than inventing them. Raw tensor/probability evidence stays local.
W&B receives aggregate metrics and permitted experiment configuration only.

## Reproducing the Train160 contrasts

Use the existing OPD command in [OPD_PLAN.md](OPD_PLAN.md), with fresh output directories,
and the following common options. Dataset/tool/output paths remain command arguments.
The four variants must use the same manifest, seed, and evaluation settings.

```bash
--train-examples 160 --batch-size 8 --epochs 1 --rank 8 \
--warmup-ratio 0.1 --eval-every 10 --generate-dev-every 10 --generate-dev \
--diagnostic-every 5 --train-probe-examples 32 --inference-concurrency 8 \
--max-estimated-usd 2.0 --wandb-mode online
```

| Variant | Additional options |
| --- | --- |
| Control160 | `--objective sampled_reverse_kl --learning-rate 0.0001` |
| H2 | `--objective sampled_reverse_kl --learning-rate 0.00003` |
| H3 | `--objective topk_forward_kl --learning-rate 0.0001` |
| H4 | `--objective hybrid_gold --gold-weight 0.25 --learning-rate 0.0001` |

Remove `--execute` for a local preflight. The length scenario retains maximum-length
Dev generation reserves, so it is deliberately more conservative than expected spend;
it is not a promise that a $2 job can finish under all output lengths. Per-request
reservations stop execution before the next call would exceed its allowance.

For the shared campaign cap, save each command as a private JSON argv list and invoke
`experiment_budget` with `--campaign-dir`, a unique `--operation-id`, `--reservation`,
the fresh `--report` path, and `--command-json`. It runs commands sequentially without
a shell, retains uncertain reservations, and verifies a completed report before settling.
All data, reports, command files and raw likelihoods stay in ignored local directories.

Full fixed Test100 uses `checkpoint_eval --split test --stage after --seed 20260913`;
H1 Dev checkpoints explicitly use `--seed 20260914`, matching the original OPD Dev
generation. The two protocols must not be mixed into one evaluation comparison.
Publish verified Test aggregates with `log_checkpoint_eval --run-label control160`
(or the corresponding variant label) to distinguish runs in the shared W&B group.
The optional public display label changes neither run identity nor comparison protocol.

Training W&B namespaces separate `training`, `dev`, `train`, `optimizer`,
`opd_diagnostics`, and `runtime`. A diagnostic old-policy KL is a sampled estimate,
not the teacher KL or a proof of improved table accuracy. Token categories are
conservative: mixed or unalignable tokens stay `unknown`; inspect that coverage before
interpreting category shares. Gradient-norm exceedance is a derived threshold indicator,
not an API-provided clipping frequency.
