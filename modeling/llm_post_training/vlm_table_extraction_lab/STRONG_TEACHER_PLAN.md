# OPD: replacing the teacher, with a matched H2 control

Registered before paid execution. The user authorized this experiment with a $10
incremental ceiling. The candidate is Qwen3.5-397B-A17B (64K), not an assumption
that a larger model is better at this table task.

## Question and controls

Does replacing frozen Qwen3.6-35B-A3B with frozen Qwen3.5-397B-A17B improve a
Qwen3.5-4B student under the existing H2 pure-OPD recipe? Reuse the completed
`h2_low_lr160` control from [the previous campaign](OPD_HYPOTHESES_RESULTS.md).
Both students start from Base, not from an SFT or H2 checkpoint.

Keep the first160 fixed Train800 records, seeded order20260914, one epoch,
batch8,20 updates, rank8 LoRA (attention/MLP, no unembedding), LR3e-5, two-step
linear warmup, Adam betas0.9/0.95, eps1e-8, clip1, and no weight decay.
Use pure sampled immediate reverse-KL, temperature1 rollouts,8192 output cap,
16384 training context and1,048,576 image pixels. Each new batch samples its own
current student; replaying H2 trajectories for training would not be on-policy.

Keep the same pinned renderer/scorer, non-thinking prompt and token alignment
checks. Pin each processor revision; hosted weight revisions are not exposed by
Tinker, so this is a historical matched-recipe control rather than simultaneous
randomized training. Model identity changes include architecture/pretraining,
not only parameter count.

## Sequential operations

1. Verify pinned tokenizer/special tokens and actual multimodal prompt equality
   before any training. Confirm the selected teacher exists in the service.
2. Evaluate candidate teacher on complete Dev100 with the old teacher's generation
   seed20260913. Reuse the existing35B Dev100 predictions after checking protocol,
   IDs and hashes. Measure candidate gold NLL/PPL as an additional diagnostic;
   do not invent a matched35B NLL if it was not recorded.
3. Score a fixed32 previously saved H2 student trajectories with the new teacher,
   retaining their original35B scores. Select the first32 in recorded training
   order before viewing new scores. This is a diagnostic on identical historical
   student prefixes (not Base-only outputs), never training data for the new run.
4. Train the candidate-teacher student under the exact H2 settings. Fixed Train32
   gold NLL/PPL and full Dev100 NLL/PPL at0/10/20; full Dev generation at10/20
   with seed20260914; pre/post-update diagnostics at1/5/10/15/20.
5. Evaluate final step20 on complete Test100, seed20260913, and publish it in the
   existing fixed comparison group with a distinct public label. Do not select
   checkpoints or tune hyperparameters using Test results.

Proceed with the registered training even if teacher Dev is not better, provided
compatibility, finite responses and budget checks pass. In that case report a
teacher replacement experiment, not a successfully stronger-teacher intervention.

## Measurements and interpretation

Primary outcome: final student Test Cell F1 difference versus H2, paired100-image
bootstrap with10,000 resamples and seed20260914. Also report numeric F1 and valid
denominator, raw/format-gated RD similarity, exact table/structure, format pass,
truncation and gold NLL/PPL. A positive Cell F1 interval with no point-estimate
format or truncation regression supports this recipe change; mixed results stay
mixed. Report all metrics and intervals, not only passing ones.

Compare fixed-prefix teacher log probabilities/advantages, feedback disagreement,
and likelihood on student outputs. Across independently generated training runs,
KL values refer to different teachers and evolving trajectories: smaller KL alone
does not mean better teaching. Update-k3, gradient norms, generation failures and
gold likelihood help interpret outcomes, not establish a unique causal mechanism.

Save all training specifications, hashes, metrics, paired comparisons, timing and
cost receipts. W&B receives only allowlisted aggregate metrics/configuration;
images, labels, original trajectories/probabilities, checkpoints, secrets and
actual filesystem paths stay out of Git/W&B. Tinker receives the authorized model
inputs. Raw local experiment files live in an ignored output directory.

## Budget

Use a new shared campaign ledger capped at$9.90, with per-operation/request
reservations, conservative uncached token accounting and no automatic replay of
uncertain requests or optimizer updates. Planned operation ceilings are teacher
Dev$3.00, fixed-prefix diagnostic$0.60, training/Dev$4.50 and Test$0.95.
Historical H2 costs are excluded. Any unused reservation is settled to recorded
usage; insufficient budget stops execution without silently changing the recipe.

At the old H2 lengths, training/Dev/Test is about$2.10 and teacher Dev generation
adds$1.15. Candidate teacher gold NLL and the fixed-prefix diagnostic add further
forward passes; allow approximately$4–5 overall, not a guaranteed bill or runtime.
Teacher rates verified from [Tinker pricing](https://tinker-docs.thinkingmachines.ai/tinker/models/):
35B prefill/sample $0.54/$1.335 and397B $3.00/$7.50 per million tokens.

Limitations: one training seed,160 images, reused Test100, historical control,
uncontrolled hosted weight revisions and multiple unadjusted comparisons. Neither
a positive nor a negative result establishes behavior at Train800 or a universal
teacher-size rule.
