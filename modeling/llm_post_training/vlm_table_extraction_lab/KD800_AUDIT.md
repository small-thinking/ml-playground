# KD800 implementation and regression audit

2026-09-14. Read-only audit of the completed KD800 run before further work on PR #102.
No new hosted inference, training, teacher collection or W&B writes were made. Existing
Test scores remain unchanged. Full run details: [KD800 results](RD_TRAIN800_KD_RESULTS.md).

## Conclusion

No consequential implementation error was found in the audited Top10 KD path. The
experiment is valid **pure-teacher, T=1, fixed-trajectory KD**, but its settings do not
establish that KD should outperform gold-label SFT. There are concrete supervision
quality and late-training regression signals. A correct loss implementation is not
a guarantee of an appropriate teacher, recipe, or better extraction accuracy.

## Algorithm checks

The executed loss is

`L = -(1/M) sum_(completion positions t) sum_(k in teacher Top10) q_top10(k | x, y_teacher,<t) log p_student(k | x, y_teacher,<t)`.

Here `q_top10` is renormalized over the retained teacher tokens, while student
probabilities retain their full-vocabulary normalization. `M` counts supervised
completion positions across the batch. This is `KL(q_top10 || p_student) + H(q_top10)`.

An independent source/cache audit and 97 offline KD tests found:

- Teacher scoring position `P+t` maps to student training row `P-1+t` with input
  `prompt + completion[:-1]`. Original tokens are preserved; no decode/re-encode shift.
- Prompt/image weights are zero; completion weights sum to one before batch scaling.
- Tinker CE sums weighted losses; weights are divided once by batch supervised-token
  count. SFT uses the same convention. There is no factor-of-K or double-mean error.
- Teacher/student tokenization and rendered prompts are checked for compatibility.
- All 792 completed trajectories include EOS. The other 8 are deliberately truncated
  at 8192 tokens; no artificial EOS is added.
- Teacher argmax matches the generated token at 950,546/951,723 positions (99.8763%).
  Artificially shifting alignment by one position reduces agreement to about 3.66%.
- The five implementation file hashes match the executed run. Initialization is a
  fresh LoRA; evaluation uses a sampler created from the saved trained weights.

Relevant code: [targets and reduction](kd_targets.py), [teacher collection](kd_collect.py),
[training and sampler lifecycle](kd.py). Native Top-K CE follows the
[Tinker contract](https://tinker-docs.thinkingmachines.ai/tinker/losses/cross-entropy/).
These checks do not inspect hosted model internals or constitute a new paid backend
numerical-gradient test. Hosted model weight revisions remain unpinned.

## Why this is not gold SFT with additional information

Gold SFT conditions on the correct answer prefix and optimizes the gold next token.
This KD conditions on the teacher's generated prefix and optimizes the teacher's
next-token distribution. Both the prefix and supervisory target change. There is
no gold CE component in this run. Matching the teacher more accurately can preserve
its mistakes or output conventions, rather than improving agreement with gold.

The original distillation paper discusses softened teacher/student distributions
and combining soft-target CE with correct-label CE; it does not establish a general
ordering of KD above supervised training. See [Hinton et al., section 2](https://arxiv.org/html/1503.02531v1#S2).
T=1 pure-teacher KD is a valid variant, but it is a narrower experiment than that
combined recipe. Greedy rollout temperature 0 is separate from loss temperature 1.

Recomputing all 951,723 cached completion distributions gives:

| Teacher distribution diagnostic | Value |
| --- | ---: |
| Mean Top1 probability within renormalized Top10 | 0.993154 |
| Positions with Top1 probability at least 0.99 | 95.8005% |
| Positions with Top1 probability at least 0.999 | 90.1917% |
| Mean entropy | 0.020457 nats |

The targets are overwhelmingly sharp; having ten stored probabilities does not
mean this run receives much additional useful soft-target signal. This does not
prove that a higher temperature would improve results. Raising only the teacher
Top10 temperature would not implement classical equal-temperature KD: it would
also require the student's temperature-dependent full-vocabulary normalization,
and excluded teacher tail mass becomes more important at higher temperatures.

## Teacher supervision quality

Existing cached teacher rollouts were scored against the actual cleaned Train800
gold references with the unchanged deterministic evaluator. No new teacher calls
were needed. These are in-task agreement metrics, not human judgments of semantic
correctness or an independent held-out evaluation.

| Metric | Teacher on Train800 | Teacher on Dev100 |
| --- | ---: | ---: |
| Cell F1 | 0.493535 | 0.509603 |
| Numeric F1 | 0.563639 | 0.598524 |
| Exact table | 5.5% (44/800) | 4% (4/100) |
| Parse success | 97.75% | 98% |

Teacher Dev cell/numeric F1 are below SFT800's 0.633124/0.684234 on the same Dev set.
Thus the larger teacher is not empirically stronger than the gold-trained student
on these task metrics. This is a plausible contributor, not an isolated causal test.

18 format-invalid trajectories contain 81,361/951,723 supervised tokens (8.55%);
the overlapping 8 truncated trajectories contain 65,536 tokens (6.89%). Keeping all
images avoids selection by format, but does not make all teacher targets reliable.
These are corpus token shares, not measured shares of total gradient influence;
actual loss normalization is per batch. Removing these rows alone would again
change the image set relative to SFT and would require a matched control.

## What happened to exact-table accuracy

All 13 deterministic metrics for every stored Base/SFT/KD Test prediction were
recomputed in the repository uv environment and match the saved per-table scores.
Exact table requires equal dimensions, cell spans and normalized cell contents;
it is not raw HTML string equality. Normalization collapses whitespace runs but
preserves whether an internal space exists.

Base had 7 exact tables. KD retained 3, lost 4, and gained 0 from the other 93.
All four lost cases still parse successfully:

| Cause among the four lost cases | Tables |
| --- | ---: |
| Same geometry; differences only in internal cell whitespace | 2 |
| Different cell spans, with numeric/content differences | 2 |

One of the latter tables still has cell F1 0.9545, illustrating that strong partial
accuracy can coexist with a failed whole-table match. The two whitespace cases
are evaluation-convention sensitivity, not evidence that the predictions are
semantically interchangeable in every use case. Numeric tokenization can also
change when internal spaces change.

Across the 93 tables Base did not solve exactly, mean cell F1 rises from 0.3625 to
0.4270 and numeric F1 from 0.4018 to 0.5136 (excluding undefined numeric cases).
KD rescues parsing on 8 of Base's 11 parse failures, but solves none of those 93
exactly. These partial improvements coexist with losing four formerly exact tables.

As a diagnostic only, removing *all* internal whitespace would yield exact-table
rates Base 8%, SFT 12%, KD 6%. It would still not reverse the ordering. This is not
a replacement metric: removing word boundaries or numeric spacing can hide real
errors. No official metric or W&B result was changed.

The paired Base→KD exact-match changes are 4 losses and 0 gains; exact two-sided
McNemar p=0.125. This small, exploratory comparison does not establish a reliable
population-level decline, but it does not erase the four observed regressions.
Against SFT, KD has 9 losses and 1 gain (p=0.02148). Repeated Test inspection and
multiple post-hoc diagnostics limit confirmatory interpretation.

## Overfitting: evidence and limits

| Step | Full Dev gold NLL |
| ---: | ---: |
| 0 | 0.107405 |
| 25 | 0.083127 |
| 50 | 0.083643 |
| 75 | 0.081434 |
| 100 | 0.085687 |

From step75 to100, NLL worsens on **95/100** Dev examples and improves on 5, with
exactly the same reference tokens. The token-weighted increase is 0.004253; the
largest single-example contribution is only 0.00008865. This is a broad late-stage
regression, not one anomalous long table driving the mean.

Full Train teacher-target KL falls from 0.032022 initially to 0.014074 finally.
However, Train KL and Dev gold NLL use different targets, so they do not establish
a same-objective train–dev generalization gap. Train KL was not measured on the
full train set at step75, so its direction over the final25 steps is also unknown.
The evidence is consistent with late overfitting or increasing mismatch to gold
while fitting teacher behavior; it cannot isolate either cause.

Only final weights were saved, and only final Dev free generation was evaluated.
There is no saved step75 sampler or intermediate exact-table score. We cannot
claim early stopping would recover the four Test regressions without another run.
The final Dev NLL also remains below the initial model's NLL.

## Next discriminating experiments, not executed

1. Save intermediate checkpoints and evaluate full Dev under a predeclared protocol.
   Use Dev for checkpoint selection, then run the fixed Test once. This tests whether
   final-step regression also appears in free generation; it does not retroactively
   replace the existing KD800 result.
2. Train a hard-target control on the **same cached teacher token trajectories** with
   matched initialization, batches and schedule. This isolates whether the soft
   distribution adds value over teacher-answer SFT, without regenerating the teacher.
   It is distinct from the existing gold-label SFT800 baseline.
3. If the goal is gold SFT plus teacher information, compare gold CE against gold CE
   plus KD using the **same gold prefixes**. Teacher probabilities must be rescored
   on those prefixes; the cached generated-prefix distributions cannot simply be
   attached to gold tokens. Tune mixing weight on Dev, retaining the gold baseline.

Do not change temperature, LR, filtering and target source simultaneously and then
attribute the result to one factor. This audit proposes these tests; it does not
claim that any will improve extraction or authorize an additional paid run.

Local aggregate evidence is in the ignored `outputs/kd800_diagnosis.json`; raw
predictions, probability arrays, labels, tokens and identifiers remain local.
