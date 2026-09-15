# OPD smoke validation

2026-09-14. Eight training images, fresh4B rank8 LoRA, batch4, two updates,
LR1e-4/warmup1, student temperature1 and one rollout/update per visit. Complete
Dev100 gold likelihood at steps0/1/2; W&B disabled. This validates execution and
is not the formal comparison with KD800.

The first attempt was rejected by Tinker's native importance-sampling schema:
`mask` is local diagnostic metadata, while the service accepts exactly
`target_tokens`, `logprobs`, `advantages`. It completed four sampled/teacher-scored
trajectories but no optimizer update. Raw responses and the original pending
journal remain local. A separate reconciliation records the known input rejection
and conservatively counts even the rejected backward pass: at most$0.066162 of
estimated compute. No uncertain optimizer request was replayed.

The corrected boundary strips only the local mask before submission, preserving
it for normalization and metrics. Regression tests reproduce strict schema
rejection and verify that no optimizer runs in that case. The second fresh run
completed both updates and all three Dev checks:

| Step | Full Dev gold NLL | PPL |
| ---: | ---: | ---: |
| 0 | 0.107375 | 1.113351 |
| 1 | 0.106059 | 1.111887 |
| 2 | 0.104681 | 1.110357 |

The corrected run consumed19,045 rollout tokens,24,067 teacher input tokens and
24,059 training input tokens. One trajectory truncated and failed parsing; all8
were retained. Per-batch mean importance ratios were1.000230 and1.000024, with a
maximum individual ratio1.8381. The sampling and learner implementations are not
numerically identical, which is why the importance correction matters.

Both updates plus intermediate Dev took114.7seconds; this excludes initialization,
initial/final Dev checks and checkpoint work. Corrected-run compute estimate was
$0.198992. Including the conservatively accounted failed attempt, smoke total is
$0.265154, within the original$0.40 allowance. No W&B records were created.

Independent recomputation from local original tokens and teacher/learner
log probabilities confirmed dataset order, prompt identity, shift, EOS/truncation,
advantages, token normalization, logged metrics, all Dev NLLs and cost. The source
fingerprint matches the run. No independent hosted-gradient rerun was performed.

At this point all256 table tests passed, including pinned renderer/scorer and
real offline W&B privacy tests. The initial sandbox-only W&B test failed to open
a local socket; it passed when the local service was permitted. No credentials,
images, prompts, IDs or probability arrays are included in this document.
