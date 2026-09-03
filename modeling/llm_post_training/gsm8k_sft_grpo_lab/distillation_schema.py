"""Shared method and metric contracts for the GSM8K distillation ladder.

This module deliberately describes *all* distillation recipes before each one
is implemented.  It keeps three concerns separate:

* the teacher signal and learning objective (method-specific);
* provenance, cost, checkpoint selection, and behavioral evaluation (shared);
* optional diagnostics that are meaningful only for a particular method.

The schema is the source of truth for W&B naming, local report metadata, and
the human-readable metric dictionary.  A metric being present here does not
make it a valid checkpoint-selection criterion; that distinction is explicit
in ``MetricSpec.decision_role``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DISTILLATION_SCHEMA_VERSION = "gsm8k-distillation-schema-v3"

HARD_RESPONSE = "teacher-response"
TOPK_RESPONSE = "teacher-topk"
TEACHER_JUDGE = "teacher-score"
PREFERENCE = "teacher-preference"
ON_POLICY_TOPK = "on-policy-topk"
DISTILLATION_SIGNAL_KINDS = (
    HARD_RESPONSE,
    TOPK_RESPONSE,
    TEACHER_JUDGE,
    PREFERENCE,
    ON_POLICY_TOPK,
)

DECISION_DIAGNOSTIC = "diagnostic_only"
DECISION_GUARDRAIL = "guardrail"
DECISION_SELECTOR = "checkpoint_selector"
DECISION_REPORTING = "reporting_only"


@dataclass(frozen=True)
class DistillationMethodSpec:
    """Immutable training contract for one member of the KD ladder."""

    signal_kind: str
    display_name: str
    teacher_signal: str
    student_target: str
    objective: str
    training_regime: str
    on_policy: bool
    implementation_status: str
    description: str


@dataclass(frozen=True)
class MetricSpec:
    """One metric's stable name, interpretation, and permitted decision role."""

    key: str
    label: str
    group: str
    direction: str
    unit: str
    decision_role: str
    definition: str
    caveat: str


METHOD_SPECS = {
    HARD_RESPONSE: DistillationMethodSpec(
        signal_kind=HARD_RESPONSE,
        display_name="Verifier-filtered teacher-response KD",
        teacher_signal="One sampled teacher reasoning trace, filtered by verifier.",
        student_target="Hard next-token targets on teacher-response positions.",
        objective="masked cross_entropy",
        training_regime="offline supervised distillation",
        on_policy=False,
        implementation_status="implemented",
        description=(
            "Classic response distillation. The student imitates a frozen teacher's "
            "verified solution trace."
        ),
    ),
    TOPK_RESPONSE: DistillationMethodSpec(
        signal_kind=TOPK_RESPONSE,
        display_name="Frozen-teacher Top-K KD",
        teacher_signal="Top-K next-token probabilities on a teacher completion.",
        student_target="K candidate token IDs and normalized teacher weights per position.",
        objective="topk cross_entropy",
        training_regime="offline supervised distillation",
        on_policy=False,
        implementation_status="planned",
        description=(
            "Soft-target KD that preserves uncertainty among the teacher's Top-K "
            "next-token candidates."
        ),
    ),
    TEACHER_JUDGE: DistillationMethodSpec(
        signal_kind=TEACHER_JUDGE,
        display_name="Teacher-judged student rollout (RLAIF)",
        teacher_signal="Scalar teacher judgment or rubric score on a student rollout.",
        student_target="On-policy rollout tokens with scalar reward-derived advantages.",
        objective="importance_sampling, ppo, cispo, or custom policy gradient",
        training_regime="on-policy reinforcement learning",
        on_policy=True,
        implementation_status="planned",
        description=(
            "This is not hard-target KD: the teacher is a reward/judge provider and "
            "the student learns from its own sampled behavior."
        ),
    ),
    PREFERENCE: DistillationMethodSpec(
        signal_kind=PREFERENCE,
        display_name="Teacher preference distillation",
        teacher_signal="Teacher preference over chosen and rejected completions.",
        student_target="Paired chosen/rejected response log-probabilities.",
        objective="DPO-style custom pairwise loss",
        training_regime="offline preference optimization",
        on_policy=False,
        implementation_status="planned",
        description=(
            "The teacher supplies a relative comparison rather than a full response "
            "or a scalar rollout reward."
        ),
    ),
    ON_POLICY_TOPK: DistillationMethodSpec(
        signal_kind=ON_POLICY_TOPK,
        display_name="On-policy Top-K self-distillation",
        teacher_signal=(
            "Teacher Top-K distribution on prefixes sampled by the current student."
        ),
        student_target="Top-K teacher targets on student-visited prefix positions.",
        objective="topk cross_entropy",
        training_regime="on-policy distribution distillation",
        on_policy=True,
        implementation_status="planned",
        description=(
            "OPSD-style route: a teacher distribution supervises the student at the "
            "student's own prefixes rather than only on gold/teacher prefixes."
        ),
    ),
}


COMMON_METRICS = (
    MetricSpec(
        key="train/optimizer_step",
        label="Optimizer step",
        group="train",
        direction="none",
        unit="step",
        decision_role=DECISION_REPORTING,
        definition="Completed student optimizer updates.",
        caveat="Use alongside optimized tokens; a step is not comparable across methods.",
    ),
    MetricSpec(
        key="train/optimized_input_tokens",
        label="Student optimized input tokens",
        group="train",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Cumulative student input tokens that reached an optimizer update.",
        caveat="This is the primary student-compute ledger, not total experiment cost.",
    ),
    MetricSpec(
        key="train/supervised_or_weighted_tokens",
        label="Supervised or weighted tokens",
        group="train",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Cumulative token positions with nonzero CE weight or RL advantage.",
        caveat="Its exact meaning depends on the method's target representation.",
    ),
    MetricSpec(
        key="train/learning_rate",
        label="Learning rate",
        group="train",
        direction="none",
        unit="scalar",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Learning rate passed to the student optimizer at this update.",
        caveat="A matching LR does not make objectives or update magnitudes comparable.",
    ),
    MetricSpec(
        key="cost/teacher_generation_usd",
        label="Teacher generation cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition=(
            "Cumulative token-priced teacher cost from observed teacher input and "
            "output tokens."
        ),
        caveat=(
            "Uses the run's configured per-million rates; reconcile against an "
            "external invoice if a provider applies other charges."
        ),
    ),
    MetricSpec(
        key="cost/teacher_input_usd",
        label="Teacher input-token cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition="Teacher prompt-token component of the observed token-priced cost.",
        caveat="The configured teacher input rate may differ from the output rate.",
    ),
    MetricSpec(
        key="cost/teacher_output_usd",
        label="Teacher output-token cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition="Teacher completion-token component of the observed token-priced cost.",
        caveat="Includes rejected teacher completions because they still incurred cost.",
    ),
    MetricSpec(
        key="cost/student_training_usd",
        label="Student training cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition="Cumulative token-priced student forward/backward cost.",
        caveat=(
            "Uses the configured flat training-token rate; it does not use "
            "inference input/output prices."
        ),
    ),
    MetricSpec(
        key="cost/dev_inference_usd",
        label="Development inference cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition=(
            "Cumulative token-priced checkpoint-selection inference cost from "
            "observed development input and output tokens."
        ),
        caveat="Development evaluation must remain separate from formal evaluation.",
    ),
    MetricSpec(
        key="cost/dev_input_usd",
        label="Development input-token cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition="Development prompt-token component of cumulative inference cost.",
        caveat="Uses the student inference input rate, not the KD training rate.",
    ),
    MetricSpec(
        key="cost/dev_output_usd",
        label="Development output-token cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition="Development sampled-token component of cumulative inference cost.",
        caveat="Uses the student inference output rate, which can be higher than input.",
    ),
    MetricSpec(
        key="cost/cumulative_usd",
        label="Cumulative experiment cost",
        group="cost",
        direction="lower",
        unit="USD",
        decision_role=DECISION_REPORTING,
        definition=(
            "Teacher, student training, and development-inference token-priced "
            "cost combined."
        ),
        caveat="Storage and external local-compute costs may not be included.",
    ),
    MetricSpec(
        key="timing/step_seconds",
        label="Optimizer-step wall time",
        group="timing",
        direction="lower",
        unit="seconds",
        decision_role=DECISION_REPORTING,
        definition="Wall-clock duration of one student training update.",
        caveat="Infrastructure-dependent; never use as a quality comparison alone.",
    ),
    MetricSpec(
        key="timing/elapsed_seconds",
        label="Training elapsed wall time",
        group="timing",
        direction="lower",
        unit="seconds",
        decision_role=DECISION_REPORTING,
        definition="Wall-clock time elapsed since student updates began.",
        caveat="Excludes or includes setup only as documented by the recipe.",
    ),
    MetricSpec(
        key="dev/checkpoint_step",
        label="Checkpoint step",
        group="dev",
        direction="none",
        unit="step",
        decision_role=DECISION_REPORTING,
        definition="Student optimizer step represented by a behavioral development result.",
        caveat="This is an x-axis, not an optimization target.",
    ),
    MetricSpec(
        key="dev/optimized_input_tokens",
        label="Development checkpoint input tokens",
        group="dev",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition=(
            "Cumulative student optimized input tokens when the development "
            "rollouts were sampled."
        ),
        caveat="Use this token x-axis rather than optimizer step across KD routes.",
    ),
    MetricSpec(
        key="dev/generated_rollouts",
        label="Development generated rollouts",
        group="dev",
        direction="none",
        unit="rollouts",
        decision_role=DECISION_REPORTING,
        definition="Number of student completions sampled for this development result.",
        caveat="Pass@4 requires exactly four rollouts per development prompt.",
    ),
    MetricSpec(
        key="dev/prompt_tokens",
        label="Development prompt tokens",
        group="dev",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Observed prompt tokens in one G4 development-evaluation event.",
        caveat="This is event-local; use cost/dev_input_usd for the cumulative ledger.",
    ),
    MetricSpec(
        key="dev/output_tokens",
        label="Development output tokens",
        group="dev",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Observed sampled completion tokens in one G4 development event.",
        caveat="This is event-local; use cost/dev_output_usd for the cumulative ledger.",
    ),
    MetricSpec(
        key="dev/pass_at_1",
        label="Development Pass@1",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_SELECTOR,
        definition="Mean exact-answer correctness across the fixed G4 development rollouts.",
        caveat="Secondary checkpoint selector after Pass@4; never a formal result.",
    ),
    MetricSpec(
        key="dev/pass_at_4",
        label="Development Pass@4",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_SELECTOR,
        definition="Fraction of fixed development prompts with at least one correct G4 rollout.",
        caveat="Primary within-run checkpoint selector, not a cross-run conclusion.",
    ),
    MetricSpec(
        key="dev/format_accuracy",
        label="Development boxed-format accuracy",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_GUARDRAIL,
        definition="Fraction of development rollouts with a parseable boxed numeric answer.",
        caveat="Cleaner format alone does not establish better problem solving.",
    ),
    MetricSpec(
        key="dev/truncation_rate",
        label="Development truncation rate",
        group="dev",
        direction="lower",
        unit="fraction",
        decision_role=DECISION_GUARDRAIL,
        definition="Fraction of development rollouts that reach the output-token limit.",
        caveat="A lower rate is a guardrail, not a primary success metric.",
    ),
    MetricSpec(
        key="dev/avg_output_tokens",
        label="Development mean output length",
        group="dev",
        direction="none",
        unit="tokens",
        decision_role=DECISION_GUARDRAIL,
        definition="Mean generated output-token count on the fixed development set.",
        caveat="Changes can reveal length collapse or runaway reasoning but are not targets.",
    ),
    MetricSpec(
        key="dev/is_initialization_policy",
        label="Development result is the initialization policy",
        group="dev",
        direction="none",
        unit="boolean",
        decision_role=DECISION_REPORTING,
        definition=(
            "Whether the behavioral development result was sampled before KD updates "
            "from its declared initialization policy."
        ),
        caveat=(
            "Makes the initialization-versus-trained selection comparison explicit, "
            "whether initialization is Base or a checkpoint."
        ),
    ),
    MetricSpec(
        key="dev/group_all_correct_frac",
        label="Development all-correct group fraction",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of G4 development prompt groups whose four outputs are correct.",
        caveat="A distribution diagnostic, not the declared selection criterion.",
    ),
    MetricSpec(
        key="dev/group_all_wrong_frac",
        label="Development all-wrong group fraction",
        group="dev",
        direction="lower",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of G4 development prompt groups whose four outputs are wrong.",
        caveat="Interpret jointly with Pass@1 and Pass@4.",
    ),
    MetricSpec(
        key="dev/group_mixed_frac",
        label="Development mixed group fraction",
        group="dev",
        direction="none",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of G4 development groups containing both correct and wrong outputs.",
        caveat="Describes sampling variability, not a desired target on its own.",
    ),
    MetricSpec(
        key="dev/group_reward_std_mean",
        label="Development group reward standard deviation",
        group="dev",
        direction="none",
        unit="standard deviation",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Mean within-prompt standard deviation of binary correctness over G4.",
        caveat="Useful for rollout diversity diagnostics, not an outcome metric.",
    ),
    MetricSpec(
        key="dev/group_unique_response_frac",
        label="Development unique-response fraction",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition=(
            "Mean fraction of distinct decoded responses within each G4 development "
            "rollout group."
        ),
        caveat=(
            "Detects sampling collapse; textual diversity alone is not problem-solving "
            "quality."
        ),
    ),
    MetricSpec(
        key="dev/process_check_coverage",
        label="Development arithmetic-check coverage",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of outputs containing arithmetic equations checkable by the lightweight parser.",
        caveat="The parser is not a complete reasoning verifier.",
    ),
    MetricSpec(
        key="dev/process_validity_rate",
        label="Development arithmetic-check validity",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Validity rate among arithmetic steps that the lightweight parser can check.",
        caveat="No checked equations means the metric has limited interpretability.",
    ),
    MetricSpec(
        key="dev/final_correct_process_invalid",
        label="Correct final answer with invalid checked process",
        group="dev",
        direction="lower",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of outputs with a correct final answer but an invalid checked arithmetic step.",
        caveat="Only reflects explicit equations caught by the lightweight parser.",
    ),
    MetricSpec(
        key="dev/final_correct_process_valid",
        label="Correct final answer with valid checked process",
        group="dev",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of outputs with a correct final answer and valid checked arithmetic steps.",
        caveat="Not equivalent to full chain-of-thought correctness.",
    ),
    MetricSpec(
        key="selection/selected_checkpoint_step",
        label="Selected checkpoint step",
        group="selection",
        direction="none",
        unit="step",
        decision_role=DECISION_REPORTING,
        definition="Checkpoint chosen by the declared development selection policy.",
        caveat="Selection provenance, not a performance metric.",
    ),
    MetricSpec(
        key="selection/selected_dev_pass_at_1",
        label="Selected checkpoint development Pass@1",
        group="selection",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_REPORTING,
        definition="Development Pass@1 of the checkpoint retained by the declared selector.",
        caveat="A development statistic, never a formal result.",
    ),
    MetricSpec(
        key="selection/selected_dev_pass_at_4",
        label="Selected checkpoint development Pass@4",
        group="selection",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_REPORTING,
        definition="Development Pass@4 of the checkpoint retained by the declared selector.",
        caveat="A development statistic, never a formal result.",
    ),
    MetricSpec(
        key="selection/selected_is_initialization",
        label="Initialization retained",
        group="selection",
        direction="lower",
        unit="boolean",
        decision_role=DECISION_REPORTING,
        definition=(
            "Whether the declared initialization policy beat every trained checkpoint "
            "on development behavior."
        ),
        caveat="If true, do not spend on a new formal evaluation for this recipe.",
    ),
)


HARD_RESPONSE_METRICS = (
    MetricSpec(
        key="data/teacher_candidate_count",
        label="Teacher candidate count",
        group="data",
        direction="none",
        unit="examples",
        decision_role=DECISION_REPORTING,
        definition="Teacher completions sampled before verifier filtering.",
        caveat="A candidate is not necessarily a usable training trace.",
    ),
    MetricSpec(
        key="data/teacher_accepted_count",
        label="Accepted teacher trace count",
        group="data",
        direction="higher",
        unit="examples",
        decision_role=DECISION_REPORTING,
        definition="Teacher responses retained as student hard targets after all filters.",
        caveat="More traces do not imply higher trace quality or better student behavior.",
    ),
    MetricSpec(
        key="data/teacher_accept_rate",
        label="Teacher trace acceptance rate",
        group="data",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Accepted teacher traces divided by sampled teacher candidates.",
        caveat="Interpret with rejection reasons and response-length distribution.",
    ),
    MetricSpec(
        key="data/teacher_rejected_incorrect_count",
        label="Teacher incorrect-trace rejections",
        group="data",
        direction="lower",
        unit="examples",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Teacher responses rejected by the exact GSM8K answer verifier.",
        caveat="Measures outcome correctness only, not reasoning soundness.",
    ),
    MetricSpec(
        key="data/teacher_rejected_format_count",
        label="Teacher format rejections",
        group="data",
        direction="lower",
        unit="examples",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Correct responses rejected because they lack the required boxed format.",
        caveat="A formatting policy can change this without changing math capability.",
    ),
    MetricSpec(
        key="data/teacher_rejected_truncated_count",
        label="Teacher truncation rejections",
        group="data",
        direction="lower",
        unit="examples",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Responses rejected because generation reached the output limit.",
        caveat="May indicate a decoding budget issue rather than teacher weakness.",
    ),
    MetricSpec(
        key="data/teacher_prompt_tokens",
        label="Teacher prompt tokens",
        group="data",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Teacher input tokens consumed while generating candidate traces.",
        caveat="Required to report total distillation compute fairly.",
    ),
    MetricSpec(
        key="data/teacher_output_tokens",
        label="Teacher output tokens",
        group="data",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Teacher sampled completion tokens before filtering.",
        caveat="Not the same as student optimized tokens.",
    ),
    MetricSpec(
        key="train/hard_kd_nll",
        label="Hard-KD teacher-forced NLL",
        group="train",
        direction="lower",
        unit="nats/token",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Student NLL on accepted teacher tokens in the current update.",
        caveat="Imitation fidelity only; do not select or promote on this metric.",
    ),
    MetricSpec(
        key="train/hard_kd_ppl",
        label="Hard-KD teacher-forced perplexity",
        group="train",
        direction="lower",
        unit="perplexity",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Exponentiated hard-KD NLL for readability.",
        caveat="Monotonic with NLL and carries the same selection limitation.",
    ),
)


TOPK_METRICS = (
    MetricSpec(
        key="data/teacher_topk_coverage",
        label="Teacher Top-K probability coverage",
        group="data",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Teacher probability mass retained after Top-K truncation.",
        caveat="Top-K renormalization loses tail information by construction.",
    ),
    MetricSpec(
        key="data/teacher_topk_entropy",
        label="Teacher Top-K entropy",
        group="data",
        direction="none",
        unit="nats",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Entropy of the teacher distribution after Top-K renormalization.",
        caveat="Depends directly on K and decoding/prefix policy.",
    ),
    MetricSpec(
        key="train/topk_kd_cross_entropy",
        label="Top-K KD cross-entropy",
        group="train",
        direction="lower",
        unit="nats/token",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Weighted student cross-entropy against normalized teacher Top-K targets.",
        caveat="Not numerically comparable to hard-target NLL without care.",
    ),
)


TEACHER_JUDGE_METRICS = (
    MetricSpec(
        key="data/judge_score_mean",
        label="Teacher-judge score mean",
        group="data",
        direction="higher",
        unit="score",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Mean scalar score assigned by the teacher judge to student rollouts.",
        caveat="Requires calibration against verifier outcomes; it is not the objective truth.",
    ),
    MetricSpec(
        key="data/judge_verifier_agreement",
        label="Teacher-judge verifier agreement",
        group="data",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_GUARDRAIL,
        definition="Agreement between teacher judgment and available outcome verifier labels.",
        caveat="Low agreement makes reward optimization unsafe to interpret.",
    ),
    MetricSpec(
        key="train/reward_mean",
        label="Teacher reward mean",
        group="train",
        direction="none",
        unit="score",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Mean teacher-derived reward over the current student rollout batch.",
        caveat="Reward improvement can be reward hacking; do not use for promotion.",
    ),
    MetricSpec(
        key="train/advantage_abs_mean",
        label="Mean absolute advantage",
        group="train",
        direction="none",
        unit="score",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Mean absolute policy-gradient advantage magnitude in the update.",
        caveat="Scale changes with reward normalization and estimator choice.",
    ),
)


ON_POLICY_METRICS = (
    MetricSpec(
        key="data/on_policy_prefix_tokens",
        label="On-policy prefix tokens",
        group="data",
        direction="none",
        unit="tokens",
        decision_role=DECISION_REPORTING,
        definition="Student-generated prefix tokens at which teacher signals were queried.",
        caveat="Distinct from teacher or gold-prefix token counts.",
    ),
    MetricSpec(
        key="data/on_policy_prefix_coverage",
        label="On-policy prefix coverage",
        group="data",
        direction="higher",
        unit="fraction",
        decision_role=DECISION_DIAGNOSTIC,
        definition="Fraction of sampled student prefix positions with a usable teacher target.",
        caveat="Coverage depends on filtering, stopping rules, and teacher availability.",
    ),
)


METHOD_METRICS = {
    HARD_RESPONSE: HARD_RESPONSE_METRICS,
    TOPK_RESPONSE: TOPK_METRICS,
    TEACHER_JUDGE: TEACHER_JUDGE_METRICS,
    PREFERENCE: (),
    ON_POLICY_TOPK: TOPK_METRICS + ON_POLICY_METRICS,
}


def method_spec(signal_kind: str) -> DistillationMethodSpec:
    """Return a method contract or fail before a run can become ambiguous."""
    try:
        return METHOD_SPECS[signal_kind]
    except KeyError as exc:
        allowed = ", ".join(DISTILLATION_SIGNAL_KINDS)
        raise ValueError(
            f"unknown distillation signal_kind={signal_kind}; use {allowed}"
        ) from exc


def metric_specs(signal_kind: str) -> Tuple[MetricSpec, ...]:
    """Return shared metrics plus the method's meaningful extension metrics."""
    method_spec(signal_kind)
    return COMMON_METRICS + METHOD_METRICS[signal_kind]


def metric_schema_dict(signal_kind: str) -> Dict[str, Any]:
    """Produce a JSON-serializable metric dictionary for reports and W&B config."""
    method = method_spec(signal_kind)
    return {
        "schema_version": DISTILLATION_SCHEMA_VERSION,
        "method": asdict(method),
        "metrics": [asdict(spec) for spec in metric_specs(signal_kind)],
        "selection_policy": {
            "primary": "dev/pass_at_4",
            "tie_break": "dev/pass_at_1",
            "prohibited": [
                spec.key
                for spec in metric_specs(signal_kind)
                if spec.decision_role == DECISION_DIAGNOSTIC
            ],
            "formal_rule": (
                "Run the algorithm-independent frozen formal inference protocol only "
                "for the preselected checkpoint."
            ),
        },
    }


def metric_dictionary_markdown(signal_kind: str) -> str:
    """Render a compact Markdown dictionary for a W&B Report or local artifact."""
    schema = metric_schema_dict(signal_kind)
    method = schema["method"]
    lines = [
        "# Distillation metric dictionary",
        "",
        f"- Schema: `{schema['schema_version']}`",
        f"- Method: **{method['display_name']}** (`{method['signal_kind']}`)",
        f"- Objective: `{method['objective']}`",
        f"- Regime: {method['training_regime']}",
        "- Checkpoint selector: `dev/pass_at_4`, then `dev/pass_at_1`.",
        "- Formal result: algorithm-independent frozen inference protocol; never tune on it.",
        "",
        "| Metric | Label | Role | Direction | Definition | Caveat |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec in metric_specs(signal_kind):
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} |".format(
                spec.key,
                spec.label,
                spec.decision_role,
                spec.direction,
                spec.definition,
                spec.caveat,
            )
        )
    return "\n".join(lines) + "\n"


def write_metric_dictionary(path: Path, signal_kind: str) -> None:
    """Write the portable chart-explanation artifact next to a local run report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metric_dictionary_markdown(signal_kind))


def configure_wandb_metrics(run: Any, signal_kind: str) -> None:
    """Configure chart axes and summary behavior when the W&B run supports it.

    W&B's SDK has no per-metric description/tooltip field.  The companion
    metric dictionary is therefore the explanatory source, while this function
    makes the dashboard's axes and best-value summaries consistent.
    """
    method_spec(signal_kind)
    define_metric = getattr(run, "define_metric", None)
    if not callable(define_metric):
        return
    define_metric("train/*", step_metric="train/optimizer_step")
    define_metric("dev/*", step_metric="dev/optimized_input_tokens")
    define_metric("dev/pass_at_4", summary="max")
    define_metric("dev/pass_at_1", summary="max")
    metric_keys = {spec.key for spec in metric_specs(signal_kind)}
    if "train/hard_kd_nll" in metric_keys:
        define_metric("train/hard_kd_nll", summary="min")
    if "train/topk_kd_cross_entropy" in metric_keys:
        define_metric("train/topk_kd_cross_entropy", summary="min")
    define_metric("cost/cumulative_usd", summary="max")


def validate_logged_metric_keys(
    signal_kind: str, keys: Iterable[str]
) -> Tuple[str, ...]:
    """Return unregistered keys, excluding W&B's internal bookkeeping metrics."""
    allowed = {spec.key for spec in metric_specs(signal_kind)}
    allowed.update(
        {
            "checkpoint/state_path",
            "checkpoint/sampler_path",
            "tables/development_rollouts",
        }
    )
    return tuple(sorted(set(keys) - allowed))
