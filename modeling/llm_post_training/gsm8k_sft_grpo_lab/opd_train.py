"""Cost-gated E10 on-policy Top-K distribution distillation for GSM8K.

E10 is intentionally a small *second-phase* experiment, not another full E9
run.  It restores E9's selected weights with a fresh optimizer, asks the
current student for G=4 rollouts on a frozen training subset, and lets the
frozen external teacher provide Top-K next-token distributions at selected
student-visited prefixes.  The verified E9 teacher trace is privileged teacher
context only; it is never added to the student's prompt.

The update is exact cross-entropy against the teacher distribution truncated
and renormalized on its returned Top-K support.  We additionally log
``KL(q_topk || p_student) = CE - H(q_topk)``.  It is consequently a valid KL
for the *declared truncated teacher target*, not a claim about a full-vocabulary
teacher-to-student KL.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from dotenv import load_dotenv

from modeling.llm_post_training.gsm8k_sft_grpo_lab.base_eval import (
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_TOKENS,
    MODEL_ID,
    PROMPT_VERSION,
    SEED,
    SUITE_ID,
    WANDB_PROJECT,
    build_prompt,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.data import (
    SplitManifest,
    content_id,
    load_official_train_rows,
    read_manifest,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.distillation_schema import (
    DISTILLATION_SCHEMA_VERSION,
    ON_POLICY_TOPK,
    configure_wandb_metrics,
    metric_schema_dict,
    method_spec,
    validate_logged_metric_keys,
    write_metric_dictionary,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.evaluation import (
    Completion,
    evaluate_groups,
)
from modeling.llm_post_training.gsm8k_sft_grpo_lab.kd_train import (
    CHECKPOINT_TTL_SECONDS,
    DEFAULT_DEVELOPMENT_EXAMPLES,
    DEFAULT_DEVELOPMENT_GROUP_SIZE,
    ENV_FILE,
    E4_COMPARISON_CHECKPOINT,
    E4_FORMAL_PASS_AT_1,
    E4_FORMAL_PASS_AT_4,
    InferenceCostLedger,
    KDTrainingError,
    TrainingCostLedger,
    _development_metrics,
    _format_inference_cost,
    _format_training_cost,
    _generation_development,
    _inference_cost_ledger,
    _loss_sum,
    _student_monitor_cost_ledger,
    _student_training_cost_ledger,
    _teacher_cost_ledger,
)


EXPERIMENT_ID = "e10"
PARENT_EXPERIMENT_ID = "e9"
PARENT_SELECTED_STEP = 204
PARENT_STATE_PATH = (
    "tinker://7dc258cd-7920-5cbf-93b2-080ca38d75de:train:0/weights/"
    "e9-kd-teacher-response-qwen-qwen3-5-9b-base-teacher-qwen3-5-397b-a17b-"
    "from-base-fresh-lora-r32-b8-lr3e-4-full-allowed-train-once-devsft-"
    "validation64-di500000-a01-seed20260901-step204"
)
PARENT_SAMPLER_PATH = (
    "tinker://7dc258cd-7920-5cbf-93b2-080ca38d75de:train:0/sampler_weights/"
    "e9-kd-teacher-response-qwen-qwen3-5-9b-base-teacher-qwen3-5-397b-a17b-"
    "from-base-fresh-lora-r32-b8-lr3e-4-full-allowed-train-once-devsft-"
    "validation64-di500000-a01-seed20260901-step204"
)
TEACHER_MODEL_ID = "Qwen/Qwen3.5-397B-A17B"
TEACHER_PREFILL_USD_PER_MILLION = 3.0
TEACHER_SAMPLE_USD_PER_MILLION = 7.5
TRAIN_USD_PER_MILLION = 1.463
OUTPUT_DIR = Path(__file__).parent / "outputs"
DEFAULT_REFERENCE_TRACE_PATH = OUTPUT_DIR / "e9_teacher_traces_5e2xrzla.jsonl"

DEFAULT_ON_POLICY_PARTITION = "rl_train"
DEFAULT_ON_POLICY_EXAMPLES = 64
DEFAULT_ROLLOUT_GROUP_SIZE = 4
DEFAULT_ROLLOUT_BATCH_SIZE = 8
DEFAULT_ROLLOUT_MAX_OUTPUT_TOKENS = 256
DEFAULT_PREFIXES_PER_ROLLOUT = 2
DEFAULT_TEACHER_TOPK = 4
DEFAULT_MAX_SEQUENCE_TOKENS = 512
DEFAULT_TEACHER_MAX_CONTEXT_TOKENS = 1024
DEFAULT_REFERENCE_MAX_TOKENS = 512
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_HARD_CAP_USD = 5.0


@dataclass(frozen=True)
class E10Config:
    """Frozen E10 micro-experiment protocol and cost envelope."""

    model_id: str = MODEL_ID
    project: str = WANDB_PROJECT
    suite_id: str = SUITE_ID
    experiment_id: str = EXPERIMENT_ID
    signal_kind: str = ON_POLICY_TOPK
    attempt: int = 1
    parent_state_path: str = PARENT_STATE_PATH
    parent_sampler_path: str = PARENT_SAMPLER_PATH
    parent_experiment_id: str = PARENT_EXPERIMENT_ID
    parent_selected_step: int = PARENT_SELECTED_STEP
    initialization_source: str = "e9_hard_kd_step204_weights_fresh_optimizer"
    teacher_model_id: str = TEACHER_MODEL_ID
    teacher_prefill_usd_per_million: float = TEACHER_PREFILL_USD_PER_MILLION
    teacher_sample_usd_per_million: float = TEACHER_SAMPLE_USD_PER_MILLION
    reference_trace_path: str = str(DEFAULT_REFERENCE_TRACE_PATH)
    on_policy_partition: str = DEFAULT_ON_POLICY_PARTITION
    on_policy_examples: int = DEFAULT_ON_POLICY_EXAMPLES
    rollout_group_size: int = DEFAULT_ROLLOUT_GROUP_SIZE
    rollout_batch_size: int = DEFAULT_ROLLOUT_BATCH_SIZE
    rollout_max_output_tokens: int = DEFAULT_ROLLOUT_MAX_OUTPUT_TOKENS
    rollout_temperature: float = 1.0
    prefixes_per_rollout: int = DEFAULT_PREFIXES_PER_ROLLOUT
    teacher_topk: int = DEFAULT_TEACHER_TOPK
    max_sequence_tokens: int = DEFAULT_MAX_SEQUENCE_TOKENS
    teacher_max_context_tokens: int = DEFAULT_TEACHER_MAX_CONTEXT_TOKENS
    reference_max_tokens: int = DEFAULT_REFERENCE_MAX_TOKENS
    learning_rate: float = DEFAULT_LEARNING_RATE
    development_partition: str = "sft_validation"
    development_examples: int = DEFAULT_DEVELOPMENT_EXAMPLES
    development_group_size: int = DEFAULT_DEVELOPMENT_GROUP_SIZE
    development_every_steps: int = 8
    checkpoint_ttl_seconds: int = CHECKPOINT_TTL_SECONDS
    hard_cap_usd: float = DEFAULT_HARD_CAP_USD
    train_usd_per_million: float = TRAIN_USD_PER_MILLION
    seed: int = SEED
    trace_output_dir: str = str(OUTPUT_DIR)

    def validate(self, manifest: Optional[SplitManifest] = None) -> None:
        if self.experiment_id != EXPERIMENT_ID:
            raise KDTrainingError("this on-policy Top-K recipe is reserved for e10")
        if self.signal_kind != ON_POLICY_TOPK:
            raise KDTrainingError("E10 must use signal_kind=on-policy-topk")
        if method_spec(self.signal_kind).implementation_status != "implemented":
            raise KDTrainingError("the on-policy Top-K adapter is not implemented")
        if (
            self.parent_experiment_id != PARENT_EXPERIMENT_ID
            or self.parent_selected_step != PARENT_SELECTED_STEP
            or self.parent_state_path != PARENT_STATE_PATH
            or self.parent_sampler_path != PARENT_SAMPLER_PATH
        ):
            raise KDTrainingError("E10 is locked to the declared E9 selected checkpoint")
        if self.initialization_source != "e9_hard_kd_step204_weights_fresh_optimizer":
            raise KDTrainingError("E10 must restore E9 weights with a fresh optimizer")
        if self.teacher_model_id != TEACHER_MODEL_ID:
            raise KDTrainingError("E10 is locked to the declared frozen external teacher")
        if self.on_policy_partition != DEFAULT_ON_POLICY_PARTITION:
            raise KDTrainingError("E10 on-policy prompts must come from frozen rl_train")
        if self.development_partition != "sft_validation":
            raise KDTrainingError("E10 development must use frozen sft_validation")
        if self.on_policy_examples != DEFAULT_ON_POLICY_EXAMPLES:
            raise KDTrainingError("E10 is a fixed 64-prompt micro-experiment")
        if self.rollout_group_size != 4 or self.development_group_size != 4:
            raise KDTrainingError("E10 retains G=4 for rollout-diversity and Pass@4")
        if self.rollout_batch_size != DEFAULT_ROLLOUT_BATCH_SIZE:
            raise KDTrainingError("E10 uses eight frozen prompts per on-policy update")
        if self.development_every_steps != self.training_steps:
            raise KDTrainingError("E10 evaluates only the parent and terminal checkpoint")
        positive_ints = {
            "attempt": self.attempt,
            "rollout_max_output_tokens": self.rollout_max_output_tokens,
            "prefixes_per_rollout": self.prefixes_per_rollout,
            "teacher_topk": self.teacher_topk,
            "max_sequence_tokens": self.max_sequence_tokens,
            "teacher_max_context_tokens": self.teacher_max_context_tokens,
            "reference_max_tokens": self.reference_max_tokens,
            "development_examples": self.development_examples,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
        }
        if any(value <= 0 for value in positive_ints.values()):
            raise KDTrainingError("E10 integer budgets must be positive")
        if min(
            self.rollout_temperature,
            self.learning_rate,
            self.hard_cap_usd,
            self.teacher_prefill_usd_per_million,
            self.teacher_sample_usd_per_million,
            self.train_usd_per_million,
        ) <= 0:
            raise KDTrainingError("E10 rates, prices, and hard cap must be positive")
        if self.max_sequence_tokens > self.teacher_max_context_tokens:
            raise KDTrainingError("student sequence budget cannot exceed teacher context budget")
        if manifest is not None:
            manifest.validate()
            if self.on_policy_examples > len(manifest.rl_train_ids):
                raise KDTrainingError("on_policy_examples exceeds frozen rl_train")
            if self.development_examples > len(manifest.sft_validation_ids):
                raise KDTrainingError("development_examples exceeds frozen sft_validation")
            if set(manifest.rl_train_ids) & set(manifest.sft_validation_ids):
                raise KDTrainingError("on-policy and development partitions must be disjoint")

    @property
    def training_steps(self) -> int:
        return math.ceil(self.on_policy_examples / self.rollout_batch_size)

    @property
    def run_name(self) -> str:
        model_slug = self.model_id.lower().replace("/", "-").replace(".", "-")
        teacher_slug = self.teacher_model_id.rsplit("/", 1)[-1].lower().replace(".", "-")
        return (
            f"{self.experiment_id}-opd-topk-{model_slug}-teacher-{teacher_slug}"
            f"-from-e9-step{self.parent_selected_step}-freshopt"
            f"-rltrain{self.on_policy_examples}-g{self.rollout_group_size}"
            f"-p{self.prefixes_per_rollout}-k{self.teacher_topk}"
            f"-ms{self.max_sequence_tokens}-lr{self.learning_rate:.0e}"
            f"-a{self.attempt:02d}-seed{self.seed}"
        )


@dataclass(frozen=True)
class ReferenceTrace:
    example_id: str
    response: str


@dataclass(frozen=True)
class TopKPrefixTarget:
    example_id: str
    rollout_id: int
    prefix_position: int
    student_input_tokens: Tuple[int, ...]
    teacher_token_ids: Tuple[int, ...]
    teacher_probs: Tuple[float, ...]
    teacher_topk_mass: float
    teacher_entropy: float


@dataclass(frozen=True)
class OnPolicyCollection:
    targets: Tuple[TopKPrefixTarget, ...]
    rollout_metrics: Dict[str, float]
    rollout_table_rows: Tuple[Tuple[Any, ...], ...]
    rollout_prompt_tokens: int
    rollout_output_tokens: int
    teacher_prompt_tokens: int
    teacher_output_tokens: int
    selected_prefix_positions: int
    usable_prefix_positions: int
    topk_mass_sum: float
    topk_entropy_sum: float
    topk_size_sum: int


def _progress(message: str) -> None:
    print(f"[gsm8k-opd] {message}", file=sys.stderr, flush=True)


def _encode(tokenizer: Any, text: str) -> list[int]:
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        tokens = tokenizer.encode(text)
    return [int(token) for token in tokens]


def _ids_hash(example_ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(example_ids).encode()).hexdigest()


def _trace_digest(traces: Mapping[str, ReferenceTrace]) -> str:
    payload = [
        {"example_id": key, "response": traces[key].response}
        for key in sorted(traces)
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_reference_traces(path: Path) -> Dict[str, ReferenceTrace]:
    if not path.is_file():
        raise KDTrainingError(f"E10 requires the E9 trace artifact: {path}")
    traces: Dict[str, ReferenceTrace] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            record = json.loads(line)
            example_id = str(record["example_id"])
            response = str(record["response"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise KDTrainingError(
                f"invalid E9 reference trace at {path}:{line_number}"
            ) from exc
        if not example_id or not response or example_id in traces:
            raise KDTrainingError(f"ambiguous E9 reference trace at {path}:{line_number}")
        traces[example_id] = ReferenceTrace(example_id=example_id, response=response)
    if not traces:
        raise KDTrainingError("E9 reference trace artifact is empty")
    return traces


def _tokenizer_vocab_hash(tokenizer: Any) -> str:
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise KDTrainingError("Top-K KD requires tokenizers that expose get_vocab()")
    vocab = get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise KDTrainingError("tokenizer get_vocab() did not return a nonempty mapping")
    encoded = json.dumps(sorted((str(key), int(value)) for key, value in vocab.items()), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_identical_tokenizers(student_tokenizer: Any, teacher_tokenizer: Any) -> str:
    student_hash = _tokenizer_vocab_hash(student_tokenizer)
    teacher_hash = _tokenizer_vocab_hash(teacher_tokenizer)
    if student_hash != teacher_hash:
        raise KDTrainingError(
            "teacher/student tokenizer vocabularies differ; Top-K token IDs cannot "
            "be used as student targets safely"
        )
    return student_hash


def _prefix_positions(response_tokens: Sequence[int], count: int) -> Tuple[int, ...]:
    if not response_tokens:
        return ()
    return tuple(
        sorted({min(len(response_tokens) - 1, (index + 1) * len(response_tokens) // (count + 1)) for index in range(count)})
    )


def _normalized_topk(entries: Any) -> Optional[Tuple[Tuple[int, ...], Tuple[float, ...], float, float]]:
    if not entries:
        return None
    by_token: Dict[int, float] = {}
    for token_id, logprob in entries:
        token_id = int(token_id)
        logprob = float(logprob)
        if math.isfinite(logprob):
            by_token[token_id] = max(by_token.get(token_id, -math.inf), logprob)
    if not by_token:
        return None
    token_ids = tuple(sorted(by_token))
    raw = tuple(math.exp(by_token[token_id]) for token_id in token_ids)
    mass = sum(raw)
    if not math.isfinite(mass) or mass <= 0.0 or mass > 1.00001:
        raise KDTrainingError("teacher returned invalid Top-K probability mass")
    probs = tuple(value / mass for value in raw)
    entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0.0)
    return token_ids, probs, mass, entropy


def _student_topk_datums(
    targets: Sequence[TopKPrefixTarget], tinker_module: Any
) -> Tuple[list[Any], int, float]:
    data = []
    optimized_input_tokens = 0
    weighted_positions = 0.0
    for target in targets:
        if not target.student_input_tokens:
            raise KDTrainingError("on-policy Top-K target has an empty student context")
        for token_id, probability in zip(
            target.teacher_token_ids, target.teacher_probs, strict=True
        ):
            input_tokens = target.student_input_tokens
            data.append(
                tinker_module.types.Datum(
                    model_input=tinker_module.ModelInput.from_ints(tokens=list(input_tokens)),
                    loss_fn_inputs={
                        "target_tokens": list(input_tokens[1:]) + [int(token_id)],
                        "weights": [0.0] * (len(input_tokens) - 1) + [float(probability)],
                    },
                )
            )
            optimized_input_tokens += len(input_tokens)
            weighted_positions += float(probability)
    if not data:
        raise KDTrainingError("no usable Top-K targets were materialized")
    return data, optimized_input_tokens, weighted_positions


def _teacher_context_prefix(
    row: Mapping[str, object],
    trace: ReferenceTrace,
    teacher_tokenizer: Any,
    config: E10Config,
) -> Tuple[int, ...]:
    header = (
        build_prompt(str(row["question"]))
        + "\n\n<verified_teacher_reference>\n"
    )
    footer = "\n</verified_teacher_reference>\n<student_draft>\n"
    header_tokens = _encode(teacher_tokenizer, header)
    footer_tokens = _encode(teacher_tokenizer, footer)
    available_reference = (
        config.teacher_max_context_tokens
        - config.rollout_max_output_tokens
        - len(header_tokens)
        - len(footer_tokens)
    )
    if available_reference <= 0:
        raise KDTrainingError("teacher context budget leaves no room for the reference")
    reference_tokens = _encode(teacher_tokenizer, trace.response)[
        : min(config.reference_max_tokens, available_reference)
    ]
    context = tuple(header_tokens + reference_tokens + footer_tokens)
    if not context or len(context) + config.rollout_max_output_tokens > config.teacher_max_context_tokens:
        raise KDTrainingError("teacher context construction exceeded its declared budget")
    return context


async def _sample_student_rollouts(
    sampler: Any,
    rows: Sequence[Mapping[str, object]],
    tinker_module: Any,
    config: E10Config,
    step: int,
) -> Tuple[Tuple[Dict[str, Any], ...], int, int]:
    tokenizer = sampler.get_tokenizer()

    async def one(index: int, row: Mapping[str, object]) -> Dict[str, Any]:
        prompt_tokens = _encode(tokenizer, build_prompt(str(row["question"])))
        if not prompt_tokens or len(prompt_tokens) > MAX_PROMPT_TOKENS:
            raise KDTrainingError("an on-policy student prompt exceeds the token limit")
        result = await sampler.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=prompt_tokens),
            num_samples=config.rollout_group_size,
            sampling_params=tinker_module.SamplingParams(
                max_tokens=config.rollout_max_output_tokens,
                temperature=config.rollout_temperature,
                seed=config.seed + step * 10_000 + index,
            ),
        )
        if len(result.sequences) != config.rollout_group_size:
            raise KDTrainingError("on-policy sampler returned the wrong G=4 size")
        sequences = tuple(tuple(int(token) for token in item.tokens) for item in result.sequences)
        return {
            "row": row,
            "example_id": content_id(row),
            "prompt_tokens": tuple(prompt_tokens),
            "sequences": sequences,
            "responses": tuple(tokenizer.decode(list(tokens)) for tokens in sequences),
        }

    tasks = [asyncio.create_task(one(index, row)) for index, row in enumerate(rows)]
    try:
        samples = tuple(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    return (
        samples,
        sum(len(sample["prompt_tokens"]) * config.rollout_group_size for sample in samples),
        sum(len(tokens) for sample in samples for tokens in sample["sequences"]),
    )


async def _query_teacher_targets(
    samples: Sequence[Mapping[str, Any]],
    references: Mapping[str, ReferenceTrace],
    teacher_client: Any,
    tinker_module: Any,
    config: E10Config,
) -> Tuple[Tuple[TopKPrefixTarget, ...], int, int, int, int, float, float, int]:
    teacher_tokenizer = teacher_client.get_tokenizer()

    async def one(sample: Mapping[str, Any], rollout_id: int) -> Dict[str, Any]:
        example_id = str(sample["example_id"])
        trace = references[example_id]
        context = _teacher_context_prefix(sample["row"], trace, teacher_tokenizer, config)
        response_tokens = tuple(sample["sequences"][rollout_id])
        positions = _prefix_positions(response_tokens, config.prefixes_per_rollout)
        if not positions:
            return {"targets": (), "selected": 0, "teacher_input": 0, "teacher_output": 0}
        teacher_prompt = tuple(context + response_tokens)
        result = await teacher_client.sample_async(
            prompt=tinker_module.ModelInput.from_ints(tokens=list(teacher_prompt)),
            num_samples=1,
            sampling_params=tinker_module.SamplingParams(max_tokens=1, temperature=1.0),
            topk_prompt_logprobs=config.teacher_topk,
        )
        topk = getattr(result, "topk_prompt_logprobs", None)
        if topk is None or len(topk) != len(teacher_prompt):
            raise KDTrainingError("teacher did not return aligned prompt Top-K log-probabilities")
        targets = []
        for position in positions:
            normalized = _normalized_topk(topk[len(context) + position])
            if normalized is None:
                continue
            token_ids, probabilities, mass, entropy = normalized
            student_input = tuple(sample["prompt_tokens"] + response_tokens[:position])
            if not student_input or len(student_input) > config.max_sequence_tokens:
                continue
            targets.append(
                TopKPrefixTarget(
                    example_id=str(sample["example_id"]),
                    rollout_id=rollout_id,
                    prefix_position=position,
                    student_input_tokens=student_input,
                    teacher_token_ids=token_ids,
                    teacher_probs=probabilities,
                    teacher_topk_mass=mass,
                    teacher_entropy=entropy,
                )
            )
        return {
            "targets": tuple(targets),
            "selected": len(positions),
            "teacher_input": len(teacher_prompt),
            "teacher_output": sum(len(sequence.tokens) for sequence in result.sequences),
        }

    tasks = [
        asyncio.create_task(one(sample, rollout_id))
        for sample in samples
        for rollout_id in range(config.rollout_group_size)
    ]
    try:
        queried = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    targets = tuple(target for item in queried for target in item["targets"])
    return (
        targets,
        sum(item["teacher_input"] for item in queried),
        sum(item["teacher_output"] for item in queried),
        sum(item["selected"] for item in queried),
        len(targets),
        sum(target.teacher_topk_mass for target in targets),
        sum(target.teacher_entropy for target in targets),
        sum(len(target.teacher_token_ids) for target in targets),
    )


ON_POLICY_ROLLOUT_COLUMNS = (
    "example_id",
    "optimizer_step",
    "rollout_id",
    "question",
    "ground_truth",
    "generated_response",
    "parsed_answer",
    "correct",
    "output_tokens",
    "format_valid",
    "truncated",
)


async def _collect_on_policy_targets(
    sampler: Any,
    rows: Sequence[Mapping[str, object]],
    references: Mapping[str, ReferenceTrace],
    teacher_client: Any,
    tinker_module: Any,
    config: E10Config,
    step: int,
) -> OnPolicyCollection:
    samples, rollout_prompt_tokens, rollout_output_tokens = await _sample_student_rollouts(
        sampler, rows, tinker_module, config, step
    )
    groups = {
        str(sample["example_id"]): tuple(
            Completion(
                example_id=str(sample["example_id"]),
                response=response,
                ground_truth=str(sample["row"]["answer"]),
                output_tokens=len(tokens),
                max_output_tokens=config.rollout_max_output_tokens,
            )
            for response, tokens in zip(sample["responses"], sample["sequences"], strict=True)
        )
        for sample in samples
    }
    rollout_report = evaluate_groups(groups, pass_k=config.rollout_group_size)
    sample_by_id = {str(sample["example_id"]): sample for sample in samples}
    table_rows = tuple(
        (
            row.example_id,
            step,
            rollout_id % config.rollout_group_size,
            str(sample_by_id[row.example_id]["row"]["question"]),
            str(sample_by_id[row.example_id]["row"]["answer"]),
            row.response,
            row.parsed_answer,
            row.correct,
            row.output_tokens,
            row.format_valid,
            row.truncated,
        )
        for rollout_id, row in enumerate(rollout_report.rows)
    )
    (
        targets,
        teacher_prompt_tokens,
        teacher_output_tokens,
        selected_positions,
        usable_positions,
        topk_mass_sum,
        topk_entropy_sum,
        topk_size_sum,
    ) = await _query_teacher_targets(
        samples, references, teacher_client, tinker_module, config
    )
    metrics = {
        "data/on_policy_rollout_count": float(len(rollout_report.rows)),
        "data/on_policy_rollout_correct_frac": float(rollout_report.metrics["eval/pass_at_1"]),
        "data/on_policy_rollout_format_accuracy": float(rollout_report.metrics["eval/format_accuracy"]),
        "data/on_policy_rollout_truncation_rate": float(rollout_report.metrics["eval/truncation_rate"]),
        "data/on_policy_group_unique_response_frac": float(rollout_report.metrics["eval/group_unique_response_frac"]),
        "data/on_policy_group_all_wrong_frac": float(rollout_report.metrics["eval/group_all_wrong_frac"]),
        "data/on_policy_group_mixed_frac": float(rollout_report.metrics["eval/group_mixed_frac"]),
        "data/on_policy_prefix_tokens": float(selected_positions),
        "data/on_policy_prefix_coverage": usable_positions / selected_positions if selected_positions else 0.0,
        "data/teacher_topk_coverage": topk_mass_sum / usable_positions if usable_positions else 0.0,
        "data/teacher_topk_entropy": topk_entropy_sum / usable_positions if usable_positions else 0.0,
        "data/teacher_topk_mean_k": topk_size_sum / usable_positions if usable_positions else 0.0,
        "data/teacher_topk_unusable_position_frac": (
            1.0 - usable_positions / selected_positions if selected_positions else 1.0
        ),
    }
    return OnPolicyCollection(
        targets=targets,
        rollout_metrics=metrics,
        rollout_table_rows=table_rows,
        rollout_prompt_tokens=rollout_prompt_tokens,
        rollout_output_tokens=rollout_output_tokens,
        teacher_prompt_tokens=teacher_prompt_tokens,
        teacher_output_tokens=teacher_output_tokens,
        selected_prefix_positions=selected_positions,
        usable_prefix_positions=usable_positions,
        topk_mass_sum=topk_mass_sum,
        topk_entropy_sum=topk_entropy_sum,
        topk_size_sum=topk_size_sum,
    )


def _rollout_cost_payload(
    teacher: InferenceCostLedger,
    student_training: TrainingCostLedger,
    student_rollout: InferenceCostLedger,
    development: InferenceCostLedger,
) -> Dict[str, float]:
    return {
        "cost/teacher_generation_usd": teacher.total_usd,
        "cost/teacher_input_usd": teacher.input_usd,
        "cost/teacher_output_usd": teacher.output_usd,
        "cost/student_training_usd": student_training.total_usd,
        "cost/on_policy_student_rollout_inference_usd": student_rollout.total_usd,
        "cost/on_policy_student_rollout_input_usd": student_rollout.input_usd,
        "cost/on_policy_student_rollout_output_usd": student_rollout.output_usd,
        "cost/dev_inference_usd": development.total_usd,
        "cost/dev_input_usd": development.input_usd,
        "cost/dev_output_usd": development.output_usd,
        "cost/cumulative_usd": (
            teacher.total_usd
            + student_training.total_usd
            + student_rollout.total_usd
            + development.total_usd
        ),
    }


def _max_cost_breakdown(config: E10Config) -> Dict[str, float]:
    rollout_count = config.on_policy_examples * config.rollout_group_size
    teacher = _inference_cost_ledger(
        rollout_count * config.teacher_max_context_tokens,
        rollout_count,
        config.teacher_prefill_usd_per_million,
        config.teacher_sample_usd_per_million,
    )
    student_rollout = _student_monitor_cost_ledger(
        rollout_count * MAX_PROMPT_TOKENS,
        rollout_count * config.rollout_max_output_tokens,
    )
    student_training = _student_training_cost_ledger(
        rollout_count
        * config.prefixes_per_rollout
        * config.teacher_topk
        * config.max_sequence_tokens,
        rollout_count * config.prefixes_per_rollout,
        config,
    )
    development_runs = 2
    development = _student_monitor_cost_ledger(
        development_runs
        * config.development_examples
        * config.development_group_size
        * MAX_PROMPT_TOKENS,
        development_runs
        * config.development_examples
        * config.development_group_size
        * MAX_OUTPUT_TOKENS,
    )
    return {
        "teacher_topk_query_max_usd": teacher.total_usd,
        "student_rollout_max_usd": student_rollout.total_usd,
        "student_topk_training_max_usd": student_training.total_usd,
        "development_inference_max_usd": development.total_usd,
        "total_max_usd": (
            teacher.total_usd
            + student_rollout.total_usd
            + student_training.total_usd
            + development.total_usd
        ),
    }


def estimate_max_token_cost_usd(config: E10Config, manifest: SplitManifest) -> float:
    config.validate(manifest)
    return _max_cost_breakdown(config)["total_max_usd"]


def _tracking_config(
    config: E10Config,
    manifest: SplitManifest,
    reference_ids: Sequence[str],
    reference_digest: str,
    tokenizer_vocab_hash: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "attempt": config.attempt,
        "suite_id": config.suite_id,
        "signal_kind": config.signal_kind,
        "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
        "distillation_method": asdict(method_spec(config.signal_kind)),
        "metric_schema": metric_schema_dict(config.signal_kind),
        "student_model_id": config.model_id,
        "initialization_source": config.initialization_source,
        "parent_experiment_id": config.parent_experiment_id,
        "parent_selected_step": config.parent_selected_step,
        "parent_state_path": config.parent_state_path,
        "parent_sampler_path": config.parent_sampler_path,
        "fresh_optimizer": True,
        "teacher_model_id": config.teacher_model_id,
        "teacher_signal": "topk_prompt_logprobs_on_student_prefixes",
        "teacher_reference_policy": "verified_e9_trace_teacher_context_only",
        "reference_trace_path": config.reference_trace_path,
        "reference_trace_digest": reference_digest,
        "on_policy_partition": config.on_policy_partition,
        "on_policy_examples": config.on_policy_examples,
        "on_policy_ids_hash": _ids_hash(reference_ids),
        "on_policy_rollout_group_size": config.rollout_group_size,
        "prefixes_per_rollout": config.prefixes_per_rollout,
        "teacher_topk": config.teacher_topk,
        "topk_target_policy": "teacher_logprobs_renormalized_within_returned_topk",
        "tokenizer_vocab_hash": tokenizer_vocab_hash,
        "dataset_id": manifest.dataset_id,
        "dataset_revision": manifest.dataset_revision,
        "manifest_hash": manifest.manifest_hash,
        "formal_overlap_count": len(set(reference_ids) & set(manifest.formal_test_ids)),
        "development_partition": config.development_partition,
        "development_examples": config.development_examples,
        "development_ids_hash": _ids_hash(manifest.sft_validation_ids[: config.development_examples]),
        "development_group_size": config.development_group_size,
        "checkpoint_selection": "development_pass_at_4_then_pass_at_1",
        "prompt_version": PROMPT_VERSION,
        "learning_rate": config.learning_rate,
        "max_sequence_tokens": config.max_sequence_tokens,
        "hard_cap_usd": config.hard_cap_usd,
        "reference_e4_checkpoint": E4_COMPARISON_CHECKPOINT,
        "reference_e4_formal_pass_at_1": E4_FORMAL_PASS_AT_1,
        "reference_e4_formal_pass_at_4": E4_FORMAL_PASS_AT_4,
        "hypothesis": (
            "Teacher-distribution supervision on E9 student-visited prefixes can "
            "repair residual on-policy errors without another full-corpus teacher run."
        ),
        "expected_failure": (
            "The E9 initialization already saturates this GSM8K protocol, so the "
            "terminal G4 development result fails to beat the parent."
        ),
    }


def build_doctor_report(
    config: E10Config,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
) -> Dict[str, Any]:
    manifest = read_manifest() if manifest is None else manifest
    config.validate(manifest)
    reference_path = Path(config.reference_trace_path)
    reference_traces = _read_reference_traces(reference_path)
    reference_ids = tuple(
        example_id
        for example_id in manifest.rl_train_ids
        if example_id in reference_traces
    )[: config.on_policy_examples]
    if len(reference_ids) != config.on_policy_examples:
        raise KDTrainingError("E9 trace lacks enough rl_train references for E10")
    estimate = _max_cost_breakdown(config)
    return {
        "mode": "local-e10-opd-preflight",
        "network_called": False,
        "run_name": config.run_name,
        "signal_kind": config.signal_kind,
        "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
        "distillation_method": asdict(method_spec(config.signal_kind)),
        "metric_schema": metric_schema_dict(config.signal_kind),
        "parent_state_path": config.parent_state_path,
        "parent_sampler_path": config.parent_sampler_path,
        "fresh_optimizer": True,
        "reference_trace_path": str(reference_path),
        "reference_trace_digest": _trace_digest(reference_traces),
        "on_policy_partition": config.on_policy_partition,
        "on_policy_examples": len(reference_ids),
        "on_policy_ids_hash": _ids_hash(reference_ids),
        "formal_overlap_count": len(set(reference_ids) & set(manifest.formal_test_ids)),
        "development_overlap_count": len(set(reference_ids) & set(manifest.sft_validation_ids)),
        "estimated_cost_breakdown_usd": estimate,
        "hard_cap_usd": config.hard_cap_usd,
        "tinker_api_key_configured": bool(environ.get("TINKER_API_KEY")),
        "wandb_api_key_configured": bool(environ.get("WANDB_API_KEY")),
        "ready_for_paid_run": (
            bool(environ.get("TINKER_API_KEY"))
            and bool(environ.get("WANDB_API_KEY"))
            and environ.get("WANDB_MODE", "").lower() != "offline"
            and estimate["total_max_usd"] <= config.hard_cap_usd
        ),
    }


def _authorize(config: E10Config, manifest: SplitManifest, allow_paid: bool, environ: Mapping[str, str]) -> None:
    config.validate(manifest)
    if not allow_paid:
        raise KDTrainingError("training is blocked; pass --allow-paid after preflight")
    if not environ.get("TINKER_API_KEY") or not environ.get("WANDB_API_KEY"):
        raise KDTrainingError("TINKER_API_KEY and WANDB_API_KEY are required")
    if environ.get("WANDB_MODE", "").lower() == "offline":
        raise KDTrainingError("WANDB_MODE=offline cannot produce the required dashboard")
    if estimate_max_token_cost_usd(config, manifest) > config.hard_cap_usd:
        raise KDTrainingError("estimated maximum token cost exceeds the hard cap")


def _select_rows_and_references(
    rows: Sequence[Mapping[str, object]],
    manifest: SplitManifest,
    traces: Mapping[str, ReferenceTrace],
    config: E10Config,
) -> Tuple[Tuple[Mapping[str, object], ...], Tuple[str, ...]]:
    if tuple(content_id(row) for row in rows) != manifest.rl_train_ids:
        raise KDTrainingError("on-policy rows do not exactly match the frozen rl_train split")
    selected = tuple(row for row in rows if content_id(row) in traces)[: config.on_policy_examples]
    ids = tuple(content_id(row) for row in selected)
    if len(selected) != config.on_policy_examples:
        raise KDTrainingError("E9 trace lacks enough accepted rl_train references")
    if set(ids) & (set(manifest.sft_validation_ids) | set(manifest.formal_test_ids)):
        raise KDTrainingError("on-policy E10 rows overlap development or formal data")
    return selected, ids


def _metric_dictionary_path(config: E10Config, run_id: str) -> Path:
    return Path(config.trace_output_dir) / f"{config.experiment_id}_metric_dictionary_{run_id}.md"


def _report_path(config: E10Config, run_id: str) -> Path:
    return Path(config.trace_output_dir) / f"{config.experiment_id}_opd_report_{run_id}.json"


def _log(wandb_run: Any, config: E10Config, payload: Mapping[str, Any], step: int) -> None:
    extra = validate_logged_metric_keys(config.signal_kind, payload.keys())
    if extra:
        raise KDTrainingError("distillation metric schema is missing: " + ", ".join(extra))
    wandb_run.log(dict(payload), step=step)


def _is_better(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    return (candidate["pass_at_4"], candidate["pass_at_1"]) > (
        current["pass_at_4"], current["pass_at_1"]
    )


async def run_e10_training(
    config: E10Config,
    allow_paid: bool,
    manifest: Optional[SplitManifest] = None,
    environ: Mapping[str, str] = os.environ,
    tinker_module: Any = None,
    wandb_module: Any = None,
    service_client: Any = None,
    rl_rows: Optional[Sequence[Mapping[str, object]]] = None,
    development_rows: Optional[Sequence[Mapping[str, object]]] = None,
    clock: Callable[[], float] = time.monotonic,
    progress: Callable[[str], None] = _progress,
) -> Dict[str, Any]:
    """Run the fixed E10 protocol after an explicit paid-run acknowledgement."""
    manifest = read_manifest() if manifest is None else manifest
    _authorize(config, manifest, allow_paid, environ)
    if tinker_module is None:
        try:
            import tinker as tinker_module
        except ImportError as exc:
            raise KDTrainingError("Tinker SDK is unavailable; run with `uv run --extra tinker`") from exc
    if wandb_module is None:
        try:
            import wandb as wandb_module
        except ImportError as exc:
            raise KDTrainingError("Weights & Biases is unavailable") from exc
    references = _read_reference_traces(Path(config.reference_trace_path))
    if rl_rows is None:
        progress("loading frozen rl_train on-policy rows")
        rl_rows = load_official_train_rows(manifest, config.on_policy_partition)
    selected_rows, selected_ids = _select_rows_and_references(rl_rows, manifest, references, config)
    if development_rows is None:
        progress("loading frozen sft_validation development rows")
        development_rows = load_official_train_rows(manifest, config.development_partition)
    if tuple(content_id(row) for row in development_rows) != manifest.sft_validation_ids:
        raise KDTrainingError("development rows do not exactly match frozen sft_validation")
    development_rows = tuple(development_rows[: config.development_examples])

    owned_http_client = None
    if service_client is None:
        import httpx

        owned_http_client = httpx.AsyncClient(follow_redirects=True)
        service_client = tinker_module.ServiceClient(
            user_metadata={"experiment_id": config.experiment_id, "suite_id": config.suite_id},
            http_client=owned_http_client,
        )

    wandb_run = None
    try:
        run_started = clock()
        max_cost = estimate_max_token_cost_usd(config, manifest)
        progress(
            f"authorized run={config.run_name} parent=e9-step{config.parent_selected_step} "
            f"rl_train={len(selected_rows)} g={config.rollout_group_size} "
            f"topk={config.teacher_topk} max_cost=${max_cost:.4f}"
        )
        progress("restoring E9 selected weights with a fresh E10 optimizer")
        training_client = await service_client.create_training_client_from_state_async(
            config.parent_state_path,
            base_model=config.model_id,
            user_metadata={"experiment_id": config.experiment_id},
        )
        progress("creating frozen external-teacher Top-K client")
        teacher_client = await service_client.create_sampling_client_async(base_model=config.teacher_model_id)
        tokenizer_hash = _require_identical_tokenizers(
            training_client.get_tokenizer(), teacher_client.get_tokenizer()
        )
        reference_digest = _trace_digest(references)
        wandb_run = wandb_module.init(
            project=config.project,
            entity=environ.get("WANDB_ENTITY") or None,
            name=config.run_name,
            group=config.suite_id,
            job_type="on-policy-topk-kd",
            tags=["gsm8k", "kd", "opd", "topk", config.experiment_id, "from-e9"],
            config=_tracking_config(config, manifest, selected_ids, reference_digest, tokenizer_hash),
        )
        configure_wandb_metrics(wandb_run, config.signal_kind)
        metric_dictionary_path = _metric_dictionary_path(config, str(getattr(wandb_run, "id", "run")))
        write_metric_dictionary(metric_dictionary_path, config.signal_kind)
        progress(f"started W&B run={getattr(wandb_run, 'url', None)}")

        teacher_ledger = _teacher_cost_ledger(0, 0, config)
        rollout_ledger = _student_monitor_cost_ledger(0, 0)
        student_ledger = _student_training_cost_ledger(0, 0, config)
        development_ledger = _student_monitor_cost_ledger(0, 0)
        completed_input_tokens = 0
        completed_weighted_positions = 0.0
        teacher_prompt_tokens = teacher_output_tokens = 0
        rollout_prompt_tokens = rollout_output_tokens = 0
        development_prompt_tokens = development_output_tokens = 0

        baseline = await _generation_development(
            service_client,
            config.parent_sampler_path,
            development_rows,
            tinker_module,
            config,
            0,
            "step=0 initialization=E9",
            progress,
        )
        development_prompt_tokens += baseline.prompt_tokens
        development_output_tokens += baseline.output_tokens
        development_ledger = _student_monitor_cost_ledger(development_prompt_tokens, development_output_tokens)
        checkpoints = [
            {
                "step": 0,
                "state_path": config.parent_state_path,
                "sampler_path": config.parent_sampler_path,
                "pass_at_1": baseline.metrics["eval/pass_at_1"],
                "pass_at_4": baseline.metrics["eval/pass_at_4"],
                "development_metrics": _development_metrics(baseline),
            }
        ]
        initial_payload: Dict[str, Any] = {
            "train/optimizer_step": 0.0,
            "train/optimized_input_tokens": 0.0,
            "train/supervised_or_weighted_tokens": 0.0,
            "dev/checkpoint_step": 0.0,
            "dev/optimized_input_tokens": 0.0,
            "dev/generated_rollouts": float(len(baseline.table_rows)),
            "dev/prompt_tokens": float(baseline.prompt_tokens),
            "dev/output_tokens": float(baseline.output_tokens),
            "dev/is_initialization_policy": 1.0,
            **_development_metrics(baseline),
            **_rollout_cost_payload(teacher_ledger, student_ledger, rollout_ledger, development_ledger),
        }
        if hasattr(wandb_module, "Table"):
            initial_payload["tables/development_rollouts"] = wandb_module.Table(
                columns=[
                    "example_id", "checkpoint_step", "rollout_id", "question", "ground_truth",
                    "generated_response", "parsed_answer", "correct", "output_tokens", "format_valid",
                    "truncated", "process_checked_steps", "process_valid_steps", "process_invalid_steps",
                ],
                data=list(baseline.table_rows),
            )
        _log(wandb_run, config, initial_payload, step=0)

        for step, start in enumerate(range(0, len(selected_rows), config.rollout_batch_size), start=1):
            step_started = clock()
            batch_rows = selected_rows[start : start + config.rollout_batch_size]
            current_sampler = await training_client.save_weights_and_get_sampling_client_async()
            collection = await _collect_on_policy_targets(
                current_sampler, batch_rows, references, teacher_client, tinker_module, config, step
            )
            if not collection.targets:
                raise KDTrainingError("E10 collected no usable Top-K targets for an update")
            data, step_input_tokens, step_weighted_positions = _student_topk_datums(collection.targets, tinker_module)
            forward_backward = await training_client.forward_backward_async(data, "cross_entropy")
            result = await forward_backward.result_async()
            optimizer = await training_client.optim_step_async(
                tinker_module.types.AdamParams(learning_rate=config.learning_rate)
            )
            await optimizer.result_async()
            completed_input_tokens += step_input_tokens
            completed_weighted_positions += step_weighted_positions
            teacher_prompt_tokens += collection.teacher_prompt_tokens
            teacher_output_tokens += collection.teacher_output_tokens
            rollout_prompt_tokens += collection.rollout_prompt_tokens
            rollout_output_tokens += collection.rollout_output_tokens
            teacher_ledger = _teacher_cost_ledger(teacher_prompt_tokens, teacher_output_tokens, config)
            rollout_ledger = _student_monitor_cost_ledger(rollout_prompt_tokens, rollout_output_tokens)
            student_ledger = _student_training_cost_ledger(completed_input_tokens, int(round(completed_weighted_positions)), config)
            cross_entropy = _loss_sum(result) / collection.usable_prefix_positions
            teacher_entropy = collection.topk_entropy_sum / collection.usable_prefix_positions
            payload: Dict[str, Any] = {
                "train/optimizer_step": float(step),
                "train/optimized_input_tokens": float(completed_input_tokens),
                "train/supervised_or_weighted_tokens": float(completed_weighted_positions),
                "train/learning_rate": config.learning_rate,
                "train/topk_kd_cross_entropy": cross_entropy,
                "train/topk_teacher_to_student_kl": cross_entropy - teacher_entropy,
                "timing/step_seconds": clock() - step_started,
                "timing/elapsed_seconds": clock() - run_started,
                **collection.rollout_metrics,
                **_rollout_cost_payload(teacher_ledger, student_ledger, rollout_ledger, development_ledger),
            }
            if hasattr(wandb_module, "Table"):
                payload["tables/on_policy_rollouts"] = wandb_module.Table(
                    columns=list(ON_POLICY_ROLLOUT_COLUMNS), data=list(collection.rollout_table_rows)
                )
            _log(wandb_run, config, payload, step=step)
            progress(
                f"step={step}/{config.training_steps} ce={cross_entropy:.5f} "
                f"topk_kl={cross_entropy - teacher_entropy:.5f} prefixes="
                f"{collection.usable_prefix_positions}/{collection.selected_prefix_positions} "
                f"unique={collection.rollout_metrics['data/on_policy_group_unique_response_frac']:.3f} "
                f"actual_token_priced_cost=${_rollout_cost_payload(teacher_ledger, student_ledger, rollout_ledger, development_ledger)['cost/cumulative_usd']:.4f}"
            )
            progress(_format_inference_cost("teacher Top-K actual", teacher_ledger))
            progress(_format_inference_cost("student on-policy rollout actual", rollout_ledger))
            progress(_format_training_cost("student Top-K KD actual", student_ledger))

        checkpoint_name = f"{config.run_name}-step{config.training_steps}"
        state_future = await training_client.save_state_async(checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds)
        state_result = await state_future.result_async()
        sampler_future = await training_client.save_weights_for_sampler_async(checkpoint_name, ttl_seconds=config.checkpoint_ttl_seconds)
        sampler_result = await sampler_future.result_async()
        terminal = await _generation_development(
            service_client,
            str(sampler_result.path),
            development_rows,
            tinker_module,
            config,
            config.training_steps,
            f"step={config.training_steps}",
            progress,
        )
        development_prompt_tokens += terminal.prompt_tokens
        development_output_tokens += terminal.output_tokens
        development_ledger = _student_monitor_cost_ledger(development_prompt_tokens, development_output_tokens)
        terminal_record = {
            "step": config.training_steps,
            "state_path": str(state_result.path),
            "sampler_path": str(sampler_result.path),
            "pass_at_1": terminal.metrics["eval/pass_at_1"],
            "pass_at_4": terminal.metrics["eval/pass_at_4"],
            "development_metrics": _development_metrics(terminal),
        }
        checkpoints.append(terminal_record)
        terminal_payload: Dict[str, Any] = {
            "dev/checkpoint_step": float(config.training_steps),
            "dev/optimized_input_tokens": float(completed_input_tokens),
            "dev/generated_rollouts": float(len(terminal.table_rows)),
            "dev/prompt_tokens": float(terminal.prompt_tokens),
            "dev/output_tokens": float(terminal.output_tokens),
            "dev/is_initialization_policy": 0.0,
            "checkpoint/state_path": terminal_record["state_path"],
            "checkpoint/sampler_path": terminal_record["sampler_path"],
            **terminal_record["development_metrics"],
            **_rollout_cost_payload(teacher_ledger, student_ledger, rollout_ledger, development_ledger),
        }
        if hasattr(wandb_module, "Table"):
            terminal_payload["tables/development_rollouts"] = wandb_module.Table(
                columns=[
                    "example_id", "checkpoint_step", "rollout_id", "question", "ground_truth",
                    "generated_response", "parsed_answer", "correct", "output_tokens", "format_valid",
                    "truncated", "process_checked_steps", "process_valid_steps", "process_invalid_steps",
                ],
                data=list(terminal.table_rows),
            )
        _log(wandb_run, config, terminal_payload, step=config.training_steps)
        selected = checkpoints[0]
        for candidate in checkpoints[1:]:
            if _is_better(candidate, selected):
                selected = candidate
        final_cost = _rollout_cost_payload(teacher_ledger, student_ledger, rollout_ledger, development_ledger)
        if final_cost["cost/cumulative_usd"] > config.hard_cap_usd:
            raise KDTrainingError("observed token cost exceeded the configured hard cap")
        report = {
            "distillation_schema_version": DISTILLATION_SCHEMA_VERSION,
            "distillation_method": asdict(method_spec(config.signal_kind)),
            "metric_schema": metric_schema_dict(config.signal_kind),
            "mode": "on-policy-topk-external-teacher-kd",
            "network_called": True,
            "run_name": config.run_name,
            "parent_state_path": config.parent_state_path,
            "parent_sampler_path": config.parent_sampler_path,
            "fresh_optimizer": True,
            "teacher_model_id": config.teacher_model_id,
            "reference_trace_path": config.reference_trace_path,
            "reference_trace_digest": reference_digest,
            "on_policy_ids_hash": _ids_hash(selected_ids),
            "on_policy_examples": len(selected_ids),
            "formal_overlap_count": len(set(selected_ids) & set(manifest.formal_test_ids)),
            "development_overlap_count": len(set(selected_ids) & set(manifest.sft_validation_ids)),
            "student_optimized_input_tokens": completed_input_tokens,
            "student_weighted_positions": completed_weighted_positions,
            "training_steps": config.training_steps,
            "selected_checkpoint": selected,
            "selected_checkpoint_is_initialization": selected["step"] == 0,
            "checkpoints": checkpoints,
            "metric_dictionary_path": str(metric_dictionary_path),
            "actual_token_priced_cost_ledger": {
                "teacher_topk_queries": asdict(teacher_ledger),
                "student_on_policy_rollouts": asdict(rollout_ledger),
                "student_training": asdict(student_ledger),
                "development_inference": asdict(development_ledger),
                "total_usd": final_cost["cost/cumulative_usd"],
            },
            "actual_token_priced_total_usd": final_cost["cost/cumulative_usd"],
            "hard_cap_usd": config.hard_cap_usd,
            "wandb_run_url": getattr(wandb_run, "url", None),
        }
        report_path = _report_path(config, str(getattr(wandb_run, "id", "run")))
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        summary = {
            "schema/version": DISTILLATION_SCHEMA_VERSION,
            "schema/method": config.signal_kind,
            "train/optimized_input_tokens": completed_input_tokens,
            "selection/selected_checkpoint_step": selected["step"],
            "selection/selected_is_initialization": float(selected["step"] == 0),
            "selection/selected_dev_pass_at_1": selected["pass_at_1"],
            "selection/selected_dev_pass_at_4": selected["pass_at_4"],
            "dev/pass_at_1": selected["pass_at_1"],
            "dev/pass_at_4": selected["pass_at_4"],
            **final_cost,
        }
        wandb_run.summary.update(summary)
        progress(_format_inference_cost("teacher Top-K final actual", teacher_ledger))
        progress(_format_inference_cost("student on-policy rollout final actual", rollout_ledger))
        progress(_format_training_cost("student Top-K KD final actual", student_ledger))
        progress(_format_inference_cost("development cumulative actual", development_ledger))
        progress(
            f"complete selected_step={selected['step']} selected_initialization="
            f"{selected['step'] == 0} optimized_input_tokens={completed_input_tokens} "
            f"actual_token_priced_cost=${final_cost['cost/cumulative_usd']:.4f}"
        )
        return report
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if owned_http_client is not None:
            await owned_http_client.aclose()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight or run E10 on-policy Top-K distillation.")
    parser.add_argument("--run", action="store_true", help="Start the paid E10 run.")
    parser.add_argument("--allow-paid", action="store_true", help="Acknowledge the bounded Tinker cost.")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--hard-cap-usd", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--reference-trace-path")
    parser.add_argument("--trace-output-dir")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> E10Config:
    overrides = {
        key: value
        for key, value in {
            "attempt": args.attempt,
            "hard_cap_usd": args.hard_cap_usd,
            "learning_rate": args.learning_rate,
            "reference_trace_path": args.reference_trace_path,
            "trace_output_dir": args.trace_output_dir,
        }.items()
        if value is not None
    }
    return E10Config(**overrides)


async def _async_main(args: argparse.Namespace) -> Dict[str, Any]:
    config = _config_from_args(args)
    if args.run:
        return await run_e10_training(config, allow_paid=args.allow_paid)
    if args.allow_paid:
        raise KDTrainingError("--allow-paid requires --run")
    return build_doctor_report(config)


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv(dotenv_path=ENV_FILE, override=False)
    report = asyncio.run(_async_main(parse_args(argv)))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
