"""Synchronous on-policy distillation: fresh student rollout, teacher score, one update."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from time import perf_counter

from .evaluate import digest, evaluate, read_jsonl
from .inference import PROMPT
from .kd import assess
from .kd_collect import (
    CollectionBudget,
    TOP_K,
    completion_topk,
    TEACHER_FORWARD_RATE,
    TEACHER_SAMPLE_RATE,
    json_hash,
    tokenizer_identity,
)
from .metrics import VERSION, parse_table, score_tables
from .kd_collection_budget import ConcurrentBudget
from .official import OfficialScorer, REVISION
from .opd_sampling_budget import BudgetedSampler
from .opd_targets import native_batch, policy_datum, summarize_policy
from .opd_ablation_objectives import hybrid_batches, normalize_ce_batch
from .opd_diagnostics import classify_token_categories, summarize_diagnostics
from .kd_targets import soft_targets, summarize_soft, topk_diagnostics
from .sft_metrics import summarize_nll
from .sft import (
    MODEL,
    TRAIN_RATE,
    FORWARD_RATE,
    SAMPLE_RATE,
    evaluation_steps,
    generate,
    prepare_examples,
    scheduled_lr,
    validate_records,
    write_json,
)
from .tinker_inference import (
    COOKBOOK_REVISION,
    PROCESSOR_REVISION,
    TEACHER_MODEL,
    image_message,
    load_renderer,
)
from .teacher_config import TEACHERS, teacher_spec
from .training_logging import TrainingLogger, evaluation_metrics


def prepare_prompts(rows, root, renderer, teacher_renderer, args):
    prompts = []
    for row in rows:
        message = image_message(root / row["image"], args.max_pixels)
        prompt = renderer.build_generation_prompt([message])
        teacher_prompt = teacher_renderer.build_generation_prompt([message])
        if json_hash(prompt.model_dump(mode="json")) != json_hash(
            teacher_prompt.model_dump(mode="json")
        ):
            raise ValueError("Teacher/student multimodal prompts differ")
        if prompt.length + args.max_new_tokens > args.max_sequence_tokens:
            raise ValueError("Prompt plus rollout cap exceeds context")
        prompts.append((row, prompt))
    return prompts


def estimate_cost(train, dev, args, gold_train=()):
    """Report both a length scenario and the worst output-cap bound, not a quote."""
    selected_teacher = teacher_spec(getattr(args, "teacher_model", TEACHER_MODEL))
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    prefill = sum(p.length for _, p in train) * args.epochs
    forward = len(evaluation_steps(steps, args.eval_every)) * sum(
        d.model_input.length for _, _, d in dev
    )
    eval_steps = evaluation_steps(steps, args.eval_every)
    every = getattr(args, "generate_dev_every", 0)
    generations = int(args.generate_dev) + sum(
        bool(every and 0 < step < steps and step % every == 0) for step in eval_steps
    )
    dev_prefill = generations * sum(p.length for _, p, _ in dev)
    dev_output = generations * len(dev) * args.max_new_tokens
    objective = getattr(args, "objective", "sampled_reverse_kl")
    gold_tokens = (
        sum(d.model_input.length for _, _, d in gold_train) * args.epochs
        if objective == "hybrid_gold"
        else 0
    )
    probe_tokens = len(eval_steps) * sum(
        d.model_input.length
        for _, _, d in gold_train[: getattr(args, "train_probe_examples", 0)]
    )
    diag_every = getattr(args, "diagnostic_every", 0)
    diag_steps = sum(
        bool(diag_every and (step == 1 or step % diag_every == 0 or step == steps))
        for step in range(1, steps + 1)
    )
    diag_passes = diag_steps * (2 if objective == "topk_forward_kl" else 1)
    diag_prefill_bound = args.batch_size * max(p.length for _, p in train)

    def cost(length):
        output = len(train) * args.epochs * length
        return (
            prefill * FORWARD_RATE
            + output * SAMPLE_RATE
            + (prefill + output) * selected_teacher.forward_rate
            + len(train) * args.epochs * selected_teacher.sample_rate
            + (prefill + output - len(train) * args.epochs) * TRAIN_RATE
            + gold_tokens * TRAIN_RATE
            + (
                forward
                + dev_prefill
                + probe_tokens
                + diag_passes * (diag_prefill_bound + args.batch_size * (length - 1))
            )
            * FORWARD_RATE
            + dev_output * SAMPLE_RATE
        ) / 1e6

    return {
        "training_tokens": prefill
        + len(train) * args.epochs * args.max_new_tokens
        - len(train) * args.epochs,
        "nll_forward_tokens": forward,
        "gold_training_tokens": gold_tokens,
        "train_probe_forward_tokens": probe_tokens,
        "diagnostic_forward_tokens_bound": diag_passes
        * (diag_prefill_bound + args.batch_size * (args.max_new_tokens - 1)),
        "dev_generation_stages": generations,
        "generation_prefill_tokens": dev_prefill,
        "generation_output_token_bound": dev_output,
        "estimated_usd_bound": cost(args.max_new_tokens),
        "estimated_usd_length_scenario": cost(args.expected_output_tokens),
        "expected_output_tokens_per_rollout": args.expected_output_tokens,
        "execution_budget_usd": args.max_estimated_usd,
    }


def collect_one(sampler, teacher, row, prompt, tokenizer, stop, args, step, position):
    """Persist each paid response before the next request; no application retries."""
    import tinker

    path = args.output_dir / "rollouts" / f"{step:04d}_{position:02d}.json"
    sequence = (
        sampler.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                temperature=1,
                top_p=1,
                top_k=-1,
                stop=stop,
                max_tokens=args.max_new_tokens,
                seed=args.seed + 1000 * step + position,
            ),
        )
        .result(timeout=600)
        .sequences[0]
    )
    record = {
        "id": row["id"],
        "image_sha256": row["image_sha256"],
        "prompt_sha256": json_hash(prompt.model_dump(mode="json")),
        "optimizer_step_before_rollout": step - 1,
        "tokens": sequence.tokens,
        "sampling_logprobs": sequence.logprobs,
        "stop_reason": sequence.stop_reason,
        "prompt_tokens": prompt.length,
    }
    write_json(path, record)
    if not sequence.tokens or sequence.stop_reason not in {"stop", "length"}:
        raise ValueError("Empty or unknown student completion")
    # Validate sampled IDs/logprobs before spending on teacher scoring.
    policy_datum(
        prompt, sequence.tokens, sequence.logprobs, sequence.logprobs, len(tokenizer)
    )
    if len(sequence.tokens) > args.max_new_tokens:
        raise ValueError("Student exceeded output cap")
    full = prompt.append(tinker.EncodedTextChunk(tokens=sequence.tokens))
    if getattr(args, "objective", "sampled_reverse_kl") == "topk_forward_kl":
        response = teacher.sample(
            prompt=full,
            num_samples=1,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=TOP_K,
            sampling_params=tinker.SamplingParams(
                max_tokens=1, temperature=0, seed=args.seed
            ),
        ).result(timeout=600)
        arrays = response.topk_prompt_logprobs_np
        if arrays is not None:
            np.savez_compressed(
                path.with_suffix(".raw-topk.npz"),
                token_ids=arrays.token_ids,
                logprobs=arrays.logprobs,
            )
        teacher_logprobs = response.prompt_logprobs
        record["teacher_full_logprobs"] = teacher_logprobs
        write_json(path, record)
        ids, probs = completion_topk(response, prompt.length, len(sequence.tokens))
        soft_targets(prompt, sequence.tokens, ids, probs, len(tokenizer))
        np.savez_compressed(
            path.with_suffix(".topk.npz"), token_ids=ids, logprobs=probs
        )
        record["teacher_topk_sha256"] = digest(path.with_suffix(".topk.npz"))
        record["retained_mass"] = topk_diagnostics(probs)
    else:
        teacher_logprobs = teacher.compute_logprobs(full).result(timeout=600)
    record["teacher_full_logprobs"] = teacher_logprobs
    write_json(path, record)
    if len(teacher_logprobs) != full.length:
        raise ValueError("Teacher logprob length differs from full sequence")
    datum = policy_datum(
        prompt,
        sequence.tokens,
        sequence.logprobs,
        teacher_logprobs[prompt.length :],
        len(tokenizer),
    )
    html = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    try:
        parse_table(html)
        valid = True
    except ValueError:
        valid = False
    record.update(html=html, format_valid=valid)
    write_json(path, record)
    return datum, {
        "rollout_tokens": len(sequence.tokens),
        "prefill_tokens": prompt.length,
        "teacher_forward_tokens": full.length,
        "training_tokens": datum.model_input.length,
        "format_invalid": int(not valid),
        "truncated": int(sequence.stop_reason == "length"),
    }


def run(args, train, dev, tokenizer, renderer, scorer, report, logger):
    import tinker
    from tinker.lib.retry_handler import RetryConfig

    selected_teacher = teacher_spec(getattr(args, "teacher_model", TEACHER_MODEL))
    budget = CollectionBudget(args.output_dir / "usage.json", args.max_estimated_usd)
    service = tinker.ServiceClient()
    retry = RetryConfig(enable_retry_logic=False)
    teacher = service.create_sampling_client(
        base_model=selected_teacher.model, retry_config=retry
    )
    client = service.create_lora_training_client(
        base_model=MODEL,
        rank=args.rank,
        seed=args.seed,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    objective = getattr(args, "objective", "sampled_reverse_kl")
    diagnostic_every = getattr(args, "diagnostic_every", 0)
    probe_count = getattr(args, "train_probe_examples", 0)
    gold_rows = [row for row, _ in train]
    if objective != "hybrid_gold":
        gold_rows = gold_rows[:probe_count]
    gold = (
        prepare_examples(
            gold_rows,
            args.data_root,
            renderer,
            args.max_pixels,
            args.max_sequence_tokens,
        )
        if gold_rows
        else []
    )
    gold_by_id = {row["id"]: datum for row, _, datum in gold}
    probe = gold[:probe_count]
    report.update(
        status="running", training_info=client.get_info().model_dump(mode="json")
    )
    report["intermediate_sampler_paths"] = {}
    report["extra_usage"] = dict.fromkeys(
        (
            "gold_training_tokens",
            "diagnostic_forward_tokens",
            "train_probe_forward_tokens",
        ),
        0,
    )
    write_json(args.output_dir / "run.json", report)

    def dev_assess(stage, step):
        amount = sum(d.model_input.length for _, _, d in dev) * FORWARD_RATE / 1e6
        budget.reserve(f"{stage}:dev_nll", amount)
        assess(client, [], dev, stage, step, args, report, logger, False)
        budget.settle(amount)
        if probe:
            count = sum(d.model_input.length for _, _, d in probe)
            budget.reserve(f"{stage}:train_probe_nll", count * FORWARD_RATE / 1e6)
            output = client.forward(
                [d for _, _, d in probe], loss_fn="cross_entropy"
            ).result(timeout=600)
            raw = []
            for (row, _, datum), result in zip(
                probe, output.loss_fn_outputs, strict=True
            ):
                weights = datum.loss_fn_inputs["weights"].data
                raw.append(
                    {
                        "id": row["id"],
                        "token_logprobs": [
                            v
                            for v, w in zip(
                                result["logprobs"].data, weights, strict=True
                            )
                            if w
                        ],
                        "target_tokens": [
                            v
                            for v, w in zip(
                                datum.loss_fn_inputs["target_tokens"].data,
                                weights,
                                strict=True,
                            )
                            if w
                        ],
                    }
                )
            write_json(args.output_dir / f"{stage}_train_probe_likelihoods.json", raw)
            summary = summarize_nll(output, [d for _, _, d in probe])
            summary.pop("per_example", None)
            report[stage]["train"] = summary
            report["extra_usage"]["train_probe_forward_tokens"] += count
            budget.settle(count * FORWARD_RATE / 1e6)
            logger.log(evaluation_metrics(report[stage]))
        write_json(args.output_dir / f"{stage}_metrics.json", report[stage])

    def generate_dev(stage, step, sampler_path):
        nonlocal budget
        budget = ConcurrentBudget(
            args.output_dir / "usage.json", args.max_estimated_usd
        )
        sampler = service.create_sampling_client(
            model_path=sampler_path, retry_config=retry
        )
        predictions = generate(
            BudgetedSampler(
                sampler, budget, args.output_dir / f"{stage}_dev_sampling_receipts"
            ),
            dev,
            tokenizer,
            renderer.get_stop_sequences(),
            args,
            args.output_dir / f"{stage}_dev_predictions.jsonl",
        )
        details, metrics = evaluate(
            [r for r, _, _ in dev], args.data_root, predictions, scorer
        )
        write_json(args.output_dir / f"{stage}_dev_details.json", details)
        report[stage]["dev"].update(metrics)
        write_json(args.output_dir / f"{stage}_metrics.json", report[stage])
        logger.log(evaluation_metrics(report[stage]))
        report["estimated_compute_usd"] = budget.state["estimated_compute_usd"]
        write_json(args.output_dir / "run.json", report)

    dev_assess("before", 0)
    rng, step = random.Random(args.seed), 0
    totals = dict.fromkeys(
        (
            "rollout_tokens",
            "prefill_tokens",
            "teacher_forward_tokens",
            "training_tokens",
            "format_invalid",
            "truncated",
        ),
        0,
    )
    started = perf_counter()
    with (args.output_dir / "steps.jsonl").open("x") as stream:
        for epoch in range(args.epochs):
            order = list(train)
            rng.shuffle(order)
            for start in range(0, len(order), args.batch_size):
                step += 1
                batch = order[start : start + args.batch_size]
                tick = perf_counter()
                diagnostic = bool(
                    diagnostic_every
                    and (
                        step == 1
                        or step % diagnostic_every == 0
                        or step == report["optimizer_steps"]
                    )
                )
                prefill = sum(p.length for _, p in batch)
                cap = len(batch) * args.max_new_tokens
                gold_batch = (
                    [gold_by_id[row["id"]] for row, _ in batch]
                    if objective == "hybrid_gold"
                    else []
                )
                gold_tokens = sum(d.model_input.length for d in gold_batch)
                diagnostic_passes = int(diagnostic) * (
                    2 if objective == "topk_forward_kl" else 1
                )
                bound = (
                    prefill * FORWARD_RATE
                    + cap * SAMPLE_RATE
                    + (prefill + cap) * selected_teacher.forward_rate
                    + len(batch) * selected_teacher.sample_rate
                    + (prefill + cap - len(batch) + gold_tokens) * TRAIN_RATE
                    + diagnostic_passes * (prefill + cap - len(batch)) * FORWARD_RATE
                ) / 1e6
                budget.reserve(f"step_{step}:rollout_teacher_backward", bound)
                # Fresh weights each batch; no trajectory replay or cross-batch prefetch.
                sampler = client.save_weights_and_get_sampling_client(
                    retry_config=retry
                )
                with ThreadPoolExecutor(max_workers=args.inference_concurrency) as pool:
                    futures = [
                        pool.submit(
                            collect_one,
                            sampler,
                            teacher,
                            row,
                            prompt,
                            tokenizer,
                            renderer.get_stop_sequences(),
                            args,
                            step,
                            j,
                        )
                        for j, (row, prompt) in enumerate(batch)
                    ]
                    try:
                        collected = [f.result() for f in futures]
                    except Exception:
                        for future in futures:
                            future.cancel()
                        raise
                datums = [d for d, _ in collected]
                native = native_batch(datums)
                pre_policy = None
                metrics = {}
                if objective == "topk_forward_kl":
                    soft = []
                    masses = []
                    for j, ((_, prompt), datum) in enumerate(
                        zip(batch, datums, strict=True)
                    ):
                        path = args.output_dir / "rollouts" / f"{step:04d}_{j:02d}.json"
                        row = json.loads(path.read_text())
                        assert (
                            digest(path.with_suffix(".topk.npz"))
                            == row["teacher_topk_sha256"]
                        )
                        with np.load(
                            path.with_suffix(".topk.npz"), allow_pickle=False
                        ) as arrays:
                            soft.append(
                                soft_targets(
                                    prompt,
                                    row["tokens"],
                                    arrays["token_ids"],
                                    arrays["logprobs"],
                                    len(tokenizer),
                                )
                            )
                            masses.extend(np.exp(arrays["logprobs"]).sum(axis=1))
                    if diagnostic:
                        pre_policy = client.forward(
                            native, loss_fn="importance_sampling"
                        ).result(timeout=600)
                    output = client.forward_backward(
                        normalize_ce_batch(soft), loss_fn="cross_entropy"
                    ).result(timeout=600)
                    metrics.update(summarize_soft(output, soft))
                    metrics.update(
                        retained_mass_mean=float(np.mean(masses)),
                        retained_mass_p05=float(np.quantile(masses, 0.05)),
                    )
                else:
                    if objective == "hybrid_gold":
                        native, gold_native = hybrid_batches(
                            datums, gold_batch, opd_weight=1 - args.gold_weight
                        )
                    output = client.forward_backward(
                        native, loss_fn="importance_sampling"
                    ).result(timeout=600)
                    pre_policy = output
                    if gold_batch:
                        gold_output = client.forward_backward(
                            gold_native, loss_fn="cross_entropy"
                        ).result(timeout=600)
                        write_json(
                            args.output_dir
                            / "rollouts"
                            / f"{step:04d}_gold_learner.json",
                            [
                                r["logprobs"].to_numpy().tolist()
                                for r in gold_output.loss_fn_outputs
                            ],
                        )
                        metrics["gold_batch_nll"] = summarize_nll(
                            gold_output, gold_batch
                        )["assistant_nll"]
                write_json(
                    args.output_dir / "rollouts" / f"{step:04d}_learner.json",
                    [r["logprobs"].to_numpy().tolist() for r in output.loss_fn_outputs],
                )
                if pre_policy is not None:
                    metrics.update(summarize_policy(pre_policy, datums))
                lr = scheduled_lr(step, args.learning_rate, report["warmup_steps"])
                optim = client.optim_step(
                    tinker.AdamParams(
                        learning_rate=lr,
                        beta1=0.9,
                        beta2=0.95,
                        eps=1e-8,
                        weight_decay=0,
                        grad_clip_norm=1,
                    )
                ).result(timeout=600)
                returned = getattr(optim, "metrics", None) or {}
                grad = returned.get("unclipped_grad_l2:mean")
                available = (
                    isinstance(grad, (int, float))
                    and not isinstance(grad, bool)
                    and math.isfinite(grad)
                    and grad >= 0
                )
                metrics["gradient_norm_available"] = int(available)
                if available:
                    metrics.update(
                        unclipped_gradient_norm=float(grad),
                        gradient_norm_exceeds_clip_threshold=int(grad > 1),
                    )
                diagnostic_metrics = {}
                if diagnostic:
                    post = client.forward(
                        native_batch(datums), loss_fn="importance_sampling"
                    ).result(timeout=600)
                    for name, value in (("pre", pre_policy), ("post", post)):
                        write_json(
                            args.output_dir
                            / "rollouts"
                            / f"{step:04d}_{name}_policy.json",
                            [
                                r["logprobs"].to_numpy().tolist()
                                for r in value.loss_fn_outputs
                            ],
                        )
                    metadata = []
                    for j, (_, usage) in enumerate(collected):
                        raw = json.loads(
                            (
                                args.output_dir
                                / "rollouts"
                                / f"{step:04d}_{j:02d}.json"
                            ).read_text()
                        )
                        metadata.append(
                            {
                                **usage,
                                "token_categories": classify_token_categories(
                                    tokenizer, raw["tokens"]
                                ),
                            }
                        )
                    diagnostic_metrics.update(
                        summarize_diagnostics(
                            datums, pre_policy, post, metadata, near_zero=0.001
                        )
                    )
                if diagnostic_every:
                    quality = []
                    for j, (row, _) in enumerate(batch):
                        raw = json.loads(
                            (
                                args.output_dir
                                / "rollouts"
                                / f"{step:04d}_{j:02d}.json"
                            ).read_text()
                        )
                        reference = parse_table(
                            (args.data_root / row["label"]).read_text()
                        )
                        try:
                            parsed = parse_table(raw["html"])
                        except ValueError:
                            parsed = None
                        quality.append(score_tables(reference, parsed))
                    for key in (
                        "cell_f1",
                        "numeric_f1",
                        "table_exact",
                        "structure_exact",
                    ):
                        values = [q[key] for q in quality if q[key] is not None]
                        if values:
                            metrics[f"rollout_{key}"] = sum(values) / len(values)
                            metrics[f"rollout_{key}_count"] = len(values)
                usage = {key: sum(u[key] for _, u in collected) for key in totals}
                diagnostic_tokens = diagnostic_passes * usage["training_tokens"]
                amount = (
                    usage["prefill_tokens"] * FORWARD_RATE
                    + usage["rollout_tokens"] * SAMPLE_RATE
                    + usage["teacher_forward_tokens"] * selected_teacher.forward_rate
                    + len(batch) * selected_teacher.sample_rate
                    + (usage["training_tokens"] + gold_tokens) * TRAIN_RATE
                    + diagnostic_tokens * FORWARD_RATE
                ) / 1e6
                budget.settle(amount)
                report["extra_usage"]["gold_training_tokens"] += gold_tokens
                report["extra_usage"]["diagnostic_forward_tokens"] += diagnostic_tokens
                for key, value in usage.items():
                    totals[key] += value
                optimizer_metrics = {
                    key: metrics.pop(key)
                    for key in (
                        "gradient_norm_available",
                        "unclipped_gradient_norm",
                        "gradient_norm_exceeds_clip_threshold",
                    )
                    if key in metrics
                }
                event = {
                    "training/optimizer_step": step,
                    "training/epoch": epoch + 1,
                    "training/learning_rate": lr,
                    "training/step_seconds": perf_counter() - tick,
                    **{f"training/{k}": v for k, v in metrics.items()},
                    **{f"optimizer/{k}": v for k, v in optimizer_metrics.items()},
                    **{f"training/{k}": v for k, v in usage.items()},
                    **{
                        f"opd_diagnostics/{k}": v for k, v in diagnostic_metrics.items()
                    },
                    "runtime/estimated_compute_usd": budget.state[
                        "estimated_compute_usd"
                    ],
                }
                stream.write(json.dumps(event, allow_nan=False) + "\n")
                stream.flush()
                logger.log(event)
                print(json.dumps(event), flush=True)
                report.update(
                    completed_optimizer_steps=step,
                    usage=totals,
                    estimated_compute_usd=budget.state["estimated_compute_usd"],
                )
                write_json(args.output_dir / "run.json", report)
                if step < report["optimizer_steps"] and step % args.eval_every == 0:
                    stage = f"step_{step:04d}"
                    saved = client.save_weights_for_sampler(
                        stage, ttl_seconds=7 * 86400
                    ).result()
                    report["intermediate_sampler_paths"][stage] = saved.path
                    dev_assess(stage, step)
                    every = getattr(args, "generate_dev_every", 0)
                    if every and step % every == 0:
                        generate_dev(stage, step, saved.path)
    report["training_loop_seconds"] = perf_counter() - started
    report["state_path"] = (
        client.save_state("final", ttl_seconds=7 * 86400).result().path
    )
    saved = client.save_weights_for_sampler("after", ttl_seconds=7 * 86400).result()
    report["sampler_paths"] = {"after": saved.path}
    dev_assess("after", step)
    if args.generate_dev:
        generate_dev("after", step, saved.path)
    report.update(
        status="completed",
        estimated_compute_usd=budget.state["estimated_compute_usd"],
        checks={
            "completed_optimizer_steps": step,
            "full_dev_examples": len(dev),
            "rollouts": len(train) * args.epochs,
        },
    )
    write_json(args.output_dir / "run.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "train-manifest",
        "dev-manifest",
        "data-root",
        "output-dir",
        "tinker-cookbook-dir",
        "official-repo",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--teacher-model", choices=tuple(TEACHERS), default=TEACHER_MODEL)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--train-examples", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--generate-dev", action="store_true")
    p.add_argument(
        "--objective",
        choices=["sampled_reverse_kl", "topk_forward_kl", "hybrid_gold"],
        default="sampled_reverse_kl",
    )
    p.add_argument("--gold-weight", type=float, default=0.25)
    p.add_argument("--diagnostic-every", type=int, default=0)
    p.add_argument("--train-probe-examples", type=int, default=0)
    p.add_argument("--generate-dev-every", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-sequence-tokens", type=int, default=16384)
    p.add_argument("--max-pixels", type=int, default=1048576)
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--expected-output-tokens", type=int, default=1600)
    p.add_argument("--max-estimated-usd", type=float, default=0.3)
    p.add_argument("--wandb-mode", choices=["disabled", "online"], default="disabled")
    p.add_argument("--wandb-project", default="vlm-table-extraction")
    p.add_argument("--dataset-label", default="rd-opd-smoke8")
    p.add_argument("--inference-concurrency", type=int, default=4)
    args = p.parse_args()
    selected_teacher = teacher_spec(args.teacher_model)
    for key, value in vars(args).items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and key
            not in {
                "warmup_ratio",
                "diagnostic_every",
                "train_probe_examples",
                "generate_dev_every",
            }
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
    if any(
        getattr(args, k) < 0
        for k in ("diagnostic_every", "train_probe_examples", "generate_dev_every")
    ):
        raise ValueError("Diagnostic counts must be nonnegative")
    if not 0 < args.gold_weight < 1 or args.train_probe_examples > args.train_examples:
        raise ValueError("Invalid hybrid weight or training probe size")
    if args.generate_dev_every and (
        not args.generate_dev or args.generate_dev_every % args.eval_every
    ):
        raise ValueError("Generation cadence must coincide with full Dev checks")
    if not 0 <= args.warmup_ratio <= 1 or not re.fullmatch(
        r"[A-Za-z0-9_.-]{1,80}", args.dataset_label
    ):
        raise ValueError("Invalid warmup or public label")
    if args.expected_output_tokens > args.max_new_tokens:
        raise ValueError("Expected length exceeds output cap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Fresh output directory required; paid runs never auto-resume")
    rows, dev_rows = read_jsonl(args.train_manifest), read_jsonl(args.dev_manifest)
    if len(rows) != 800 or len(dev_rows) != 100 or not 1 <= args.train_examples <= 800:
        raise ValueError("Expected fixed Train800 and complete Dev100")
    validate_records(rows, dev_rows, args.data_root)
    tokenizer, renderer = load_renderer(
        MODEL, PROCESSOR_REVISION, args.tinker_cookbook_dir
    )
    teacher_tokenizer, teacher_renderer = load_renderer(
        selected_teacher.model, selected_teacher.revision, args.tinker_cookbook_dir
    )
    identity = tokenizer_identity(tokenizer, teacher_tokenizer)
    train = prepare_prompts(
        rows[: args.train_examples], args.data_root, renderer, teacher_renderer, args
    )
    dev = prepare_examples(
        dev_rows, args.data_root, renderer, args.max_pixels, args.max_sequence_tokens
    )
    scorer = OfficialScorer(args.official_repo)
    gold_count = (
        len(train) if args.objective == "hybrid_gold" else args.train_probe_examples
    )
    gold_for_cost = (
        prepare_examples(
            [row for row, _ in train[:gold_count]],
            args.data_root,
            renderer,
            args.max_pixels,
            args.max_sequence_tokens,
        )
        if gold_count
        else []
    )
    cost = estimate_cost(train, dev, args, gold_for_cost)
    steps = math.ceil(len(train) / args.batch_size) * args.epochs
    args.generate_train, args.train_nll_endpoints_only, args.generate_every = (
        False,
        True,
        steps,
    )
    args.rollouts_per_example, args.updates_per_rollout = 1, 1
    implementation_files = (
        "opd.py",
        "teacher_config.py",
        "tinker_inference.py",
        "opd_targets.py",
        "opd_sampling_budget.py",
        "opd_diagnostics.py",
        "opd_ablation_objectives.py",
        "kd_targets.py",
        "sft_metrics.py",
        "training_logging.py",
        "kd.py",
        "sft.py",
        "kd_collect.py",
        "kd_collection_budget.py",
    )
    report = {
        "status": "preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": {
            "sampled_reverse_kl": "on_policy_reverse_kl_opd",
            "topk_forward_kl": "on_policy_topk_distillation",
            "hybrid_gold": "on_policy_reverse_kl_gold",
        }[args.objective],
        "model": MODEL,
        "initialization": "fresh_lora_on_hosted_model",
        "hosted_weight_revision": None,
        "teacher_model": selected_teacher.model,
        "teacher_processor_revision": selected_teacher.revision,
        "teacher_hosted_weight_revision": None,
        "processor_revision": PROCESSOR_REVISION,
        "cookbook_revision": COOKBOOK_REVISION,
        "official_revision": REVISION,
        "metrics_version": VERSION,
        "tinker_version": version("tinker"),
        "prompt": PROMPT,
        "train_manifest_sha256": digest(args.train_manifest),
        "dev_manifest_sha256": digest(args.dev_manifest),
        "train_examples": len(train),
        "dev_examples": len(dev),
        "optimizer_steps": steps,
        "tokenizer_sha256": identity,
        "rollout_temperature": 1.0,
        "loss_temperature": 1.0,
        "opd_objective": args.objective,
        "gold_weight": args.gold_weight if args.objective == "hybrid_gold" else 0,
        "advantage_discount": 0.0,
        "rollout_seed_policy": "seed_plus_1000_step_plus_position",
        "lora": {
            "rank": args.rank,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": False,
        },
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "weight_decay": 0,
            "grad_clip_norm": 1.0,
        },
        "loss_reduction": {
            "sampled_reverse_kl": "batch_supervised_position_mean_importance_sampling",
            "topk_forward_kl": "batch_supervised_position_mean_topk_ce",
            "hybrid_gold": "separate_token_means_weighted_opd_and_gold_ce",
        }[args.objective],
        "learning_rate_schedule": "linear_warmup_then_constant",
        "warmup_steps": math.ceil(args.warmup_ratio * steps),
        "rates_usd_per_million": {
            "training": TRAIN_RATE,
            "forward": FORWARD_RATE,
            "sample": SAMPLE_RATE,
            "teacher_forward": selected_teacher.forward_rate,
            "teacher_scoring_output": selected_teacher.sample_rate,
        },
        "implementation_sha256": json_hash(
            {f: digest(Path(__file__).with_name(f)) for f in implementation_files}
        ),
        "implementation_files": implementation_files,
        "config": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "cost": cost,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rollouts").mkdir()
    write_json(args.output_dir / "run.json", report)
    print(
        json.dumps({"status": "preflight", "optimizer_steps": steps, **cost}),
        flush=True,
    )
    if not args.execute:
        return
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    logger = None
    try:
        logger = TrainingLogger(report)
        run(args, train, dev, tokenizer, renderer, scorer, report, logger)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
        raise
    finally:
        if logger:
            logger.log({"training/status": report["status"]})
            if "estimated_compute_usd" in report:
                logger.log(
                    {
                        "runtime/estimated_training_compute_usd": report[
                            "estimated_compute_usd"
                        ]
                    }
                )
            report["wandb_receipt"] = logger.finish()
        write_json(args.output_dir / "run.json", report)


if __name__ == "__main__":
    main()
