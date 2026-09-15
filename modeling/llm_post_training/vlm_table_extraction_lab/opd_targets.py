"""Immediate reverse-KL feedback on fresh student trajectories, at temperature 1."""

import numpy as np


def _logprobs(values, length):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (length,) or not np.isfinite(values).all():
        raise ValueError("Expected finite completion-aligned log probabilities")
    if (values > 0).any() or (values == -99999).any():
        raise ValueError("Invalid or missing completion log probability")
    return values


def policy_datum(
    prompt, sampled_tokens, sampling_logprobs, teacher_logprobs, vocab_size
):
    """A_t = log teacher(y_t|prefix) - log old_student(y_t|prefix), detached.

    Prompt/image rows are masked. Inputs end one token before targets, retaining
    the original rollout including EOS or the final token of a capped trajectory.
    """
    import tinker

    tokens = np.asarray(sampled_tokens)
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 1
        or prompt.length < 1
        or tokens.ndim != 1
        or not tokens.size
        or tokens.dtype.kind not in "iu"
        or (tokens < 0).any()
        or (tokens >= vocab_size).any()
    ):
        raise ValueError("Invalid prompt or completion tokens")
    old = _logprobs(sampling_logprobs, len(tokens))
    teacher = _logprobs(teacher_logprobs, len(tokens))
    start = prompt.length - 1
    size = start + len(tokens)
    targets = np.zeros(size, dtype=np.int64)
    logprobs = np.zeros(size, dtype=np.float32)
    advantages = np.zeros(size, dtype=np.float32)
    mask = np.zeros(size, dtype=np.float32)
    targets[start:], logprobs[start:] = tokens, old
    advantages[start:], mask[start:] = teacher - old, 1
    chunks = list(prompt.chunks)
    if len(tokens) > 1:
        chunks.append(tinker.EncodedTextChunk(tokens=tokens[:-1].tolist()))
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=chunks),
        loss_fn_inputs={
            key: tinker.TensorData.from_numpy(value)
            for key, value in {
                "target_tokens": targets,
                "logprobs": logprobs,
                "advantages": advantages,
                "mask": mask,
            }.items()
        },
    )


def _arrays(datum):
    arrays = {k: v.to_numpy() for k, v in datum.loss_fn_inputs.items()}
    if any(v.shape != (datum.model_input.length,) for v in arrays.values()):
        raise ValueError("Policy tensor length differs from input")
    mask = arrays["mask"]
    if not np.isin(mask, [0, 1]).all() or not mask.any():
        raise ValueError("Expected a nonempty binary completion mask")
    if any(not np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Non-finite policy input")
    if (arrays["advantages"][mask == 0] != 0).any():
        raise ValueError("Prompt positions must have zero advantage")
    return arrays


def normalize_batch(datums):
    """Copy and divide advantages once by batch completion-token count."""
    import tinker

    arrays = [_arrays(d) for d in datums]
    count = sum(int(a["mask"].sum()) for a in arrays)
    if not count:
        raise ValueError("Empty policy batch")
    return [
        tinker.Datum(
            model_input=d.model_input,
            loss_fn_inputs={
                k: tinker.TensorData.from_numpy(
                    (v / count if k == "advantages" else v).astype(v.dtype).copy()
                )
                for k, v in a.items()
            },
        )
        for d, a in zip(datums, arrays, strict=True)
    ]


def native_batch(datums):
    """The service accepts only its three loss fields; mask is local metadata."""
    import tinker

    return [
        tinker.Datum(
            model_input=d.model_input,
            loss_fn_inputs={
                k: d.loss_fn_inputs[k]
                for k in ("target_tokens", "logprobs", "advantages")
            },
        )
        for d in normalize_batch(datums)
    ]


def summarize_policy(output, unnormalized_datums):
    """Sampled KL can be negative; never label its exponential as perplexity."""
    old, advantage, current = [], [], []
    for result, datum in zip(output.loss_fn_outputs, unnormalized_datums, strict=True):
        a = _arrays(datum)
        active = a["mask"] > 0
        values = result["logprobs"].to_numpy()
        if values.shape != active.shape:
            raise ValueError("Learner log probability shape differs")
        current.extend(_logprobs(values[active], int(active.sum())))
        old.extend(a["logprobs"][active])
        advantage.extend(a["advantages"][active])
    if not old:
        raise ValueError("Empty policy output")
    old, advantage, current = map(
        lambda a: np.asarray(a, dtype=np.float64), (old, advantage, current)
    )
    ratio = np.exp(current - old)
    if not np.isfinite(ratio).all():
        raise ValueError("Non-finite importance ratio before optimizer")
    return {
        "supervised_tokens": len(old),
        "sampled_reverse_kl": float(-advantage.mean()),
        "sampling_entropy_estimate": float(-old.mean()),
        "teacher_nll_on_student_samples": float(-(old + advantage).mean()),
        "importance_ratio_mean": float(ratio.mean()),
        "importance_ratio_max": float(ratio.max()),
        "importance_log_ratio_abs_max": float(np.abs(current - old).max()),
        "advantage_mean": float(advantage.mean()),
        "advantage_min": float(advantage.min()),
        "advantage_max": float(advantage.max()),
    }
