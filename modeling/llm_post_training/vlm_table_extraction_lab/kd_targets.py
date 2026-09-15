"""Completion-aligned, truncated teacher distributions for off-policy KD."""

import numpy as np


def _logprobs(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape:
        raise ValueError("Expected nonempty [completion_length, K] logprobs")
    if not np.isfinite(values).all() or (values > 0).any():
        raise ValueError("Teacher logprobs must be finite and nonpositive")
    if (values == -99999.0).any():
        raise ValueError("Missing Top-K sentinel is unsupported")
    # Genuine full-vocabulary logprobs cannot assign more than total mass one.
    if (np.exp(values).sum(axis=1) > 1 + 1e-5).any():
        raise ValueError("Top-K probability mass exceeds one")
    return values


def soft_targets(prompt, sampled_tokens, topk_token_ids, topk_logprobs, vocab_size):
    """Build [N,K] CE targets; response row zero predicts the first completion token.

    Teacher arrays must already align with the *unmodified* sampled completion.
    Masked prompt/image rows contain dummy token zero and zero weight. Full K
    is required at every completion position; missing/padded targets fail closed.
    """
    import tinker

    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 1
    ):
        raise ValueError("vocab_size must be a positive integer")
    sampled = np.asarray(sampled_tokens)
    if sampled.ndim != 1 or not sampled.size or sampled.dtype.kind not in "iu":
        raise ValueError("Expected a nonempty integer completion")
    if (sampled < 0).any() or (sampled >= vocab_size).any():
        raise ValueError("Completion token outside vocabulary")
    if prompt.length < 1:
        raise ValueError("Prompt must contain at least one position")
    ids = np.asarray(topk_token_ids)
    logprobs = _logprobs(topk_logprobs)
    if ids.shape != logprobs.shape or ids.shape[0] != len(sampled):
        raise ValueError("Top-K shape must match completion length")
    if ids.dtype.kind not in "iu" or (ids < 0).any() or (ids >= vocab_size).any():
        raise ValueError("Top-K token IDs must be integers inside the vocabulary")
    if ids.shape[1] > 1 and (np.diff(np.sort(ids, axis=1), axis=1) == 0).any():
        raise ValueError("Duplicate Top-K token IDs")
    probabilities = np.exp(logprobs - logprobs.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    first_response = prompt.length - 1
    shape = (first_response + len(sampled), ids.shape[1])
    targets = np.zeros(shape, dtype=np.int64)
    weights = np.zeros(shape, dtype=np.float32)
    targets[first_response:] = ids
    weights[first_response:] = probabilities
    chunks = list(prompt.chunks)
    if len(sampled) > 1:
        chunks.append(tinker.EncodedTextChunk(tokens=sampled[:-1].tolist()))
    return tinker.Datum(
        model_input=tinker.ModelInput(chunks=chunks),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(targets),
            "weights": tinker.TensorData.from_numpy(weights),
        },
    )


def _weights(datum):
    weights = np.asarray(datum.loss_fn_inputs["weights"].to_numpy(), dtype=np.float64)
    targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    if (
        weights.ndim != 2
        or 0 in weights.shape
        or weights.shape != targets.shape
        or weights.shape[0] != datum.model_input.length
    ):
        raise ValueError("Expected matching [N,K] targets and weights")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Weights must be finite and nonnegative")
    active = weights.sum(axis=1) > 0
    if not active.any():
        raise ValueError("Datum has no supervised positions")
    if not np.allclose(weights[active].sum(axis=1), 1, rtol=1e-6, atol=1e-7):
        raise ValueError("Expected unnormalized datums with unit mass per active row")
    return weights, active


def normalize_batch(datums):
    """Return copies with summed CE normalized by supervised positions, not N*K."""
    import tinker

    datums = list(datums)
    if not datums:
        raise ValueError("Batch has no supervised positions")
    arrays = [_weights(datum) for datum in datums]
    count = sum(int(active.sum()) for _, active in arrays)
    return [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                **datum.loss_fn_inputs,
                "target_tokens": tinker.TensorData.from_numpy(
                    datum.loss_fn_inputs["target_tokens"].to_numpy().copy()
                ),
                "weights": tinker.TensorData.from_numpy(
                    (weights / count).astype(np.float32)
                ),
            },
        )
        for datum, (weights, _) in zip(datums, arrays, strict=True)
    ]


def summarize_soft(output, unnormalized_datums):
    """Token-mean CE, entropy and KL(q_topK || p_full_vocab).

    Student logprobs are full-vocabulary log probabilities for the selected K
    IDs. Never renormalize the student over K. This KL is against the truncated,
    renormalized teacher distribution, not the unknown full teacher distribution.
    """
    datums = list(unnormalized_datums)
    if not datums or len(output.loss_fn_outputs) != len(datums):
        raise ValueError("Output and datum counts must match and be nonempty")
    arrays = [_weights(datum) for datum in datums]
    count = sum(int(active.sum()) for _, active in arrays)
    ce, entropy = 0.0, 0.0
    for result, (weights, _) in zip(output.loss_fn_outputs, arrays, strict=True):
        logprobs = np.asarray(result["logprobs"].to_numpy(), dtype=np.float64)
        if logprobs.shape != weights.shape:
            raise ValueError("Student logprobs must have matching [N,K] shape")
        # Ignore prompt/image rows (and zero-weight slots) before validating LPs.
        selected = weights > 0
        values = logprobs[selected]
        if not np.isfinite(values).all() or (values > 0).any():
            raise ValueError("Active student logprobs must be finite and nonpositive")
        probabilities = weights[selected]
        ce -= float(np.sum(probabilities * (values / count)))
        entropy -= float(np.sum(probabilities * np.log(probabilities) / count))
    if not np.isfinite([ce, entropy]).all():
        raise ValueError("Non-finite soft-target summary")
    return {
        "soft_cross_entropy": ce,
        "teacher_entropy": entropy,
        "truncated_forward_kl": ce - entropy,
        "supervised_tokens": count,
    }


def topk_diagnostics(logprobs):
    """Retained full-teacher mass before Top-K renormalization, T=1."""
    masses = np.exp(_logprobs(logprobs)).sum(axis=1)
    return {
        "retained_mass_mean": float(masses.mean()),
        "retained_mass_p05": float(np.quantile(masses, 0.05)),
        "retained_mass_min": float(masses.min()),
        "retained_mass_fraction_below_095": float((masses < 0.95).mean()),
    }
