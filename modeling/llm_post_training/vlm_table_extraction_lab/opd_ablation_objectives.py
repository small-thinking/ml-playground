"""Token-mean native objectives for student-prefix Top-K and hybrid gold CE."""

import math

import numpy as np

from .kd_targets import normalize_batch as normalize_soft_batch
from .opd_targets import native_batch


def normalize_ce_batch(datums):
    """Normalize scalar gold or [N,K] soft CE by total supervised positions.

    Gold weights must be binary; soft rows must have unit mass. A temporary
    singleton K dimension reuses KD validation without changing native schemas.
    """
    import tinker

    datums = list(datums)
    expanded, dimensions = [], []
    for datum in datums:
        targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
        weights = datum.loss_fn_inputs["weights"].to_numpy()
        if weights.ndim not in (1, 2) or weights.shape != targets.shape:
            raise ValueError("Expected matching scalar or [N,K] CE arrays")
        dimensions.append(weights.ndim)
        if weights.ndim == 1:
            targets, weights = targets[:, None], weights[:, None]
        expanded.append(
            tinker.Datum(
                model_input=datum.model_input,
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData.from_numpy(targets.copy()),
                    "weights": tinker.TensorData.from_numpy(weights.copy()),
                },
            )
        )
    normalized = normalize_soft_batch(expanded)
    return [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                key: tinker.TensorData.from_numpy(
                    (
                        value.to_numpy()[:, 0] if dimension == 1 else value.to_numpy()
                    ).copy()
                )
                for key, value in datum.loss_fn_inputs.items()
            },
        )
        for datum, dimension in zip(normalized, dimensions, strict=True)
    ]


def _scaled_copies(datums, key, factor):
    import tinker

    return [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={
                name: tinker.TensorData.from_numpy(
                    (
                        value.to_numpy() * factor if name == key else value.to_numpy()
                    ).copy()
                )
                for name, value in datum.loss_fn_inputs.items()
            },
        )
        for datum in datums
    ]


def hybrid_batches(policy_datums, gold_datums, opd_weight=0.75):
    """Return native OPD and CE batches for two backward calls, one optimizer.

    Each objective has its own token denominator. Accumulate the two gradient
    contributions before a single optim_step; do not divide by batch size again.
    """
    if isinstance(opd_weight, bool) or not np.isscalar(opd_weight):
        raise ValueError("OPD mixture weight must be a finite scalar in [0,1]")
    if not math.isfinite(opd_weight) or not 0 <= opd_weight <= 1:
        raise ValueError("OPD mixture weight must be a finite scalar in [0,1]")
    policy = native_batch(list(policy_datums))
    gold = normalize_ce_batch(gold_datums)
    return (
        _scaled_copies(policy, "advantages", opd_weight),
        _scaled_copies(gold, "weights", 1 - opd_weight),
    )
