"""Aggregate OPD diagnostics; no decoded text, token IDs, or per-example outputs.

Signal proxies measure absolute scalar surrogate coefficients |ratio * A|,
not parameter gradients. All token averages weight completion tokens equally.
Post-update statistics reuse the original samples: the unweighted sampled KL
is an old-policy estimate only when the sampling and pre-update learner policies
agree. Importance-weighted variants correct the conditional action distribution
at the sampled prefixes, not the distribution of whole trajectories/prefixes.
"""

import math
import re

import numpy as np

from .opd_targets import _arrays, _logprobs


CATEGORIES = ("numeric", "content", "markup", "eos", "unknown")


def _learner_values(output, arrays):
    values = []
    for result, a in zip(output.loss_fn_outputs, arrays, strict=True):
        active = a["mask"] > 0
        lp = np.asarray(result["logprobs"].to_numpy(), dtype=np.float64)
        if lp.shape != active.shape:
            raise ValueError("Learner log probability shape differs")
        values.append(_logprobs(lp[active], int(active.sum())))
    return np.concatenate(values)


def _k3(delta):
    # expm1 removes the exp(d)-1 cancellation; a series also avoids subtracting
    # nearly identical expm1(d) and d for tiny optimizer updates.
    result = np.expm1(delta) - delta
    small = np.abs(delta) < 1e-4
    d = delta[small]
    result[small] = d * d * (0.5 + d * (1 / 6 + d * (1 / 24 + d / 120)))
    return result


def summarize_diagnostics(
    unnormalized_datums,
    pre_output,
    post_output=None,
    metadata=None,
    *,
    near_zero=1e-6,
    ratio_tail=2.0,
):
    """Return flat, finite aggregate scalars from unnormalized policy datums.

    metadata is optional and example-aligned. Allowed fields are format_invalid
    and truncated (bool or 0/1), and token_categories (completion-aligned labels
    from CATEGORIES). Missing flags remain unknown: coverage is reported and
    they do not contribute to flagged shares. Missing categories become unknown.
    Shares divide by total batch |ratio*A|; *_per_batch_token divides by the
    entire batch completion-token count. Zero signal yields zero shares plus
    signal_proxy_zero_total=1, rather than a claim of uniform contributions.

    ratio_tail is a multiplicative threshold: ratios outside [1/tail, tail].
    The caller must supply pre/post outputs for the same datums in the same order.
    No IDs/text/paths from metadata are included in the result.
    """
    if not math.isfinite(near_zero) or near_zero < 0:
        raise ValueError("Expected a finite nonnegative near-zero threshold")
    if not math.isfinite(ratio_tail) or ratio_tail <= 1:
        raise ValueError("Expected a finite ratio tail threshold greater than one")
    arrays = [_arrays(d) for d in unnormalized_datums]
    if not arrays:
        raise ValueError("Empty policy batch")
    counts = [int(a["mask"].sum()) for a in arrays]
    advantage = np.concatenate([a["advantages"][a["mask"] > 0] for a in arrays]).astype(
        np.float64
    )
    sampled = np.concatenate(
        [_logprobs(a["logprobs"][a["mask"] > 0], n) for a, n in zip(arrays, counts)]
    )
    pre = _learner_values(pre_output, arrays)
    log_ratio = pre - sampled
    with np.errstate(over="raise", invalid="raise"):
        try:
            ratio = np.exp(log_ratio)
            signal = np.abs(ratio * advantage)
            total = float(signal.sum())
            scaled = np.exp(log_ratio - log_ratio.max())
            ess = float(scaled.sum() ** 2 / np.square(scaled).sum())
        except FloatingPointError as error:
            raise ValueError("Non-finite diagnostic importance or signal") from error
    n = len(advantage)

    def share(mask):
        return float(signal[mask].sum() / total) if total else 0.0

    metrics = {
        "diagnostic_completion_tokens": n,
        "advantage_std": float(advantage.std()),
        "advantage_near_zero_fraction": float((np.abs(advantage) <= near_zero).mean()),
        "advantage_near_zero_threshold": near_zero,
        "advantage_positive_fraction": float((advantage > 0).mean()),
        "signal_proxy_abs_mean": total / n,
        "signal_proxy_zero_total": int(total == 0),
        "importance_ess": ess,
        "importance_ess_fraction": ess / n,
        "importance_tail_fraction": float(
            (np.abs(log_ratio) > math.log(ratio_tail)).mean()
        ),
        "importance_tail_threshold": ratio_tail,
        "sampler_to_pre_learner_log_ratio_mean": float(log_ratio.mean()),
        "sampler_to_pre_learner_log_ratio_abs_mean": float(np.abs(log_ratio).mean()),
    }
    for percentile in (1, 5, 25, 50, 75, 95, 99):
        metrics[f"advantage_p{percentile:02d}"] = float(
            np.percentile(advantage, percentile)
        )
    ordered = np.sort(signal)
    for percent in (1, 5):
        count = math.ceil(n * percent / 100)
        metrics[f"signal_proxy_top_{percent}pct_share"] = (
            float(ordered[-count:].sum() / total) if total else 0.0
        )

    if metadata is not None:
        metadata = list(metadata)
        if len(metadata) != len(arrays):
            raise ValueError("Metadata length differs from policy batch")
        for key in ("format_invalid", "truncated"):
            flags, known = [], []
            for item, count in zip(metadata, counts):
                value = item.get(key)
                if value is not None and (
                    type(value) not in (bool, int) or value not in (0, 1)
                ):
                    raise ValueError("Expected boolean diagnostic flags")
                flags.extend([value == 1] * count)
                known.extend([value is not None] * count)
            mask = np.asarray(flags, dtype=bool)
            metrics[f"{key}_token_coverage"] = float(np.mean(known))
            metrics[f"signal_proxy_{key}_share"] = share(mask)
            metrics[f"signal_proxy_{key}_per_batch_token"] = float(
                signal[mask].sum() / n
            )
        labels = []
        for item, count in zip(metadata, counts):
            categories = item.get("token_categories", ["unknown"] * count)
            if len(categories) != count or any(c not in CATEGORIES for c in categories):
                raise ValueError("Invalid or misaligned token category labels")
            labels.extend(categories)
        labels = np.asarray(labels)
        for category in CATEGORIES:
            mask = labels == category
            metrics[f"category_{category}_token_fraction"] = float(mask.mean())
            metrics[f"signal_proxy_category_{category}_share"] = share(mask)
            if mask.any():
                metrics[f"category_{category}_advantage_mean"] = float(
                    advantage[mask].mean()
                )

    if post_output is not None:
        delta = _learner_values(post_output, arrays) - pre
        try:
            with np.errstate(over="raise", invalid="raise"):
                k3 = _k3(delta)
                metrics.update(
                    update_log_ratio_mean=float(delta.mean()),
                    update_sampled_old_policy_kl=float(-delta.mean()),
                    update_k3_mean=float(k3.mean()),
                    update_importance_weighted_old_policy_kl=float(
                        np.mean(ratio * -delta)
                    ),
                    update_importance_weighted_k3_mean=float(np.mean(ratio * k3)),
                )
        except FloatingPointError as error:
            raise ValueError("Non-finite post-update diagnostic") from error
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite diagnostic aggregate")
    return metrics


def classify_token_categories(tokenizer, token_ids):
    """Return labels only; lexical classification is not exact semantic parsing.

    Require an exact decode/re-encode token-ID round trip and offset mapping.
    Unavailable/misaligned offsets yield unknown except explicitly known EOS IDs.
    Tags/comments count as markup; number-like spans outside tags as numeric;
    other text as content. Mixed-boundary tokens remain unknown. This heuristic
    does not interpret HTML entities, malformed markup, script, or CSS semantics.
    No text, IDs, offsets, or exception content is returned.
    """
    token_ids = list(token_ids)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos_ids = set(eos if isinstance(eos, (list, tuple, set)) else [eos])
    labels = ["eos" if t in eos_ids else "unknown" for t in token_ids]
    try:
        text = tokenizer.decode(
            token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if list(encoded["input_ids"]) != token_ids or len(offsets) != len(token_ids):
            return labels
        chars = np.full(len(text), "content", dtype=object)
        for match in re.finditer(
            r"<!--.*?-->|</?[A-Za-z](?:[^<>\"']|\"[^\"]*\"|'[^']*')*>", text, re.DOTALL
        ):
            chars[match.start() : match.end()] = "markup"
        for match in re.finditer(
            r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?!\w)",
            text,
        ):
            start, end = match.span()
            if (chars[start:end] == "content").all():
                chars[start:end] = "numeric"
        for index, (start, end) in enumerate(offsets):
            if labels[index] == "eos" or not (0 <= start < end <= len(text)):
                continue
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            kinds = set(chars[start:end])
            if len(kinds) == 1:
                labels[index] = kinds.pop()
    except (AttributeError, KeyError, TypeError, ValueError, NotImplementedError):
        return ["eos" if t in eos_ids else "unknown" for t in token_ids]
    return labels
