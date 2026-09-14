"""Teacher-forced target NLL diagnostics; target logprobs do not give accuracy."""

import math


def summarize_nll(output, datums):
    """Summarize original binary assistant masks, ignoring prompt/padding values.

    The aggregate is a supervised-token mean; the example mean is a separate
    macro diagnostic. Empty supervision is invalid, including within an example.
    """
    results = output.loss_fn_outputs
    if len(results) != len(datums):
        raise ValueError("Inconsistent output and datum counts")
    losses, per_example = [], []
    for result, datum in zip(results, datums):
        logprobs = result["logprobs"].data
        masks = datum.loss_fn_inputs["weights"].data
        if len(logprobs) != len(masks):
            raise ValueError("Inconsistent logprob and mask sizes")
        example_losses = []
        for logprob, mask in zip(logprobs, masks):
            if mask not in (0, 1):
                raise ValueError("NLL requires original binary supervision masks")
            if mask == 0:
                continue
            if not math.isfinite(logprob):
                raise ValueError("Non-finite supervised log probability")
            if logprob > 0:
                raise ValueError("Positive supervised log probability")
            example_losses.append(-float(logprob))
        count = len(example_losses)
        if not count:
            raise ValueError("No supervised tokens in example NLL")
        per_example.append(
            {
                "nll": math.fsum(loss / count for loss in example_losses),
                "supervised_tokens": count,
            }
        )
        losses.extend(example_losses)
    count = len(losses)
    if not count:
        raise ValueError("No supervised tokens in NLL")
    # Divide before summing so finite but extreme NLLs do not overflow the sum.
    nll = math.fsum(loss / count for loss in losses)
    try:
        perplexity = math.exp(nll)
    except OverflowError:
        perplexity = None
    return {
        "assistant_nll": nll,
        "assistant_perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
        "supervised_tokens": count,
        "example_nll_mean": math.fsum(
            example["nll"] / len(per_example) for example in per_example
        ),
        "example_nll_max": max(example["nll"] for example in per_example),
        "zero_logprob_fraction": sum(loss == 0 for loss in losses) / count,
        "per_example": per_example,
    }
