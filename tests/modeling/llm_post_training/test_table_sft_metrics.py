"""NLL reporting tests use plain objects and never import a training SDK."""

import json
import math
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.sft_metrics import (
    summarize_nll,
)


def inputs(logprobs, masks):
    output = SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": SimpleNamespace(data=values)} for values in logprobs
        ]
    )
    datums = [
        SimpleNamespace(loss_fn_inputs={"weights": SimpleNamespace(data=values)})
        for values in masks
    ]
    return output, datums


def test_unequal_lengths_separate_micro_and_macro_and_exponentiate_aggregate():
    result = summarize_nll(*inputs([[-1], [-2, -3, 0]], [[1], [1, 1, 1]]))
    assert result["supervised_tokens"] == 4
    assert result["assistant_nll"] == pytest.approx(6 / 4)
    assert result["example_nll_mean"] == pytest.approx((1 + 5 / 3) / 2)
    assert result["example_nll_max"] == pytest.approx(5 / 3)
    assert result["per_example"][0] == {"nll": 1, "supervised_tokens": 1}
    assert result["per_example"][1]["nll"] == pytest.approx(5 / 3)
    assert result["per_example"][1]["supervised_tokens"] == 3
    assert result["assistant_perplexity"] == pytest.approx(math.exp(6 / 4))
    assert result["assistant_perplexity"] != pytest.approx(
        (math.exp(1) + math.exp(5 / 3)) / 2
    )
    assert result["perplexity_overflow"] is False
    assert result["zero_logprob_fraction"] == 0.25
    json.dumps(result, allow_nan=False)


def test_prompt_and_padding_values_never_enter_statistics():
    result = summarize_nll(
        *inputs(
            [[math.nan, math.inf, 1, 0, -2, 0, -math.inf]],
            [[0, 0, 0, 0, 1, 1, 0]],
        )
    )
    assert result["assistant_nll"] == 1
    assert result["assistant_perplexity"] == math.e
    assert result["supervised_tokens"] == 2
    assert result["zero_logprob_fraction"] == 0.5


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_supervised_logprobs_rejected(value):
    with pytest.raises(ValueError, match="Non-finite"):
        summarize_nll(*inputs([[value]], [[1]]))


@pytest.mark.parametrize("mask", [0.5, -1, 2, math.nan, math.inf])
def test_original_binary_mask_required(mask):
    with pytest.raises(ValueError, match="binary"):
        summarize_nll(*inputs([[-1]], [[mask]]))


@pytest.mark.parametrize(
    "logprobs,masks,message",
    [
        ([[-1]], [], "counts"),
        ([[-1, -2]], [[1]], "sizes"),
        ([[0.1]], [[1]], "Positive"),
        ([], [], "No supervised"),
        ([[-1]], [[0]], "No supervised"),
        ([[-1], [-2]], [[1], [0]], "No supervised"),
    ],
)
def test_invalid_shapes_probabilities_and_empty_supervision(logprobs, masks, message):
    with pytest.raises(ValueError, match=message):
        summarize_nll(*inputs(logprobs, masks))


@pytest.mark.parametrize("nll", [1000, 1e308])
def test_perplexity_overflow_is_explicit_and_json_safe(nll):
    result = summarize_nll(*inputs([[-nll, -nll]], [[1, 1]]))
    assert result["assistant_nll"] == nll
    assert result["assistant_perplexity"] is None
    assert result["perplexity_overflow"] is True
    json.dumps(result, allow_nan=False)
