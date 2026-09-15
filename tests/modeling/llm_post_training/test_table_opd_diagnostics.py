from types import SimpleNamespace
import json

import numpy as np
import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.opd_diagnostics import (
    classify_token_categories,
    summarize_diagnostics,
)
from modeling.llm_post_training.vlm_table_extraction_lab.opd_targets import policy_datum

tinker = pytest.importorskip("tinker")


def datum(advantages):
    old = np.full(len(advantages), -5.0)
    return policy_datum(
        tinker.ModelInput.from_ints([8, 9]),
        [1] * len(old),
        old,
        old + advantages,
        10,
    )


def output(*values):
    return SimpleNamespace(
        loss_fn_outputs=[
            {
                "logprobs": tinker.TensorData.from_numpy(
                    np.array([np.nan, *v], dtype=np.float64)
                )
            }
            for v in values
        ]
    )


def test_advantages_concentration_and_importance_ess_are_token_weighted():
    ds = [datum(np.array([-2, 0, 1, 3]))]
    ratios = np.array([1, 1, 1, 4])
    metrics = summarize_diagnostics(ds, output(-5 + np.log(ratios)))
    assert metrics["advantage_std"] == pytest.approx(np.std([-2, 0, 1, 3]))
    assert metrics["advantage_p50"] == 0.5
    assert metrics["advantage_near_zero_fraction"] == 0.25
    assert metrics["advantage_positive_fraction"] == 0.5
    assert metrics["signal_proxy_top_1pct_share"] == pytest.approx(12 / 15)
    assert metrics["signal_proxy_top_5pct_share"] == pytest.approx(12 / 15)
    assert metrics["importance_ess"] == pytest.approx(49 / 19)
    assert metrics["importance_ess_fraction"] == pytest.approx(49 / 76)
    assert metrics["importance_tail_fraction"] == 0.25


def test_invalid_truncated_signal_is_batch_normalized_and_private_metadata_excluded():
    ds = [datum(np.array([1])), datum(np.array([2, 3, 4]))]
    metadata = [
        {
            "format_invalid": 1,
            "truncated": 0,
            "token_categories": ["eos"],
            "text": "PRIVATE",
        },
        {
            "format_invalid": 0,
            "truncated": 1,
            "token_categories": ["numeric", "markup", "content"],
        },
    ]
    metrics = summarize_diagnostics(ds, output([-5], [-5, -5, -5]), metadata=metadata)
    assert metrics["signal_proxy_format_invalid_share"] == 0.1
    assert metrics["signal_proxy_format_invalid_per_batch_token"] == 0.25
    assert metrics["signal_proxy_truncated_share"] == 0.9
    assert metrics["signal_proxy_truncated_per_batch_token"] == 2.25
    assert metrics["signal_proxy_category_numeric_share"] == 0.2
    assert metrics["category_numeric_token_fraction"] == 0.25
    assert metrics["format_invalid_token_coverage"] == 1
    assert "PRIVATE" not in json.dumps(metrics)


def test_update_uses_pre_learner_not_sampling_policy_and_k3_is_stable():
    ds = [datum(np.array([1, -1]))]
    pre = np.array([-4.0, -4.0])  # sampler is -5: deliberate drift
    delta = np.array([1e-10, -1e-10])
    post = pre + delta
    metrics = summarize_diagnostics(ds, output(pre), output(post))
    actual = post - pre
    assert metrics["sampler_to_pre_learner_log_ratio_mean"] == 1
    assert metrics["update_log_ratio_mean"] == pytest.approx(actual.mean(), abs=1e-20)
    assert metrics["update_sampled_old_policy_kl"] == pytest.approx(
        -actual.mean(), abs=1e-20
    )
    assert metrics["update_k3_mean"] == pytest.approx(np.mean(actual**2 / 2), rel=1e-8)
    assert metrics["update_importance_weighted_k3_mean"] == pytest.approx(
        np.e * np.mean(actual**2 / 2), rel=1e-8
    )


def test_zero_signal_and_missing_metadata_are_explicit():
    metrics = summarize_diagnostics(
        [datum(np.zeros(2))], output([-5, -5]), metadata=[{}]
    )
    assert metrics["signal_proxy_zero_total"] == 1
    assert metrics["signal_proxy_top_1pct_share"] == 0
    assert metrics["advantage_near_zero_fraction"] == 1
    assert metrics["format_invalid_token_coverage"] == 0
    assert metrics["category_unknown_token_fraction"] == 1


@pytest.mark.parametrize(
    "metadata", [[], [{"token_categories": ["numeric"]}], [{"truncated": "PRIVATE"}]]
)
def test_misaligned_or_invalid_metadata_rejected(metadata):
    with pytest.raises(ValueError):
        summarize_diagnostics([datum(np.ones(2))], output([-5, -5]), metadata=metadata)


def test_empty_nonfinite_or_misaligned_inputs_rejected():
    with pytest.raises(ValueError):
        summarize_diagnostics([], output())
    ds = [datum(np.ones(2))]
    for bad in (output([-5]), output([-5, np.nan]), output([-5, 1])):
        with pytest.raises(ValueError):
            summarize_diagnostics(ds, bad)
    with pytest.raises(ValueError):
        summarize_diagnostics(ds, output([-5, -5]), output([-5]))


class OffsetTokenizer:
    eos_token_id = 99

    def decode(self, ids, **kwargs):
        return "<td>12.5</td>word!<eos>"

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [1, 2, 3, 4, 99],
            "offset_mapping": [(0, 4), (4, 8), (8, 13), (13, 18), (18, 23)],
        }


def test_offset_classification_returns_only_categories_and_known_eos():
    tokenizer = OffsetTokenizer()
    assert classify_token_categories(tokenizer, [1, 2, 3, 4, 99]) == [
        "markup",
        "numeric",
        "markup",
        "content",
        "eos",
    ]
    assert classify_token_categories(tokenizer, [8, 2, 3, 4, 99]) == [
        "unknown",
        "unknown",
        "unknown",
        "unknown",
        "eos",
    ]


def test_mixed_boundary_or_unavailable_offsets_are_unknown():
    class MixedTokenizer(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 99], "offset_mapping": [(2, 8), (0, 0)]}

    class NoOffsets(OffsetTokenizer):
        def __call__(self, text, **kwargs):
            raise NotImplementedError("PRIVATE")

    assert classify_token_categories(MixedTokenizer(), [1, 99]) == ["unknown", "eos"]
    assert classify_token_categories(NoOffsets(), [1, 99]) == ["unknown", "eos"]
