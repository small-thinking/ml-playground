"""Offline distribution/gradient and multimodal-alignment checks for KD."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modeling.llm_post_training.vlm_table_extraction_lab.kd_targets import (
    normalize_batch,
    soft_targets,
    summarize_soft,
    topk_diagnostics,
)

tinker = pytest.importorskip("tinker")


def make_datum(probabilities=((0.6, 0.3),), sampled=None):
    p = np.asarray(probabilities)
    return soft_targets(
        tinker.ModelInput.from_ints([8, 9]),
        sampled if sampled is not None else [1] * len(p),
        np.tile(np.arange(p.shape[1]), (len(p), 1)),
        np.log(p),
        10,
    )


def output(values):
    return SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": tinker.TensorData.from_numpy(np.asarray(v, dtype=np.float32))}
            for v in values
        ]
    )


def test_shift_keeps_image_identity_and_all_completion_targets():
    image = tinker.types.ImageChunk(data=b"fixture", format="png", expected_tokens=3)
    prompt = tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[8]),
            image,
            tinker.EncodedTextChunk(tokens=[9]),
        ]
    )
    datum = soft_targets(
        prompt, [2, 3], [[2, 1], [3, 4]], np.log([[0.8, 0.1], [0.6, 0.3]]), 10
    )
    assert datum.model_input.chunks[1] is image
    assert datum.model_input.length == prompt.length + 1
    assert datum.model_input.chunks[-1].tokens == [2]
    assert prompt.chunks[-1].tokens == [9]
    targets = datum.loss_fn_inputs["target_tokens"].to_numpy()
    weights = datum.loss_fn_inputs["weights"].to_numpy()
    assert targets.shape == weights.shape == (6, 2)
    np.testing.assert_array_equal(targets[:4], 0)
    np.testing.assert_array_equal(weights[:4], 0)
    np.testing.assert_array_equal(targets[4:], [[2, 1], [3, 4]])
    np.testing.assert_allclose(weights[4:], [[8 / 9, 1 / 9], [2 / 3, 1 / 3]])
    assert datum.loss_fn_inputs["weights"].shape == [6, 2]


def test_single_completion_token_keeps_entire_prompt_without_empty_extra_chunk():
    prompt = tinker.ModelInput.from_ints([4])
    datum = soft_targets(prompt, [2], [[2]], [[-0.2]], 5)
    assert datum.model_input.chunks == prompt.chunks
    np.testing.assert_array_equal(datum.loss_fn_inputs["weights"].to_numpy(), [[1]])


@pytest.mark.parametrize(
    "change",
    [
        {"sampled_tokens": []},
        {"sampled_tokens": [1.0]},
        {"sampled_tokens": [True]},
        {"sampled_tokens": [10]},
        {"topk_token_ids": [[0, 0]]},
        {"topk_token_ids": [[0, -1]]},
        {"topk_token_ids": [[0, 10]]},
        {"topk_token_ids": [[0.0, 1.0]]},
        {"topk_token_ids": [[0]]},
        {"topk_logprobs": [[-0.1, None]]},
        {"topk_logprobs": [[-0.1, float("nan")]]},
        {"topk_logprobs": [[-0.1, float("-inf")]]},
        {"topk_logprobs": [[-0.1, -99999.0]]},
        {"topk_logprobs": [[-0.1, 0.1]]},
        {"topk_logprobs": [[-0.01, -0.01]]},
        {"topk_logprobs": []},
        {"topk_logprobs": [-1.0, -2.0]},
        {"topk_logprobs": [[-1.0, -2.0], [-1.0, -2.0]]},
        {"prompt": tinker.ModelInput.empty()},
        {"vocab_size": True},
    ],
)
def test_invalid_targets_fail_closed(change):
    args = dict(
        prompt=tinker.ModelInput.from_ints([8]),
        sampled_tokens=[1],
        topk_token_ids=[[0, 1]],
        topk_logprobs=[[-1.0, -2.0]],
        vocab_size=10,
    )
    args.update(change)
    with pytest.raises(ValueError):
        soft_targets(**args)


def test_batch_normalization_is_over_tokens_and_does_not_mutate_inputs():
    datums = [make_datum(), make_datum(((0.4, 0.2), (0.3, 0.2)))]
    originals = [d.loss_fn_inputs["weights"].to_numpy().copy() for d in datums]
    normalized = normalize_batch(datums)
    assert sum(
        d.loss_fn_inputs["weights"].to_numpy().sum() for d in normalized
    ) == pytest.approx(1)
    for original, datum, norm in zip(originals, datums, normalized):
        np.testing.assert_array_equal(
            datum.loss_fn_inputs["weights"].to_numpy(), original
        )
        np.testing.assert_allclose(
            norm.loss_fn_inputs["weights"].to_numpy(), original / 3
        )
        assert norm is not datum
    with pytest.raises(ValueError, match="unit mass"):
        normalize_batch(normalized)
    with pytest.raises(ValueError, match="no supervised"):
        normalize_batch([])


@pytest.mark.parametrize(
    "ids,teacher,expected_q",
    [
        ([0, 1, 2], [0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
        ([2], [0.5], [0.0, 0.0, 1.0]),
    ],
)
def test_k_equals_vocab_and_k_one_gradients_match_closed_form(ids, teacher, expected_q):
    datum = soft_targets(
        tinker.ModelInput.from_ints([0]), [2], [ids], [np.log(teacher)], 3
    )
    norm = normalize_batch([datum])[0]
    logits = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    weights = torch.tensor(
        norm.loss_fn_inputs["weights"].to_numpy(), dtype=torch.float64
    )
    loss = -(weights * logp[torch.tensor([ids])]).sum()
    loss.backward()
    expected = logits.detach().softmax(-1) - torch.tensor(expected_q)
    torch.testing.assert_close(logits.grad, expected, rtol=1e-6, atol=1e-7)


def test_soft_summary_uses_full_student_probabilities_and_masks_nan_prompt():
    datum = make_datum(((0.6, 0.3), (0.4, 0.2)))
    # Both teacher rows normalize to (2/3,1/3). Student only allocates .3 to K.
    native = np.array([[np.nan, np.nan], np.log([0.1, 0.2]), np.log([0.2, 0.1])])
    result = summarize_soft(output([native]), [datum])
    q = np.array([2 / 3, 1 / 3])
    expected_ce = -np.mean(
        [np.dot(q, np.log([0.1, 0.2])), np.dot(q, np.log([0.2, 0.1]))]
    )
    expected_entropy = -np.dot(q, np.log(q))
    assert result["supervised_tokens"] == 2
    assert result["soft_cross_entropy"] == pytest.approx(expected_ce)
    assert result["teacher_entropy"] == pytest.approx(expected_entropy)
    assert result["truncated_forward_kl"] == pytest.approx(
        expected_ce - expected_entropy
    )
    # Renormalizing student over its selected .3 mass would erroneously remove this.
    assert result["soft_cross_entropy"] > -np.log(0.3)
    norm = normalize_batch([datum])[0]
    direct_loss = -np.sum(norm.loss_fn_inputs["weights"].to_numpy()[1:] * native[1:])
    assert direct_loss == pytest.approx(result["soft_cross_entropy"])


def test_summary_matches_token_mean_across_unequal_example_lengths():
    datums = [make_datum(((0.6,),)), make_datum(((0.6,), (0.6,)))]
    result = summarize_soft(output([[[np.nan], [-1]], [[np.nan], [-2], [-6]]]), datums)
    assert result == {
        "soft_cross_entropy": 3.0,
        "teacher_entropy": 0.0,
        "truncated_forward_kl": 3.0,
        "supervised_tokens": 3,
    }


@pytest.mark.parametrize(
    "native",
    [
        [[0, 0], [float("nan"), -1]],
        [[0, 0], [float("inf"), -1]],
        [[0, 0], [0.1, -1]],
        [[0, 0], [-1, float("-inf")]],
        [[-1, -2]],
        [-1, -2, -3, -4],
    ],
)
def test_invalid_active_probabilities_and_shapes_are_rejected(native):
    with pytest.raises(ValueError):
        summarize_soft(output([native]), [make_datum()])


def test_summary_rejects_wrong_output_count_and_normalized_input():
    with pytest.raises(ValueError, match="counts"):
        summarize_soft(output([]), [make_datum()])
    datums = [make_datum(), make_datum()]
    with pytest.raises(ValueError, match="unit mass"):
        summarize_soft(output([[[0, 0], [-1, -2]]] * 2), normalize_batch(datums))


def test_retained_mass_is_measured_before_normalization():
    result = topk_diagnostics(np.log([[0.6, 0.3], [0.6, 0.38], [0.2, 0.3]]))
    masses = np.array([0.9, 0.98, 0.5])
    assert result["retained_mass_mean"] == pytest.approx(masses.mean())
    assert result["retained_mass_p05"] == pytest.approx(np.quantile(masses, 0.05))
    assert result["retained_mass_min"] == pytest.approx(0.5)
    assert result["retained_mass_fraction_below_095"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "weights",
    [
        [[0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [float("nan"), 1.0]],
        [[0.0, 0.0], [-0.1, 1.1]],
        [[0.0, 0.0], [0.2, 0.2]],
    ],
)
def test_invalid_native_weights_rejected(weights):
    datum = make_datum()
    datum.loss_fn_inputs["weights"] = tinker.TensorData.from_numpy(
        np.asarray(weights, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        normalize_batch([datum])


def test_masked_underflow_slot_ignores_missing_student_logprob():
    datum = make_datum(((1.0, 1e-300),))
    result = summarize_soft(output([[[np.nan, np.nan], [-2.0, np.nan]]]), [datum])
    assert result["soft_cross_entropy"] == 2
    assert result["teacher_entropy"] == 0
