"""Offline alignment, importance-sampling, and reverse-KL gradient checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modeling.llm_post_training.vlm_table_extraction_lab.opd_targets import (
    native_batch,
    normalize_batch,
    policy_datum,
    summarize_policy,
)

tinker = pytest.importorskip("tinker")


def make_datum(old=(-2.0,), teacher=(-1.0,)):
    return policy_datum(
        tinker.ModelInput.from_ints([8, 9]), [1] * len(old), old, teacher, 10
    )


def array(datum, key):
    return datum.loss_fn_inputs[key].to_numpy()


def output(values):
    return SimpleNamespace(
        loss_fn_outputs=[
            {"logprobs": tinker.TensorData.from_numpy(np.asarray(v, dtype=np.float32))}
            for v in values
        ]
    )


def native_loss(datum, current_logprobs):
    """Tinker importance_sampling's summed surrogate, with detached old LP/A."""
    old = torch.tensor(array(datum, "logprobs"), dtype=torch.float64)
    advantage = torch.tensor(array(datum, "advantages"), dtype=torch.float64)
    return -(torch.exp(current_logprobs - old) * advantage).sum()


def test_multimodal_shift_preserves_image_and_scores_first_token_through_eos():
    image = tinker.types.ImageChunk(data=b"fixture", format="png", expected_tokens=3)
    prompt = tinker.ModelInput(
        chunks=[
            tinker.EncodedTextChunk(tokens=[8]),
            image,
            tinker.EncodedTextChunk(tokens=[9]),
        ]
    )
    eos = 7
    datum = policy_datum(prompt, [2, 3, eos], [-2, -3, -4], [-1, -5, -4], 10)
    assert datum.model_input.chunks[1] is image
    assert datum.model_input.length == prompt.length + 2
    assert datum.model_input.chunks[-1].tokens == [2, 3]
    assert prompt.chunks[-1].tokens == [9]
    assert set(datum.loss_fn_inputs) == {
        "target_tokens",
        "logprobs",
        "advantages",
        "mask",
    }
    for key in datum.loss_fn_inputs:
        assert array(datum, key).shape == (7,)
        np.testing.assert_array_equal(array(datum, key)[:4], 0)
    np.testing.assert_array_equal(array(datum, "target_tokens")[4:], [2, 3, eos])
    np.testing.assert_array_equal(array(datum, "logprobs")[4:], [-2, -3, -4])
    np.testing.assert_array_equal(array(datum, "advantages")[4:], [1, -2, 0])
    # A zero-reward EOS remains a sampled token in all denominators.
    np.testing.assert_array_equal(array(datum, "mask")[4:], 1)


def test_single_eos_completion_has_no_empty_input_chunk():
    prompt = tinker.ModelInput.from_ints([4])
    datum = policy_datum(prompt, [2], [-0.2], [-0.2], 5)
    assert datum.model_input.chunks == prompt.chunks
    np.testing.assert_array_equal(array(datum, "target_tokens"), [2])
    np.testing.assert_array_equal(array(datum, "mask"), [1])
    np.testing.assert_array_equal(array(datum, "advantages"), [0])


@pytest.mark.parametrize(
    "change",
    [
        {"sampled_tokens": []},
        {"sampled_tokens": [1.0]},
        {"sampled_tokens": [True]},
        {"sampled_tokens": [-1]},
        {"sampled_tokens": [10]},
        {"sampled_tokens": [[1]]},
        {"prompt": tinker.ModelInput.empty()},
        {"vocab_size": True},
        {"vocab_size": 0},
        {"vocab_size": 10.0},
    ]
    + [
        {key: value}
        for key in ("sampling_logprobs", "teacher_logprobs")
        for value in (
            [],
            [-1, -2],
            [[-1]],
            [None],
            [float("nan")],
            [float("inf")],
            [float("-inf")],
            [-99999.0],
            [0.1],
        )
    ],
)
def test_invalid_tokens_logprob_vectors_and_missing_sentinels_fail_closed(change):
    args = dict(
        prompt=tinker.ModelInput.from_ints([8]),
        sampled_tokens=[1],
        sampling_logprobs=[-2.0],
        teacher_logprobs=[-1.0],
        vocab_size=10,
    )
    args.update(change)
    with pytest.raises(ValueError):
        policy_datum(**args)


def test_batch_scaling_counts_zero_advantage_tokens_and_does_not_mutate():
    datums = [make_datum(), make_datum((-2, -2, -2), (-2, -4, -1))]
    originals = [
        {key: array(d, key).copy() for key in d.loss_fn_inputs} for d in datums
    ]
    normalized = normalize_batch(datums)
    assert sum(array(d, "mask").sum() for d in normalized) == 4
    for original, datum, scaled in zip(originals, datums, normalized, strict=True):
        assert scaled is not datum
        for key in original:
            np.testing.assert_array_equal(array(datum, key), original[key])
            expected = original[key] / 4 if key == "advantages" else original[key]
            np.testing.assert_array_equal(array(scaled, key), expected)
    # Unequal lengths must not introduce per-example weighting or a second /B.
    current = [
        torch.tensor([0, -2.0], dtype=torch.float64, requires_grad=True),
        torch.tensor([0, -2.0, -2.0, -2.0], dtype=torch.float64, requires_grad=True),
    ]
    loss = sum(native_loss(d, lp) for d, lp in zip(normalized, current, strict=True))
    loss.backward()
    torch.testing.assert_close(
        current[0].grad, torch.tensor([0, -0.25], dtype=torch.float64)
    )
    torch.testing.assert_close(
        current[1].grad, torch.tensor([0, 0, 0.5, -0.25], dtype=torch.float64)
    )


def test_all_zero_advantages_still_have_a_valid_batch_and_zero_gradient():
    datum = normalize_batch([make_datum((-2, -3), (-2, -3))])[0]
    current = torch.tensor([0, -1, -4], dtype=torch.float64, requires_grad=True)
    loss = native_loss(datum, current)
    loss.backward()
    assert loss.item() == 0
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_native_batch_strips_diagnostic_mask_and_normalizes_exactly_once():
    datums = [make_datum(), make_datum((-2, -2, -2), (-2, -4, -1))]
    native = native_batch(datums)
    assert len(native) == 2
    for original, submitted in zip(datums, native, strict=True):
        assert set(submitted.loss_fn_inputs) == {
            "target_tokens",
            "logprobs",
            "advantages",
        }
        assert "mask" in original.loss_fn_inputs
        np.testing.assert_array_equal(
            array(submitted, "advantages"), array(original, "advantages") / 4
        )
        for key in ("target_tokens", "logprobs"):
            np.testing.assert_array_equal(array(submitted, key), array(original, key))
    # Zero-advantage tokens contributed to the count even though mask is removed.
    np.testing.assert_array_equal(array(native[0], "advantages"), [0, 0.25])
    np.testing.assert_array_equal(array(datums[0], "advantages"), [0, 1])
    np.testing.assert_array_equal(array(datums[1], "mask"), [0, 1, 1, 1])


@pytest.mark.parametrize(
    "mask", [[0, 0], [0, 0.5], [0, -1], [0, float("nan")], [[0, 1]]]
)
def test_invalid_masks_are_rejected(mask):
    datum = make_datum()
    datum.loss_fn_inputs["mask"] = tinker.TensorData.from_numpy(
        np.asarray(mask, dtype=np.float32)
    )
    with pytest.raises(ValueError):
        normalize_batch([datum])


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        normalize_batch([])


def test_enumerated_expected_surrogate_gradient_equals_reverse_kl_gradient():
    # Exhaustive action enumeration, not a noisy Monte Carlo approximation.
    logits = torch.tensor([0.4, -0.2, 0.7], dtype=torch.float64, requires_grad=True)
    teacher = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    old = logp.detach()
    old_probabilities = old.exp()
    expected_surrogate = torch.zeros((), dtype=torch.float64)
    for action in range(3):
        datum = policy_datum(
            tinker.ModelInput.from_ints([0]),
            [action],
            [old[action].item()],
            [teacher[action].log().item()],
            3,
        )
        expected_surrogate += old_probabilities[action] * native_loss(
            normalize_batch([datum])[0], logp[action : action + 1]
        )
    actual = torch.autograd.grad(expected_surrogate, logits, retain_graph=True)[0]
    reverse_kl = (logp.exp() * (logp - teacher.log())).sum()
    expected = torch.autograd.grad(reverse_kl, logits)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert expected.sum().item() == pytest.approx(0, abs=1e-14)
    assert expected.norm().item() > 0.1


@pytest.mark.parametrize("teacher_lp,expected_sign", [(-1.0, -1), (-3.0, 1)])
def test_reward_sign_and_off_policy_ratio_follow_teacher_preference(
    teacher_lp, expected_sign
):
    datum = normalize_batch([make_datum((-2.0,), (teacher_lp,))])[0]
    current = torch.tensor([0, -1.0], dtype=torch.float64, requires_grad=True)
    loss = native_loss(datum, current)
    loss.backward()
    # Negative gradient raises the sampled token's LP when teacher prefers it.
    assert current.grad[1].item() == pytest.approx(expected_sign * np.e)
    assert current.grad[0].item() == 0


def test_summary_uses_token_means_and_masks_prompt_nan_including_zero_rewards():
    datums = [make_datum(), make_datum((-3, -4), (-3, -2))]
    result = summarize_policy(output([[np.nan, -1], [np.nan, -4, -4]]), datums)
    assert result["supervised_tokens"] == 3
    assert result["sampled_reverse_kl"] == pytest.approx(-1)
    assert result["sampling_entropy_estimate"] == pytest.approx(3)
    assert result["teacher_nll_on_student_samples"] == pytest.approx(2)
    assert result["importance_ratio_mean"] == pytest.approx((np.e + 1 / np.e + 1) / 3)
    assert result["importance_ratio_max"] == pytest.approx(np.e)
    # Sampled reverse KL can be negative; do not clamp a finite MC estimate.
    assert result["sampled_reverse_kl"] < 0


@pytest.mark.parametrize(
    "native",
    [
        [0, float("nan")],
        [0, float("inf")],
        [0, float("-inf")],
        [0, 0.1],
        [0, -99999],
        [-1],
        [[0, -1]],
    ],
)
def test_summary_rejects_invalid_active_learner_logprobs_or_shape(native):
    with pytest.raises(ValueError):
        summarize_policy(output([native]), [make_datum()])


def test_summary_rejects_mismatched_output_counts():
    with pytest.raises(ValueError):
        summarize_policy(output([]), [make_datum()])
