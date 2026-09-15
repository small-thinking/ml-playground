"""Offline objective checks for student-prefix Top-K and hybrid OPD/gold CE."""

import numpy as np
import pytest
import torch

from modeling.llm_post_training.vlm_table_extraction_lab.kd_targets import soft_targets
from modeling.llm_post_training.vlm_table_extraction_lab.opd_ablation_objectives import (
    hybrid_batches,
    normalize_ce_batch,
)
from modeling.llm_post_training.vlm_table_extraction_lab.opd_targets import policy_datum

tinker = pytest.importorskip("tinker")


def array(datum, key):
    return datum.loss_fn_inputs[key].to_numpy()


def gold(targets, weights=None):
    targets = np.asarray(targets, dtype=np.int64)
    if weights is None:
        weights = np.ones(targets.shape, dtype=np.float32)
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints([0] * targets.shape[0]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(targets),
            "weights": tinker.TensorData.from_numpy(
                np.asarray(weights, dtype=np.float32)
            ),
        },
    )


def policy(tokens, old, teacher):
    return policy_datum(tinker.ModelInput.from_ints([0]), tokens, old, teacher, 3)


def test_ce_normalization_preserves_scalar_and_topk_native_shapes_and_tokens():
    originals = [gold([0, 1], [0, 1]), gold([[0, 1], [1, 2]], [[0.6, 0.4], [0.3, 0.7]])]
    normalized = normalize_ce_batch(iter(originals))
    assert array(normalized[0], "weights").shape == (2,)
    assert array(normalized[1], "weights").shape == (2, 2)
    for original, result in zip(originals, normalized, strict=True):
        assert set(result.loss_fn_inputs) == {"target_tokens", "weights"}
        np.testing.assert_array_equal(
            array(original, "target_tokens"), array(result, "target_tokens")
        )
        np.testing.assert_allclose(
            array(result, "weights"), array(original, "weights") / 3
        )
    np.testing.assert_array_equal(array(originals[0], "weights"), [0, 1])
    np.testing.assert_allclose(array(originals[1], "weights"), [[0.6, 0.4], [0.3, 0.7]])


def test_topk_ce_uses_student_prefixes_and_teacher_distribution_at_each_position():
    prompt = tinker.ModelInput.from_ints([2, 2])
    datum = soft_targets(
        prompt,
        [0, 1, 2],
        [[2, 1], [0, 2], [1, 0]],
        np.log([[0.6, 0.2], [0.4, 0.3], [0.5, 0.2]]),
        3,
    )
    assert datum.model_input.chunks[-1].tokens == [0, 1]
    np.testing.assert_array_equal(
        array(datum, "target_tokens")[1:], [[2, 1], [0, 2], [1, 0]]
    )
    normalized = normalize_ce_batch([datum])[0]
    logits = torch.tensor([0.2, -0.4, 0.5], dtype=torch.float64, requires_grad=True)
    logp = logits.log_softmax(-1)
    weights = torch.tensor(array(normalized, "weights"), dtype=torch.float64)
    targets = torch.tensor(array(normalized, "target_tokens"))
    loss = -(weights * logp[targets]).sum()
    loss.backward()
    teacher = torch.tensor(
        [[(0.4 / 0.7), 0, (0.3 / 0.7)], [0, 0.25, 0.75], [(0.2 / 0.7), (0.5 / 0.7), 0]],
        dtype=torch.float64,
    ).mean(0)
    torch.testing.assert_close(
        logits.grad, logits.detach().softmax(-1) - teacher, rtol=1e-6, atol=1e-7
    )


def test_hybrid_scales_each_token_mean_and_accumulates_gradients_before_one_update():
    logits = torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64, requires_grad=True)
    initial = logits.detach().clone()
    old = initial.log_softmax(-1).numpy()
    policies = [
        policy([0], old[[0]], [-0.5]),
        policy([1, 2], old[[1, 2]], [old[1], -2.0]),
    ]
    golds = [gold([2, 2]), gold([0, 1, 1])]
    originals = [
        {k: array(d, k).copy() for k in d.loss_fn_inputs} for d in policies + golds
    ]
    native_policy, native_gold = hybrid_batches(policies, golds)
    assert all(
        set(d.loss_fn_inputs) == {"target_tokens", "logprobs", "advantages"}
        for d in native_policy
    )
    assert sum(array(d, "weights").sum() for d in native_gold) == pytest.approx(0.25)
    for before, datum in zip(originals, policies + golds, strict=True):
        for key in before:
            np.testing.assert_array_equal(array(datum, key), before[key])
    for original, scaled in zip(policies, native_policy, strict=True):
        np.testing.assert_allclose(
            array(scaled, "advantages"), array(original, "advantages") * 0.75 / 3
        )

    logp = logits.log_softmax(-1)
    policy_loss = sum(
        -(
            torch.exp(
                logp[torch.tensor(array(d, "target_tokens"))]
                - torch.tensor(array(d, "logprobs"))
            )
            * torch.tensor(array(d, "advantages"))
        ).sum()
        for d in native_policy
    )
    policy_loss.backward()
    logp = logits.log_softmax(-1)
    gold_loss = sum(
        -(
            torch.tensor(array(d, "weights"))
            * logp[torch.tensor(array(d, "target_tokens"))]
        ).sum()
        for d in native_gold
    )
    gold_loss.backward()

    expected_logits = initial.clone().requires_grad_()
    expected_lp = expected_logits.log_softmax(-1)
    advantage = torch.tensor(np.concatenate([array(d, "advantages") for d in policies]))
    expected_loss = (
        -0.75 * (expected_lp * advantage).mean()
        - 0.25 * expected_lp[torch.tensor([2, 2, 0, 1, 1])].mean()
    )
    expected_loss.backward()
    torch.testing.assert_close(logits.grad, expected_logits.grad, rtol=1e-6, atol=1e-7)
    optimizer = torch.optim.SGD([logits], lr=0.1)
    optimizer.step()
    torch.testing.assert_close(
        logits, initial - 0.1 * expected_logits.grad, rtol=1e-6, atol=1e-7
    )


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_mixture_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        hybrid_batches([policy([1], [-2], [-1])], [gold([1])], weight)


@pytest.mark.parametrize(
    "weights",
    [[0, 0], [0, 0.5], [0, -1], [0, float("nan")], [[0.3, 0.3], [0, 0]], [[[1]]]],
)
def test_gold_and_soft_weights_must_define_unnormalized_unit_mass_tokens(weights):
    weights = np.asarray(weights)
    datum = gold(np.zeros(weights.shape, dtype=np.int64), weights)
    with pytest.raises(ValueError):
        normalize_ce_batch([datum])


def test_hybrid_supports_soft_ce_and_retains_zero_reward_policy_tokens():
    native_policy, native_gold = hybrid_batches(
        [policy([0, 1], [-2, -2], [-2, -1])],
        [gold([[0, 1], [1, 2]], [[0.4, 0.6], [0.7, 0.3]])],
    )
    np.testing.assert_array_equal(array(native_policy[0], "advantages"), [0, 0.375])
    assert array(native_gold[0], "weights").shape == (2, 2)
    np.testing.assert_allclose(
        array(native_gold[0], "weights"), np.array([[0.4, 0.6], [0.7, 0.3]]) * 0.125
    )
