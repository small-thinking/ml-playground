"""Exercise the training protocol with a fake service, not a paid model."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import tinker

from modeling.llm_post_training.vlm_table_extraction_lab import kd
from modeling.llm_post_training.vlm_table_extraction_lab.kd_targets import soft_targets


def fixture_data():
    prompt = tinker.ModelInput.from_ints([1, 2])
    train = [
        soft_targets(
            prompt, [3, 4], [[3, 4], [4, 3]], np.log([[0.8, 0.2], [0.9, 0.1]]), 10
        )
        for _ in range(8)
    ]
    gold = tinker.Datum(
        model_input=tinker.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=[2, 3, 4], dtype="int64"),
            "weights": tinker.TensorData(data=[0, 1, 1], dtype="float32"),
        },
    )
    dev = [({"id": str(i)}, prompt, gold) for i in range(100)]
    return train, dev


def args_for(tmp_path, generate=False):
    return SimpleNamespace(
        output_dir=tmp_path,
        batch_size=4,
        epochs=1,
        rank=8,
        seed=20260913,
        learning_rate=1e-4,
        eval_every=1,
        max_new_tokens=8192,
        generate_dev=generate,
    )


def test_cost_counts_full_dev_and_both_soft_target_endpoints(tmp_path):
    train, dev = fixture_data()
    args = args_for(tmp_path, True)
    cost = kd.estimate_cost(train, dev, args)
    assert cost["training_tokens"] == 24
    assert cost["nll_forward_tokens"] == 2 * 24 + 3 * 300
    assert cost["generation_prefill_tokens"] == 200
    assert cost["generation_output_token_bound"] == 819200
    assert cost["estimated_usd_bound"] == pytest.approx(
        (24 * 0.737 + (948 + 200) * 0.33 + 819200 * 1.005) / 1e6
    )


def test_run_uses_fresh_lora_full_dev_and_normalized_soft_gradients(
    tmp_path, monkeypatch
):
    train, dev = fixture_data()
    args = args_for(tmp_path)
    calls, events = [], []

    def future(value):
        return SimpleNamespace(result=lambda **kw: value)

    class Client:
        updates = 0

        def get_info(self):
            return SimpleNamespace(model_dump=lambda **kw: {"fixture": True})

        def forward(self, datums, loss_fn):
            calls.append(("forward", len(datums), loss_fn))
            return self.output(datums)

        def forward_backward(self, datums, loss_fn):
            calls.append(("backward", len(datums), loss_fn))
            assert all(len(d.loss_fn_inputs["weights"].shape) == 2 for d in datums)
            assert sum(
                np.sum(d.loss_fn_inputs["weights"].to_numpy()) for d in datums
            ) == pytest.approx(1)
            return self.output(datums)

        def output(self, datums):
            return future(
                SimpleNamespace(
                    loss_fn_outputs=[
                        {
                            "logprobs": tinker.TensorData.from_numpy(
                                np.full(
                                    d.loss_fn_inputs["weights"].to_numpy().shape,
                                    -2.0 + self.updates * 0.1,
                                    dtype=np.float32,
                                )
                            )
                        }
                        for d in datums
                    ]
                )
            )

        def optim_step(self, params):
            calls.append(("optimizer", params.learning_rate))
            self.updates += 1
            return future(SimpleNamespace(metrics={}))

        def save_state(self, name, ttl_seconds):
            assert ttl_seconds == 7 * 86400
            return future(SimpleNamespace(path="private-state"))

        def save_weights_for_sampler(self, name, ttl_seconds):
            return future(SimpleNamespace(path="private-sampler"))

    client = Client()

    class Service:
        def create_lora_training_client(self, **kwargs):
            assert kwargs == {
                "base_model": kd.MODEL,
                "rank": 8,
                "seed": args.seed,
                "train_attn": True,
                "train_mlp": True,
                "train_unembed": False,
            }
            return client

    monkeypatch.setattr(tinker, "ServiceClient", Service)
    report = {
        "optimizer_steps": 2,
        "warmup_steps": 2,
        "cost": kd.estimate_cost(train, dev, args),
    }
    logger = SimpleNamespace(log=events.append)
    kd.run(args, train, dev, None, None, None, report, logger)
    assert client.updates == 2
    assert [c[1] for c in calls if c[0] == "optimizer"] == [5e-5, 1e-4]
    assert [c[1] for c in calls if c[0] == "forward"] == [100, 8, 100, 100, 8]
    assert [c[1] for c in calls if c[0] == "backward"] == [4, 4]
    assert report["status"] == "completed"
    assert report["checks"]["teacher_target_kl_decreased"]
    assert math.isfinite(report["after"]["dev"]["assistant_perplexity"])
    assert "private-" not in json.dumps(events)
    assert len(json.loads((tmp_path / "after_dev_likelihoods.json").read_text())) == 100
