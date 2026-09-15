from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import json
from types import SimpleNamespace

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab import opd_sampling_budget
from modeling.llm_post_training.vlm_table_extraction_lab.kd_collect import atomic_json
from modeling.llm_post_training.vlm_table_extraction_lab.kd_collection_budget import (
    ConcurrentBudget,
)
from modeling.llm_post_training.vlm_table_extraction_lab.opd_sampling_budget import (
    BudgetedSampler,
)
from modeling.llm_post_training.vlm_table_extraction_lab.sft import (
    FORWARD_RATE,
    SAMPLE_RATE,
)


class Sequence:
    def __init__(self, tokens):
        self.tokens = tokens

    def model_dump(self, mode):
        assert mode == "json"
        return {"tokens": self.tokens, "stop_reason": "stop"}


class Client:
    def __init__(self, tokens=(1, 2), error=None):
        self.calls = []
        self.tokens, self.error = tokens, error

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        future = Future()
        if self.error:
            future.set_exception(self.error)
        else:
            future.set_result(SimpleNamespace(sequences=[Sequence(list(self.tokens))]))
        return future


def sample(sampler):
    return sampler.sample(
        SimpleNamespace(length=100), 1, SimpleNamespace(max_tokens=10)
    )


def test_concurrent_reservations_settle_actual_cost_and_preserve_prior_usage(tmp_path):
    usage = tmp_path / "usage.json"
    atomic_json(usage, {"estimated_compute_usd": 0.25, "calls": 2, "pending": None})
    budget = ConcurrentBudget(usage, 1)
    sampler = BudgetedSampler(Client(), budget, tmp_path / "raw")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = list(pool.map(lambda _: sample(sampler), range(4)))
    assert len(budget.state["pending"]) == 4
    bound = (100 * FORWARD_RATE + 10 * SAMPLE_RATE) / 1e6
    assert all(
        p["estimated_usd_bound"] == bound for p in budget.state["pending"].values()
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda f: f.result(timeout=1), futures))
    assert futures[0].result() is results[0]
    actual = (100 * FORWARD_RATE + 2 * SAMPLE_RATE) / 1e6
    assert budget.state["estimated_compute_usd"] == pytest.approx(0.25 + 4 * actual)
    assert budget.state["calls"] == 6 and budget.state["pending"] is None
    assert json.loads(usage.read_text()) == budget.state
    paths = sorted((tmp_path / "raw").glob("*.json"))
    assert [p.name for p in paths] == [f"{i:06d}.json" for i in range(1, 5)]
    assert json.loads(paths[0].read_text())["sequences"][0]["tokens"] == [1, 2]


def test_budget_rejection_precedes_paid_call(tmp_path):
    budget = ConcurrentBudget(tmp_path / "usage.json", 0)
    client = Client()
    sampler = BudgetedSampler(client, budget, tmp_path / "raw")
    with pytest.raises(ValueError, match="before next paid request"):
        sample(sampler)
    assert not client.calls and budget.state["pending"] is None


@pytest.mark.parametrize("error", [TimeoutError(), CancelledError()])
def test_failed_or_cancelled_request_remains_pending_without_retry(tmp_path, error):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    client = Client(error=error)
    future = sample(BudgetedSampler(client, budget, tmp_path / "raw"))
    for _ in range(2):
        with pytest.raises(type(error)):
            future.result(timeout=1)
    assert len(client.calls) == 1 and budget.state["pending"]
    with pytest.raises(ValueError, match="Uncertain paid request"):
        ConcurrentBudget(budget.path, 1)


@pytest.mark.parametrize("tokens", [[], [-1], [True], [1.5], [1] * 11])
def test_invalid_tokens_are_saved_without_settlement(tmp_path, tokens):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    future = sample(BudgetedSampler(Client(tokens), budget, tmp_path / "raw"))
    with pytest.raises(ValueError, match="Invalid sampled tokens"):
        future.result()
    assert budget.state["pending"] and budget.state["calls"] == 0
    raw = json.loads((tmp_path / "raw" / "000001.json").read_text())
    assert raw["sequences"][0]["tokens"] == tokens


def test_raw_persistence_failure_prevents_settlement(tmp_path, monkeypatch):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    future = sample(BudgetedSampler(Client(), budget, tmp_path / "raw"))

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(opd_sampling_budget, "atomic_json", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        future.result()
    assert budget.state["pending"] and budget.state["calls"] == 0


def test_multiple_sequences_rejected_before_paid_call(tmp_path):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    client = Client()
    sampler = BudgetedSampler(client, budget, tmp_path / "raw")
    with pytest.raises(ValueError, match="exactly one"):
        sampler.sample(SimpleNamespace(length=100), 2, SimpleNamespace(max_tokens=10))
    assert not client.calls and budget.state["pending"] is None
