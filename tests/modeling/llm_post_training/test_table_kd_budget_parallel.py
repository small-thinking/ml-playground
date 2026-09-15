"""Concurrent teacher calls must share reservations and fail closed on resume."""

from concurrent.futures import ThreadPoolExecutor
import json
from threading import Barrier

import pytest

from modeling.llm_post_training.vlm_table_extraction_lab.kd_collection_budget import (
    ConcurrentBudget,
)


def test_parallel_reservations_cannot_each_spend_total_budget(tmp_path):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    barrier = Barrier(2)

    def request(key):
        account = budget.for_example()
        barrier.wait()
        try:
            account.reserve(key, 0.6)
        except ValueError:
            return "blocked"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, ["a", "b"]))
    assert sorted(results) == ["blocked", "reserved"]
    with pytest.raises(ValueError, match="Uncertain"):
        ConcurrentBudget(budget.path, 1)


def test_out_of_order_settlement_preserves_other_pending_request(tmp_path):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    a, b = budget.for_example(), budget.for_example()
    a.reserve("a", 0.2)
    b.reserve("b", 0.3)
    b.settle(0.25)
    saved = json.loads(budget.path.read_text())
    assert set(saved["pending"]) == {"a"}
    assert saved["estimated_compute_usd"] == 0.25
    with pytest.raises(ValueError):
        a.settle(0.21)
    a.settle(0.1)
    resumed = ConcurrentBudget(budget.path, 1)
    assert resumed.state == {"pending": None, "calls": 2, "estimated_compute_usd": 0.35}


def test_duplicate_reservation_does_not_discard_original(tmp_path):
    budget = ConcurrentBudget(tmp_path / "usage.json", 1)
    a = budget.for_example()
    a.reserve("a", 0.1)
    original = budget.path.read_bytes()
    with pytest.raises(ValueError):
        a.reserve("b", 0.2)
    with pytest.raises(ValueError):
        budget.for_example().reserve("a", 0.2)
    assert budget.path.read_bytes() == original
