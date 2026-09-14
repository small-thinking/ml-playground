"""One durable shared budget for concurrent teacher requests."""

from threading import Lock

from .kd_collect import CollectionBudget, atomic_json


class ConcurrentBudget(CollectionBudget):
    def __init__(self, path, maximum):
        super().__init__(path, maximum)
        self.lock = Lock()

    def for_example(self):
        return RequestBudget(self)


class RequestBudget:
    """Each example makes sequential calls; reservations share one total cap."""

    def __init__(self, parent):
        self.parent, self.key = parent, None

    def reserve(self, key, bound):
        parent = self.parent
        with parent.lock:
            pending = dict(parent.state["pending"] or {})
            if self.key is not None or key in pending:
                raise ValueError("Duplicate outstanding request")
            reserved = sum(item["estimated_usd_bound"] for item in pending.values())
            if (
                parent.state["estimated_compute_usd"] + reserved + bound
            ) * 1.1 > parent.maximum:
                raise ValueError("Collection budget reached before next paid request")
            pending[key] = {"estimated_usd_bound": bound}
            parent.state["pending"] = pending
            atomic_json(parent.path, parent.state)
            self.key = key

    def settle(self, amount):
        parent = self.parent
        with parent.lock:
            pending = dict(parent.state["pending"] or {})
            if (
                self.key not in pending
                or amount > pending[self.key]["estimated_usd_bound"] + 1e-9
            ):
                raise ValueError("Paid usage exceeds its reservation")
            del pending[self.key]
            parent.state["estimated_compute_usd"] += amount
            parent.state["calls"] += 1
            parent.state["pending"] = pending or None
            atomic_json(parent.path, parent.state)
            self.key = None
