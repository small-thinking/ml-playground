"""Durable per-request sampling reservations for OPD checkpoint generation."""

from pathlib import Path
from threading import Lock
from uuid import uuid4

from .kd_collect import atomic_json
from .sft import FORWARD_RATE, SAMPLE_RATE


class BudgetedSampler:
    def __init__(self, client, budget, output_dir):
        self.client, self.budget = client, budget
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.prefix = uuid4().hex
        self.number = max(
            (int(p.stem) for p in self.output_dir.glob("*.json") if p.stem.isdigit()),
            default=0,
        )

    def sample(self, prompt, num_samples, sampling_params):
        if num_samples != 1:
            raise ValueError("Budgeted sampling requires exactly one sequence")
        length, cap = prompt.length, sampling_params.max_tokens
        if any(type(n) is not int or n <= 0 for n in (length, cap)):
            raise ValueError("Prompt length and output cap must be positive integers")
        with self.lock:
            self.number += 1
            number = self.number
        request = self.budget.for_example()
        request.reserve(
            f"opd-eval:{self.prefix}:{number}",
            (length * FORWARD_RATE + cap * SAMPLE_RATE) / 1e6,
        )
        future = self.client.sample(
            prompt=prompt, num_samples=1, sampling_params=sampling_params
        )
        return _BudgetedFuture(
            future, request, self.output_dir / f"{number:06d}.json", length, cap
        )


class _BudgetedFuture:
    def __init__(self, future, request, path, prompt_length, cap):
        self.future, self.request, self.path = future, request, path
        self.prompt_length, self.cap = prompt_length, cap
        self.lock = Lock()
        self.response = None
        self.error = None

    def result(self, timeout=None):
        # A repeated result() never submits another call or settles twice.
        with self.lock:
            if self.error is not None:
                raise self.error
            if self.response is not None:
                return self.response
            try:
                response = self.future.result(timeout=timeout)
                atomic_json(
                    self.path,
                    {
                        "sequences": [
                            {
                                "tokens": s.tokens,
                                "logprobs": s.logprobs,
                                "stop_reason": s.stop_reason,
                            }
                            for s in response.sequences
                        ]
                    },
                )
                if len(response.sequences) != 1:
                    raise ValueError("Expected exactly one sampled sequence")
                tokens = response.sequences[0].tokens
                if (
                    not tokens
                    or len(tokens) > self.cap
                    or any(type(t) is not int or t < 0 for t in tokens)
                ):
                    raise ValueError(
                        "Invalid sampled tokens; reservation remains pending"
                    )
                self.request.settle(
                    (self.prompt_length * FORWARD_RATE + len(tokens) * SAMPLE_RATE)
                    / 1e6
                )
                self.response = response
                return response
            except Exception as error:
                self.error = error
                raise
