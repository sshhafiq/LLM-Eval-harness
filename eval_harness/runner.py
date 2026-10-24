"""Experiment runner: apply models to datasets and score with metrics.

A *model* is any callable ``(prompt: str) -> str`` (the :class:`ModelFn`
protocol), so real LLM clients, local wrappers, and test doubles all plug in
the same way.  Examples are evaluated in parallel with a thread pool because
model calls are typically I/O-bound.
"""

from __future__ import annotations

import statistics
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Protocol, Sequence

from .datasets import Example
from .metrics import Metric, MetricResult


class ModelFn(Protocol):
    """Structural type for model callables used by the runner."""

    def __call__(self, prompt: str) -> str:
        ...


# ---------------------------------------------------------------------------
# Mock models (tests, examples, offline development)
# ---------------------------------------------------------------------------


class ConstantModel:
    """Always returns the same response regardless of the prompt."""

    def __init__(self, response: str = "I don't know.") -> None:
        self.response = response

    def __call__(self, prompt: str) -> str:
        return self.response

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ConstantModel(response={self.response!r})"


class EchoModel:
    """Returns the prompt unchanged — a deterministic pseudo-model."""

    def __call__(self, prompt: str) -> str:
        return prompt


class ReverseModel:
    """Returns the prompt reversed — useful for asserting plumbing works."""

    def __call__(self, prompt: str) -> str:
        return prompt[::-1]


class FailingModel:
    """Raises on every call; used to test the runner's error handling."""

    def __init__(self, message: str = "boom") -> None:
        self.message = message

    def __call__(self, prompt: str) -> str:
        raise RuntimeError(self.message)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ExampleResult:
    """Per-example outcome: prediction plus one score per metric."""

    example_id: str
    prediction: str
    scores: Dict[str, float]
    details: Dict[str, Dict[str, object]] = field(default_factory=dict)
    error: str | None = None
    latency_s: float = 0.0


@dataclass
class ExperimentResult:
    """Complete outcome of running one model over one dataset."""

    model_name: str
    dataset_name: str
    n_examples: int
    example_results: List[ExampleResult]

    def aggregate(self) -> Dict[str, float]:
        """Mean score per metric across all examples."""
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for result in self.example_results:
            for name, score in result.scores.items():
                totals[name] = totals.get(name, 0.0) + score
                counts[name] = counts.get(name, 0) + 1
        return {name: totals[name] / counts[name] for name in totals}

    def scores_for(self, metric: str) -> List[float]:
        """Per-example scores for one metric, in example order."""
        return [r.scores[metric] for r in self.example_results if metric in r.scores]

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.example_results if r.error is not None)

    @property
    def mean_latency_s(self) -> float:
        if not self.example_results:
            return 0.0
        return statistics.mean(r.latency_s for r in self.example_results)


def _evaluate_one(
    model_fn: ModelFn, example: Example, metrics: Sequence[Metric]
) -> ExampleResult:
    started = time.perf_counter()
    try:
        prediction = model_fn(example.input)
    except Exception as exc:  # noqa: BLE001 - record, don't crash the run
        return ExampleResult(
            example_id=example.id,
            prediction="",
            scores={m.name: 0.0 for m in metrics},
            error=f"{type(exc).__name__}: {exc}",
            latency_s=time.perf_counter() - started,
        )
    scores: Dict[str, float] = {}
    details: Dict[str, Dict[str, object]] = {}
    for metric in metrics:
        result: MetricResult = metric(prediction, example.reference)
        scores[result.name] = result.score
        details[result.name] = dict(result.details)
    return ExampleResult(
        example_id=example.id,
        prediction=prediction,
        scores=scores,
        details=details,
        latency_s=time.perf_counter() - started,
    )


def run_experiment(
    model_fn: ModelFn,
    *,
    model_name: str,
    dataset: Sequence[Example],
    dataset_name: str,
    metrics: Sequence[Metric],
    num_workers: int = 4,
) -> ExperimentResult:
    """Run ``model_fn`` over ``dataset`` and score every example.

    Examples are evaluated concurrently with ``num_workers`` threads.  A model
    exception for one example is recorded on that example (score 0.0 for every
    metric) without aborting the run.  The order of ``example_results``
    matches the input dataset order.
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    dataset = list(dataset)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        results = list(
            pool.map(
                lambda ex: _evaluate_one(model_fn, ex, metrics),
                dataset,
            )
        )
    return ExperimentResult(
        model_name=model_name,
        dataset_name=dataset_name,
        n_examples=len(dataset),
        example_results=results,
    )
