"""Activities for the eval-suite workflows.

Two activity kinds:

- ``run_benchmark_shard`` — run one shard of examples through
  :func:`eval_harness.runner.run_experiment` with the benchmark's
  scorers (resolved from the metric registry by name). Model failures
  are recorded per example by the runner itself, so a shard only fails
  on infrastructure errors — exactly what the workflow's retry policy
  should retry.
- ``aggregate_benchmark`` — merge shard results and compute the
  benchmark summary with :mod:`eval_harness.stats` (means,
  ``summarize``, and a bootstrap CI for the primary metric). This lives
  in an activity because bootstrap resampling is randomized work that
  must not run inside deterministic workflow code.
"""

from __future__ import annotations

from typing import Callable

from temporalio import activity
from temporalio.exceptions import ApplicationError

from eval_harness.datasets import Example
from eval_harness.metrics import get_metric
from eval_harness.runner import (
    ConstantModel,
    EchoModel,
    ModelFn,
    ReverseModel,
    run_experiment,
)
from eval_harness.stats import bootstrap_ci, summarize
from eval_harness.temporal.models import (
    AggregateBenchmarkInput,
    BenchmarkSummary,
    ExampleScore,
    RunShardInput,
    ShardResult,
)

#: Model factory protocol: (kind, model_name) -> ModelFn. Deployments
#: with real model clients supply their own factory to EvalActivities.
ModelFactory = Callable[[str, str], ModelFn]


def default_model_factory(kind: str, model_name: str) -> ModelFn:
    """Resolve the harness's built-in deterministic models by kind."""
    if kind == "echo":
        return EchoModel()
    if kind == "reverse":
        return ReverseModel()
    if kind == "constant":
        return ConstantModel()
    raise ApplicationError(
        f"unknown model_kind: {kind!r} "
        "(built-ins: 'echo', 'reverse', 'constant')",
        type="UnknownModelKind",
        non_retryable=True,
    )


class EvalActivities:
    """Shard execution + aggregation activities."""

    def __init__(self, model_factory: ModelFactory = default_model_factory) -> None:
        self._model_factory = model_factory

    @activity.defn
    async def run_benchmark_shard(self, input: RunShardInput) -> ShardResult:
        try:
            metrics = [get_metric(name) for name in input.metric_names]
        except KeyError as exc:
            raise ApplicationError(
                str(exc), type="UnknownMetric", non_retryable=True
            ) from exc
        model = self._model_factory(input.model_kind, input.model_name)
        examples = [
            Example(id=spec.id, input=spec.input, reference=spec.reference)
            for spec in input.examples
        ]
        result = run_experiment(
            model,
            model_name=input.model_name,
            dataset=examples,
            dataset_name=input.benchmark_name,
            metrics=metrics,
        )
        return ShardResult(
            benchmark_name=input.benchmark_name,
            examples=[
                ExampleScore(
                    example_id=r.example_id,
                    prediction=r.prediction,
                    scores=dict(r.scores),
                    error=r.error,
                    latency_s=r.latency_s,
                )
                for r in result.example_results
            ],
        )

    @activity.defn
    async def aggregate_benchmark(
        self, input: AggregateBenchmarkInput
    ) -> BenchmarkSummary:
        scores: list[ExampleScore] = [
            ex for shard in input.shard_results for ex in shard.examples
        ]
        n = len(scores)
        n_errors = sum(1 for ex in scores if ex.error is not None)

        means: dict[str, float] = {}
        per_metric: dict[str, list[float]] = {}
        for name in input.metric_names:
            values = [ex.scores[name] for ex in scores if name in ex.scores]
            per_metric[name] = values
            if values:
                means[name] = sum(values) / len(values)

        primary = input.metric_names[0] if input.metric_names else ""
        primary_values = per_metric.get(primary, [])
        primary_summary: dict[str, float] = {}
        primary_ci: list[float] = []
        if primary_values:
            primary_summary = {
                k: float(v) for k, v in summarize(primary_values).items()
            }
            if len(primary_values) >= 2:
                lo, hi = bootstrap_ci(
                    primary_values,
                    n_resamples=input.bootstrap_resamples,
                    seed=input.bootstrap_seed,
                )
                primary_ci = [lo, hi]

        mean_latency = (
            sum(ex.latency_s for ex in scores) / n if n else 0.0
        )
        return BenchmarkSummary(
            name=input.benchmark_name,
            model_name=input.model_name,
            n_examples=n,
            n_errors=n_errors,
            means=means,
            primary_metric=primary,
            primary_summary=primary_summary,
            primary_ci95=primary_ci,
            mean_latency_s=mean_latency,
        )

    def all(self) -> list:
        return [self.run_benchmark_shard, self.aggregate_benchmark]
