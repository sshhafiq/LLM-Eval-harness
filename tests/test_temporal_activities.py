"""Unit tests for the Temporal eval activities (no Temporal server).

Shard activities run the harness's own ``run_experiment`` with registry
metrics, and the aggregation activity uses ``eval_harness.stats`` — both
are plain async methods, exercised directly here. Workflow tests live
in ``test_temporal_workflows.py``.
"""

from __future__ import annotations

import asyncio

import pytest
from temporalio.exceptions import ApplicationError

from eval_harness.temporal.activities import EvalActivities
from eval_harness.temporal.models import (
    AggregateBenchmarkInput,
    ExampleScore,
    ExampleSpec,
    RunShardInput,
    ShardResult,
)


def _spec(i: int, text: str | None = None) -> ExampleSpec:
    text = text if text is not None else f"answer {i}"
    return ExampleSpec(id=f"ex-{i}", input=text, reference=text)


def run(coro):
    return asyncio.run(coro)


# ----------------------------------------------------------------------
# shard execution
# ----------------------------------------------------------------------
def test_shard_echo_model_scores_perfectly():
    activities = EvalActivities()
    result = run(activities.run_benchmark_shard(RunShardInput(
        benchmark_name="sanity",
        model_kind="echo",
        model_name="echo",
        metric_names=["exact_match", "token_f1"],
        examples=[_spec(i) for i in range(4)],
    )))
    assert result.benchmark_name == "sanity"
    assert len(result.examples) == 4
    for ex in result.examples:
        assert ex.error is None
        assert ex.scores["exact_match"] == 1.0
        assert ex.scores["token_f1"] == 1.0
        assert ex.prediction == ex.example_id.replace("ex-", "answer ")


def test_shard_constant_model_scores_zero_on_exact_match():
    activities = EvalActivities()
    result = run(activities.run_benchmark_shard(RunShardInput(
        benchmark_name="baseline",
        model_kind="constant",
        model_name="constant",
        metric_names=["exact_match"],
        examples=[_spec(0), _spec(1)],
    )))
    assert all(ex.scores["exact_match"] == 0.0 for ex in result.examples)


def test_shard_unknown_metric_is_non_retryable():
    activities = EvalActivities()
    with pytest.raises(ApplicationError) as excinfo:
        run(activities.run_benchmark_shard(RunShardInput(
            benchmark_name="bad",
            model_kind="echo",
            model_name="echo",
            metric_names=["no_such_metric"],
            examples=[_spec(0)],
        )))
    assert excinfo.value.non_retryable is True


def test_shard_unknown_model_kind_is_non_retryable():
    activities = EvalActivities()
    with pytest.raises(ApplicationError) as excinfo:
        run(activities.run_benchmark_shard(RunShardInput(
            benchmark_name="bad",
            model_kind="gpt-9000",
            model_name="gpt-9000",
            metric_names=["exact_match"],
            examples=[_spec(0)],
        )))
    assert excinfo.value.non_retryable is True


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------
def _shard(scores: list[float], errors: int = 0) -> ShardResult:
    examples = [
        ExampleScore(example_id=f"ex-{i}", prediction="p",
                     scores={"exact_match": s}, latency_s=0.5)
        for i, s in enumerate(scores)
    ]
    for j in range(errors):
        examples.append(ExampleScore(
            example_id=f"err-{j}", prediction="",
            scores={"exact_match": 0.0}, error="RuntimeError: boom",
            latency_s=0.1,
        ))
    return ShardResult(benchmark_name="bench", examples=examples)


def test_aggregate_merges_shards_and_computes_stats():
    activities = EvalActivities()
    summary = run(activities.aggregate_benchmark(AggregateBenchmarkInput(
        benchmark_name="bench",
        model_name="echo",
        metric_names=["exact_match"],
        shard_results=[_shard([1.0, 1.0]), _shard([0.0, 1.0], errors=1)],
        bootstrap_resamples=200,
    )))
    assert summary.n_examples == 5
    assert summary.n_errors == 1
    assert summary.means["exact_match"] == pytest.approx(3.0 / 5.0)
    assert summary.primary_metric == "exact_match"
    assert summary.primary_summary["n"] == 5
    assert summary.primary_summary["mean"] == pytest.approx(0.6)
    assert len(summary.primary_ci95) == 2
    assert summary.primary_ci95[0] <= 0.6 <= summary.primary_ci95[1]
    assert summary.mean_latency_s == pytest.approx(
        (0.5 * 4 + 0.1) / 5
    )


def test_aggregate_empty_benchmark():
    activities = EvalActivities()
    summary = run(activities.aggregate_benchmark(AggregateBenchmarkInput(
        benchmark_name="empty",
        model_name="echo",
        metric_names=["exact_match"],
        shard_results=[],
    )))
    assert summary.n_examples == 0
    assert summary.means == {}
    assert summary.primary_ci95 == []
