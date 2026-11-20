"""The eval-suite workflows.

Two levels of durable orchestration:

- ``EvalSuiteWorkflow`` (parent) starts one ``BenchmarkWorkflow`` child
  per benchmark and collects their summaries. Children keep each
  benchmark's history small, independently inspectable, and separately
  retryable — a suite of twenty benchmarks is twenty small histories,
  not one giant one.
- ``BenchmarkWorkflow`` (child) shards its benchmark's examples, fans
  shard activities out in parallel, then calls the aggregation activity
  (bootstrap statistics are randomized and belong in an activity, not in
  deterministic workflow code).

Progress for the benchmark currently being sharded is exposed through
the child's ``benchmark_progress`` query.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from eval_harness.temporal.activities import EvalActivities
    from eval_harness.temporal.models import (
        AggregateBenchmarkInput,
        BenchmarkInput,
        BenchmarkSummary,
        EvalSuiteInput,
        EvalSuiteResult,
        RunShardInput,
        ShardResult,
    )

_SHARD_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)

_AGGREGATE_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)


@workflow.defn
class BenchmarkWorkflow:
    """Run one benchmark: shard, fan out, aggregate."""

    def __init__(self) -> None:
        self._benchmark = ""
        self._shards_total = 0
        self._shards_completed = 0

    @workflow.query
    def benchmark_progress(self) -> dict:
        return {
            "benchmark": self._benchmark,
            "shards_total": self._shards_total,
            "shards_completed": self._shards_completed,
        }

    @workflow.run
    async def run(self, input: BenchmarkInput) -> BenchmarkSummary:
        benchmark = input.benchmark
        self._benchmark = benchmark.name
        examples = benchmark.examples
        shard_size = max(1, benchmark.shard_size)
        shards = [
            examples[i : i + shard_size]
            for i in range(0, len(examples), shard_size)
        ]
        self._shards_total = len(shards)

        async def run_one(shard_examples) -> ShardResult:
            result = await workflow.execute_activity(
                EvalActivities.run_benchmark_shard,
                RunShardInput(
                    benchmark_name=benchmark.name,
                    model_kind=input.model_kind,
                    model_name=input.model_name,
                    metric_names=list(benchmark.metric_names),
                    examples=shard_examples,
                ),
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=_SHARD_RETRY,
            )
            self._shards_completed += 1
            return result

        shard_results = await asyncio.gather(*(run_one(s) for s in shards))

        return await workflow.execute_activity(
            EvalActivities.aggregate_benchmark,
            AggregateBenchmarkInput(
                benchmark_name=benchmark.name,
                model_name=input.model_name,
                metric_names=list(benchmark.metric_names),
                shard_results=list(shard_results),
                bootstrap_resamples=input.bootstrap_resamples,
                bootstrap_seed=input.bootstrap_seed,
            ),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=_AGGREGATE_RETRY,
        )


@workflow.defn
class EvalSuiteWorkflow:
    """Run a whole suite as one child workflow per benchmark."""

    @workflow.run
    async def run(self, input: EvalSuiteInput) -> EvalSuiteResult:
        async def run_benchmark(benchmark) -> BenchmarkSummary:
            return await workflow.execute_child_workflow(
                BenchmarkWorkflow.run,
                BenchmarkInput(
                    suite_id=input.suite_id,
                    model_kind=input.model_kind,
                    model_name=input.model_name,
                    benchmark=benchmark,
                    bootstrap_resamples=input.bootstrap_resamples,
                    bootstrap_seed=input.bootstrap_seed,
                ),
                id=f"{input.suite_id}-{benchmark.name}",
            )

        summaries = await asyncio.gather(
            *(run_benchmark(b) for b in input.benchmarks)
        )
        summaries = list(summaries)
        return EvalSuiteResult(
            suite_id=input.suite_id,
            model_name=input.model_name,
            benchmarks=summaries,
            total_examples=sum(s.n_examples for s in summaries),
            total_errors=sum(s.n_errors for s in summaries),
        )
