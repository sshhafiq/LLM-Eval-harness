"""Workflow tests against Temporal's time-skipping test environment.

Skipped automatically if the test-server binary cannot be downloaded
(offline sandbox); the activity tests cover shard + aggregation logic.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from eval_harness.temporal.activities import EvalActivities
from eval_harness.temporal.models import (
    BenchmarkSpec,
    EvalSuiteInput,
    ExampleSpec,
)
from eval_harness.temporal.workflows import BenchmarkWorkflow, EvalSuiteWorkflow

pytestmark = pytest.mark.asyncio

TASK_QUEUE = "test-eval-harness"


async def _start_env() -> WorkflowEnvironment:
    try:
        return await asyncio.wait_for(
            WorkflowEnvironment.start_time_skipping(), timeout=90
        )
    except Exception as exc:
        pytest.skip(f"Temporal test server unavailable: {exc}")


def _worker(env: WorkflowEnvironment) -> Worker:
    return Worker(
        env.client,
        task_queue=TASK_QUEUE,
        workflows=[EvalSuiteWorkflow, BenchmarkWorkflow],
        activities=EvalActivities().all(),
    )


def _examples(prefix: str, n: int) -> list[ExampleSpec]:
    return [
        ExampleSpec(id=f"{prefix}-{i}", input=f"text {i}",
                    reference=f"text {i}")
        for i in range(n)
    ]


async def test_suite_runs_child_per_benchmark_and_aggregates():
    env = await _start_env()
    async with env:
        async with _worker(env):
            suite_id = f"t-{uuid.uuid4().hex[:8]}"
            result = await env.client.execute_workflow(
                EvalSuiteWorkflow.run,
                EvalSuiteInput(
                    suite_id=suite_id,
                    model_kind="echo",
                    model_name="echo",
                    benchmarks=[
                        BenchmarkSpec(
                            name="alpha",
                            examples=_examples("a", 10),
                            metric_names=["exact_match", "token_f1"],
                            shard_size=4,
                        ),
                        BenchmarkSpec(
                            name="beta",
                            examples=_examples("b", 5),
                            metric_names=["exact_match"],
                            shard_size=2,
                        ),
                    ],
                    bootstrap_resamples=200,
                ),
                id=suite_id,
                task_queue=TASK_QUEUE,
            )
            assert result.total_examples == 15
            assert result.total_errors == 0
            by_name = {b.name: b for b in result.benchmarks}
            assert by_name["alpha"].n_examples == 10
            assert by_name["alpha"].means["exact_match"] == 1.0
            assert by_name["beta"].n_examples == 5
            assert by_name["beta"].primary_ci95  # CI computed for n >= 2


async def test_benchmark_workflow_alone():
    env = await _start_env()
    async with env:
        async with _worker(env):
            from eval_harness.temporal.models import BenchmarkInput

            wf_id = f"t-{uuid.uuid4().hex[:8]}"
            summary = await env.client.execute_workflow(
                BenchmarkWorkflow.run,
                BenchmarkInput(
                    suite_id="solo",
                    model_kind="constant",
                    model_name="constant",
                    benchmark=BenchmarkSpec(
                        name="solo-bench",
                        examples=_examples("s", 6),
                        metric_names=["exact_match"],
                        shard_size=3,
                    ),
                    bootstrap_resamples=200,
                ),
                id=wf_id,
                task_queue=TASK_QUEUE,
            )
            assert summary.n_examples == 6
            assert summary.means["exact_match"] == 0.0
