"""Worker + demo client for the eval-suite workflows.

Run a local Temporal server first (from the repo root)::

    docker compose up -d          # temporalio/auto-setup on :7233

then::

    python -m eval_harness.temporal.worker worker
    python -m eval_harness.temporal.worker demo

The demo builds a small suite from inline examples (an echo-model
sanity benchmark and a constant-model baseline), runs it as one parent
workflow with a child per benchmark, and prints per-benchmark means and
bootstrap CIs — the full shard -> fan-out -> aggregate path.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from temporalio.client import Client
from temporalio.worker import Worker

from eval_harness.temporal.activities import EvalActivities
from eval_harness.temporal.models import (
    BenchmarkSpec,
    EvalSuiteInput,
    ExampleSpec,
)
from eval_harness.temporal.workflows import BenchmarkWorkflow, EvalSuiteWorkflow

TASK_QUEUE = "llm-eval-harness-suites"


def build_worker(client: Client, task_queue: str = TASK_QUEUE) -> Worker:
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[EvalSuiteWorkflow, BenchmarkWorkflow],
        activities=EvalActivities().all(),
    )


async def _run_worker(address: str, namespace: str) -> None:
    client = await Client.connect(address, namespace=namespace)
    worker = build_worker(client)
    print(f"eval-harness worker listening on task queue '{TASK_QUEUE}'")
    await worker.run()


def _demo_suite(suite_id: str) -> EvalSuiteInput:
    qa_examples = [
        ExampleSpec(
            id=f"qa-{i}",
            input=f"What is {i} + {i}?",
            reference=f"What is {i} + {i}?",
        )
        for i in range(10)
    ]
    return EvalSuiteInput(
        suite_id=suite_id,
        model_kind="echo",
        model_name="echo-demo",
        benchmarks=[
            BenchmarkSpec(
                name="echo-sanity",
                examples=qa_examples,
                metric_names=["exact_match", "token_f1", "contains_match"],
                shard_size=4,
            ),
            BenchmarkSpec(
                name="echo-baseline-check",
                examples=qa_examples[:6],
                metric_names=["exact_match", "rouge_l"],
                shard_size=3,
            ),
        ],
    )


async def _run_demo(args: argparse.Namespace) -> None:
    client = await Client.connect(args.address, namespace=args.namespace)
    worker = build_worker(client)
    async with worker:
        suite_id = f"demo-{uuid.uuid4().hex[:8]}"
        result = await client.execute_workflow(
            EvalSuiteWorkflow.run,
            _demo_suite(suite_id),
            id=suite_id,
            task_queue=TASK_QUEUE,
        )
        print(f"suite {result.suite_id} ({result.model_name}): "
              f"{result.total_examples} examples, "
              f"{result.total_errors} errors")
        for bench in result.benchmarks:
            means = ", ".join(
                f"{k}={v:.3f}" for k, v in sorted(bench.means.items())
            )
            ci = bench.primary_ci95
            ci_text = f" CI95=[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
            print(f"  {bench.name}: {means}{ci_text}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval-harness-temporal")
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--namespace", default="default")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("worker", help="run the worker in the foreground")
    sub.add_parser("demo", help="run a worker + one demo suite")

    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(_run_worker(args.address, args.namespace))
    else:
        asyncio.run(_run_demo(args))


if __name__ == "__main__":
    main()
