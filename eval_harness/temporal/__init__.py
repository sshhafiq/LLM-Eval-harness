"""Temporal integration: evaluation suites as durable workflows.

A suite run in-process dies with the process. This subpackage runs a
suite on Temporal instead:

- :class:`~eval_harness.temporal.workflows.EvalSuiteWorkflow` starts one
  child workflow per benchmark, so each benchmark is independently
  observable, retryable, and cancellable in the Temporal UI.
- :class:`~eval_harness.temporal.workflows.BenchmarkWorkflow` shards a
  benchmark's examples and fans shard activities out in parallel.
- :class:`~eval_harness.temporal.activities.EvalActivities` executes a
  shard with the harness's own :func:`run_experiment` and scorers, and
  aggregates shard results with the :mod:`eval_harness.stats` module
  (bootstrap CIs included).

Requires the ``temporal`` extra: ``pip install llm-eval-harness[temporal]``.
"""

from eval_harness.temporal.models import (
    BenchmarkSpec,
    BenchmarkSummary,
    EvalSuiteInput,
    EvalSuiteResult,
    ExampleSpec,
)

__all__ = [
    "BenchmarkSpec",
    "BenchmarkSummary",
    "EvalSuiteInput",
    "EvalSuiteResult",
    "ExampleSpec",
]
