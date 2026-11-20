"""Serializable data model for the eval-suite workflows.

Plain dataclasses with JSON-friendly types. Models and metrics are
referenced *by name* (``model_kind`` / ``metric_names``) and resolved to
callables inside activities, where real code runs; workflows only ever
move data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExampleSpec:
    """One dataset example, mirroring :class:`eval_harness.datasets.Example`."""

    id: str
    input: str
    reference: str


@dataclass
class BenchmarkSpec:
    """A named benchmark: a dataset slice plus the metrics to score with."""

    name: str
    examples: list[ExampleSpec]
    metric_names: list[str] = field(
        default_factory=lambda: ["exact_match", "token_f1"]
    )
    #: Examples per shard activity.
    shard_size: int = 16


@dataclass
class EvalSuiteInput:
    """Input to :class:`EvalSuiteWorkflow`."""

    suite_id: str
    #: Built-in model kind resolved in activities: "constant" | "echo" |
    #: "reverse" (see :mod:`eval_harness.runner`). Real deployments map
    #: their own model names here via the activity's model factory.
    model_kind: str = "echo"
    model_name: str = "echo"
    benchmarks: list[BenchmarkSpec] = field(default_factory=list)
    #: Bootstrap settings for the primary metric's confidence interval.
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42


@dataclass
class BenchmarkInput:
    """Input to the :class:`BenchmarkWorkflow` child workflow."""

    suite_id: str
    model_kind: str
    model_name: str
    benchmark: BenchmarkSpec
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42


@dataclass
class RunShardInput:
    benchmark_name: str
    model_kind: str
    model_name: str
    metric_names: list[str]
    examples: list[ExampleSpec]


@dataclass
class ExampleScore:
    """One example's outcome inside a shard."""

    example_id: str
    prediction: str
    scores: dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    latency_s: float = 0.0


@dataclass
class ShardResult:
    benchmark_name: str
    examples: list[ExampleScore] = field(default_factory=list)


@dataclass
class AggregateBenchmarkInput:
    benchmark_name: str
    model_name: str
    metric_names: list[str]
    shard_results: list[ShardResult]
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 42


@dataclass
class BenchmarkSummary:
    """Aggregated outcome for one benchmark."""

    name: str
    model_name: str
    n_examples: int
    n_errors: int
    #: Mean score per metric (``ExperimentResult.aggregate`` semantics).
    means: dict[str, float] = field(default_factory=dict)
    #: ``stats.summarize`` output for the primary (first) metric.
    primary_metric: str = ""
    primary_summary: dict[str, float] = field(default_factory=dict)
    #: Bootstrap CI for the primary metric's mean: [low, high].
    primary_ci95: list[float] = field(default_factory=list)
    mean_latency_s: float = 0.0


@dataclass
class EvalSuiteResult:
    suite_id: str
    model_name: str
    benchmarks: list[BenchmarkSummary] = field(default_factory=list)
    total_examples: int = 0
    total_errors: int = 0
