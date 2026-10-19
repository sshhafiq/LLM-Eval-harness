"""LLM eval harness: benchmark and evaluate LLM outputs."""

from .datasets import DatasetError, Example, load_jsonl, sample, train_test_split
from .metrics import (
    Judgment,
    JudgeBackend,
    Metric,
    MetricResult,
    MockJudge,
    bleu,
    exact_match,
    get_metric,
    json_validity,
    length_stats,
    list_metrics,
    llm_judge_metric,
    register_metric,
    rouge_l,
)
from .report import generate_html_report
from .runner import (
    ConstantModel,
    EchoModel,
    ExampleResult,
    ExperimentResult,
    FailingModel,
    ModelFn,
    ReverseModel,
    run_experiment,
)
from .stats import bootstrap_ci, paired_permutation_test, summarize, win_rate
from .tracking import Regression, RunRecord, RunRegistry

__version__ = "0.1.0"

__all__ = [
    "DatasetError",
    "Example",
    "load_jsonl",
    "sample",
    "train_test_split",
    "Judgment",
    "JudgeBackend",
    "Metric",
    "MetricResult",
    "MockJudge",
    "bleu",
    "exact_match",
    "get_metric",
    "json_validity",
    "length_stats",
    "list_metrics",
    "llm_judge_metric",
    "register_metric",
    "rouge_l",
    "generate_html_report",
    "ConstantModel",
    "EchoModel",
    "ExampleResult",
    "ExperimentResult",
    "FailingModel",
    "ModelFn",
    "ReverseModel",
    "run_experiment",
    "bootstrap_ci",
    "paired_permutation_test",
    "summarize",
    "win_rate",
    "Regression",
    "RunRecord",
    "RunRegistry",
    "__version__",
]
