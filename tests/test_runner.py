"""Tests for eval_harness.runner."""

import pytest

from eval_harness.datasets import Example
from eval_harness.metrics import get_metric
from eval_harness.runner import (
    ConstantModel,
    EchoModel,
    FailingModel,
    ReverseModel,
    run_experiment,
)


def _examples():
    return [
        Example(id="a", input="hello", reference="hello"),
        Example(id="b", input="world", reference="world"),
        Example(id="c", input="foo", reference="bar"),
    ]


def test_constant_model_exact_match():
    result = run_experiment(
        ConstantModel("hello"),
        model_name="const",
        dataset=_examples(),
        dataset_name="test",
        metrics=[get_metric("exact_match")],
        num_workers=2,
    )
    assert result.n_examples == 3
    # "hello" matches reference only for example "a"
    assert result.aggregate()["exact_match"] == pytest.approx(1 / 3)
    assert [r.example_id for r in result.example_results] == ["a", "b", "c"]


def test_echo_model_perfect_on_matching_dataset():
    examples = [
        Example(id="a", input="Paris", reference="Paris"),
        Example(id="b", input="42", reference="42"),
    ]
    result = run_experiment(
        EchoModel(),
        model_name="echo",
        dataset=examples,
        dataset_name="test",
        metrics=[get_metric("exact_match"), get_metric("rouge_l")],
        num_workers=1,
    )
    assert result.aggregate()["exact_match"] == pytest.approx(1.0)
    assert result.aggregate()["rouge_l"] == pytest.approx(1.0)
    assert result.n_errors == 0


def test_reverse_model():
    result = run_experiment(
        ReverseModel(),
        model_name="rev",
        dataset=[Example(id="a", input="abc", reference="cba")],
        dataset_name="test",
        metrics=[get_metric("exact_match")],
    )
    assert result.aggregate()["exact_match"] == pytest.approx(1.0)


def test_failing_model_records_errors():
    result = run_experiment(
        FailingModel("kaput"),
        model_name="fail",
        dataset=_examples(),
        dataset_name="test",
        metrics=[get_metric("exact_match")],
    )
    assert result.n_errors == 3
    assert result.aggregate()["exact_match"] == pytest.approx(0.0)
    assert all(r.error and "kaput" in r.error for r in result.example_results)


def test_scores_for_and_latency():
    result = run_experiment(
        EchoModel(),
        model_name="echo",
        dataset=_examples(),
        dataset_name="test",
        metrics=[get_metric("exact_match")],
    )
    scores = result.scores_for("exact_match")
    assert scores == pytest.approx([1.0, 1.0, 0.0])
    assert result.mean_latency_s >= 0.0


def test_bad_worker_count():
    with pytest.raises(ValueError):
        run_experiment(
            EchoModel(),
            model_name="echo",
            dataset=_examples(),
            dataset_name="test",
            metrics=[get_metric("exact_match")],
            num_workers=0,
        )


def test_parallel_matches_serial():
    metrics = [get_metric("exact_match"), get_metric("bleu")]
    serial = run_experiment(
        EchoModel(), model_name="e", dataset=_examples(), dataset_name="t",
        metrics=metrics, num_workers=1,
    )
    parallel = run_experiment(
        EchoModel(), model_name="e", dataset=_examples(), dataset_name="t",
        metrics=metrics, num_workers=8,
    )
    assert serial.aggregate() == pytest.approx(parallel.aggregate())
