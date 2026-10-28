"""Tests for eval_harness.report."""

import pytest

from eval_harness.datasets import Example
from eval_harness.metrics import get_metric
from eval_harness.report import generate_html_report, svg_bar_chart
from eval_harness.runner import ConstantModel, run_experiment


def _result(name, response):
    examples = [
        Example(id="a", input="Paris", reference="Paris"),
        Example(id="b", input="Lyon", reference="Paris"),
    ]
    return run_experiment(
        ConstantModel(response),
        model_name=name,
        dataset=examples,
        dataset_name="tiny",
        metrics=[get_metric("exact_match"), get_metric("rouge_l")],
        num_workers=1,
    )


def test_svg_bar_chart_structure():
    svg = svg_bar_chart("Title", ["a", "b"], [0.5, 1.0])
    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "Title" in svg
    assert svg.count("<rect") == 2


def test_svg_clamps_values():
    svg = svg_bar_chart("t", ["a"], [2.0])
    assert "<rect" in svg  # does not raise; value clamped to 1.0


def test_report_writes_self_contained_html(tmp_path):
    out = tmp_path / "report.html"
    generate_html_report(
        [("paris-bot", _result("paris-bot", "Paris"))],
        out,
        title="Test Report",
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in text
    assert "Test Report" in text
    assert "paris-bot" in text
    assert "exact_match" in text
    assert "<svg" in text
    assert "<table>" in text
    # self-contained: no external references
    assert "http://" not in text and "https://" not in text


def test_report_multiple_models(tmp_path):
    out = tmp_path / "report.html"
    generate_html_report(
        [("m1", _result("m1", "Paris")), ("m2", _result("m2", "Lyon"))],
        out,
    )
    text = out.read_text(encoding="utf-8")
    assert "m1" in text and "m2" in text
    # m1 scores 0.5 exact_match vs m2's 0.0 -> best cell is bolded
    assert 'class="best"' in text


def test_report_empty_raises(tmp_path):
    with pytest.raises(ValueError):
        generate_html_report([], tmp_path / "r.html")
