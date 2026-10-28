"""Tests for eval_harness.tracking."""

import pytest

from eval_harness.tracking import RunRegistry


@pytest.fixture
def registry(tmp_path):
    with RunRegistry(tmp_path / "runs.db") as reg:
        yield reg


def test_record_and_get(registry):
    run_id = registry.record_run(
        name="baseline",
        model="echo",
        dataset="qa.jsonl",
        aggregate={"exact_match": 0.8, "rouge_l": 0.9},
        ci={"exact_match": (0.7, 0.9)},
        config={"seed": 42},
    )
    record = registry.get_run(run_id)
    assert record.name == "baseline"
    assert record.model == "echo"
    assert record.metrics["exact_match"]["mean"] == pytest.approx(0.8)
    assert record.metrics["exact_match"]["ci_low"] == pytest.approx(0.7)
    assert record.metrics["rouge_l"]["ci_low"] is None
    assert record.config == {"seed": 42}
    assert record.created_at  # ISO timestamp present


def test_get_missing_raises(registry):
    with pytest.raises(KeyError):
        registry.get_run(999)


def test_list_and_find(registry):
    registry.record_run(name="a", model="m", dataset="d", aggregate={"x": 1.0})
    registry.record_run(name="b", model="m", dataset="d", aggregate={"x": 2.0})
    registry.record_run(name="a", model="m", dataset="d", aggregate={"x": 3.0})
    runs = registry.list_runs()
    assert [r.name for r in runs] == ["a", "b", "a"]  # most recent first
    assert registry.find_run("a").metrics["x"]["mean"] == pytest.approx(3.0)
    with pytest.raises(KeyError):
        registry.find_run("nope")


def _two_runs(registry, base_scores, cur_scores):
    base_id = registry.record_run(
        name="base", model="m", dataset="d", aggregate=base_scores
    )
    cur_id = registry.record_run(
        name="cur", model="m", dataset="d", aggregate=cur_scores
    )
    return base_id, cur_id


def test_compare_detects_regression(registry):
    base_id, cur_id = _two_runs(
        registry, {"exact_match": 0.80, "rouge_l": 0.90}, {"exact_match": 0.70, "rouge_l": 0.90}
    )
    regressions = registry.compare(cur_id, base_id, threshold=0.02)
    assert len(regressions) == 1
    reg = regressions[0]
    assert reg.metric == "exact_match"
    assert reg.drop == pytest.approx(0.10)
    assert reg.drop_pct == pytest.approx(12.5)


def test_compare_no_regression_when_improved(registry):
    base_id, cur_id = _two_runs(registry, {"exact_match": 0.70}, {"exact_match": 0.85})
    assert registry.compare(cur_id, base_id, threshold=0.02) == []


def test_compare_below_threshold(registry):
    base_id, cur_id = _two_runs(registry, {"exact_match": 0.80}, {"exact_match": 0.795})
    assert registry.compare(cur_id, base_id, threshold=0.02) == []


def test_compare_relative_mode(registry):
    base_id, cur_id = _two_runs(registry, {"exact_match": 0.50}, {"exact_match": 0.44})
    # absolute drop 0.06 > 0.02 would flag in absolute mode too; use a case
    # where only the relative reading matters:
    base2, cur2 = _two_runs(registry, {"bleu": 0.40}, {"bleu": 0.35})
    absolute = registry.compare(cur2, base2, threshold=0.10)
    relative = registry.compare(cur2, base2, threshold=0.10, relative=True)
    assert absolute == []  # 0.05 < 0.10 absolute
    assert len(relative) == 1  # 0.05/0.40 = 12.5% > 10%
    assert relative[0].relative is True


def test_compare_ignores_unshared_metrics(registry):
    base_id, cur_id = _two_runs(registry, {"a": 0.9, "b": 0.9}, {"a": 0.1})
    regressions = registry.compare(cur_id, base_id, threshold=0.02)
    assert [r.metric for r in regressions] == ["a"]
