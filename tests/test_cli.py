"""Tests for eval_harness.cli (no network, temp files only)."""

from click.testing import CliRunner

from eval_harness.cli import harness
from eval_harness.tracking import RunRegistry

DATASET = (
    '{"id": "a", "input": "Paris", "reference": "Paris"}\n'
    '{"id": "b", "input": "Lyon", "reference": "Paris"}\n'
)


def _dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(DATASET, encoding="utf-8")
    return str(path)


def test_run_command(tmp_path):
    runner = CliRunner()
    report = tmp_path / "report.html"
    db = tmp_path / "runs.db"
    result = runner.invoke(
        harness,
        [
            "run",
            "--dataset", _dataset(tmp_path),
            "--model", "constant",
            "--constant-text", "Paris",
            "--metrics", "exact_match",
            "--workers", "1",
            "--output", str(report),
            "--db", str(db),
            "--name", "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "exact_match" in result.output
    assert report.exists()
    with RunRegistry(db) as registry:
        record = registry.find_run("cli-test")
    assert record.metrics["exact_match"]["mean"] == 1.0


def test_run_command_no_record(tmp_path):
    runner = CliRunner()
    db = tmp_path / "runs.db"
    result = runner.invoke(
        harness,
        ["run", "--dataset", _dataset(tmp_path), "--no-record",
         "--db", str(db), "--output", str(tmp_path / "r.html")],
    )
    assert result.exit_code == 0, result.output
    assert not db.exists()


def test_run_command_bad_metric(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        harness, ["run", "--dataset", _dataset(tmp_path), "--metrics", "nope"]
    )
    assert result.exit_code != 0


def test_run_command_sample(tmp_path):
    runner = CliRunner()
    result = runner.invoke(
        harness,
        ["run", "--dataset", _dataset(tmp_path), "--sample", "1",
         "--output", str(tmp_path / "r.html"), "--no-record"],
    )
    assert result.exit_code == 0, result.output
    assert "n=1" in result.output


def test_compare_command_flags_regression(tmp_path):
    db = tmp_path / "runs.db"
    with RunRegistry(db) as registry:
        registry.record_run(name="base", model="m", dataset="d",
                            aggregate={"exact_match": 0.9})
        registry.record_run(name="cur", model="m", dataset="d",
                            aggregate={"exact_match": 0.5})
    runner = CliRunner()
    result = runner.invoke(
        harness,
        ["compare", "--db", str(db), "--baseline", "base", "--current", "cur"],
    )
    assert result.exit_code == 1  # regressions -> non-zero exit for CI gates
    assert "REGRESSIONS" in result.output
    assert "exact_match" in result.output


def test_compare_command_clean(tmp_path):
    db = tmp_path / "runs.db"
    with RunRegistry(db) as registry:
        registry.record_run(name="base", model="m", dataset="d",
                            aggregate={"exact_match": 0.5})
        registry.record_run(name="cur", model="m", dataset="d",
                            aggregate={"exact_match": 0.9})
    runner = CliRunner()
    result = runner.invoke(
        harness,
        ["compare", "--db", str(db), "--baseline", "base", "--current", "cur"],
    )
    assert result.exit_code == 0, result.output
    assert "No regressions" in result.output
