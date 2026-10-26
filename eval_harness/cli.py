"""Command-line interface.

Examples:
    harness run --dataset examples/qa_sample.jsonl --model echo \\
        --metrics exact_match,rouge_l --output report.html
    harness run --dataset data.jsonl --model constant \\
        --constant-text "I don't know." --metrics bleu --sample 20
    harness compare --db runs.db --baseline baseline --current candidate
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import click

from .datasets import Example, load_jsonl, sample
from .metrics import Metric, get_metric, list_metrics
from .report import generate_html_report
from .runner import (
    ConstantModel,
    EchoModel,
    ExperimentResult,
    ModelFn,
    ReverseModel,
    run_experiment,
)
from .stats import bootstrap_ci
from .tracking import RunRegistry


def _build_model(name: str, constant_text: str) -> ModelFn:
    if name == "echo":
        return EchoModel()
    if name == "reverse":
        return ReverseModel()
    if name == "constant":
        return ConstantModel(constant_text)
    raise click.BadParameter(f"unknown model {name!r}")


MODEL_CHOICES = ["echo", "reverse", "constant"]


@click.group()
def harness() -> None:
    """LLM eval harness: benchmark model outputs with pluggable metrics."""


@harness.command("run")
@click.option("--dataset", "dataset_path", required=True, type=click.Path(exists=True),
              help="JSONL dataset file.")
@click.option("--model", "model_name", default="echo", show_default=True,
              type=click.Choice(MODEL_CHOICES), help="Mock model to evaluate.")
@click.option("--constant-text", default="I don't know.", show_default=True,
              help="Response text for --model constant.")
@click.option("--metrics", default="exact_match,rouge_l", show_default=True,
              help=f"Comma-separated metric names. Available: {', '.join(list_metrics())}")
@click.option("--workers", default=4, show_default=True, type=int,
              help="Parallel evaluation workers.")
@click.option("--sample", "sample_n", type=int, default=None,
              help="Evaluate on a random sample of N examples.")
@click.option("--seed", default=42, show_default=True, type=int, help="Random seed.")
@click.option("--name", default=None,
              help="Run name for the registry (defaults to the model name).")
@click.option("--output", "report_path", default="report.html", show_default=True,
              help="Where to write the HTML report.")
@click.option("--db", "db_path", default="runs.db", show_default=True,
              help="SQLite registry file for run tracking.")
@click.option("--no-record", is_flag=True, default=False,
              help="Skip recording the run in the registry.")
def run_command(
    dataset_path: str,
    model_name: str,
    constant_text: str,
    metrics: str,
    workers: int,
    sample_n: Optional[int],
    seed: int,
    name: Optional[str],
    report_path: str,
    db_path: str,
    no_record: bool,
) -> None:
    """Run one model over a dataset and score it with the given metrics."""
    metric_names = [m.strip() for m in metrics.split(",") if m.strip()]
    try:
        metric_fns: List[Metric] = [get_metric(m) for m in metric_names]
    except KeyError as exc:
        raise click.BadParameter(str(exc)) from exc

    examples: List[Example] = load_jsonl(dataset_path)
    if sample_n is not None:
        examples = sample(examples, sample_n, seed=seed)

    model_fn = _build_model(model_name, constant_text)
    result = run_experiment(
        model_fn,
        model_name=model_name,
        dataset=examples,
        dataset_name=Path(dataset_path).name,
        metrics=metric_fns,
        num_workers=workers,
    )

    aggregate = result.aggregate()
    click.echo(f"\nmodel={model_name}  dataset={Path(dataset_path).name}  n={result.n_examples}")
    for metric_name in metric_names:
        scores = result.scores_for(metric_name)
        low, high = bootstrap_ci(scores, seed=seed)
        click.echo(f"  {metric_name:<14} mean={aggregate[metric_name]:.4f}  95% CI=[{low:.4f}, {high:.4f}]")
    if result.n_errors:
        click.echo(f"  {result.n_errors} example(s) errored", err=True)

    generate_html_report([(model_name, result)], report_path)
    click.echo(f"\nreport written to {report_path}")

    if not no_record:
        ci = {m: bootstrap_ci(result.scores_for(m), seed=seed) for m in metric_names}
        with RunRegistry(db_path) as registry:
            run_id = registry.record_run(
                name=name or model_name,
                model=model_name,
                dataset=Path(dataset_path).name,
                aggregate=aggregate,
                ci=ci,
                config={"metrics": metric_names, "workers": workers, "seed": seed},
            )
        click.echo(f"recorded as run #{run_id} in {db_path}")


@harness.command("compare")
@click.option("--db", "db_path", default="runs.db", show_default=True,
              help="SQLite registry file.")
@click.option("--baseline", required=True, help="Baseline run name.")
@click.option("--current", required=True, help="Current run name.")
@click.option("--threshold", default=0.02, show_default=True, type=float,
              help="Absolute mean-score drop that counts as a regression.")
@click.option("--relative", is_flag=True, default=False,
              help="Measure the drop relative to the baseline mean.")
def compare_command(
    db_path: str, baseline: str, current: str, threshold: float, relative: bool
) -> None:
    """Compare two recorded runs and flag metric regressions."""
    with RunRegistry(db_path) as registry:
        try:
            baseline_rec = registry.find_run(baseline)
            current_rec = registry.find_run(current)
        except KeyError as exc:
            raise click.BadParameter(str(exc)) from exc
        regressions = registry.compare(
            current_rec.id, baseline_rec.id, threshold=threshold, relative=relative
        )
    if not regressions:
        click.echo(f"No regressions: {current!r} vs baseline {baseline!r} "
                   f"(threshold={threshold}).")
        return
    click.echo(f"REGRESSIONS in {current!r} vs baseline {baseline!r}:")
    for reg in regressions:
        click.echo(
            f"  {reg.metric}: {reg.baseline_mean:.4f} -> {reg.current_mean:.4f} "
            f"(drop {reg.drop:.4f}, {reg.drop_pct:.1f}%)"
        )
    sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    harness()
