"""End-to-end example: evaluate two mock models on the sample QA dataset.

Runs:
    python examples/run_example.py

Outputs:
    - prints a mean-score table with bootstrap confidence intervals
    - writes examples/report.html (self-contained, open in a browser)
    - records both runs in examples/runs.db and compares candidate vs baseline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_harness import (  # noqa: E402
    ConstantModel,
    EchoModel,
    bootstrap_ci,
    generate_html_report,
    get_metric,
    load_jsonl,
    run_experiment,
)
from eval_harness.tracking import RunRegistry  # noqa: E402

HERE = Path(__file__).resolve().parent
DATASET = HERE / "qa_sample.jsonl"
METRIC_NAMES = ["exact_match", "rouge_l", "bleu", "length_stats"]


def main() -> None:
    examples = load_jsonl(DATASET)
    metrics = [get_metric(name) for name in METRIC_NAMES]

    baseline = run_experiment(
        EchoModel(),
        model_name="echo-baseline",
        dataset=examples,
        dataset_name=DATASET.name,
        metrics=metrics,
        num_workers=4,
    )
    candidate = run_experiment(
        ConstantModel("I don't know."),
        model_name="constant-candidate",
        dataset=examples,
        dataset_name=DATASET.name,
        metrics=metrics,
        num_workers=4,
    )

    print(f"\n{'metric':<14}{'echo-baseline':>22}{'constant-candidate':>22}")
    print("-" * 58)
    for name in METRIC_NAMES:
        b_scores, c_scores = baseline.scores_for(name), candidate.scores_for(name)
        b_lo, b_hi = bootstrap_ci(b_scores)
        c_lo, c_hi = bootstrap_ci(c_scores)
        print(
            f"{name:<14}{baseline.aggregate()[name]:>8.4f} [{b_lo:.3f},{b_hi:.3f}]"
            f"{candidate.aggregate()[name]:>8.4f} [{c_lo:.3f},{c_hi:.3f}]"
        )

    report_path = HERE / "report.html"
    generate_html_report(
        [("echo-baseline", baseline), ("constant-candidate", candidate)],
        report_path,
        title="LLM Eval Harness — Example Report",
    )
    print(f"\nHTML report written to {report_path}")

    db_path = HERE / "runs.db"
    with RunRegistry(db_path) as registry:
        base_id = registry.record_run(
            name="echo-baseline",
            model="echo",
            dataset=DATASET.name,
            aggregate=baseline.aggregate(),
            ci={m: bootstrap_ci(baseline.scores_for(m)) for m in METRIC_NAMES},
        )
        cand_id = registry.record_run(
            name="constant-candidate",
            model="constant",
            dataset=DATASET.name,
            aggregate=candidate.aggregate(),
            ci={m: bootstrap_ci(candidate.scores_for(m)) for m in METRIC_NAMES},
        )
        regressions = registry.compare(cand_id, base_id, threshold=0.02)
    print(f"Recorded runs #{base_id} (baseline) and #{cand_id} (candidate) in {db_path}")
    if regressions:
        print("Regressions detected vs baseline:")
        for reg in regressions:
            print(
                f"  {reg.metric}: {reg.baseline_mean:.4f} -> {reg.current_mean:.4f} "
                f"(drop {reg.drop_pct:.1f}%)"
            )
    else:
        print("No regressions vs baseline.")


if __name__ == "__main__":
    main()
