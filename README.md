# LLM Eval Harness

A lightweight, dependency-free toolkit for benchmarking and evaluating LLM
outputs. Point it at a JSONL dataset and a model, and it runs every example,
scores each output with pluggable metrics, and produces confidence intervals,
regression checks, and a self-contained HTML report.

Built with only the standard library plus `click` — metrics like ROUGE-L and
BLEU are implemented from scratch, so there is nothing heavy to install and
no model API calls are ever required (a deterministic mock judge covers
LLM-as-judge flows in tests and CI).

## Design

```
JSONL dataset ──▶ runner ──▶ per-example predictions ──▶ metrics ──▶ aggregates
                     │                                            │
                     │ ModelFn protocol                           ▼
                     │ (any callable:                            stats (bootstrap CIs,
                     │  real client, wrapper,                     paired tests, win rates)
                     │  or test double)                           │
                     ▼                                            ▼
              tracking (SQLite run registry,        report (self-contained HTML
              baseline diffing, regression gates)    with inline SVG charts)
```

**Components**

| Module | Responsibility |
|---|---|
| `eval_harness/metrics.py` | Pluggable metric registry: `exact_match`, `rouge_l`, `bleu` (from scratch), `json_validity`, `length_stats`, and `llm_as_judge` with a `JudgeBackend` protocol + deterministic `MockJudge` |
| `eval_harness/datasets.py` | JSONL loading with validation, deterministic sampling, train/test splits |
| `eval_harness/runner.py` | Parallel experiment runner over any `ModelFn`; ships mock models (`EchoModel`, `ConstantModel`, `ReverseModel`) |
| `eval_harness/stats.py` | Bootstrap confidence intervals, paired permutation tests, pairwise win rates |
| `eval_harness/tracking.py` | SQLite run registry; compare a run against a baseline and flag regressions past a threshold |
| `eval_harness/report.py` | Self-contained HTML reports — tables plus inline SVG bar charts, zero external assets |
| `eval_harness/cli.py` | `harness run` / `harness compare` command-line interface |

## Quickstart

```bash
pip install -e ".[dev]"

# Evaluate a mock model on the bundled 50-question sample dataset
harness run --dataset examples/qa_sample.jsonl --model echo \
    --metrics exact_match,rouge_l,bleu --output report.html

# Or run the full end-to-end example (two models, report, registry, comparison)
python examples/run_example.py
```

## Dataset format

Each line of the JSONL dataset is one example with three required fields
and one optional field:

```json
{"id": "q1", "input": "What is the capital of France?", "reference": "Paris", "metadata": {"topic": "geography"}}
```

`id`, `input`, and `reference` are required; `metadata` is optional and
passed through untouched. A line missing a required field fails fast with
the file name and line number, rather than scoring silently against an
empty reference.

Example output (`harness run`):

```
model=echo  dataset=qa_sample.jsonl  n=50
  exact_match    mean=0.0400  95% CI=[0.0000, 0.1000]
  rouge_l        mean=0.4123  95% CI=[0.3301, 0.5012]
  bleu           mean=0.1834  95% CI=[0.1207, 0.2544]

report written to report.html
recorded as run #1 in runs.db
```

Compare a candidate run against a baseline (exits non-zero on regression —
drop it straight into CI):

```bash
harness compare --db runs.db --baseline echo-baseline --current candidate
```

## Using it as a library

```python
from eval_harness import (
    EchoModel, MockJudge, get_metric, llm_judge_metric,
    load_jsonl, run_experiment, bootstrap_ci, win_rate,
)

examples = load_jsonl("examples/qa_sample.jsonl")
metrics = [get_metric("rouge_l"), llm_judge_metric(MockJudge())]

result = run_experiment(
    EchoModel(), model_name="echo", dataset=examples,
    dataset_name="qa_sample", metrics=metrics, num_workers=4,
)
print(result.aggregate())

# Is the candidate really better than the baseline?
baseline = run_experiment(...)
p = paired_permutation_test(
    baseline.scores_for("rouge_l"), result.scores_for("rouge_l")
)
print("p-value:", p, "| win rate:", win_rate(baseline.scores_for("rouge_l"),
                                            result.scores_for("rouge_l")))
```

## Custom metrics

```python
from eval_harness import MetricResult, register_metric

class KeywordRecall:
    name = "keyword_recall"
    def __call__(self, prediction, reference):
        keywords = {"paris", "france"}
        hits = sum(k in prediction.lower() for k in keywords)
        return MetricResult(name=self.name, score=hits / len(keywords))

register_metric("keyword_recall", KeywordRecall())
```

Then use `--metrics keyword_recall` or `get_metric("keyword_recall")`.

## Development

```bash
python -m pytest          # full suite
python examples/run_example.py   # end-to-end smoke test
```
