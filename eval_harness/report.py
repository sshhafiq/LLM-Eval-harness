"""Self-contained HTML report generation.

Reports embed all CSS and SVG inline, so the output file can be opened in any
browser with no network access and no external assets.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

from .runner import ExperimentResult

_CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 2rem auto; max-width: 960px; padding: 0 1rem; color: #1f2328; }
h1 { border-bottom: 1px solid #d0d7de; padding-bottom: 0.4rem; }
h2 { margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { border: 1px solid #d0d7de; padding: 0.45rem 0.7rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f6f8fa; }
tr.best td { font-weight: 600; }
.meta { color: #57606a; font-size: 0.9rem; }
svg.chart { display: block; margin: 1rem 0; }
"""


def svg_bar_chart(
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    *,
    width: int = 640,
    bar_height: int = 26,
    pad: int = 8,
) -> str:
    """Render a simple horizontal SVG bar chart (values expected in [0, 1])."""
    labels = [str(l) for l in labels]
    values = [max(0.0, min(1.0, float(v))) for v in values]
    label_w = max([len(l) for l in labels] + [10]) * 7 + 8
    chart_w = width - label_w - 60
    height = pad * 2 + len(labels) * (bar_height + pad)
    parts = [
        f'<svg class="chart" width="{width}" height="{height}" role="img"'
        f' aria-label="{html.escape(title)}">',
        f"<title>{html.escape(title)}</title>",
    ]
    y = pad
    for label, value in zip(labels, values):
        bar_w = max(1, int(value * chart_w))
        hue = 150 if value >= 0.7 else (40 if value >= 0.4 else 0)
        parts.append(
            f'<text x="0" y="{y + bar_height - 8}" font-size="12">'
            f"{html.escape(label)}</text>"
        )
        parts.append(
            f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{bar_height}"'
            f' fill="hsl({hue}, 60%, 45%)" rx="3"/>'
        )
        parts.append(
            f'<text x="{label_w + bar_w + 6}" y="{y + bar_height - 8}"'
            f' font-size="12">{value:.3f}</text>'
        )
        y += bar_height + pad
    parts.append("</svg>")
    return "\n".join(parts)


def _results_table(
    named_results: Sequence[Tuple[str, ExperimentResult]],
    metric_names: Sequence[str],
) -> str:
    """HTML table of mean scores; the best model per metric is bolded."""
    best: Dict[str, float] = {}
    for _, result in named_results:
        agg = result.aggregate()
        for m in metric_names:
            if m in agg:
                best[m] = max(best.get(m, float("-inf")), agg[m])
    rows = ["<table>", "<tr><th>Model</th><th>Examples</th>"]
    rows += [f"<th>{html.escape(m)}</th>" for m in metric_names]
    rows.append("</tr>")
    for name, result in named_results:
        agg = result.aggregate()
        cells = [f"<td>{html.escape(name)}</td>", f"<td>{result.n_examples}</td>"]
        for m in metric_names:
            score = agg.get(m)
            text = f"{score:.4f}" if score is not None else "—"
            cls = (
                ' class="best"'
                if score is not None and score == best.get(m)
                else ""
            )
            cells.append(f"<td{cls}>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def generate_html_report(
    named_results: Sequence[Tuple[str, ExperimentResult]],
    output_path: Union[str, Path],
    title: str = "LLM Eval Report",
) -> Path:
    """Write a self-contained HTML report and return its path.

    Includes a mean-score table and one SVG bar chart per metric.  Raises
    ``ValueError`` when ``named_results`` is empty.
    """
    named_results = list(named_results)
    if not named_results:
        raise ValueError("named_results must be non-empty")
    metric_names: List[str] = []
    for _, result in named_results:
        for m in result.aggregate():
            if m not in metric_names:
                metric_names.append(m)

    sections = [f"<h1>{html.escape(title)}</h1>"]
    for _, result in named_results:
        sections.append(
            '<p class="meta">'
            f"Model <b>{html.escape(result.model_name)}</b> · "
            f"dataset <b>{html.escape(result.dataset_name)}</b> · "
            f"{result.n_examples} examples · "
            f"{result.n_errors} errors · "
            f"mean latency {result.mean_latency_s * 1000:.1f} ms/example</p>"
        )
    sections.append("<h2>Mean scores</h2>")
    sections.append(_results_table(named_results, metric_names))
    for metric in metric_names:
        labels = [name for name, _ in named_results]
        values = [res.aggregate().get(metric, 0.0) for _, res in named_results]
        sections.append(f"<h2>{html.escape(metric)}</h2>")
        sections.append(svg_bar_chart(f"Mean {metric}", labels, values))

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
{chr(10).join(sections)}
</body>
</html>
"""
    output_path = Path(output_path)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
