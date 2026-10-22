"""Pluggable evaluation metrics.

Each metric is a callable ``(prediction, reference) -> MetricResult`` with a
``name`` attribute.  Built-in metrics are registered in :data:`METRICS` and can
be retrieved with :func:`get_metric`; third-party code can add new ones with
:func:`register_metric`.

All metrics are implemented from scratch on the standard library so the
package has no heavy NLP dependencies.
"""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Protocol


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class MetricResult:
    """Outcome of applying one metric to one (prediction, reference) pair."""

    name: str
    score: float
    details: Dict[str, object] = field(default_factory=dict)


class Metric(Protocol):
    """Structural type for anything usable as a metric."""

    name: str

    def __call__(self, prediction: str, reference: str) -> MetricResult:
        ...


class _FnMetric:
    """Adapter that turns a plain function into a named Metric."""

    def __init__(self, name: str, fn: Callable[[str, str], MetricResult]) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, prediction: str, reference: str) -> MetricResult:
        return self._fn(prediction, reference)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<Metric {self.name}>"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_METRICS: Dict[str, Metric] = {}


def register_metric(name: str, metric: Metric) -> None:
    """Register ``metric`` under ``name``, replacing any existing entry."""
    _METRICS[name] = metric


def get_metric(name: str) -> Metric:
    """Return the metric registered as ``name``.

    Raises:
        KeyError: if no metric is registered under ``name``.  The message
            lists the available metrics to help with typos.
    """
    try:
        return _METRICS[name]
    except KeyError:
        available = ", ".join(sorted(_METRICS)) or "(none)"
        raise KeyError(f"unknown metric {name!r}; available: {available}") from None


def list_metrics() -> list[str]:
    """Return the sorted names of all registered metrics."""
    return sorted(_METRICS)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def tokenize(text: str) -> list[str]:
    """Lowercase whitespace tokenization used by the n-gram metrics."""
    return text.lower().split()


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of two token lists.

    Uses O(min(len(a), len(b))) space.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ---------------------------------------------------------------------------
# Built-in metrics
# ---------------------------------------------------------------------------


def exact_match(prediction: str, reference: str) -> MetricResult:
    """1.0 when prediction matches reference after normalization, else 0.0."""
    score = 1.0 if normalize(prediction) == normalize(reference) else 0.0
    return MetricResult(name="exact_match", score=score)


def rouge_l(prediction: str, reference: str) -> MetricResult:
    """ROUGE-L F1: LCS-based overlap between prediction and reference tokens."""
    p_toks, r_toks = tokenize(prediction), tokenize(reference)
    if not p_toks or not r_toks:
        return MetricResult(
            name="rouge_l", score=0.0, details={"lcs": 0, "precision": 0.0, "recall": 0.0}
        )
    lcs = _lcs_length(p_toks, r_toks)
    precision = lcs / len(p_toks)
    recall = lcs / len(r_toks)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MetricResult(
        name="rouge_l",
        score=f1,
        details={"lcs": lcs, "precision": precision, "recall": recall},
    )


def bleu(prediction: str, reference: str, max_n: int = 4) -> MetricResult:
    """Corpus-style BLEU for a single pair, implemented from scratch.

    Uses clipped n-gram precisions (n = 1..``max_n``), geometric mean, and the
    standard brevity penalty.  Orders with no n-grams in the prediction (e.g.
    4-grams for a 3-token prediction) are skipped; any available order with
    zero clipped precision yields a score of 0.0 (no smoothing), matching the
    reference BLEU behavior for degenerate outputs.
    """
    p_toks, r_toks = tokenize(prediction), tokenize(reference)
    if not p_toks or not r_toks:
        return MetricResult(name="bleu", score=0.0, details={"precisions": []})

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        p_ngrams = _ngrams(p_toks, n)
        if not p_ngrams:
            continue  # prediction shorter than n: order not applicable
        ref_counts = Counter(_ngrams(r_toks, n))
        clipped = 0
        for ng in p_ngrams:
            if ref_counts[ng] > 0:
                clipped += 1
                ref_counts[ng] -= 1
        precision = clipped / len(p_ngrams)
        if precision == 0.0:
            return MetricResult(
                name="bleu", score=0.0, details={"precisions": precisions + [0.0]}
            )
        precisions.append(precision)

    if not precisions:
        return MetricResult(name="bleu", score=0.0, details={"precisions": []})

    log_mean = sum(math.log(p) for p in precisions) / len(precisions)
    bp = 1.0 if len(p_toks) >= len(r_toks) else math.exp(1 - len(r_toks) / len(p_toks))
    return MetricResult(
        name="bleu",
        score=bp * math.exp(log_mean),
        details={"precisions": precisions, "brevity_penalty": bp},
    )


def json_validity(prediction: str, reference: str) -> MetricResult:
    """1.0 if ``prediction`` parses as JSON, else 0.0.

    The reference is unused; the metric checks output well-formedness, which
    is useful for structured-generation evals.
    """
    try:
        json.loads(prediction)
    except json.JSONDecodeError as exc:
        return MetricResult(
            name="json_validity", score=0.0, details={"error": str(exc)}
        )
    return MetricResult(name="json_validity", score=1.0, details={"error": None})


def length_stats(prediction: str, reference: str) -> MetricResult:
    """Length similarity between prediction and reference.

    Score is ``min/max`` of token counts (1.0 = identical length); the raw
    character and token counts are reported in ``details`` for analysis.
    """
    p_toks, r_toks = tokenize(prediction), tokenize(reference)
    p_len, r_len = len(p_toks), len(r_toks)
    score = min(p_len, r_len) / max(p_len, r_len) if max(p_len, r_len) else 1.0
    return MetricResult(
        name="length_stats",
        score=score,
        details={
            "pred_chars": len(prediction),
            "ref_chars": len(reference),
            "pred_tokens": p_len,
            "ref_tokens": r_len,
            "length_ratio": p_len / r_len if r_len else 0.0,
        },
    )


# ---------------------------------------------------------------------------
# LLM-as-judge
# ---------------------------------------------------------------------------


@dataclass
class Judgment:
    """A judge backend's verdict on one (prediction, reference) pair."""

    score: float  # expected in [0, 1]
    rationale: str


class JudgeBackend(Protocol):
    """Structural type for LLM judge backends."""

    name: str

    def judge(self, prediction: str, reference: str, rubric: str = "") -> Judgment:
        ...


class MockJudge:
    """Deterministic stand-in judge for tests and offline development.

    Scores the token-overlap F1 between prediction and reference so results
    are fully reproducible without any model API calls.
    """

    name = "mock"

    def judge(self, prediction: str, reference: str, rubric: str = "") -> Judgment:
        p_toks, r_toks = set(tokenize(prediction)), set(tokenize(reference))
        if not p_toks or not r_toks:
            return Judgment(score=0.0, rationale="mock-judge: empty input")
        overlap = len(p_toks & r_toks)
        precision = overlap / len(p_toks)
        recall = overlap / len(r_toks)
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return Judgment(
            score=f1, rationale=f"mock-judge: token-overlap F1={f1:.3f}"
        )


def llm_judge_metric(backend: JudgeBackend, rubric: str = "") -> Metric:
    """Build a metric that scores pairs via a :class:`JudgeBackend`.

    Example:
        >>> metric = llm_judge_metric(MockJudge())
        >>> metric("Paris", "Paris").score
        1.0
    """

    def _score(prediction: str, reference: str) -> MetricResult:
        judgment = backend.judge(prediction, reference, rubric)
        score = min(1.0, max(0.0, judgment.score))
        return MetricResult(
            name=f"llm_judge[{backend.name}]",
            score=score,
            details={"rationale": judgment.rationale},
        )

    return _FnMetric(f"llm_judge[{backend.name}]", _score)


# Register built-ins.
register_metric("exact_match", _FnMetric("exact_match", exact_match))
register_metric("rouge_l", _FnMetric("rouge_l", rouge_l))
register_metric("bleu", _FnMetric("bleu", bleu))
register_metric("json_validity", _FnMetric("json_validity", json_validity))
register_metric("length_stats", _FnMetric("length_stats", length_stats))
