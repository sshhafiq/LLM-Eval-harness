"""Dataset loading, validation, sampling, and splitting.

Datasets are JSONL files with one object per line::

    {"id": "q1", "input": "What is the capital of France?", "reference": "Paris"}
    {"id": "q2", "input": "2 + 2 = ?", "reference": "4", "metadata": {"topic": "math"}}

``id``, ``input`` and ``reference`` are required; ``metadata`` is optional and
carries any extra per-example information through the pipeline untouched.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union


class DatasetError(ValueError):
    """Raised when a dataset file fails validation."""


@dataclass
class Example:
    """A single evaluation example."""

    id: str
    input: str
    reference: str
    metadata: Dict[str, object] = field(default_factory=dict)


_REQUIRED_FIELDS = ("id", "input", "reference")


def _validate_record(record: object, line_no: int, path: str) -> Example:
    if not isinstance(record, dict):
        raise DatasetError(f"{path}:{line_no}: expected a JSON object, got {type(record).__name__}")
    for name in _REQUIRED_FIELDS:
        if name not in record:
            raise DatasetError(f"{path}:{line_no}: missing required field {name!r}")
        if not isinstance(record[name], str):
            raise DatasetError(
                f"{path}:{line_no}: field {name!r} must be a string, "
                f"got {type(record[name]).__name__}"
            )
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"{path}:{line_no}: field 'metadata' must be an object")
    return Example(
        id=record["id"],
        input=record["input"],
        reference=record["reference"],
        metadata=dict(metadata),
    )


def load_jsonl(path: Union[str, Path]) -> List[Example]:
    """Load and validate a JSONL dataset.

    Raises:
        DatasetError: on malformed JSON, missing/invalid fields, empty files,
            or duplicate example ids.
        FileNotFoundError: if ``path`` does not exist.
    """
    path = Path(path)
    examples: List[Example] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue  # tolerate blank lines
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            example = _validate_record(record, line_no, str(path))
            if example.id in seen_ids:
                raise DatasetError(f"{path}:{line_no}: duplicate id {example.id!r}")
            seen_ids.add(example.id)
            examples.append(example)
    if not examples:
        raise DatasetError(f"{path}: dataset is empty")
    return examples


def sample(examples: List[Example], n: int, seed: int = 42) -> List[Example]:
    """Return ``n`` examples sampled without replacement (deterministic seed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    rng = random.Random(seed)
    return rng.sample(examples, min(n, len(examples)))


def train_test_split(
    examples: List[Example], test_size: float = 0.2, seed: int = 42
) -> tuple[List[Example], List[Example]]:
    """Deterministically split examples into (train, test).

    ``test_size`` is the fraction of examples assigned to the test split; it
    must be in ``(0, 1)``.  The split is a partition: no example appears in
    both splits and their union is the input (order not preserved).
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_size))
    n_test = min(n_test, len(shuffled) - 1)  # keep at least one train example
    return shuffled[n_test:], shuffled[:n_test]
