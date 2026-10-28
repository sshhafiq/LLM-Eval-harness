"""Tests for eval_harness.datasets."""

import pytest

from eval_harness.datasets import (
    DatasetError,
    load_jsonl,
    sample,
    train_test_split,
)

VALID = (
    '{"id": "a", "input": "q1", "reference": "r1"}\n'
    '{"id": "b", "input": "q2", "reference": "r2", "metadata": {"topic": "x"}}\n'
    "\n"  # blank lines are tolerated
    '{"id": "c", "input": "q3", "reference": "r3"}\n'
)


def _write(tmp_path, text):
    path = tmp_path / "data.jsonl"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid(tmp_path):
    examples = load_jsonl(_write(tmp_path, VALID))
    assert [e.id for e in examples] == ["a", "b", "c"]
    assert examples[1].metadata == {"topic": "x"}
    assert examples[0].metadata == {}


def test_missing_field(tmp_path):
    with pytest.raises(DatasetError, match="missing required field"):
        load_jsonl(_write(tmp_path, '{"id": "a", "input": "q"}\n'))


def test_non_string_field(tmp_path):
    with pytest.raises(DatasetError, match="must be a string"):
        load_jsonl(_write(tmp_path, '{"id": 1, "input": "q", "reference": "r"}\n'))


def test_bad_json(tmp_path):
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_jsonl(_write(tmp_path, "{not json}\n"))


def test_non_object_line(tmp_path):
    with pytest.raises(DatasetError, match="expected a JSON object"):
        load_jsonl(_write(tmp_path, "[1, 2]\n"))


def test_duplicate_ids(tmp_path):
    with pytest.raises(DatasetError, match="duplicate id"):
        load_jsonl(
            _write(
                tmp_path,
                '{"id": "a", "input": "q1", "reference": "r1"}\n'
                '{"id": "a", "input": "q2", "reference": "r2"}\n',
            )
        )


def test_empty_file(tmp_path):
    with pytest.raises(DatasetError, match="empty"):
        load_jsonl(_write(tmp_path, "\n  \n"))


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_jsonl(tmp_path / "nope.jsonl")


def test_bad_metadata(tmp_path):
    with pytest.raises(DatasetError, match="metadata"):
        load_jsonl(
            _write(
                tmp_path,
                '{"id": "a", "input": "q", "reference": "r", "metadata": [1]}\n',
            )
        )


def test_sample_size_and_determinism(tmp_path):
    examples = load_jsonl(_write(tmp_path, VALID))
    first = sample(examples, 2, seed=1)
    second = sample(examples, 2, seed=1)
    assert [e.id for e in first] == [e.id for e in second]
    assert len(first) == 2
    # sampling more than available returns everything
    assert len(sample(examples, 99, seed=1)) == 3


def test_sample_negative():
    with pytest.raises(ValueError):
        sample([], -1)


def _many(tmp_path, n):
    lines = "\n".join(
        f'{{"id": "ex-{i}", "input": "q{i}", "reference": "r{i}"}}' for i in range(n)
    )
    return load_jsonl(_write(tmp_path, lines + "\n"))


def test_split_is_partition(tmp_path):
    examples = _many(tmp_path, 20)
    train, test = train_test_split(examples, test_size=0.25, seed=7)
    assert len(test) == 5
    assert len(train) == 15
    train_ids = {e.id for e in train}
    test_ids = {e.id for e in test}
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == {e.id for e in examples}


def test_split_deterministic(tmp_path):
    examples = _many(tmp_path, 20)
    first = train_test_split(examples, test_size=0.3, seed=3)
    second = train_test_split(examples, test_size=0.3, seed=3)
    assert [e.id for e in first[0]] == [e.id for e in second[0]]
    assert [e.id for e in first[1]] == [e.id for e in second[1]]


def test_split_bad_size(tmp_path):
    examples = _many(tmp_path, 5)
    with pytest.raises(ValueError):
        train_test_split(examples, test_size=0.0)
    with pytest.raises(ValueError):
        train_test_split(examples, test_size=1.5)
