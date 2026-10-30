"""Tests for eval_harness.metrics — correctness against hand-computed values."""

import math

import pytest

from eval_harness.metrics import (
    MetricResult,
    MockJudge,
    bleu,
    exact_match,
    get_metric,
    json_validity,
    length_stats,
    list_metrics,
    llm_judge_metric,
    register_metric,
    rouge_l,
)


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Paris", "Paris").score == 1.0

    def test_normalization(self):
        assert exact_match("  PARIS! ", "paris").score == 1.0

    def test_mismatch(self):
        assert exact_match("Paris", "London").score == 0.0

    def test_name(self):
        assert exact_match("a", "a").name == "exact_match"


class TestRougeL:
    def test_known_value(self):
        # pred tokens: the cat sat on the mat (6)
        # ref tokens:  the cat is on the mat (6)
        # LCS: the cat on the mat -> 5; P = 5/6, R = 5/6, F1 = 5/6
        result = rouge_l("the cat sat on the mat", "the cat is on the mat")
        assert result.score == pytest.approx(5 / 6)
        assert result.details["lcs"] == 5
        assert result.details["precision"] == pytest.approx(5 / 6)
        assert result.details["recall"] == pytest.approx(5 / 6)

    def test_identical_is_one(self):
        assert rouge_l("hello world", "hello world").score == pytest.approx(1.0)

    def test_no_overlap_is_zero(self):
        assert rouge_l("aaa bbb", "ccc ddd").score == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        # pred tokens: the cat sat (3); ref tokens: the cat ran fast (4)
        # LCS: the cat -> 2; P = 2/3, R = 2/4, F1 = 4/7
        result = rouge_l("the cat sat", "the cat ran fast")
        assert result.score == pytest.approx(4 / 7)
        assert result.details["lcs"] == 2

    def test_empty_prediction(self):
        assert rouge_l("", "hello").score == 0.0

    def test_empty_reference(self):
        assert rouge_l("hello", "").score == 0.0


class TestBleu:
    def test_identical_is_one(self):
        assert bleu("the cat sat", "the cat sat").score == pytest.approx(1.0)

    def test_known_brevity_penalty(self):
        # pred "the cat sat" (3 tokens), ref "the cat sat on the mat" (6).
        # All available n-gram precisions are 1.0; BP = exp(1 - 6/3) = exp(-1).
        result = bleu("the cat sat", "the cat sat on the mat")
        assert result.score == pytest.approx(math.exp(-1))
        assert result.details["brevity_penalty"] == pytest.approx(math.exp(-1))

    def test_no_brevity_penalty_when_longer_or_equal(self):
        # Identical strings: prediction not shorter than reference -> BP = 1.0.
        result = bleu("the cat sat on the mat", "the cat sat on the mat")
        assert result.score == pytest.approx(1.0)
        assert result.details["brevity_penalty"] == pytest.approx(1.0)

    def test_zero_overlap(self):
        assert bleu("aaa bbb ccc", "ddd eee fff").score == 0.0

    def test_empty_prediction(self):
        assert bleu("", "hello world").score == 0.0

    def test_short_prediction_skips_missing_orders(self):
        # Pred "the cat" (2 tokens) vs ref "the cat sat" (3 tokens): only
        # unigram and bigram orders apply, both precision 1.0, so the score
        # is just the brevity penalty exp(1 - 3/2).
        result = bleu("the cat", "the cat sat")
        assert result.score == pytest.approx(math.exp(-0.5))
        assert result.details["precisions"] == pytest.approx([1.0, 1.0])

    def test_max_n_limits_orders(self):
        # Unigram-only BLEU ignores word order: same tokens, shuffled, is 1.0.
        result = bleu("cat the sat", "the cat sat", max_n=1)
        assert result.score == pytest.approx(1.0)
        assert result.details["precisions"] == pytest.approx([1.0])

    def test_partial_overlap_between_zero_and_one(self):
        # Every n-gram order 1..4 has nonzero clipped precision here, and both
        # strings are 6 tokens, so BP = 1 and 0 < BLEU < 1.
        score = bleu("the cat sat on the mat", "the cat sat on the rug").score
        assert 0.0 < score < 1.0


class TestJsonValidity:
    def test_valid_object(self):
        assert json_validity('{"a": 1}', "").score == 1.0

    def test_valid_array(self):
        assert json_validity("[1, 2, 3]", "").score == 1.0

    def test_invalid(self):
        result = json_validity("{not json", "")
        assert result.score == 0.0
        assert result.details["error"]


class TestLengthStats:
    def test_identical_length(self):
        result = length_stats("a b c", "d e f")
        assert result.score == pytest.approx(1.0)
        assert result.details["pred_tokens"] == 3

    def test_ratio(self):
        # 2 tokens vs 4 tokens -> 0.5
        assert length_stats("a b", "c d e f").score == pytest.approx(0.5)

    def test_empty_both(self):
        assert length_stats("", "").score == pytest.approx(1.0)


class TestMockJudge:
    def test_identical_is_one(self):
        assert MockJudge().judge("Paris", "Paris").score == pytest.approx(1.0)

    def test_deterministic(self):
        judge = MockJudge()
        first = judge.judge("the cat sat", "the cat ran")
        second = judge.judge("the cat sat", "the cat ran")
        assert first.score == second.score
        assert first.rationale == second.rationale

    def test_empty_is_zero(self):
        assert MockJudge().judge("", "Paris").score == 0.0


class TestLlmJudgeMetric:
    def test_name_and_score(self):
        metric = llm_judge_metric(MockJudge())
        assert metric.name == "llm_judge[mock]"
        assert metric("Paris", "Paris").score == pytest.approx(1.0)

    def test_score_bounded(self):
        metric = llm_judge_metric(MockJudge())
        result = metric("xyz", "completely different words here")
        assert 0.0 <= result.score <= 1.0
        assert result.details["rationale"]


class TestRegistry:
    def test_builtins_registered(self):
        for name in ("exact_match", "rouge_l", "bleu", "json_validity", "length_stats"):
            assert name in list_metrics()
            assert get_metric(name).name == name

    def test_unknown_raises_helpful_keyerror(self):
        with pytest.raises(KeyError, match="available"):
            get_metric("no_such_metric")

    def test_register_custom(self):
        class AlwaysHalf:
            name = "always_half"

            def __call__(self, prediction: str, reference: str) -> MetricResult:
                return MetricResult(name="always_half", score=0.5)

        register_metric("always_half", AlwaysHalf())
        assert get_metric("always_half")("a", "b").score == 0.5
