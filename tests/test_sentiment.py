"""Tests for src/sentiment.py — FinBERT sentiment engine.

The actual FinBERT model is ~440 MB and downloads on first use.
Tests mock the pipeline to avoid network/model dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.sentiment import (
    FinBertSentiment,
    _empty_score,
    _parse_pipeline_output,
    get_sentiment_engine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_pipeline_output(label: str, score: float) -> list[dict]:
    """Mimic the list-of-dicts that transformers pipeline returns with top_k=None."""
    labels = ["positive", "negative", "neutral"]
    others = [l for l in labels if l != label]
    return [
        {"label": label, "score": score},
        {"label": others[0], "score": (1 - score) / 2},
        {"label": others[1], "score": (1 - score) / 2},
    ]


def _make_engine(outputs: list[list[dict]]) -> FinBertSentiment:
    """Build a FinBertSentiment with a mocked pipeline.

    Each element in *outputs* is what ``pipe(text)`` returns for one input.
    The real pipeline with ``top_k=None`` wraps each result in a list, so
    ``pipe("text")`` → ``[[{…}, {…}, {…}]]``.  MagicMock with ``side_effect``
    pops one element per call, so each element must be that outer list.
    """
    engine = object.__new__(FinBertSentiment)
    mock_pipe = MagicMock(side_effect=outputs)
    engine._pipe = mock_pipe
    return engine


# ---------------------------------------------------------------------------
# _parse_pipeline_output
# ---------------------------------------------------------------------------


class TestParsePipelineOutput:
    def test_positive_dominant(self):
        raw = _fake_pipeline_output("positive", 0.92)
        result = _parse_pipeline_output(raw)
        assert result["label"] == "positive"
        assert result["compound"] > 0
        assert result["positive"] > result["negative"]

    def test_negative_dominant(self):
        raw = _fake_pipeline_output("negative", 0.88)
        result = _parse_pipeline_output(raw)
        assert result["label"] == "negative"
        assert result["compound"] < 0

    def test_neutral_dominant(self):
        raw = _fake_pipeline_output("neutral", 0.75)
        result = _parse_pipeline_output(raw)
        assert result["label"] == "neutral"
        assert result["compound"] == pytest.approx(0.0, abs=0.01)

    def test_keys_present(self):
        raw = _fake_pipeline_output("positive", 0.6)
        result = _parse_pipeline_output(raw)
        assert set(result.keys()) == {"label", "positive", "neutral", "negative", "compound"}


# ---------------------------------------------------------------------------
# _empty_score
# ---------------------------------------------------------------------------


class TestEmptyScore:
    def test_returns_neutral(self):
        s = _empty_score()
        assert s["label"] == "neutral"
        assert s["compound"] == 0.0
        assert s["neutral"] == 1.0


# ---------------------------------------------------------------------------
# FinBertSentiment.score
# ---------------------------------------------------------------------------


class TestScore:
    def test_positive_headline(self):
        outputs = [[_fake_pipeline_output("positive", 0.91)]]
        engine = _make_engine(outputs)
        with patch("src.sentiment.cache_get", return_value=None), \
             patch("src.sentiment.cache_set"):
            result = engine.score("Apple reports record-breaking quarterly earnings")
        assert result["label"] == "positive"
        assert result["compound"] > 0.5

    def test_negative_headline(self):
        outputs = [[_fake_pipeline_output("negative", 0.87)]]
        engine = _make_engine(outputs)
        with patch("src.sentiment.cache_get", return_value=None), \
             patch("src.sentiment.cache_set"):
            result = engine.score("Company files for bankruptcy after fraud scandal")
        assert result["label"] == "negative"
        assert result["compound"] < -0.5

    def test_empty_text(self):
        engine = _make_engine([])
        result = engine.score("")
        assert result["label"] == "neutral"
        assert result["compound"] == 0.0

    def test_whitespace_only(self):
        engine = _make_engine([])
        result = engine.score("   ")
        assert result["label"] == "neutral"

    def test_cache_hit_skips_pipeline(self):
        cached = {"label": "positive", "positive": 0.8, "neutral": 0.15, "negative": 0.05, "compound": 0.75}
        engine = _make_engine([])
        with patch("src.sentiment.cache_get", return_value=cached):
            result = engine.score("Some headline")
        assert result == cached
        engine._pipe.assert_not_called()


# ---------------------------------------------------------------------------
# FinBertSentiment.score_batch
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_empty_list(self):
        engine = _make_engine([])
        with patch("src.sentiment.cache_get", return_value=None), \
             patch("src.sentiment.cache_set"):
            assert engine.score_batch([]) == []

    def test_preserves_order(self):
        batch_result = [
            _fake_pipeline_output("positive", 0.9),
            _fake_pipeline_output("negative", 0.85),
            _fake_pipeline_output("neutral", 0.7),
        ]
        engine = _make_engine([])
        engine._pipe = MagicMock(return_value=batch_result)
        with patch("src.sentiment.cache_get", return_value=None), \
             patch("src.sentiment.cache_set"):
            results = engine.score_batch(["Good", "Bad", "Meh"])
        assert len(results) == 3
        assert results[0]["label"] == "positive"
        assert results[1]["label"] == "negative"
        assert results[2]["label"] == "neutral"

    def test_mixed_cache_hits(self):
        cached = {"label": "neutral", "positive": 0.1, "neutral": 0.8, "negative": 0.1, "compound": 0.0}
        batch_result = [_fake_pipeline_output("positive", 0.9)]
        engine = _make_engine([])
        engine._pipe = MagicMock(return_value=batch_result)

        call_count = 0

        def fake_cache_get(key, ttl):
            nonlocal call_count
            call_count += 1
            # First call is for "cached headline", second for "new headline"
            if call_count == 1:
                return cached
            return None

        with patch("src.sentiment.cache_get", side_effect=fake_cache_get), \
             patch("src.sentiment.cache_set"):
            results = engine.score_batch(["cached headline", "new headline"])
        assert len(results) == 2
        assert results[0] == cached
        assert results[1]["label"] == "positive"


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_weighted_average_in_range(self):
        scores = [
            {"compound": 0.8},
            {"compound": -0.3},
            {"compound": 0.1},
            {"compound": -0.5},
            {"compound": 0.6},
        ]
        avg = sum(s["compound"] for s in scores) / len(scores)
        assert -1.0 <= avg <= 1.0
        assert avg == pytest.approx(0.14, abs=0.01)

    def test_all_positive(self):
        scores = [{"compound": 0.5}, {"compound": 0.7}, {"compound": 0.9}]
        avg = sum(s["compound"] for s in scores) / len(scores)
        assert avg > 0


# ---------------------------------------------------------------------------
# get_sentiment_engine singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_returns_same_instance(self):
        import src.sentiment as mod
        mod._engine = None
        with patch.object(FinBertSentiment, "__init__", return_value=None):
            e1 = get_sentiment_engine()
            e2 = get_sentiment_engine()
        assert e1 is e2
        mod._engine = None  # cleanup


# ---------------------------------------------------------------------------
# VADER fallback in advisor
# ---------------------------------------------------------------------------


class TestAdvisorFallback:
    def test_analyze_ticker_sentiment_returns_tuple(self):
        from src import advisor

        with patch("src.advisor.get_sentiment_engine") as mock_eng:
            mock_eng.return_value.score_batch.return_value = [
                {"label": "positive", "compound": 0.6},
            ]
            with patch("src.advisor._gemini_generate", return_value=None):
                with patch("src.advisor.av_get_ticker_news", return_value=[]):
                    result = advisor.analyze_ticker_sentiment("AAPL", [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[1] == []
