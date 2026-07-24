"""
FinBERT sentiment engine.

Wraps ProsusAI/finbert for finance-tuned sentiment scoring.
Lazily loads the model once per process; batch-inferences efficiently.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from .database import cache_get, cache_set

log = logging.getLogger(__name__)

_MODEL_NAME = "ProsusAI/finbert"
_TTL_SENTIMENT = 86400  # 24 h


class FinBertSentiment:
    """Score text with FinBERT and return structured sentiment dict."""

    def __init__(self) -> None:
        from transformers import (  # noqa: F811 – deferred import
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )

        tok = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            top_k=None,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def score(self, text: str) -> dict[str, Any]:
        """Return sentiment dict for a single piece of text."""
        if not text or not text.strip():
            return _empty_score()

        cache_key = _cache_key(text)
        cached = cache_get(cache_key, _TTL_SENTIMENT)
        if cached is not None:
            return cached

        result = self._score_raw(text)
        cache_set(cache_key, result)
        return result

    def score_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """Score a list of texts, using cache where possible.

        Uncached texts are batched through the pipeline in one call
        for efficiency, then results are merged back in order.
        """
        if not texts:
            return []

        results: list[dict[str, Any] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = _empty_score()
                continue
            cache_key = _cache_key(text)
            cached = cache_get(cache_key, _TTL_SENTIMENT)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            raw_results = self._pipe([t[:512] for t in uncached_texts])
            for idx, raw in zip(uncached_indices, raw_results):
                score_dict = _parse_pipeline_output(raw)
                cache_set(_cache_key(texts[idx]), score_dict)
                results[idx] = score_dict

        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_raw(self, text: str) -> dict[str, Any]:
        raw = self._pipe(text[:512])[0]
        return _parse_pipeline_output(raw)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _parse_pipeline_output(raw: list[dict]) -> dict[str, Any]:
    """Convert pipeline output to our standard schema."""
    scores = {item["label"].lower(): item["score"] for item in raw}
    label = max(scores, key=scores.get)  # type: ignore[arg-type]
    return {
        "label": label,
        "positive": round(scores.get("positive", 0.0), 4),
        "neutral": round(scores.get("neutral", 0.0), 4),
        "negative": round(scores.get("negative", 0.0), 4),
        "compound": round(
            scores.get("positive", 0.0) - scores.get("negative", 0.0), 4
        ),
    }


def _empty_score() -> dict[str, Any]:
    return {
        "label": "neutral",
        "positive": 0.0,
        "neutral": 1.0,
        "negative": 0.0,
        "compound": 0.0,
    }


def _cache_key(text: str) -> str:
    h = hashlib.sha256(text.encode()).hexdigest()[:32]
    return f"sentiment:{h}"


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_engine: FinBertSentiment | None = None


def get_sentiment_engine() -> FinBertSentiment:
    """Return the singleton FinBertSentiment, loading the model on first call."""
    global _engine
    if _engine is None:
        log.info("Loading FinBERT model (first call) ...")
        _engine = FinBertSentiment()
        log.info("FinBERT model loaded.")
    return _engine
