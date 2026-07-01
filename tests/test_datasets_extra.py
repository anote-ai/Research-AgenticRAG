"""Tests for the FRAMES + CRAG adapters and the dense retriever."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

import agenticrag.datasets as datasets_mod
from agenticrag.datasets import (
    QASample,
    _build_frames_corpus,
    _chunk_text,
    _crag_fallback,
    _frames_fallback,
    _normalize_links,
    _wiki_title_from_url,
    load_dataset,
)
from agenticrag.retrievers import DenseRetriever


# --------------------------------------------------------------------------- #
# FRAMES / CRAG fallbacks
# --------------------------------------------------------------------------- #

class TestFramesFallback:
    def test_returns_qasamples(self):
        samples = _frames_fallback()
        assert samples and all(isinstance(s, QASample) for s in samples)

    def test_variable_hop_depth(self):
        depths = {s.hop_count for s in _frames_fallback()}
        # FRAMES' value is variable hop depth — the fallback spans >1 depth.
        assert len(depths) > 1

    def test_has_supporting_docs(self):
        assert all(s.supporting_docs for s in _frames_fallback())


class TestCragFallback:
    def test_returns_qasamples(self):
        samples = _crag_fallback()
        assert samples and all(isinstance(s, QASample) for s in samples)

    def test_includes_false_premise_type(self):
        ids = " ".join(s.sample_id for s in _crag_fallback())
        assert "false_premise" in ids


# --------------------------------------------------------------------------- #
# FRAMES Wikipedia passage step (deterministic, network-free)
# --------------------------------------------------------------------------- #

class TestFramesPassageHelpers:
    def test_normalize_native_list(self):
        assert _normalize_links(["http://a", "http://b"]) == ["http://a", "http://b"]

    def test_normalize_stringified_list(self):
        out = _normalize_links("['https://en.wikipedia.org/wiki/A', 'https://x/B']")
        assert out == ["https://en.wikipedia.org/wiki/A", "https://x/B"]

    def test_normalize_newline_separated(self):
        out = _normalize_links("https://en.wikipedia.org/wiki/A\nhttps://en.wikipedia.org/wiki/B")
        assert len(out) == 2

    def test_normalize_none(self):
        assert _normalize_links(None) == []

    def test_title_from_url(self):
        assert _wiki_title_from_url("https://en.wikipedia.org/wiki/Albert_Einstein") == "Albert Einstein"

    def test_title_url_decoded(self):
        assert _wiki_title_from_url("https://en.wikipedia.org/wiki/Niels_Bohr%27s_house").startswith("Niels Bohr")

    def test_chunk_text_prefixes_title_and_bounds_size(self):
        text = "First paragraph.\n== Section ==\nSecond paragraph here.\n" + ("x" * 50)
        chunks = _chunk_text(text, "Title", max_chars=30)
        assert chunks  # produced something
        assert all(c.startswith("Title: ") for c in chunks)
        # Header line is dropped.
        assert all("== Section ==" not in c for c in chunks)

    def test_build_corpus_uses_cache_and_chunks(self):
        # Fake cache: returns canned page text without any network.
        class _FakeCache:
            def extract(self, title):
                return f"{title} was a notable subject. It had several facts."

        docs = _build_frames_corpus(
            ["https://en.wikipedia.org/wiki/Marie_Curie",
             "https://en.wikipedia.org/wiki/Pierre_Curie"],
            _FakeCache(), passage_chars=200,
        )
        assert any("Marie Curie" in d for d in docs)
        assert any("Pierre Curie" in d for d in docs)

    def test_build_corpus_respects_max_passages(self):
        class _FakeCache:
            def extract(self, title):
                return "\n".join(f"Fact {i} about it." for i in range(20))

        docs = _build_frames_corpus(
            ["https://en.wikipedia.org/wiki/X"], _FakeCache(),
            passage_chars=20, max_passages=3,
        )
        assert len(docs) <= 3

    def test_build_corpus_degrades_to_title_when_empty(self):
        class _EmptyCache:
            def extract(self, title):
                return ""

        docs = _build_frames_corpus(
            ["https://en.wikipedia.org/wiki/Obscure_Page"], _EmptyCache()
        )
        assert docs == ["Obscure Page"]


# --------------------------------------------------------------------------- #
# Unified loader routing
# --------------------------------------------------------------------------- #

class TestLoadDatasetRouting:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            load_dataset("not_a_dataset")

    def test_known_names_route(self, monkeypatch):
        # Force the built-in fallbacks so the test never touches the network.
        monkeypatch.setattr(datasets_mod, "_hf_available", lambda: False)
        for name in ("hotpotqa", "musique", "frames", "crag"):
            samples = load_dataset(name, max_samples=1)
            assert isinstance(samples, list)
            assert all(isinstance(s, QASample) for s in samples)


# --------------------------------------------------------------------------- #
# DenseRetriever (deterministic fallback path)
# --------------------------------------------------------------------------- #

class TestDenseRetrieverFallback:
    def _forced_fallback(self):
        # A bogus model name makes construction take the except branch (no model
        # download), so the test deterministically exercises the offline
        # token-overlap fallback and stays fast/network-free in CI.
        dr = DenseRetriever(model_name="__agenticrag_nonexistent_test_model__")
        dr._available = False
        dr._model = None
        return dr

    def test_returns_ranked_tuples(self):
        dr = self._forced_fallback()
        corpus = ["apple makes iphone", "banana yellow fruit", "eiffel tower paris"]
        out = dr.retrieve("who makes the iphone", corpus, top_k=2)
        assert len(out) == 2
        assert all(isinstance(doc, str) and isinstance(score, float) for doc, score in out)
        # Most relevant doc ranks first.
        assert out[0][0] == "apple makes iphone"

    def test_empty_corpus(self):
        assert self._forced_fallback().retrieve("q", [], top_k=3) == []

    def test_scores_non_negative(self):
        dr = self._forced_fallback()
        out = dr.retrieve("apple", ["apple pie", "banana"], top_k=2)
        assert all(score >= 0.0 for _, score in out)
