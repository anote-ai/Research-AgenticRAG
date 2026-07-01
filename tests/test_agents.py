"""Tests for agenticrag.agents — LLMAgent, providers, decision parsing."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agenticrag.agents import (
    AgentDecision,
    HopState,
    LLMAgent,
    LLMResponse,
    MockProvider,
    make_provider,
    parse_decision,
)
from agenticrag.core import PipelineTrace
from agenticrag.retrievers import BM25Retriever, TokenOverlapRetriever

CORPUS = [
    "Apple Inc. produces the iPhone.",
    "Tim Cook is the CEO of Apple Inc.",
    "Bananas are a good source of potassium.",
    "The Eiffel Tower is in Paris.",
]
QUESTION = "Who is the CEO of the company that produces the iPhone?"


# --------------------------------------------------------------------------- #
# parse_decision
# --------------------------------------------------------------------------- #

class TestParseDecision:
    def test_parses_search(self):
        d = parse_decision('{"action": "search", "query": "next fact"}')
        assert d.action == "search" and d.query == "next fact"

    def test_parses_answer(self):
        d = parse_decision('{"action": "answer", "answer": "Tim Cook"}')
        assert d.action == "answer" and d.answer == "Tim Cook"

    def test_search_without_query_falls_back_to_answer(self):
        d = parse_decision('{"action": "search"}')
        assert d.action == "answer"

    def test_non_json_treated_as_answer(self):
        d = parse_decision("The answer is Tim Cook.")
        assert d.action == "answer" and "Tim Cook" in d.answer

    def test_embedded_json_extracted(self):
        d = parse_decision('Sure! {"action": "answer", "answer": "Paris"} done')
        assert d.action == "answer" and d.answer == "Paris"


# --------------------------------------------------------------------------- #
# MockProvider
# --------------------------------------------------------------------------- #

class TestMockProvider:
    def test_deterministic(self):
        p = MockProvider()
        sys_p = "sys"
        user = "QUESTION: q\nHOP: 1 / 3\n\nEVIDENCE:\n(none retrieved yet)\n\ngo"
        r1 = p.generate(sys_p, user)
        r2 = p.generate(sys_p, user)
        assert r1.text == r2.text

    def test_reports_token_usage(self):
        r = MockProvider().generate("sys", "QUESTION: q\nHOP: 1 / 3\n\nEVIDENCE:\n(none)")
        assert isinstance(r, LLMResponse)
        assert r.total_tokens == r.input_tokens + r.output_tokens
        assert r.total_tokens > 0

    def test_no_evidence_triggers_search(self):
        user = "QUESTION: who\nHOP: 1 / 3\n\nEVIDENCE:\n(none retrieved yet)"
        d = parse_decision(MockProvider().generate("s", user).text)
        assert d.action == "search"

    def test_answer_prompt_without_hop_always_answers(self):
        # No HOP line => forced-answer prompt.
        user = "QUESTION: who\n\nEVIDENCE:\n[1] who is bob\n\nanswer now"
        d = parse_decision(MockProvider().generate("s", user).text)
        assert d.action == "answer"


# --------------------------------------------------------------------------- #
# make_provider
# --------------------------------------------------------------------------- #

class TestMakeProvider:
    def test_mock(self):
        assert isinstance(make_provider("mock"), MockProvider)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            make_provider("nonsense-provider")

    def test_colon_form_splits_model(self):
        # Should not construct a real client for mock; just verify name handling.
        assert isinstance(make_provider("mock:whatever"), MockProvider)


# --------------------------------------------------------------------------- #
# LLMAgent.run
# --------------------------------------------------------------------------- #

class TestLLMAgentRun:
    def _agent(self, retriever=None, max_iter=3):
        return LLMAgent(
            provider=MockProvider(),
            retriever=retriever or BM25Retriever(),
            max_iterations=max_iter,
        )

    def test_returns_pipeline_trace(self):
        t = self._agent().run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert isinstance(t, PipelineTrace)

    def test_populates_hop_fields(self):
        t = self._agent().run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert len(t.hop_queries) == len(t.hop_docs) == t.iterations_used
        assert len(t.tool_calls) >= 1
        assert all(tc["name"] == "retrieve" for tc in t.tool_calls)

    def test_finds_answer(self):
        t = self._agent().run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert "Tim Cook" in t.final_answer

    def test_tracks_token_cost(self):
        t = self._agent().run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert t.tokens_used > 0

    def test_empty_corpus_does_not_crash(self):
        t = self._agent().run(QUESTION, [], reference_answer="Tim Cook")
        assert isinstance(t, PipelineTrace)

    def test_respects_iteration_budget(self):
        t = self._agent(max_iter=2).run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert t.iterations_used <= 2

    def test_falls_back_to_token_overlap_without_retriever(self):
        agent = LLMAgent(provider=MockProvider(), retriever=None, max_iterations=3)
        t = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        assert t.retrieved_docs  # something retrieved


# --------------------------------------------------------------------------- #
# Resumability (live-suffix substrate)
# --------------------------------------------------------------------------- #

class TestResume:
    def test_resume_continues_from_prefix(self):
        agent = LLMAgent(provider=MockProvider(), retriever=BM25Retriever(), max_iterations=3)
        base = agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")
        # Corrupt hop 1 to empty and let the agent react.
        prefix = [HopState(query=base.hop_queries[0], docs=[])]
        resumed = agent.resume_from_hops(QUESTION, CORPUS, prefix=prefix, reference_answer="Tim Cook")
        assert resumed.hop_docs[0] == []          # corruption preserved in the prefix
        assert resumed.iterations_used >= 2        # agent searched again

    def test_resume_default_start_hop(self):
        agent = LLMAgent(provider=MockProvider(), retriever=BM25Retriever())
        prefix = [HopState(query="q1", docs=["Tim Cook is the CEO of Apple Inc."])]
        resumed = agent.resume_from_hops(QUESTION, CORPUS, prefix=prefix, reference_answer="Tim Cook")
        assert resumed.hop_queries[0] == "q1"

    def test_force_answer_uses_only_prefix_evidence(self):
        agent = LLMAgent(provider=MockProvider(), retriever=BM25Retriever())
        prefix = [HopState(query=QUESTION, docs=["Tim Cook is the CEO of Apple Inc."])]
        t = agent.force_answer(QUESTION, prefix=prefix, reference_answer="Tim Cook")
        assert t.iterations_used == 1
        assert t.hop_docs == [["Tim Cook is the CEO of Apple Inc."]]
        assert "Tim Cook" in t.final_answer

    def test_force_answer_empty_prefix_yields_empty_answer(self):
        agent = LLMAgent(provider=MockProvider(), retriever=BM25Retriever())
        t = agent.force_answer(QUESTION, prefix=[], reference_answer="Tim Cook")
        assert t.final_answer == ""
