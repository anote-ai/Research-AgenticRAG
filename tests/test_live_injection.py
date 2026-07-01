"""Tests for agenticrag.injection.LiveFailureInjector (W2 — interventional injection)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.agents import LLMAgent, MockProvider
from agenticrag.core import FailureStage, FailureType, PipelineTrace
from agenticrag.injection import LIVE_INJECTIONS, LiveFailureInjector
from agenticrag.retrievers import BM25Retriever

CORPUS = [
    "Apple Inc. produces the iPhone.",
    "Tim Cook is the CEO of Apple Inc.",
    "Bananas are a good source of potassium.",
    "The Eiffel Tower is in Paris.",
]
QUESTION = "Who is the CEO of the company that produces the iPhone?"


def _agent():
    return LLMAgent(provider=MockProvider(), retriever=BM25Retriever(), max_iterations=3)


def _base(agent):
    return agent.run(QUESTION, CORPUS, reference_answer="Tim Cook")


class TestLiveInjectionSchema:
    def test_all_methods_listed_are_callable(self):
        agent = _agent()
        inj = LiveFailureInjector(agent)
        base = _base(agent)
        for name in LIVE_INJECTIONS:
            res = getattr(inj, name)(base, CORPUS, hop=1)
            assert isinstance(res.injected_trace, PipelineTrace)
            assert res.injected_at_hop >= 1
            assert res.original_trace_id == base.trace_id
            assert res.injected_trace.trace_id == base.trace_id

    def test_failure_types_match_methods(self):
        agent = _agent()
        inj = LiveFailureInjector(agent)
        base = _base(agent)
        expected = {
            "inject_empty_retrieval": FailureType.EMPTY_RETRIEVAL,
            "inject_irrelevant_docs": FailureType.IRRELEVANT_RETRIEVAL,
            "inject_query_drift": FailureType.QUERY_DRIFT,
            "inject_false_premise": FailureType.FALSE_PREMISE,
            "inject_stale_evidence": FailureType.STALE_EVIDENCE,
            "inject_early_termination": FailureType.EARLY_TERMINATION,
        }
        for name, ftype in expected.items():
            res = getattr(inj, name)(base, CORPUS, hop=1)
            assert res.injected_failure_type == ftype
            assert res.injected_stage == FailureStage.RETRIEVAL


class TestLiveInjectionBehavior:
    def test_empty_retrieval_clears_target_hop_in_prefix(self):
        agent = _agent()
        base = _base(agent)
        res = LiveFailureInjector(agent).inject_empty_retrieval(base, CORPUS, hop=1)
        # The injected hop-1 evidence is empty; the agent then re-ran the suffix.
        assert res.injected_trace.hop_docs[0] == []

    def test_irrelevant_docs_places_noise_at_hop(self):
        agent = _agent()
        base = _base(agent)
        noise = ["Completely unrelated penguin document."]
        res = LiveFailureInjector(agent).inject_irrelevant_docs(
            base, CORPUS, hop=1, noise_docs=noise
        )
        assert res.injected_trace.hop_docs[0] == noise

    def test_false_premise_injects_premise_doc(self):
        agent = _agent()
        base = _base(agent)
        res = LiveFailureInjector(agent).inject_false_premise(
            base, CORPUS, hop=1, premise="FALSE: nothing is true here."
        )
        assert "FALSE: nothing is true here." in res.injected_trace.hop_docs[0]

    def test_does_not_mutate_original_trace(self):
        agent = _agent()
        base = _base(agent)
        original = [list(h) for h in base.hop_docs]
        LiveFailureInjector(agent).inject_empty_retrieval(base, CORPUS, hop=1)
        assert base.hop_docs == original

    def test_hop_is_clamped_to_trace_depth(self):
        agent = _agent()
        base = _base(agent)
        # base is typically 1 hop; injecting at hop 9 clamps in-range.
        res = LiveFailureInjector(agent).inject_empty_retrieval(base, CORPUS, hop=9)
        assert 1 <= res.injected_at_hop <= max(1, base.iterations_used)

    def test_early_termination_truncates_and_answers(self):
        agent = _agent()
        base = _base(agent)
        res = LiveFailureInjector(agent).inject_early_termination(base, CORPUS, hop=1)
        # Truncated to zero prior hops -> answers from no evidence.
        assert res.injected_trace.iterations_used <= base.iterations_used
        assert res.injected_failure_type == FailureType.EARLY_TERMINATION
