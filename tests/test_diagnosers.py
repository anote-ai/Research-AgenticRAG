"""Tests for agenticrag.diagnosers — baselines + the propagation-aware method (C3)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.agents import LLMAgent, MockProvider
from agenticrag.core import FailureRecord, FailureStage, PipelineTrace
from agenticrag.diagnosers import (
    Diagnosis,
    DoctorRAGDiagnoser,
    LLMJudgeDiagnoser,
    PropagationAwareDiagnoser,
    RuleBasedDiagnoser,
    batch_diagnose,
)
from agenticrag.retrievers import TokenOverlapRetriever

# A corpus where the CEO doc directly overlaps the question (so the Mock can
# actually answer it) — lets us build deterministic single-hop scenarios.
CORPUS = [
    "who is the ceo of apple tim cook",
    "apple makes the iphone device",
    "bananas are a yellow fruit",
    "eiffel tower paris france",
]
QUESTION = "who is the ceo of apple"
GOLD = "tim cook"


def _correct_trace() -> PipelineTrace:
    return PipelineTrace(
        query=QUESTION,
        retrieved_docs=["who is the ceo of apple tim cook"],
        tool_calls=[{"name": "retrieve", "args": {"q": QUESTION}, "iteration": 1}],
        final_answer="who is the ceo of apple tim cook",
        reference_answer=GOLD,
        hop_queries=[QUESTION],
        hop_docs=[["who is the ceo of apple tim cook"]],
        iterations_used=1,
    )


def _wrong_retrieval_trace() -> PipelineTrace:
    """A trace whose hop-1 retrieved off-topic docs and answered wrong."""
    return PipelineTrace(
        query=QUESTION,
        retrieved_docs=["bananas are a yellow fruit"],
        tool_calls=[{"name": "retrieve", "args": {"q": QUESTION}, "iteration": 1}],
        final_answer="bananas are a yellow fruit",
        reference_answer=GOLD,
        hop_queries=[QUESTION],
        hop_docs=[["bananas are a yellow fruit"]],
        iterations_used=1,
    )


def _ref():
    return {"answer": GOLD, "corpus": CORPUS}


# --------------------------------------------------------------------------- #
# RuleBasedDiagnoser
# --------------------------------------------------------------------------- #

class TestRuleBased:
    def test_empty_retrieval_attributed_to_retrieval(self):
        trace = PipelineTrace(
            query=QUESTION, retrieved_docs=[], tool_calls=[],
            final_answer="", reference_answer=GOLD,
            hop_queries=[QUESTION], hop_docs=[[]], iterations_used=1,
        )
        d = RuleBasedDiagnoser().diagnose(trace, _ref())
        assert d.stage == FailureStage.RETRIEVAL

    def test_wrong_answer_attributed_to_answer_generation(self):
        d = RuleBasedDiagnoser().diagnose(_wrong_retrieval_trace(), _ref())
        # Post-hoc: a wrong-but-grounded answer reads as an answer-stage fault,
        # missing the retrieval root cause — the weakness C3 addresses.
        assert d.stage == FailureStage.ANSWER_GENERATION
        assert d.predicted_hop == 0

    def test_zero_cost(self):
        d = RuleBasedDiagnoser().diagnose(_wrong_retrieval_trace(), _ref())
        assert d.cost_tokens == 0


# --------------------------------------------------------------------------- #
# DoctorRAGDiagnoser
# --------------------------------------------------------------------------- #

class TestDoctorRAG:
    def test_correct_trace_is_none(self):
        d = DoctorRAGDiagnoser().diagnose(_correct_trace(), _ref())
        assert d.stage == FailureStage.NONE

    def test_localizes_low_coverage_hop(self):
        d = DoctorRAGDiagnoser().diagnose(_wrong_retrieval_trace(), _ref())
        assert d.stage == FailureStage.RETRIEVAL
        assert d.predicted_hop == 1


# --------------------------------------------------------------------------- #
# LLMJudgeDiagnoser (offline mock path)
# --------------------------------------------------------------------------- #

class TestLLMJudge:
    def test_correct_trace_is_none(self):
        d = LLMJudgeDiagnoser(provider=MockProvider()).diagnose(_correct_trace(), _ref())
        assert d.stage == FailureStage.NONE

    def test_reports_cost(self):
        d = LLMJudgeDiagnoser(provider=MockProvider()).diagnose(_wrong_retrieval_trace(), _ref())
        assert d.cost_tokens > 0


# --------------------------------------------------------------------------- #
# PropagationAwareDiagnoser (C3)
# --------------------------------------------------------------------------- #

class TestPropagationAware:
    def _agent(self):
        return LLMAgent(provider=MockProvider(), retriever=TokenOverlapRetriever(), max_iterations=3)

    def test_correct_trace_is_none(self):
        agent = self._agent()
        d = PropagationAwareDiagnoser(agent).diagnose(_correct_trace(), _ref())
        assert d.stage == FailureStage.NONE
        assert d.predicted_hop == 0

    def test_localizes_retrieval_fault_via_counterfactual(self):
        agent = self._agent()
        d = PropagationAwareDiagnoser(agent).diagnose(_wrong_retrieval_trace(), _ref())
        # Repairing hop 1 (re-retrieving the CEO doc) flips the answer -> hop 1.
        assert d.stage == FailureStage.RETRIEVAL
        assert d.predicted_hop == 1

    def test_beats_rule_based_on_hop_localization(self):
        agent = self._agent()
        trace, ref = _wrong_retrieval_trace(), _ref()
        prop = PropagationAwareDiagnoser(agent).diagnose(trace, ref)
        rule = RuleBasedDiagnoser().diagnose(trace, ref)
        # Truth: the fault is at hop 1. Propagation-aware localizes it; rule-based
        # attributes to the answer level (hop 0).
        assert prop.predicted_hop == 1
        assert rule.predicted_hop != 1

    def test_reports_reexecution_cost(self):
        agent = self._agent()
        d = PropagationAwareDiagnoser(agent).diagnose(_wrong_retrieval_trace(), _ref())
        assert d.cost_tokens > 0

    def test_corpus_from_constructor_when_absent_in_ref(self):
        agent = self._agent()
        diag = PropagationAwareDiagnoser(agent, corpus=CORPUS)
        d = diag.diagnose(_wrong_retrieval_trace(), {"answer": GOLD})  # no corpus key
        assert d.predicted_hop == 1


# --------------------------------------------------------------------------- #
# Diagnosis adapters / batch
# --------------------------------------------------------------------------- #

class TestDiagnosisAdapters:
    def test_to_record_sets_root_cause_stage(self):
        diag = Diagnosis(
            trace_id="t", stage=FailureStage.RETRIEVAL,
            failure_type="irrelevant_retrieval", predicted_hop=2,
        )
        rec = diag.to_record()
        assert isinstance(rec, FailureRecord)
        assert rec.root_cause == FailureStage.RETRIEVAL.value

    def test_batch_diagnose(self):
        traces = [_correct_trace(), _wrong_retrieval_trace()]
        refs = [_ref(), _ref()]
        out = batch_diagnose(DoctorRAGDiagnoser(), traces, refs)
        assert len(out) == 2
        assert out[0].stage == FailureStage.NONE
