"""Tests for agenticrag.injection — FailureInjector and injection_sensitivity."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agenticrag.core import DiagnosticBenchmark, FailureStage, FailureType, PipelineTrace
from agenticrag.injection import FailureInjector, InjectionResult, injection_sensitivity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_success_trace(
    query: str = "What is revenue?",
    answer: str = "Revenue is $500 million.",
    hops: int = 2,
) -> PipelineTrace:
    hop_docs = [
        ["Doc A: Financial report.", "Doc B: Revenue data."],
        ["Doc C: Earnings summary."],
    ][:hops]
    hop_queries = [query] + [f"{query} [follow-up]"] * (hops - 1)
    return PipelineTrace(
        trace_id="trace-success",
        query=query,
        retrieved_docs=["Doc A: Financial report.", "Doc B: Revenue data.", "Doc C: Earnings summary."],
        tool_calls=[{"name": "retrieve", "args": {"q": q}} for q in hop_queries],
        final_answer=answer,
        reference_answer=answer,
        hop_queries=hop_queries,
        hop_docs=hop_docs,
        iterations_used=hops,
    )


def _ref(trace: PipelineTrace) -> dict:
    return {"answer": trace.reference_answer}


# ---------------------------------------------------------------------------
# inject_empty_retrieval
# ---------------------------------------------------------------------------

class TestInjectEmptyRetrieval:
    def test_clears_target_hop_docs(self):
        trace = _make_success_trace(hops=2)
        injector = FailureInjector()
        result = injector.inject_empty_retrieval(trace, hop=2)

        assert result.injected_trace.hop_docs[1] == []

    def test_preserves_earlier_hop_docs(self):
        trace = _make_success_trace(hops=2)
        injector = FailureInjector()
        result = injector.inject_empty_retrieval(trace, hop=2)

        assert result.injected_trace.hop_docs[0] == trace.hop_docs[0]

    def test_clears_all_subsequent_hops(self):
        trace = _make_success_trace(hops=2)
        injector = FailureInjector()
        result = injector.inject_empty_retrieval(trace, hop=1)

        assert result.injected_trace.hop_docs[0] == []
        assert result.injected_trace.hop_docs[1] == []

    def test_clears_answer_when_no_docs_survive(self):
        trace = _make_success_trace(hops=2)
        injector = FailureInjector()
        result = injector.inject_empty_retrieval(trace, hop=1)

        assert result.injected_trace.final_answer == ""
        assert result.injected_trace.retrieved_docs == []

    def test_preserves_answer_when_some_docs_survive(self):
        trace = _make_success_trace(hops=2)
        injector = FailureInjector()
        result = injector.inject_empty_retrieval(trace, hop=2)

        assert result.injected_trace.final_answer != ""
        assert len(result.injected_trace.retrieved_docs) > 0

    def test_does_not_mutate_original(self):
        trace = _make_success_trace(hops=2)
        original_docs = [list(h) for h in trace.hop_docs]
        FailureInjector().inject_empty_retrieval(trace, hop=1)
        assert trace.hop_docs == original_docs

    def test_metadata_fields(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_empty_retrieval(trace, hop=1)
        assert result.original_trace_id == trace.trace_id
        assert result.injected_stage == FailureStage.RETRIEVAL
        assert result.injected_failure_type == FailureType.EMPTY_RETRIEVAL
        assert result.injected_at_hop == 1


# ---------------------------------------------------------------------------
# inject_irrelevant_docs
# ---------------------------------------------------------------------------

class TestInjectIrrelevantDocs:
    NOISE = ["Unrelated doc about penguins.", "Another off-topic doc."]

    def test_replaces_docs_at_target_hop(self):
        trace = _make_success_trace(hops=2)
        result = FailureInjector().inject_irrelevant_docs(trace, self.NOISE, hop=1)
        assert result.injected_trace.hop_docs[0] == self.NOISE

    def test_later_hops_untouched(self):
        trace = _make_success_trace(hops=2)
        result = FailureInjector().inject_irrelevant_docs(trace, self.NOISE, hop=1)
        assert result.injected_trace.hop_docs[1] == trace.hop_docs[1]

    def test_answer_untouched(self):
        trace = _make_success_trace(hops=2)
        result = FailureInjector().inject_irrelevant_docs(trace, self.NOISE, hop=1)
        assert result.injected_trace.final_answer == trace.final_answer

    def test_metadata_fields(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_irrelevant_docs(trace, self.NOISE, hop=1)
        assert result.injected_stage == FailureStage.RETRIEVAL
        assert result.injected_failure_type == FailureType.IRRELEVANT_RETRIEVAL
        assert result.injected_at_hop == 1


# ---------------------------------------------------------------------------
# inject_no_tool_calls
# ---------------------------------------------------------------------------

class TestInjectNoToolCalls:
    def test_clears_all_tool_calls(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_no_tool_calls(trace)
        assert result.injected_trace.tool_calls == []

    def test_retrieval_unchanged(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_no_tool_calls(trace)
        assert result.injected_trace.retrieved_docs == trace.retrieved_docs

    def test_metadata_fields(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_no_tool_calls(trace)
        assert result.injected_stage == FailureStage.TOOL_CALL
        assert result.injected_failure_type == FailureType.NO_TOOL_CALLS
        assert result.injected_at_hop == 0


# ---------------------------------------------------------------------------
# inject_empty_answer
# ---------------------------------------------------------------------------

class TestInjectEmptyAnswer:
    def test_clears_final_answer(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_empty_answer(trace)
        assert result.injected_trace.final_answer == ""

    def test_retrieval_unchanged(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_empty_answer(trace)
        assert result.injected_trace.retrieved_docs == trace.retrieved_docs

    def test_metadata_fields(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_empty_answer(trace)
        assert result.injected_stage == FailureStage.ANSWER_GENERATION
        assert result.injected_failure_type == FailureType.EMPTY_ANSWER
        assert result.injected_at_hop == 0


# ---------------------------------------------------------------------------
# inject_hallucinated_answer
# ---------------------------------------------------------------------------

class TestInjectHallucinatedAnswer:
    def test_replaces_answer_with_fabrication(self):
        trace = _make_success_trace()
        fabricated = "Xylophone revenues exceeded expectations last quarter."
        result = FailureInjector().inject_hallucinated_answer(trace, fabricated)
        assert result.injected_trace.final_answer == fabricated

    def test_uses_default_fabrication_when_not_specified(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_hallucinated_answer(trace)
        assert result.injected_trace.final_answer != ""
        assert result.injected_trace.final_answer != trace.final_answer

    def test_retrieved_docs_untouched(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_hallucinated_answer(trace)
        assert result.injected_trace.retrieved_docs == trace.retrieved_docs

    def test_metadata_fields(self):
        trace = _make_success_trace()
        result = FailureInjector().inject_hallucinated_answer(trace)
        assert result.injected_stage == FailureStage.ANSWER_GENERATION
        assert result.injected_failure_type == FailureType.HALLUCINATION
        assert result.injected_at_hop == 0


# ---------------------------------------------------------------------------
# injection_sensitivity
# ---------------------------------------------------------------------------

class TestInjectionSensitivity:
    def _clean_traces_and_refs(self, n: int = 4):
        traces = [_make_success_trace() for _ in range(n)]
        # Give each a unique trace_id to avoid confusion
        for i, t in enumerate(traces):
            object.__setattr__(t, "trace_id", f"trace-{i}")
        refs = [_ref(t) for t in traces]
        return traces, refs

    def test_empty_retrieval_detected_at_100_pct(self):
        traces, refs = self._clean_traces_and_refs()
        injector = FailureInjector()
        benchmark = DiagnosticBenchmark()
        sens = injection_sensitivity(
            traces, refs, injector, benchmark, method="inject_empty_retrieval", hop=1
        )
        assert sens == 1.0

    def test_empty_answer_detected_at_100_pct(self):
        traces, refs = self._clean_traces_and_refs()
        sens = injection_sensitivity(
            traces, refs, FailureInjector(), DiagnosticBenchmark(),
            method="inject_empty_answer",
        )
        assert sens == 1.0

    def test_no_tool_calls_detected_at_100_pct(self):
        # Build traces with no retrieved docs so tool-call failure is primary
        traces = []
        for i in range(3):
            t = PipelineTrace(
                trace_id=f"t-{i}",
                query="q",
                retrieved_docs=["some doc"],
                tool_calls=[{"name": "retrieve", "args": {}}],
                final_answer="some doc answer",
                reference_answer="some doc answer",
                hop_queries=["q"],
                hop_docs=[["some doc"]],
                iterations_used=1,
            )
            traces.append(t)
        refs = [{"answer": t.reference_answer} for t in traces]
        sens = injection_sensitivity(
            traces, refs, FailureInjector(), DiagnosticBenchmark(),
            method="inject_no_tool_calls",
        )
        assert sens == 1.0

    def test_returns_zero_for_empty_input(self):
        sens = injection_sensitivity(
            [], [], FailureInjector(), DiagnosticBenchmark(),
            method="inject_empty_retrieval",
        )
        assert sens == 0.0

    def test_raises_on_length_mismatch(self):
        traces, refs = self._clean_traces_and_refs(n=3)
        with pytest.raises(ValueError):
            injection_sensitivity(
                traces, refs[:2], FailureInjector(), DiagnosticBenchmark(),
                method="inject_empty_retrieval",
            )

    def test_sensitivity_is_float_in_unit_interval(self):
        traces, refs = self._clean_traces_and_refs(n=5)
        sens = injection_sensitivity(
            traces, refs, FailureInjector(), DiagnosticBenchmark(),
            method="inject_empty_retrieval", hop=1,
        )
        assert 0.0 <= sens <= 1.0
