"""Tests for agenticrag.core."""

import pytest
from agenticrag.core import (
    FailureStage,
    PipelineTrace,
    FailureRecord,
    DiagnosticBenchmark,
)


def test_failure_stage_values():
    assert FailureStage.RETRIEVAL.value == "retrieval"
    assert FailureStage.TOOL_CALL.value == "tool_call"
    assert FailureStage.ANSWER_GENERATION.value == "answer_generation"
    assert FailureStage.NONE.value == "none"


def test_failure_stage_members():
    stages = {s.value for s in FailureStage}
    assert stages == {"retrieval", "tool_call", "answer_generation", "none"}


def test_pipeline_trace_construction():
    trace = PipelineTrace(
        trace_id="t1",
        query="What is the revenue?",
        retrieved_docs=["doc1", "doc2"],
        tool_calls=[{"name": "search", "args": {}}],
        final_answer="$100M",
        reference_answer="$100M",
    )
    assert trace.trace_id == "t1"
    assert len(trace.retrieved_docs) == 2
    assert len(trace.tool_calls) == 1


def test_pipeline_trace_defaults():
    trace = PipelineTrace(
        trace_id="t2",
        query="Who is the CEO?",
        final_answer="Alice",
        reference_answer="Alice",
    )
    assert trace.retrieved_docs == []
    assert trace.tool_calls == []


def test_failure_record_construction():
    rec = FailureRecord(
        trace_id="t1",
        stage=FailureStage.RETRIEVAL,
        failure_type="empty_retrieval",
        propagated=True,
        root_cause="No documents retrieved.",
    )
    assert rec.stage == FailureStage.RETRIEVAL
    assert rec.propagated is True


def test_attribute_failures_counts():
    records = [
        FailureRecord(trace_id="t1", stage=FailureStage.RETRIEVAL,
                      failure_type="empty", propagated=True, root_cause=""),
        FailureRecord(trace_id="t2", stage=FailureStage.RETRIEVAL,
                      failure_type="empty", propagated=True, root_cause=""),
        FailureRecord(trace_id="t3", stage=FailureStage.TOOL_CALL,
                      failure_type="error", propagated=False, root_cause=""),
        FailureRecord(trace_id="t4", stage=FailureStage.NONE,
                      failure_type="none", propagated=False, root_cause=""),
    ]
    benchmark = DiagnosticBenchmark()
    counts = benchmark.attribute_failures(records)
    assert counts["retrieval"] == 2
    assert counts["tool_call"] == 1
    assert counts["none"] == 1
    assert counts["answer_generation"] == 0


def test_diagnose_trace_empty_retrieval():
    trace = PipelineTrace(
        trace_id="t1", query="q", retrieved_docs=[],
        final_answer="ans", reference_answer="ans",
    )
    bench = DiagnosticBenchmark()
    rec = bench.diagnose_trace(trace, {})
    assert rec.stage == FailureStage.RETRIEVAL


def test_diagnose_trace_correct():
    trace = PipelineTrace(
        trace_id="t2", query="q", retrieved_docs=["doc"],
        final_answer="correct", reference_answer="correct",
    )
    bench = DiagnosticBenchmark()
    rec = bench.diagnose_trace(trace, {})
    assert rec.stage == FailureStage.NONE
