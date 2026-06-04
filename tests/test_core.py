"""Tests for agenticrag.core — 10 tests."""
from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.core import (
    FailureStage,
    PipelineTrace,
    FailureRecord,
    DiagnosticBenchmark,
)


def test_failure_stage_values():
    assert FailureStage.RETRIEVAL == "retrieval"
    assert FailureStage.TOOL_CALL == "tool_call"
    assert FailureStage.ANSWER_GENERATION == "answer_generation"
    assert FailureStage.NONE == "none"


def test_pipeline_trace_construction_and_default_trace_id():
    trace = PipelineTrace(
        query="test",
        retrieved_docs=["doc1"],
        tool_calls=[{"name": "search"}],
        final_answer="answer",
        reference_answer="answer",
    )
    assert trace.trace_id  # non-empty
    assert trace.query == "test"


def test_failure_record_severity_validator_raises():
    with pytest.raises(Exception):
        FailureRecord(
            trace_id="t1",
            stage=FailureStage.RETRIEVAL,
            failure_type="test",
            severity=1.5,
        )


def test_diagnose_trace_empty_retrieval():
    bench = DiagnosticBenchmark()
    trace = PipelineTrace(
        trace_id="t1",
        query="q",
        retrieved_docs=[],
        tool_calls=[],
        final_answer="",
        reference_answer="ref",
    )
    record = bench.diagnose_trace(trace, {"answer": "ref"})
    assert record.stage == FailureStage.RETRIEVAL
    assert record.failure_type == "empty_retrieval"
    assert record.propagated is True


def test_diagnose_trace_no_tool_calls():
    bench = DiagnosticBenchmark()
    trace = PipelineTrace(
        trace_id="t2",
        query="q",
        retrieved_docs=["doc1"],
        tool_calls=[],
        final_answer="",
        reference_answer="ref",
    )
    record = bench.diagnose_trace(trace, {"answer": "ref"})
    assert record.stage == FailureStage.TOOL_CALL
    assert record.failure_type == "no_tool_calls"


def test_diagnose_trace_empty_answer():
    bench = DiagnosticBenchmark()
    trace = PipelineTrace(
        trace_id="t3",
        query="q",
        retrieved_docs=["doc1"],
        tool_calls=[{"name": "search"}],
        final_answer="   ",
        reference_answer="ref",
    )
    record = bench.diagnose_trace(trace, {"answer": "ref"})
    assert record.stage == FailureStage.ANSWER_GENERATION
    assert record.failure_type == "empty_answer"


def test_diagnose_trace_success():
    bench = DiagnosticBenchmark()
    # Answer must share tokens with retrieved doc to avoid hallucination detection.
    trace = PipelineTrace(
        trace_id="t4",
        query="q",
        retrieved_docs=["CompanyA revenue was 500 million in fiscal year 2023."],
        tool_calls=[{"name": "search"}],
        final_answer="CompanyA revenue was 500 million.",
        reference_answer="CompanyA revenue was 500 million.",
    )
    record = bench.diagnose_trace(trace, {"answer": "CompanyA revenue was 500 million."})
    assert record.stage == FailureStage.NONE
    assert record.failure_type == "success"


def test_batch_diagnose_length():
    bench = DiagnosticBenchmark()
    traces = [
        PipelineTrace(
            trace_id=f"t{i}",
            query="q",
            retrieved_docs=["doc"],
            tool_calls=[{"name": "search"}],
            final_answer="correct",
            reference_answer="correct",
        )
        for i in range(5)
    ]
    refs = [{"answer": "correct"}] * 5
    records = bench.batch_diagnose(traces, refs)
    assert len(records) == 5


def test_attribute_failures_structure():
    bench = DiagnosticBenchmark()
    records = [
        FailureRecord(trace_id="t1", stage=FailureStage.RETRIEVAL, failure_type="empty_retrieval", propagated=True),
        FailureRecord(trace_id="t2", stage=FailureStage.NONE, failure_type="success"),
    ]
    attr = bench.attribute_failures(records)
    assert "by_stage" in attr
    assert "total_failures" in attr
    assert "propagation_rate" in attr
    assert attr["total_failures"] == 1


def test_attribute_failures_propagation_rate():
    bench = DiagnosticBenchmark()
    records = [
        FailureRecord(trace_id="t1", stage=FailureStage.RETRIEVAL, failure_type="er", propagated=True),
        FailureRecord(trace_id="t2", stage=FailureStage.RETRIEVAL, failure_type="er", propagated=True),
        FailureRecord(trace_id="t3", stage=FailureStage.NONE, failure_type="success", propagated=False),
    ]
    attr = bench.attribute_failures(records)
    assert abs(attr["propagation_rate"] - 2/3) < 1e-9
