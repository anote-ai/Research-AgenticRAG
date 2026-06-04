"""Tests for agenticrag.evaluate."""

import pytest
from agenticrag.core import FailureStage, FailureRecord, PipelineTrace
from agenticrag.evaluate import (
    stage_attribution_rate,
    propagation_rate,
    failure_confusion_matrix,
    end_to_end_accuracy,
)


def _make_record(stage: FailureStage, propagated: bool = False) -> FailureRecord:
    return FailureRecord(
        trace_id="t",
        stage=stage,
        failure_type="test",
        propagated=propagated,
        root_cause="",
    )


def test_stage_attribution_rate():
    records = [
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.TOOL_CALL),
    ]
    rate = stage_attribution_rate(records, FailureStage.RETRIEVAL)
    assert abs(rate - 2 / 3) < 1e-9


def test_stage_attribution_rate_empty():
    assert stage_attribution_rate([], FailureStage.RETRIEVAL) == 0.0


def test_propagation_rate():
    records = [
        _make_record(FailureStage.RETRIEVAL, propagated=True),
        _make_record(FailureStage.RETRIEVAL, propagated=True),
        _make_record(FailureStage.TOOL_CALL, propagated=False),
    ]
    rate = propagation_rate(records)
    assert abs(rate - 2 / 3) < 1e-9


def test_propagation_rate_none():
    records = [_make_record(FailureStage.NONE, propagated=False)] * 3
    assert propagation_rate(records) == 0.0


def test_end_to_end_accuracy_full():
    traces = [
        PipelineTrace(trace_id="t1", query="q", final_answer="A", reference_answer="A"),
        PipelineTrace(trace_id="t2", query="q", final_answer="B", reference_answer="B"),
    ]
    assert end_to_end_accuracy(traces) == 1.0


def test_end_to_end_accuracy_partial():
    traces = [
        PipelineTrace(trace_id="t1", query="q", final_answer="A", reference_answer="A"),
        PipelineTrace(trace_id="t2", query="q", final_answer="wrong", reference_answer="B"),
    ]
    assert end_to_end_accuracy(traces) == 0.5


def test_end_to_end_accuracy_empty():
    assert end_to_end_accuracy([]) == 0.0


def test_failure_confusion_matrix():
    pred = ["retrieval", "tool_call", "retrieval"]
    true = ["retrieval", "retrieval", "tool_call"]
    matrix = failure_confusion_matrix(pred, true)
    assert matrix["retrieval"]["retrieval"] == 1
    assert matrix["retrieval"]["tool_call"] == 1
    assert matrix["tool_call"]["retrieval"] == 1


def test_failure_confusion_matrix_mismatch():
    with pytest.raises(ValueError):
        failure_confusion_matrix(["a"], ["a", "b"])
