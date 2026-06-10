"""Tests for agenticrag.evaluate — 7 tests."""
from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.core import FailureRecord, FailureStage
from agenticrag.evaluate import (
    stage_attribution_rate,
    propagation_rate,
    end_to_end_accuracy,
    severity_weighted_failure_rate,
    failure_confusion_matrix,
    root_cause_accuracy,
    root_cause_stage,
)


def _make_record(stage: FailureStage, propagated: bool = False, severity: float = 0.5) -> FailureRecord:
    ft = "success" if stage == FailureStage.NONE else "error"
    return FailureRecord(trace_id="t", stage=stage, failure_type=ft, propagated=propagated, severity=severity)


def test_stage_attribution_rate():
    records = [
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.TOOL_CALL),
    ]
    rate = stage_attribution_rate(records, FailureStage.RETRIEVAL)
    assert abs(rate - 2/3) < 1e-9


def test_propagation_rate():
    records = [
        _make_record(FailureStage.RETRIEVAL, propagated=True),
        _make_record(FailureStage.NONE, propagated=False),
    ]
    assert abs(propagation_rate(records) - 0.5) < 1e-9


def test_end_to_end_accuracy_all_none():
    records = [_make_record(FailureStage.NONE) for _ in range(5)]
    assert end_to_end_accuracy(records) == 1.0


def test_end_to_end_accuracy_no_none():
    records = [_make_record(FailureStage.RETRIEVAL) for _ in range(5)]
    assert end_to_end_accuracy(records) == 0.0


def test_severity_weighted_failure_rate():
    records = [
        _make_record(FailureStage.RETRIEVAL, severity=0.8),
        _make_record(FailureStage.TOOL_CALL, severity=0.6),
    ]
    rate = severity_weighted_failure_rate(records)
    assert abs(rate - 0.7) < 1e-9


def test_failure_confusion_matrix_structure():
    predicted = ["retrieval", "tool_call", "retrieval"]
    true = ["retrieval", "retrieval", "tool_call"]
    cm = failure_confusion_matrix(predicted, true)
    assert "retrieval" in cm
    assert "tool_call" in cm
    assert "tp" in cm["retrieval"]
    assert "fp" in cm["retrieval"]
    assert "fn" in cm["retrieval"]


def test_root_cause_stage_uses_canonical_root_cause_string():
    record = FailureRecord(
        trace_id="t1",
        stage=FailureStage.ANSWER_GENERATION,
        failure_type="empty_retrieval",
        propagated=True,
        root_cause="retrieval",
    )

    assert root_cause_stage(record) == FailureStage.RETRIEVAL


def test_root_cause_stage_falls_back_to_record_stage_for_descriptive_text():
    record = FailureRecord(
        trace_id="t1",
        stage=FailureStage.TOOL_CALL,
        failure_type="no_tool_calls",
        propagated=True,
        root_cause="No tool calls made",
    )

    assert root_cause_stage(record) == FailureStage.TOOL_CALL


def test_root_cause_accuracy_scores_stage_values():
    records = [
        FailureRecord(
            trace_id="t1",
            stage=FailureStage.ANSWER_GENERATION,
            failure_type="empty_retrieval",
            propagated=True,
            root_cause="retrieval",
        ),
        FailureRecord(
            trace_id="t2",
            stage=FailureStage.ANSWER_GENERATION,
            failure_type="empty_answer",
            propagated=False,
            root_cause="answer_generation",
        ),
        FailureRecord(
            trace_id="t3",
            stage=FailureStage.TOOL_CALL,
            failure_type="no_tool_calls",
            propagated=True,
            root_cause="tool_call",
        ),
    ]

    acc = root_cause_accuracy(
        records,
        [
            FailureStage.RETRIEVAL,
            FailureStage.ANSWER_GENERATION,
            FailureStage.RETRIEVAL,
        ],
    )

    assert abs(acc - 2 / 3) < 1e-9


def test_root_cause_accuracy_can_exclude_successes():
    records = [
        FailureRecord(
            trace_id="t1",
            stage=FailureStage.NONE,
            failure_type="success",
        ),
        FailureRecord(
            trace_id="t2",
            stage=FailureStage.ANSWER_GENERATION,
            failure_type="empty_retrieval",
            propagated=True,
            root_cause="retrieval",
        ),
    ]

    acc = root_cause_accuracy(
        records,
        [FailureStage.NONE, FailureStage.RETRIEVAL],
        include_success=False,
    )

    assert acc == 1.0


def test_root_cause_accuracy_requires_equal_lengths():
    records = [_make_record(FailureStage.RETRIEVAL)]

    with pytest.raises(ValueError):
        root_cause_accuracy(records, [])


def test_batch_mix_accuracy():
    records = [
        _make_record(FailureStage.NONE),
        _make_record(FailureStage.NONE),
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.TOOL_CALL),
    ]
    acc = end_to_end_accuracy(records)
    assert abs(acc - 0.5) < 1e-9
