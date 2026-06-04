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


def test_batch_mix_accuracy():
    records = [
        _make_record(FailureStage.NONE),
        _make_record(FailureStage.NONE),
        _make_record(FailureStage.RETRIEVAL),
        _make_record(FailureStage.TOOL_CALL),
    ]
    acc = end_to_end_accuracy(records)
    assert abs(acc - 0.5) < 1e-9
