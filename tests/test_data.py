"""Tests for agenticrag.data — 4 tests."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.core import PipelineTrace, DiagnosticBenchmark, FailureStage
from agenticrag.data import make_trace, make_dataset


def test_make_trace_success_returns_pipeline_trace():
    trace, ref = make_trace(success=True)
    assert isinstance(trace, PipelineTrace)
    assert trace.final_answer == ref["answer"]


def test_make_dataset_total_length():
    traces, refs = make_dataset(n_success=10, n_retrieval_fail=5, n_tool_fail=3, n_answer_fail=7)
    assert len(traces) == 25
    assert len(refs) == 25


def test_make_trace_retrieval_failure():
    trace, ref = make_trace(success=False, failure_stage="retrieval")
    assert trace.retrieved_docs == []


def test_diagnose_dataset_has_correct_failure_counts():
    traces, refs = make_dataset(n_success=5, n_retrieval_fail=3, n_tool_fail=2, n_answer_fail=4)
    bench = DiagnosticBenchmark()
    records = bench.batch_diagnose(traces, refs)
    n_success = sum(1 for r in records if r.stage == FailureStage.NONE)
    n_retrieval = sum(1 for r in records if r.stage == FailureStage.RETRIEVAL)
    assert n_success == 5
    assert n_retrieval == 3
