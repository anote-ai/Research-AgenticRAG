from __future__ import annotations

import pytest

from agenticrag.core import FailureStage, FailureType
from agenticrag.data import make_multi_hop_trace
from agenticrag.injection import (
    InjectionSpec,
    group_records_by_hop,
    inject_failure,
    inject_failures,
    make_injection_grid,
)


def test_inject_retrieval_failure_labels_root_and_observed_stage() -> None:
    trace, _ = make_multi_hop_trace(question_idx=0, seed=7)
    injected = inject_failure(
        trace,
        InjectionSpec(
            stage=FailureStage.RETRIEVAL,
            failure_type=FailureType.EMPTY_RETRIEVAL,
            hop=2,
            severity=0.9,
        ),
    )

    assert injected.trace.hop_docs[1] == []
    assert injected.trace.final_answer == ""
    assert injected.label.root_cause_stage == FailureStage.RETRIEVAL
    assert injected.label.observed_stage == FailureStage.ANSWER_GENERATION
    assert injected.label.propagated is True

    record = injected.record
    assert record.stage == FailureStage.ANSWER_GENERATION
    assert record.root_cause == "retrieval"
    assert record.failure_type == "empty_retrieval"


def test_inject_tool_failure_removes_target_hop_call() -> None:
    trace, _ = make_multi_hop_trace(question_idx=1, seed=9)
    injected = inject_failure(
        trace,
        InjectionSpec(
            stage=FailureStage.TOOL_CALL,
            failure_type=FailureType.NO_TOOL_CALLS,
            hop=1,
            severity=0.7,
        ),
    )

    assert injected.trace.final_answer == ""
    assert all(call.get("iteration") != 1 for call in injected.trace.tool_calls)
    assert injected.label.root_cause_stage == FailureStage.TOOL_CALL
    assert injected.label.observed_stage == FailureStage.ANSWER_GENERATION


def test_inject_answer_generation_failure_is_local() -> None:
    trace, _ = make_multi_hop_trace(question_idx=2, seed=11)
    injected = inject_failure(
        trace,
        InjectionSpec(
            stage=FailureStage.ANSWER_GENERATION,
            failure_type=FailureType.HALLUCINATION,
            hop=1,
        ),
    )

    assert injected.trace.final_answer != trace.final_answer
    assert injected.label.root_cause_stage == FailureStage.ANSWER_GENERATION
    assert injected.label.observed_stage == FailureStage.ANSWER_GENERATION
    assert injected.label.propagated is False


def test_make_injection_grid_and_group_records_by_hop() -> None:
    trace, _ = make_multi_hop_trace(question_idx=0, seed=13)
    specs = make_injection_grid(
        max_hops=2,
        stages=[FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION],
    )
    injected = inject_failures(trace, specs)
    grouped = group_records_by_hop(injected)

    assert len(specs) == 4
    assert sorted(grouped) == [1, 2]
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 2


def test_injection_spec_rejects_zero_hop() -> None:
    with pytest.raises(ValueError):
        InjectionSpec(
            stage=FailureStage.RETRIEVAL,
            failure_type=FailureType.EMPTY_RETRIEVAL,
            hop=0,
        )
