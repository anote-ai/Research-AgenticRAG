from __future__ import annotations

from agenticrag.core import FailureRecord, FailureStage
from agenticrag.data import make_multi_hop_trace
from agenticrag.injection import InjectionSpec, inject_failure
from agenticrag.core import FailureType
from agenticrag.propagation import PropagationGraph


def test_propagation_graph_counts_edges_from_labels() -> None:
    trace, _ = make_multi_hop_trace(question_idx=0, seed=17)
    labels = [
        inject_failure(
            trace,
            InjectionSpec(
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.EMPTY_RETRIEVAL,
                hop=1,
            ),
        ).label,
        inject_failure(
            trace,
            InjectionSpec(
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.EMPTY_RETRIEVAL,
                hop=2,
            ),
        ).label,
        inject_failure(
            trace,
            InjectionSpec(
                stage=FailureStage.ANSWER_GENERATION,
                failure_type=FailureType.EMPTY_ANSWER,
                hop=1,
            ),
        ).label,
    ]

    graph = PropagationGraph.from_labels(labels)

    assert graph.edge_count(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) == 2
    assert graph.edge_count(FailureStage.ANSWER_GENERATION, FailureStage.ANSWER_GENERATION) == 1
    assert graph.edge_probability(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) == 1.0


def test_propagation_graph_transition_matrix_uses_stage_values() -> None:
    graph = PropagationGraph()
    graph.add_edge(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION, count=3)
    graph.add_edge(FailureStage.RETRIEVAL, FailureStage.TOOL_CALL, count=1)

    matrix = graph.transition_matrix()

    assert matrix["retrieval"]["answer_generation"] == 0.75
    assert matrix["retrieval"]["tool_call"] == 0.25


def test_propagation_graph_fits_records_using_root_cause() -> None:
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
            stage=FailureStage.TOOL_CALL,
            failure_type="no_tool_calls",
            propagated=False,
            root_cause="tool_call",
        ),
        FailureRecord(
            trace_id="t3",
            stage=FailureStage.NONE,
            failure_type="success",
        ),
    ]

    graph = PropagationGraph.from_records(records)

    assert graph.edge_count(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) == 1
    assert graph.edge_count(FailureStage.TOOL_CALL, FailureStage.TOOL_CALL) == 1
    assert graph.outgoing_total(FailureStage.NONE) == 0
