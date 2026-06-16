"""Tests for agenticrag.propagation — PropagationGraph and PropagationEdge."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agenticrag.core import FailureRecord, FailureStage, FailureType
from agenticrag.propagation import PropagationEdge, PropagationGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(stage: FailureStage, failure_type: FailureType = FailureType.EMPTY_RETRIEVAL) -> FailureRecord:
    return FailureRecord(
        trace_id="t",
        stage=stage,
        failure_type=failure_type,
        propagated=(stage != FailureStage.NONE),
        severity=0.5 if stage != FailureStage.NONE else 0.0,
    )


def _success() -> FailureRecord:
    return _record(FailureStage.NONE, FailureType.SUCCESS)


def _retrieval_failure() -> FailureRecord:
    return _record(FailureStage.RETRIEVAL, FailureType.EMPTY_RETRIEVAL)


def _answer_failure() -> FailureRecord:
    return _record(FailureStage.ANSWER_GENERATION, FailureType.EMPTY_ANSWER)


def _tool_failure() -> FailureRecord:
    return _record(FailureStage.TOOL_CALL, FailureType.NO_TOOL_CALLS)


# ---------------------------------------------------------------------------
# PropagationEdge
# ---------------------------------------------------------------------------

class TestPropagationEdge:
    def test_frozen(self):
        edge = PropagationEdge(
            source_stage=FailureStage.RETRIEVAL,
            target_stage=FailureStage.ANSWER_GENERATION,
            weight=0.8,
            count=4,
        )
        with pytest.raises((AttributeError, TypeError)):
            edge.weight = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        edge = PropagationEdge(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION, 0.75, 3)
        assert edge.source_stage == FailureStage.RETRIEVAL
        assert edge.target_stage == FailureStage.ANSWER_GENERATION
        assert edge.weight == 0.75
        assert edge.count == 3


# ---------------------------------------------------------------------------
# PropagationGraph — empty state
# ---------------------------------------------------------------------------

class TestPropagationGraphEmpty:
    def setup_method(self):
        self.g = PropagationGraph()

    def test_edges_empty(self):
        assert self.g.edges() == []

    def test_stage_coupling_zero_when_no_data(self):
        assert self.g.stage_coupling(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) == 0.0

    def test_propagation_depth_zero(self):
        assert self.g.propagation_depth() == 0.0

    def test_propagation_depth_with_stage_filter_zero(self):
        assert self.g.propagation_depth(FailureStage.RETRIEVAL) == 0.0

    def test_critical_path_empty(self):
        assert self.g.critical_path() == []

    def test_failure_rate_by_stage_empty(self):
        assert self.g.failure_rate_by_stage() == {}

    def test_self_correction_rate_zero(self):
        assert self.g.self_correction_rate() == 0.0

    def test_summary_keys(self):
        s = self.g.summary()
        expected_keys = {
            "total_traces", "failure_rate_by_stage", "coupling_matrix",
            "mean_propagation_depth", "critical_path", "self_correction_rate", "n_edges",
        }
        assert set(s.keys()) == expected_keys

    def test_summary_zero_totals(self):
        s = self.g.summary()
        assert s["total_traces"] == 0
        assert s["n_edges"] == 0
        assert s["mean_propagation_depth"] == 0.0
        assert s["self_correction_rate"] == 0.0

    def test_infer_from_empty_dict(self):
        self.g.infer_from_hops({})
        assert self.g._total_traces == 0

    def test_propagation_depth_distribution_empty(self):
        assert self.g.propagation_depth_distribution() == {}


# ---------------------------------------------------------------------------
# PropagationGraph — single hop (no propagation possible)
# ---------------------------------------------------------------------------

class TestPropagationGraphSingleHop:
    def test_no_edges_from_single_hop(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure(), _success()]})
        assert g.edges() == []

    def test_total_traces_counted(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure(), _success(), _answer_failure()]})
        assert g._total_traces == 3

    def test_failure_counts_updated(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure(), _success(), _retrieval_failure()]})
        rate = g.failure_rate_by_stage()
        assert rate.get("retrieval", 0.0) == pytest.approx(2 / 3)

    def test_propagation_depth_from_single_hop(self):
        g = PropagationGraph()
        # All failures at hop 1 — path length == 1 for failing traces
        g.infer_from_hops({1: [_retrieval_failure(), _retrieval_failure()]})
        assert g.propagation_depth() == pytest.approx(1.0)

    def test_critical_path_empty_when_no_propagation(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure(), _answer_failure()]})
        # No edges → critical_path is empty
        assert g.critical_path() == []


# ---------------------------------------------------------------------------
# PropagationGraph — multi-hop propagation
# ---------------------------------------------------------------------------

class TestPropagationGraphMultiHop:
    def _two_hop_cascade(self):
        """4 traces: 3 cascade retrieval→answer, 1 succeeds at both hops."""
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure(), _retrieval_failure(), _success()],
            2: [_answer_failure(),    _answer_failure(),    _answer_failure(),    _success()],
        })
        return g

    def test_edge_exists_retrieval_to_answer(self):
        g = self._two_hop_cascade()
        edges = {(e.source_stage, e.target_stage): e for e in g.edges()}
        assert (FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) in edges

    def test_stage_coupling_retrieval_to_answer(self):
        g = self._two_hop_cascade()
        # 3 retrieval failures, all followed by answer failures → coupling = 1.0
        coupling = g.stage_coupling(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION)
        assert coupling == pytest.approx(1.0)

    def test_stage_coupling_zero_for_unobserved_pair(self):
        g = self._two_hop_cascade()
        assert g.stage_coupling(FailureStage.TOOL_CALL, FailureStage.ANSWER_GENERATION) == 0.0

    def test_propagation_depth_two_hops(self):
        g = self._two_hop_cascade()
        # 3 failing traces each have depth 2 (failure at hop1 + hop2)
        assert g.propagation_depth() == pytest.approx(2.0)

    def test_propagation_depth_with_stage_filter(self):
        g = self._two_hop_cascade()
        # Filtering on RETRIEVAL: only traces that start with retrieval failure
        assert g.propagation_depth(FailureStage.RETRIEVAL) == pytest.approx(2.0)

    def test_propagation_depth_filter_no_match(self):
        g = self._two_hop_cascade()
        # No traces start with TOOL_CALL failure
        assert g.propagation_depth(FailureStage.TOOL_CALL) == 0.0

    def test_propagation_depth_distribution(self):
        g = self._two_hop_cascade()
        dist = g.propagation_depth_distribution()
        # 3 traces with depth 2
        assert dist.get(2, 0) == 3

    def test_critical_path_lists_retrieval_first(self):
        g = self._two_hop_cascade()
        # RETRIEVAL is the source of all 3 propagation edges
        path = g.critical_path()
        assert len(path) > 0
        assert path[0] == FailureStage.RETRIEVAL

    def test_total_traces_correct(self):
        g = self._two_hop_cascade()
        assert g._total_traces == 4

    def test_failure_rate_retrieval(self):
        g = self._two_hop_cascade()
        rates = g.failure_rate_by_stage()
        # 3 of 4 traces had retrieval failures
        assert rates.get("retrieval", 0.0) == pytest.approx(3 / 4)

    def test_failure_rate_answer(self):
        g = self._two_hop_cascade()
        rates = g.failure_rate_by_stage()
        assert rates.get("answer_generation", 0.0) == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# PropagationGraph — self-correction
# ---------------------------------------------------------------------------

class TestSelfCorrection:
    def test_no_self_correction_when_all_cascade(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure()],
            2: [_answer_failure(),    _answer_failure()],
        })
        # Every failing trace continues to fail at hop 2 → rate = 0
        assert g.self_correction_rate() == pytest.approx(0.0)

    def test_full_self_correction(self):
        g = PropagationGraph()
        # Failures at hop 1, success at hop 2 — paths are length 1,
        # max_hop = 2, so path[-1][0]=1 < 2 → corrected
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure()],
            2: [_success(),           _success()],
        })
        assert g.self_correction_rate() == pytest.approx(1.0)

    def test_partial_self_correction(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure()],
            2: [_answer_failure(),    _success()],
        })
        # 1 of 2 failing traces recovered
        assert g.self_correction_rate() == pytest.approx(0.5)

    def test_no_failing_traces_returns_zero(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_success(), _success()],
            2: [_success(), _success()],
        })
        assert g.self_correction_rate() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# PropagationGraph — coupling_matrix
# ---------------------------------------------------------------------------

class TestCouplingMatrix:
    def test_matrix_contains_observed_source_stages(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_answer_failure()],
        })
        matrix = g.coupling_matrix()
        assert "retrieval" in matrix

    def test_matrix_values_in_unit_interval(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure()],
            2: [_answer_failure(),    _success()],
        })
        for src_row in g.coupling_matrix().values():
            for v in src_row.values():
                assert 0.0 <= v <= 1.0

    def test_matrix_row_keys_are_all_stages(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_answer_failure()],
        })
        matrix = g.coupling_matrix()
        all_stage_values = {s.value for s in FailureStage}
        for row in matrix.values():
            assert set(row.keys()) == all_stage_values

    def test_matrix_diagonal_possible(self):
        g = PropagationGraph()
        # Same stage failure at both hops
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_retrieval_failure()],
        })
        matrix = g.coupling_matrix()
        coupling = matrix.get("retrieval", {}).get("retrieval", 0.0)
        assert coupling == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PropagationGraph — multiple infer_from_hops calls accumulate
# ---------------------------------------------------------------------------

class TestAccumulation:
    def test_multiple_calls_accumulate_traces(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_answer_failure()]})
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_answer_failure()]})
        assert g._total_traces == 2

    def test_multiple_calls_accumulate_edges(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_answer_failure()]})
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_answer_failure()]})
        edges = {(e.source_stage, e.target_stage): e for e in g.edges()}
        assert edges[(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION)].count == 2

    def test_coupling_recomputed_after_accumulation(self):
        g = PropagationGraph()
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_answer_failure()]})
        g.infer_from_hops({1: [_retrieval_failure()], 2: [_success()]})
        # After 2 calls: 2 retrieval failures, 1 propagated to answer → 0.5
        assert g.stage_coupling(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# PropagationGraph — three-hop chains
# ---------------------------------------------------------------------------

class TestThreeHopChain:
    def test_three_stage_chain_edges(self):
        # retrieval → tool → answer across 3 hops
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_tool_failure()],
            3: [_answer_failure()],
        })
        edge_pairs = {(e.source_stage, e.target_stage) for e in g.edges()}
        assert (FailureStage.RETRIEVAL, FailureStage.TOOL_CALL) in edge_pairs
        assert (FailureStage.TOOL_CALL, FailureStage.ANSWER_GENERATION) in edge_pairs

    def test_three_hop_depth(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_tool_failure()],
            3: [_answer_failure()],
        })
        assert g.propagation_depth() == pytest.approx(3.0)

    def test_depth_distribution_three_hop(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure(), _retrieval_failure()],
            2: [_tool_failure(),      _success()],
            3: [_answer_failure(),    _success()],
        })
        dist = g.propagation_depth_distribution()
        # Trace 0: path length 3 (hop1 ret, hop2 tool, hop3 answer)
        assert dist.get(3, 0) == 1
        # Trace 1: path length 1 (hop1 ret only, then success)
        assert dist.get(1, 0) == 1

    def test_summary_n_edges_three_hop(self):
        g = PropagationGraph()
        g.infer_from_hops({
            1: [_retrieval_failure()],
            2: [_tool_failure()],
            3: [_answer_failure()],
        })
        s = g.summary()
        # ret→tool and tool→answer
        assert s["n_edges"] == 2
