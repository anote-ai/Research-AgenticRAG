"""Tests for agenticrag.experiment — AblationResult and run_ablation."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agenticrag.core import DiagnosticBenchmark, FailureStage, PipelineTrace
from agenticrag.experiment import (
    ANSWER_METHODS,
    ALL_METHODS,
    HOP_METHODS,
    AblationCell,
    AblationResult,
    run_ablation,
    _DEFAULT_NOISE_DOCS,
)
from agenticrag.injection import FailureInjector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _trace(
    *,
    query: str = "What is revenue?",
    answer: str = "Revenue is $500 million.",
    hops: int = 3,
    trace_id: str = "t0",
) -> PipelineTrace:
    hop_docs = [
        ["Doc A: Revenue data.", "Doc B: Financial report."],
        ["Doc C: Earnings summary."],
        ["Doc D: Annual report."],
    ][:hops]
    hop_queries = [query] + [f"{query} follow-up {i}" for i in range(1, hops)]
    return PipelineTrace(
        trace_id=trace_id,
        query=query,
        retrieved_docs=["Doc A: Revenue data.", "Doc B: Financial report.",
                        "Doc C: Earnings summary.", "Doc D: Annual report."][:hops * 2],
        tool_calls=[{"name": "retrieve", "args": {"q": q}} for q in hop_queries],
        final_answer=answer,
        reference_answer=answer,
        hop_queries=hop_queries,
        hop_docs=hop_docs,
        iterations_used=hops,
    )


def _traces(n: int = 3, hops: int = 3) -> list[PipelineTrace]:
    return [_trace(trace_id=f"t{i}", hops=hops) for i in range(n)]


def _refs(traces: list[PipelineTrace]) -> list[dict]:
    return [{"answer": t.reference_answer} for t in traces]


def _injector() -> FailureInjector:
    return FailureInjector()


def _bench() -> DiagnosticBenchmark:
    return DiagnosticBenchmark()


# ---------------------------------------------------------------------------
# run_ablation — argument validation
# ---------------------------------------------------------------------------

class TestRunAblationValidation:
    def test_raises_on_empty_traces(self):
        with pytest.raises(ValueError, match="non-empty"):
            run_ablation([], [], _injector(), _bench())

    def test_raises_on_length_mismatch(self):
        ts = _traces(3)
        with pytest.raises(ValueError, match="same length"):
            run_ablation(ts, _refs(ts)[:2], _injector(), _bench())

    def test_raises_on_unknown_method(self):
        ts = _traces(2)
        with pytest.raises(ValueError, match="Unknown injection method"):
            run_ablation(ts, _refs(ts), _injector(), _bench(), methods=["nonexistent"])


# ---------------------------------------------------------------------------
# run_ablation — basic structure
# ---------------------------------------------------------------------------

class TestRunAblationStructure:
    def setup_method(self):
        self.ts = _traces(3)
        self.refs = _refs(self.ts)
        self.result = run_ablation(
            self.ts, self.refs, _injector(), _bench(),
            methods=["inject_empty_retrieval", "inject_empty_answer"],
            hops=[1, 2],
        )

    def test_n_samples(self):
        assert self.result.n_samples == 3

    def test_baseline_records_length(self):
        assert len(self.result.baseline_records) == 3

    def test_cells_count(self):
        # 2 hop methods × 2 hops = 4 cells
        # but methods=['inject_empty_retrieval'(hop), 'inject_empty_answer'(answer)]
        # inject_empty_retrieval with hops=[1,2] → 2 cells
        # inject_empty_answer (answer method) → 1 cell
        assert len(self.result.cells) == 3

    def test_cell_records_length(self):
        for cell in self.result.cells:
            assert len(cell.records) == 3

    def test_injected_stage_set(self):
        for cell in self.result.cells:
            assert isinstance(cell.injected_stage, FailureStage)

    def test_sensitivity_in_unit_interval(self):
        for cell in self.result.cells:
            assert 0.0 <= cell.sensitivity <= 1.0

    def test_rca_in_unit_interval(self):
        for cell in self.result.cells:
            assert 0.0 <= cell.root_cause_accuracy_score <= 1.0

    def test_severity_in_unit_interval(self):
        for cell in self.result.cells:
            assert 0.0 <= cell.severity_rate <= 1.0

    def test_stage_rates_keys(self):
        non_none_stages = {s.value for s in FailureStage if s != FailureStage.NONE}
        for cell in self.result.cells:
            assert set(cell.stage_rates.keys()) == non_none_stages


# ---------------------------------------------------------------------------
# run_ablation — default methods include all five
# ---------------------------------------------------------------------------

class TestRunAblationDefaultMethods:
    def test_default_runs_all_methods(self):
        ts = _traces(2)
        result = run_ablation(ts, _refs(ts), _injector(), _bench(), hops=[1])
        method_names = {c.injection_method for c in result.cells}
        assert method_names == set(ALL_METHODS)

    def test_hop_method_cells_have_hop_gt_zero(self):
        ts = _traces(2)
        result = run_ablation(ts, _refs(ts), _injector(), _bench(), hops=[1, 2])
        for c in result.cells:
            if c.injection_method in HOP_METHODS:
                assert c.hop > 0

    def test_answer_method_cells_have_hop_zero(self):
        ts = _traces(2)
        result = run_ablation(ts, _refs(ts), _injector(), _bench(), hops=[1])
        for c in result.cells:
            if c.injection_method in ANSWER_METHODS:
                assert c.hop == 0


# ---------------------------------------------------------------------------
# AblationResult — lookup helpers
# ---------------------------------------------------------------------------

class TestAblationResultLookup:
    def setup_method(self):
        ts = _traces(2)
        self.result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval", "inject_empty_answer"],
            hops=[1, 2],
        )

    def test_cell_found_by_method_and_hop(self):
        c = self.result.cell("inject_empty_retrieval", hop=1)
        assert c is not None
        assert c.injection_method == "inject_empty_retrieval"
        assert c.hop == 1

    def test_cell_returns_none_for_missing(self):
        c = self.result.cell("inject_empty_retrieval", hop=99)
        assert c is None

    def test_cell_answer_method_hop_zero(self):
        c = self.result.cell("inject_empty_answer", hop=0)
        assert c is not None

    def test_cells_for_method_ordered_by_hop(self):
        cells = self.result.cells_for_method("inject_empty_retrieval")
        hops = [c.hop for c in cells]
        assert hops == sorted(hops)

    def test_cells_for_method_empty_for_unknown(self):
        assert self.result.cells_for_method("nonexistent") == []


# ---------------------------------------------------------------------------
# AblationResult — paper-ready outputs
# ---------------------------------------------------------------------------

class TestAblationResultOutputs:
    def setup_method(self):
        ts = _traces(3)
        self.result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval", "inject_empty_answer"],
            hops=[1, 2],
        )

    def test_sensitivity_table_keys(self):
        st = self.result.sensitivity_table()
        assert "inject_empty_retrieval@hop1" in st
        assert "inject_empty_retrieval@hop2" in st
        assert "inject_empty_answer" in st

    def test_sensitivity_table_values_in_unit_interval(self):
        for v in self.result.sensitivity_table().values():
            assert 0.0 <= v <= 1.0

    def test_metrics_table_has_all_cells(self):
        mt = self.result.metrics_table()
        assert "inject_empty_retrieval@hop1" in mt
        assert "inject_empty_answer" in mt

    def test_metrics_table_row_keys(self):
        mt = self.result.metrics_table()
        row = mt["inject_empty_retrieval@hop1"]
        assert "sensitivity" in row
        assert "root_cause_accuracy" in row
        assert "severity_rate" in row

    def test_records_by_hop_keys(self):
        rbh = self.result.records_by_hop("inject_empty_retrieval")
        assert set(rbh.keys()) == {1, 2}

    def test_records_by_hop_record_count(self):
        rbh = self.result.records_by_hop("inject_empty_retrieval")
        for hop, records in rbh.items():
            assert len(records) == 3

    def test_records_by_hop_answer_method(self):
        rbh = self.result.records_by_hop("inject_empty_answer")
        assert 0 in rbh

    def test_baseline_accuracy_in_unit_interval(self):
        acc = self.result.baseline_accuracy()
        assert 0.0 <= acc <= 1.0

    def test_baseline_severity_in_unit_interval(self):
        sev = self.result.baseline_severity()
        assert 0.0 <= sev <= 1.0


# ---------------------------------------------------------------------------
# Detection correctness
# ---------------------------------------------------------------------------

class TestDetectionCorrectness:
    def test_empty_retrieval_detected_at_100_pct(self):
        ts = _traces(4)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval"], hops=[1],
        )
        cell = result.cell("inject_empty_retrieval", hop=1)
        assert cell is not None
        assert cell.sensitivity == pytest.approx(1.0)

    def test_empty_answer_detected_at_100_pct(self):
        ts = _traces(4)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_answer"],
        )
        cell = result.cell("inject_empty_answer", hop=0)
        assert cell is not None
        assert cell.sensitivity == pytest.approx(1.0)

    def test_no_tool_calls_detected_at_100_pct(self):
        ts = _traces(4)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_no_tool_calls"],
        )
        cell = result.cell("inject_no_tool_calls", hop=0)
        assert cell is not None
        assert cell.sensitivity == pytest.approx(1.0)

    def test_injected_stage_matches_expected_for_retrieval(self):
        ts = _traces(2)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval"], hops=[1],
        )
        cell = result.cell("inject_empty_retrieval", hop=1)
        assert cell is not None
        assert cell.injected_stage == FailureStage.RETRIEVAL

    def test_injected_stage_matches_expected_for_answer(self):
        ts = _traces(2)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_answer"],
        )
        cell = result.cell("inject_empty_answer", hop=0)
        assert cell is not None
        assert cell.injected_stage == FailureStage.ANSWER_GENERATION

    def test_injected_stage_matches_expected_for_tool(self):
        ts = _traces(2)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_no_tool_calls"],
        )
        cell = result.cell("inject_no_tool_calls", hop=0)
        assert cell is not None
        assert cell.injected_stage == FailureStage.TOOL_CALL


# ---------------------------------------------------------------------------
# Custom noise docs
# ---------------------------------------------------------------------------

class TestCustomNoiseDocs:
    def test_custom_noise_docs_accepted(self):
        ts = _traces(2)
        noise = ["Penguins live in Antarctica.", "Seaweed grows in the ocean."]
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_irrelevant_docs"], hops=[1],
            noise_docs=noise,
        )
        cell = result.cell("inject_irrelevant_docs", hop=1)
        assert cell is not None
        assert cell.sensitivity >= 0.0

    def test_default_noise_docs_non_empty(self):
        assert len(_DEFAULT_NOISE_DOCS) > 0
        assert all(isinstance(d, str) and d for d in _DEFAULT_NOISE_DOCS)


# ---------------------------------------------------------------------------
# records_by_hop integration with PropagationGraph (structural only)
# ---------------------------------------------------------------------------

class TestRecordsByHopStructure:
    def test_records_by_hop_values_are_failure_records(self):
        from agenticrag.core import FailureRecord
        ts = _traces(3)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval"], hops=[1, 2, 3],
        )
        rbh = result.records_by_hop("inject_empty_retrieval")
        for hop, records in rbh.items():
            assert all(isinstance(r, FailureRecord) for r in records)

    def test_records_by_hop_all_hops_present(self):
        ts = _traces(3)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval"], hops=[1, 2, 3],
        )
        rbh = result.records_by_hop("inject_empty_retrieval")
        assert set(rbh.keys()) == {1, 2, 3}

    def test_metrics_table_stage_rate_keys_valid(self):
        ts = _traces(2)
        result = run_ablation(
            ts, _refs(ts), _injector(), _bench(),
            methods=["inject_empty_retrieval"], hops=[1],
        )
        mt = result.metrics_table()
        row = mt["inject_empty_retrieval@hop1"]
        stage_keys = [k for k in row if k.startswith("stage_")]
        non_none_values = [s.value for s in FailureStage if s != FailureStage.NONE]
        assert set(k[6:] for k in stage_keys) == set(non_none_values)
