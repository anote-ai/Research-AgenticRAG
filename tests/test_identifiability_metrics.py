"""Tests for the W4 metrics: identifiability, localization, cost, counterfactual recovery."""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from agenticrag.core import FailureStage, PipelineTrace
from agenticrag.diagnosers import Diagnosis
from agenticrag.evaluate import (
    attribution_identifiability,
    cost_per_correct_diagnosis,
    diagnosis_correct,
    localization_accuracy,
    mean_localization_error,
    rescore_identifiability,
)
from agenticrag.injection import InjectionResult
from agenticrag.propagation import counterfactual_recovery_rate


def _diag(hop, stage=FailureStage.RETRIEVAL, cost=10):
    return Diagnosis(trace_id="t", stage=stage, failure_type="x", predicted_hop=hop, cost_tokens=cost)


def _truth(hop, stage=FailureStage.RETRIEVAL):
    return (stage, hop)


# --------------------------------------------------------------------------- #
# diagnosis_correct
# --------------------------------------------------------------------------- #

class TestDiagnosisCorrect:
    def test_hop_match(self):
        assert diagnosis_correct(_diag(2), _truth(2), criterion="hop")
        assert not diagnosis_correct(_diag(2), _truth(3), criterion="hop")

    def test_hop_tolerance(self):
        assert diagnosis_correct(_diag(2), _truth(3), criterion="hop", hop_tolerance=1)

    def test_stage_match(self):
        assert diagnosis_correct(
            _diag(2, FailureStage.RETRIEVAL), _truth(9, FailureStage.RETRIEVAL), criterion="stage"
        )

    def test_both(self):
        assert diagnosis_correct(_diag(2), _truth(2), criterion="both")
        assert not diagnosis_correct(_diag(2), _truth(2, FailureStage.ANSWER_GENERATION), criterion="both")

    def test_accepts_injection_result_as_truth(self):
        ir = InjectionResult(
            original_trace_id="o",
            injected_trace=PipelineTrace(
                query="q", retrieved_docs=[], tool_calls=[], final_answer="",
                reference_answer="", hop_queries=[], hop_docs=[],
            ),
            injected_stage=FailureStage.RETRIEVAL,
            injected_failure_type="empty_retrieval",
            injected_at_hop=3,
        )
        assert diagnosis_correct(_diag(3), ir, criterion="hop")

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            diagnosis_correct(_diag(1), _truth(1), criterion="bogus")


# --------------------------------------------------------------------------- #
# localization_accuracy / mean_localization_error
# --------------------------------------------------------------------------- #

class TestLocalization:
    def test_accuracy(self):
        diags = [_diag(1), _diag(2), _diag(3)]
        truths = [_truth(1), _truth(2), _truth(9)]
        assert localization_accuracy(diags, truths) == pytest.approx(2 / 3)

    def test_empty(self):
        assert localization_accuracy([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            localization_accuracy([_diag(1)], [])

    def test_mean_error(self):
        diags = [_diag(1), _diag(4)]
        truths = [_truth(1), _truth(2)]
        assert mean_localization_error(diags, truths) == pytest.approx(1.0)  # (0 + 2)/2


# --------------------------------------------------------------------------- #
# attribution_identifiability (the C2 curve)
# --------------------------------------------------------------------------- #

class TestAttributionIdentifiability:
    def test_decaying_curve(self):
        # Post-hoc-like diagnoser: perfect at hop 1, wrong deeper.
        diagnoses_by_depth = {
            1: [_diag(1), _diag(1)],
            2: [_diag(1), _diag(1)],  # predicts hop 1 even when truth is 2
            3: [_diag(1), _diag(1)],
        }
        truths_by_depth = {
            1: [_truth(1), _truth(1)],
            2: [_truth(2), _truth(2)],
            3: [_truth(3), _truth(3)],
        }
        curve = attribution_identifiability(diagnoses_by_depth, truths_by_depth)
        assert curve[1] == 1.0
        assert curve[2] == 0.0
        assert curve[3] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            attribution_identifiability({1: [_diag(1)]}, {1: []})


# --------------------------------------------------------------------------- #
# cost_per_correct_diagnosis
# --------------------------------------------------------------------------- #

class TestCostPerCorrect:
    def test_basic(self):
        diags = [_diag(1, cost=10), _diag(2, cost=30)]
        truths = [_truth(1), _truth(9)]  # one correct
        out = cost_per_correct_diagnosis(diags, truths)
        assert out["total_cost"] == 40.0
        assert out["n_correct"] == 1.0
        assert out["cost_per_correct"] == 40.0

    def test_no_correct_is_inf(self):
        diags = [_diag(1, cost=10)]
        truths = [_truth(5)]
        out = cost_per_correct_diagnosis(diags, truths)
        assert math.isinf(out["cost_per_correct"])


# --------------------------------------------------------------------------- #
# rescore_identifiability (re-score persisted raw diagnoses)
# --------------------------------------------------------------------------- #

class TestRescore:
    RESULT = {
        "diagnosers": ["a", "b"],
        "raw_by_depth": {
            "1": {
                "truth": [["retrieval", 1], ["retrieval", 1]],
                "predictions": {
                    "a": [["retrieval", 1, 10], ["answer_generation", 0, 5]],
                    "b": [["retrieval", 2, 0], ["retrieval", 1, 0]],
                },
            }
        },
    }

    def test_stage_criterion(self):
        out = rescore_identifiability(self.RESULT, criterion="stage")
        assert out["a"][1] == 0.5   # one retrieval, one answer_generation
        assert out["b"][1] == 1.0   # both retrieval

    def test_hop_criterion_exact(self):
        out = rescore_identifiability(self.RESULT, criterion="hop", hop_tolerance=0)
        assert out["a"][1] == 0.5   # hops [1,0] vs [1,1]
        assert out["b"][1] == 0.5   # hops [2,1] vs [1,1]

    def test_hop_tolerance_recovers(self):
        out = rescore_identifiability(self.RESULT, criterion="hop", hop_tolerance=1)
        assert out["b"][1] == 1.0

    def test_invalid_criterion_raises(self):
        with pytest.raises(ValueError):
            rescore_identifiability(self.RESULT, criterion="bogus")


# --------------------------------------------------------------------------- #
# counterfactual_recovery_rate
# --------------------------------------------------------------------------- #

class TestCounterfactualRecovery:
    def _ir(self, answer):
        trace = PipelineTrace(
            query="q", retrieved_docs=["d"], tool_calls=[{"name": "retrieve"}],
            final_answer=answer, reference_answer="paris",
            hop_queries=["q"], hop_docs=[["d"]], iterations_used=1,
        )
        return InjectionResult(
            original_trace_id="o", injected_trace=trace,
            injected_stage=FailureStage.RETRIEVAL,
            injected_failure_type="empty_retrieval", injected_at_hop=1,
        )

    def test_half_recovered(self):
        results = [self._ir("paris"), self._ir("london")]
        refs = [{"answer": "paris"}, {"answer": "paris"}]
        assert counterfactual_recovery_rate(results, refs) == 0.5

    def test_empty(self):
        assert counterfactual_recovery_rate([], []) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            counterfactual_recovery_rate([self._ir("paris")], [])
