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
    ancestor_hit,
    ancestor_hit_rate,
    attribution_identifiability,
    bootstrap_localization_ci,
    cost_per_correct_diagnosis,
    diagnosis_correct,
    localization_accuracy,
    mean_localization_error,
    rescore_identifiability,
    slice_identifiability,
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

    def test_require_actual_depth_filters_clamped_truths(self):
        result = {
            "diagnosers": ["a"],
            "raw_by_depth": {
                "2": {
                    "truth": [["retrieval", 1], ["retrieval", 2]],
                    "predictions": {
                        "a": [["retrieval", 1, 0], ["retrieval", 2, 0]],
                    },
                }
            },
        }
        loose = rescore_identifiability(result, criterion="hop")
        strict = rescore_identifiability(
            result, criterion="hop", require_actual_depth=True
        )
        assert loose["a"][2] == 1.0
        assert strict["a"][2] == 1.0

        result["raw_by_depth"]["2"]["predictions"]["a"][0][1] = 2
        loose = rescore_identifiability(result, criterion="hop")
        strict = rescore_identifiability(
            result, criterion="hop", require_actual_depth=True
        )
        assert loose["a"][2] == 0.5
        assert strict["a"][2] == 1.0


# --------------------------------------------------------------------------- #
# ancestor_hit / ancestor_hit_rate
# --------------------------------------------------------------------------- #

class TestAncestorHit:
    def test_at_true_hop_same_stage(self):
        assert ancestor_hit(_diag(2), _truth(2))

    def test_before_true_hop_same_stage(self):
        # Prediction at hop 1, true at hop 3 — still a causal ancestor
        assert ancestor_hit(_diag(1), _truth(3))

    def test_after_true_hop_fails(self):
        # Predicting too late (hop 3 when true is hop 2) is not an ancestor
        assert not ancestor_hit(_diag(3), _truth(2))

    def test_wrong_stage_fails(self):
        assert not ancestor_hit(
            _diag(1, FailureStage.ANSWER_GENERATION), _truth(2, FailureStage.RETRIEVAL)
        )

    def test_hop_zero_fails(self):
        # hop=0 means "no hop prediction" — not a valid ancestor
        assert not ancestor_hit(_diag(0), _truth(2))

    def test_ancestor_hit_rate_batch(self):
        diags = [_diag(1), _diag(3), _diag(1)]
        truths = [_truth(3), _truth(2), _truth(1)]
        # hop 1 ≤ 3 ✓, hop 3 > 2 ✗, hop 1 = 1 ✓  → 2/3
        assert ancestor_hit_rate(diags, truths) == pytest.approx(2 / 3)

    def test_ancestor_hit_rate_empty(self):
        assert ancestor_hit_rate([], []) == 0.0

    def test_ancestor_hit_rate_length_mismatch(self):
        with pytest.raises(ValueError):
            ancestor_hit_rate([_diag(1)], [])


# --------------------------------------------------------------------------- #
# bootstrap_localization_ci
# --------------------------------------------------------------------------- #

class TestBootstrapCI:
    def test_perfect_accuracy_ci(self):
        diags = [_diag(1)] * 20
        truths = [_truth(1)] * 20
        ci = bootstrap_localization_ci(diags, truths, n_boot=200, seed=0)
        assert ci["mean"] == pytest.approx(1.0)
        assert ci["ci_low"] == pytest.approx(1.0)
        assert ci["ci_high"] == pytest.approx(1.0)
        assert ci["n"] == 20

    def test_zero_accuracy_ci(self):
        diags = [_diag(1)] * 10
        truths = [_truth(5)] * 10
        ci = bootstrap_localization_ci(diags, truths, n_boot=200, seed=0)
        assert ci["mean"] == pytest.approx(0.0)
        assert ci["ci_low"] == pytest.approx(0.0)

    def test_deterministic_with_seed(self):
        diags = [_diag(1), _diag(2), _diag(3)] * 5
        truths = [_truth(1), _truth(3), _truth(3)] * 5
        ci1 = bootstrap_localization_ci(diags, truths, n_boot=500, seed=42)
        ci2 = bootstrap_localization_ci(diags, truths, n_boot=500, seed=42)
        assert ci1 == ci2

    def test_ci_widens_with_small_n(self):
        # 5 samples → wide CI; 50 samples → narrower CI
        diags_small = [_diag(1), _diag(1), _diag(2), _diag(2), _diag(3)]
        truths_small = [_truth(1), _truth(2), _truth(2), _truth(3), _truth(3)]
        diags_large = diags_small * 10
        truths_large = truths_small * 10
        ci_s = bootstrap_localization_ci(diags_small, truths_small, n_boot=500, seed=0)
        ci_l = bootstrap_localization_ci(diags_large, truths_large, n_boot=500, seed=0)
        width_small = ci_s["ci_high"] - ci_s["ci_low"]
        width_large = ci_l["ci_high"] - ci_l["ci_low"]
        assert width_small >= width_large

    def test_empty_returns_zeros(self):
        ci = bootstrap_localization_ci([], [], n_boot=100, seed=0)
        assert ci["n"] == 0
        assert ci["mean"] == 0.0

    def test_ci_bounds_between_0_and_1(self):
        diags = [_diag(i % 3 + 1) for i in range(15)]
        truths = [_truth((i % 3 + 1 + 1) % 3 + 1) for i in range(15)]
        ci = bootstrap_localization_ci(diags, truths, n_boot=300, seed=7)
        assert 0.0 <= ci["ci_low"] <= ci["mean"] <= ci["ci_high"] <= 1.0


# --------------------------------------------------------------------------- #
# slice_identifiability
# --------------------------------------------------------------------------- #

_SLICE_RESULT = {
    "diagnosers": ["a", "b"],
    "raw_by_depth": {
        "1": {
            "truth": [["retrieval", 1], ["retrieval", 1], ["retrieval", 1]],
            "predictions": {
                "a": [["retrieval", 1, 0], ["retrieval", 1, 0], ["answer_generation", 0, 0]],
                "b": [["retrieval", 1, 0], ["answer_generation", 0, 0], ["retrieval", 1, 0]],
            },
            "metadata": [
                {"intervention_method": "inject_empty_retrieval", "injected_failure_type": "empty"},
                {"intervention_method": "inject_empty_retrieval", "injected_failure_type": "empty"},
                {"intervention_method": "inject_irrelevant_docs", "injected_failure_type": "irrelevant"},
            ],
        }
    },
}


class TestSliceIdentifiability:
    def test_slice_by_method_separates_cases(self):
        slices = slice_identifiability(_SLICE_RESULT, slice_by="intervention_method")
        # "inject_empty_retrieval" has 2 cases
        assert "inject_empty_retrieval" in slices
        assert "inject_irrelevant_docs" in slices

    def test_pooled_vs_sliced_denominators(self):
        # Pooled across all cases: diagnoser "a" gets 2/3 correct at depth 1
        from agenticrag.evaluate import rescore_identifiability
        pooled = rescore_identifiability(_SLICE_RESULT, criterion="hop")
        assert pooled["a"][1] == pytest.approx(2 / 3)

        # Sliced by method: "inject_empty_retrieval" → 2/2 = 1.0 for "a"
        slices = slice_identifiability(_SLICE_RESULT, slice_by="intervention_method")
        assert slices["inject_empty_retrieval"]["a"][1] == pytest.approx(1.0)
        # "inject_irrelevant_docs" → 0/1 = 0.0 for "a"
        assert slices["inject_irrelevant_docs"]["a"][1] == pytest.approx(0.0)

    def test_slice_by_failure_type(self):
        slices = slice_identifiability(_SLICE_RESULT, slice_by="injected_failure_type")
        assert "empty" in slices
        assert "irrelevant" in slices

    def test_missing_metadata_key_groups_as_unknown(self):
        result = {
            "diagnosers": ["a"],
            "raw_by_depth": {
                "1": {
                    "truth": [["retrieval", 1]],
                    "predictions": {"a": [["retrieval", 1, 0]]},
                    "metadata": [{}],  # no "intervention_method" key
                }
            },
        }
        slices = slice_identifiability(result, slice_by="intervention_method")
        assert "unknown" in slices

    def test_empty_raw_by_depth(self):
        result = {"diagnosers": ["a"], "raw_by_depth": {}}
        slices = slice_identifiability(result)
        assert slices == {}


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
