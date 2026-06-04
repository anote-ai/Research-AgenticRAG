from __future__ import annotations

from typing import Dict, List

from .core import FailureRecord, FailureStage


def stage_attribution_rate(records: List[FailureRecord], stage: FailureStage) -> float:
    """Fraction of failed records (non-NONE) attributed to the given stage."""
    failed = [r for r in records if r.stage != FailureStage.NONE]
    if not failed:
        return 0.0
    return sum(1 for r in failed if r.stage == stage) / len(failed)


def propagation_rate(records: List[FailureRecord]) -> float:
    """Fraction of all records where propagated=True."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.propagated) / len(records)


def failure_confusion_matrix(
    predicted_stages: List[str], true_stages: List[str]
) -> Dict[str, Dict[str, int]]:
    """Compute per-stage TP, FP, FN."""
    all_stages = set(predicted_stages) | set(true_stages)
    result: Dict[str, Dict[str, int]] = {}

    for stage in all_stages:
        tp = sum(
            1 for p, t in zip(predicted_stages, true_stages) if p == stage and t == stage
        )
        fp = sum(
            1 for p, t in zip(predicted_stages, true_stages) if p == stage and t != stage
        )
        fn = sum(
            1 for p, t in zip(predicted_stages, true_stages) if p != stage and t == stage
        )
        result[stage] = {"tp": tp, "fp": fp, "fn": fn}

    return result


def end_to_end_accuracy(records: List[FailureRecord]) -> float:
    """Fraction of records where stage == NONE (i.e., successful)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.stage == FailureStage.NONE) / len(records)


def severity_weighted_failure_rate(records: List[FailureRecord]) -> float:
    """Mean severity for non-NONE records."""
    failed = [r for r in records if r.stage != FailureStage.NONE]
    if not failed:
        return 0.0
    return sum(r.severity for r in failed) / len(failed)
