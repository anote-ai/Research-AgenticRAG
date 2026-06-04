"""Evaluation utilities for the agentic RAG diagnostic benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenticrag.core import FailureRecord, FailureStage, PipelineTrace


def stage_attribution_rate(
    records: list["FailureRecord"], stage: "FailureStage"
) -> float:
    """Fraction of failure records attributed to *stage*."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.stage == stage) / len(records)


def propagation_rate(records: list["FailureRecord"]) -> float:
    """Fraction of records where the failure propagated downstream."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.propagated) / len(records)


def failure_confusion_matrix(
    predicted_stages: list[str], true_stages: list[str]
) -> dict[str, dict[str, int]]:
    """Compute a confusion matrix between predicted and true failure stages.

    Returns a nested dict: matrix[true_stage][predicted_stage] = count.
    """
    if len(predicted_stages) != len(true_stages):
        raise ValueError("predicted_stages and true_stages must have the same length.")

    all_stages = sorted(set(predicted_stages) | set(true_stages))
    matrix: dict[str, dict[str, int]] = {
        t: {p: 0 for p in all_stages} for t in all_stages
    }
    for pred, true in zip(predicted_stages, true_stages):
        matrix[true][pred] += 1
    return matrix


def end_to_end_accuracy(traces: list["PipelineTrace"]) -> float:
    """Exact-match accuracy: fraction of traces where final_answer == reference_answer."""
    if not traces:
        return 0.0
    correct = sum(
        1 for t in traces
        if t.final_answer.strip().lower() == t.reference_answer.strip().lower()
    )
    return correct / len(traces)
