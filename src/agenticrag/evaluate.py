from __future__ import annotations

from typing import Dict, List

from .core import FailureRecord, FailureStage, PipelineTrace


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


def multi_hop_accuracy(traces: List[PipelineTrace]) -> float:
    """Fraction of traces that required >1 retrieval hop and produced a non-empty answer.

    A trace is counted as a multi-hop success when it used more than one hop
    (iterations_used > 1) and the final answer is non-empty.
    Returns 0 if no multi-hop traces exist.
    """
    multi_hop = [t for t in traces if t.iterations_used > 1]
    if not multi_hop:
        return 0.0
    successful = sum(1 for t in multi_hop if t.final_answer.strip() != "")
    return successful / len(multi_hop)


def retrieval_loop_efficiency(traces: List[PipelineTrace], max_iterations: int = 3) -> float:
    """Mean fraction of the iteration budget *not* consumed.

    A value of 1.0 means every trace resolved in a single hop.
    A value of 0.0 means every trace exhausted the full iteration budget.
    """
    if not traces or max_iterations <= 0:
        return 0.0
    savings = [(max_iterations - t.iterations_used) / max_iterations for t in traces]
    return sum(savings) / len(savings)


def mean_hops_per_trace(traces: List[PipelineTrace]) -> float:
    """Average number of retrieval iterations across all traces."""
    if not traces:
        return 0.0
    return sum(t.iterations_used for t in traces) / len(traces)


def hop_doc_coverage(traces: List[PipelineTrace]) -> float:
    """Mean fraction of hops that retrieved at least one document."""
    if not traces:
        return 0.0
    coverages: List[float] = []
    for trace in traces:
        if not trace.hop_docs:
            coverages.append(0.0)
            continue
        filled = sum(1 for hop in trace.hop_docs if len(hop) > 0)
        coverages.append(filled / len(trace.hop_docs))
    return sum(coverages) / len(coverages)
