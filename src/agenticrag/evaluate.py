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


def failure_amplification_rate(
    records_by_hop: Dict[int, List[FailureRecord]],
) -> Dict[int, float]:
    """Failure rate at each hop depth, showing whether errors amplify or attenuate.

    Parameters
    ----------
    records_by_hop:
        Mapping from hop index (1-based) to the FailureRecords for traces
        that had a failure *injected or observed* at that hop.

    Returns
    -------
    A dict mapping hop index → failure rate (fraction of non-NONE records).
    A rising curve means failures compound; a falling curve means the pipeline
    self-corrects.
    """
    result: Dict[int, float] = {}
    for hop, records in sorted(records_by_hop.items()):
        if not records:
            result[hop] = 0.0
        else:
            failed = sum(1 for r in records if r.stage != FailureStage.NONE)
            result[hop] = failed / len(records)
    return result


def recovery_rate(records_by_hop: Dict[int, List[FailureRecord]]) -> float:
    """Fraction of mid-pipeline failures that eventually resolved (stage == NONE).

    A trace is considered a mid-pipeline failure if it had at least one
    non-NONE record at an early hop but a NONE record at the final hop.
    This requires records grouped by hop; the last hop's record is the outcome.

    Parameters
    ----------
    records_by_hop:
        Same format as failure_amplification_rate: hop → records.
        The highest hop key is treated as the final outcome.

    Returns the fraction of traces with a mid-pipeline failure that ultimately
    recovered (final-hop record has stage == NONE).
    """
    if not records_by_hop:
        return 0.0

    hops = sorted(records_by_hop.keys())
    if len(hops) < 2:
        return 0.0

    final_hop = hops[-1]
    final_records = records_by_hop[final_hop]

    # For each position index: check whether any earlier hop had a failure
    min_len = min(len(records_by_hop[h]) for h in hops)
    if min_len == 0:
        return 0.0

    recovered = 0
    had_mid_failure = 0
    for idx in range(min_len):
        early_failed = any(
            records_by_hop[h][idx].stage != FailureStage.NONE for h in hops[:-1]
        )
        if not early_failed:
            continue
        had_mid_failure += 1
        if idx < len(final_records) and final_records[idx].stage == FailureStage.NONE:
            recovered += 1

    return recovered / had_mid_failure if had_mid_failure else 0.0
