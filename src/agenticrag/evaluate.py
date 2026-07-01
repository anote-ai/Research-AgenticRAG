from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import FailureRecord, FailureStage, PipelineTrace, _normalize_answer, _token_recall


def answer_token_recall(prediction: str, reference: str) -> float:
    """Token recall of *prediction* against *reference* after SQuAD-style normalization."""
    return _token_recall(prediction, reference)


def answer_em_score(prediction: str, reference: str) -> float:
    """Exact-match score (1.0 or 0.0) after SQuAD-style normalization."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def mean_answer_recall(
    traces: List[PipelineTrace],
) -> float:
    """Mean token recall of final_answer vs reference_answer across traces."""
    if not traces:
        return 0.0
    return sum(
        _token_recall(t.final_answer, t.reference_answer) for t in traces
    ) / len(traces)


def mean_answer_em(traces: List[PipelineTrace]) -> float:
    """Mean exact-match score of final_answer vs reference_answer across traces."""
    if not traces:
        return 0.0
    return sum(
        answer_em_score(t.final_answer, t.reference_answer) for t in traces
    ) / len(traces)


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


def root_cause_stage(record: FailureRecord) -> FailureStage:
    """Return the predicted root-cause stage for a failure record.

    FailureRecord.root_cause is currently a free-form string in some diagnostic
    paths. When it stores a canonical stage value, use it. Otherwise fall back
    to the record's attributed stage.
    """
    try:
        return FailureStage(record.root_cause)
    except ValueError:
        return record.stage


def root_cause_accuracy(
    predicted_records: Sequence[FailureRecord],
    true_root_causes: Sequence[FailureRecord | FailureStage | str],
    include_success: bool = True,
) -> float:
    """Fraction of records where the earliest failing stage is identified.

    Parameters
    ----------
    predicted_records:
        Diagnostic output whose root cause should be scored.
    true_root_causes:
        Ground-truth root causes as FailureRecords, FailureStage values, or
        canonical stage strings.
    include_success:
        Whether successful traces (stage == NONE) count in the denominator.
    """
    if len(predicted_records) != len(true_root_causes):
        raise ValueError(
            "predicted_records and true_root_causes must have the same length"
        )

    pairs = [
        (root_cause_stage(predicted), _coerce_root_cause_stage(true))
        for predicted, true in zip(predicted_records, true_root_causes)
    ]
    if not include_success:
        pairs = [
            (predicted, true)
            for predicted, true in pairs
            if true != FailureStage.NONE
        ]

    if not pairs:
        return 0.0

    correct = sum(1 for predicted, true in pairs if predicted == true)
    return correct / len(pairs)


def end_to_end_accuracy(records: List[FailureRecord]) -> float:
    """Fraction of records where stage == NONE (i.e., successful)."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.stage == FailureStage.NONE) / len(records)


def _coerce_root_cause_stage(value: FailureRecord | FailureStage | str) -> FailureStage:
    if isinstance(value, FailureRecord):
        return root_cause_stage(value)
    if isinstance(value, FailureStage):
        return value
    return FailureStage(value)


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


# --------------------------------------------------------------------------- #
# Attribution-identifiability + cost metrics (W4 / C2 / C3)                      #
# --------------------------------------------------------------------------- #

def _coerce_truth(item: Any) -> Tuple[FailureStage, int]:
    """Normalize a ground-truth root cause to ``(stage, hop)``.

    Accepts an ``InjectionResult`` (``injected_stage`` / ``injected_at_hop``), a
    ``FailureRecord`` (``stage`` / no hop -> 0), a ``(stage, hop)`` tuple, or a
    bare ``FailureStage`` / stage string (hop -> 0).
    """
    if hasattr(item, "injected_stage") and hasattr(item, "injected_at_hop"):
        return _as_stage(item.injected_stage), int(item.injected_at_hop)
    if isinstance(item, FailureRecord):
        return item.stage, 0
    if isinstance(item, tuple) and len(item) == 2:
        return _as_stage(item[0]), int(item[1])
    return _as_stage(item), 0


def _as_stage(value: Any) -> FailureStage:
    if isinstance(value, FailureStage):
        return value
    return FailureStage(str(value))


def _diagnosis_pair(diag: Any) -> Tuple[FailureStage, int]:
    """Extract ``(stage, predicted_hop)`` from a Diagnosis-like or FailureRecord."""
    if hasattr(diag, "predicted_hop"):
        return _as_stage(diag.stage), int(diag.predicted_hop)
    if isinstance(diag, FailureRecord):
        return diag.stage, 0
    raise TypeError(f"Cannot read a diagnosis from {type(diag)!r}")


def diagnosis_correct(
    diag: Any,
    truth: Any,
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> bool:
    """Whether *diag* matches ground-truth *truth* under *criterion*.

    criterion:
        ``"stage"`` — predicted stage equals injected stage.
        ``"hop"``   — predicted hop within ``hop_tolerance`` of injected hop.
        ``"both"``  — stage and hop both match.
    """
    pred_stage, pred_hop = _diagnosis_pair(diag)
    true_stage, true_hop = _coerce_truth(truth)
    stage_ok = pred_stage == true_stage
    hop_ok = abs(pred_hop - true_hop) <= hop_tolerance
    if criterion == "stage":
        return stage_ok
    if criterion == "hop":
        return hop_ok
    if criterion == "both":
        return stage_ok and hop_ok
    raise ValueError("criterion must be 'stage', 'hop', or 'both'")


def localization_accuracy(
    diagnoses: Sequence[Any],
    truths: Sequence[Any],
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> float:
    """Fraction of diagnoses that correctly localize the root cause."""
    if len(diagnoses) != len(truths):
        raise ValueError("diagnoses and truths must have the same length")
    if not diagnoses:
        return 0.0
    correct = sum(
        1
        for d, t in zip(diagnoses, truths)
        if diagnosis_correct(d, t, criterion=criterion, hop_tolerance=hop_tolerance)
    )
    return correct / len(diagnoses)


def mean_localization_error(diagnoses: Sequence[Any], truths: Sequence[Any]) -> float:
    """Mean absolute hop distance between predicted and injected hop."""
    if len(diagnoses) != len(truths):
        raise ValueError("diagnoses and truths must have the same length")
    if not diagnoses:
        return 0.0
    errs = [
        abs(_diagnosis_pair(d)[1] - _coerce_truth(t)[1])
        for d, t in zip(diagnoses, truths)
    ]
    return sum(errs) / len(errs)


def attribution_identifiability(
    diagnoses_by_depth: Dict[int, Sequence[Any]],
    truths_by_depth: Dict[int, Sequence[Any]],
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> Dict[int, float]:
    """Root-cause-attribution accuracy as a function of injection depth (the C2 curve).

    Returns ``{depth: accuracy}``.  The headline finding is that for post-hoc
    diagnosers this decays toward 0 as depth grows (the failure becomes
    unidentifiable from the final trace once it propagates and is masked), while
    the propagation-aware diagnoser stays high.

    Parameters
    ----------
    diagnoses_by_depth:
        ``{injection_depth: [Diagnosis, ...]}``.
    truths_by_depth:
        ``{injection_depth: [InjectionResult | (stage, hop) | ...]}`` aligned with
        ``diagnoses_by_depth``.
    criterion / hop_tolerance:
        Forwarded to :func:`diagnosis_correct`.
    """
    out: Dict[int, float] = {}
    for depth in sorted(diagnoses_by_depth):
        diags = diagnoses_by_depth[depth]
        truths = truths_by_depth.get(depth, [])
        if len(diags) != len(truths):
            raise ValueError(
                f"depth {depth}: diagnoses ({len(diags)}) and truths ({len(truths)}) differ"
            )
        out[depth] = localization_accuracy(
            diags, truths, criterion=criterion, hop_tolerance=hop_tolerance
        )
    return out


def rescore_identifiability(
    result: Dict[str, Any],
    criterion: str = "stage",
    hop_tolerance: int = 0,
    require_actual_depth: bool = False,
) -> Dict[str, Dict[int, float]]:
    """Recompute per-diagnoser accuracy vs depth from persisted raw diagnoses.

    Lets you evaluate an already-run experiment under a *different* criterion
    (``stage`` / ``hop`` / ``both``) or hop tolerance without re-spending tokens.
    When ``require_actual_depth`` is True, only cases whose persisted truth hop
    equals the depth bucket are scored; this audits older runs whose live injector
    may have clamped short traces into shallower actual intervention hops.

    Returns ``{diagnoser: {depth: accuracy}}``.
    """
    raw = result.get("raw_by_depth", {})
    names = result.get("diagnosers", [])
    out: Dict[str, Dict[int, float]] = {n: {} for n in names}
    for depth_str, entry in raw.items():
        depth = int(depth_str)
        truth = entry.get("truth", [])
        preds_by_name = entry.get("predictions", {})
        indices = list(range(len(truth)))
        if require_actual_depth:
            indices = [
                i for i in indices
                if len(truth[i]) >= 2 and int(truth[i][1]) == depth
            ]
        for name in names:
            preds = preds_by_name.get(name, [])
            valid_indices = [i for i in indices if i < len(preds)]
            n = len(valid_indices)
            if n == 0:
                out[name][depth] = 0.0
                continue
            correct = 0
            for i in valid_indices:
                pred = preds[i]
                tru = truth[i]
                pred_stage, pred_hop = _as_stage(pred[0]), int(pred[1])
                true_stage, true_hop = _as_stage(tru[0]), int(tru[1])
                stage_ok = pred_stage == true_stage
                hop_ok = abs(pred_hop - true_hop) <= hop_tolerance
                if criterion == "stage":
                    ok = stage_ok
                elif criterion == "hop":
                    ok = hop_ok
                elif criterion == "both":
                    ok = stage_ok and hop_ok
                else:
                    raise ValueError("criterion must be 'stage', 'hop', or 'both'")
                correct += int(ok)
            out[name][depth] = correct / n
    return out


def ancestor_hit(diag: Any, truth: Any) -> bool:
    """True when the prediction is at or before the true hop and stages match.

    Useful for propagation paths where an early-hop prediction is causal —
    the diagnoser correctly identified a hop that did contribute to the failure,
    even if not the exact injected hop.
    """
    pred_stage, pred_hop = _diagnosis_pair(diag)
    true_stage, true_hop = _coerce_truth(truth)
    return pred_stage == true_stage and 0 < pred_hop <= true_hop


def ancestor_hit_rate(diagnoses: Sequence[Any], truths: Sequence[Any]) -> float:
    """Fraction of diagnoses that hit a causal ancestor (at/before true hop, same stage)."""
    if len(diagnoses) != len(truths):
        raise ValueError("diagnoses and truths must have the same length")
    if not diagnoses:
        return 0.0
    return sum(1 for d, t in zip(diagnoses, truths) if ancestor_hit(d, t)) / len(diagnoses)


def bootstrap_localization_ci(
    diagnoses: Sequence[Any],
    truths: Sequence[Any],
    criterion: str = "hop",
    hop_tolerance: int = 0,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap 95% CI for localization accuracy.

    Returns ``{"mean": float, "ci_low": float, "ci_high": float, "n": int}``.
    Fully deterministic via ``seed``. Small-n conditions (n < 30) naturally
    produce wide CIs — no additional flag needed to signal low confidence.
    """
    import random

    n = len(diagnoses)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}

    point = localization_accuracy(
        list(diagnoses), list(truths), criterion=criterion, hop_tolerance=hop_tolerance
    )
    rng = random.Random(seed)
    pairs = list(zip(diagnoses, truths))
    boot_means: List[float] = []
    for _ in range(n_boot):
        sample = rng.choices(pairs, k=n)
        ds, ts = zip(*sample)
        boot_means.append(
            localization_accuracy(list(ds), list(ts), criterion=criterion, hop_tolerance=hop_tolerance)
        )
    boot_means.sort()
    low_idx = max(0, int(0.025 * n_boot))
    high_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "mean": point,
        "ci_low": boot_means[low_idx],
        "ci_high": boot_means[high_idx],
        "n": n,
    }


def slice_identifiability(
    result: Dict[str, Any],
    slice_by: str = "intervention_method",
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Compute per-diagnoser accuracy sliced by a metadata field.

    Reads the ``raw_by_depth`` block of a persisted result JSON and groups
    cases by ``slice_by`` (e.g. ``"intervention_method"``,
    ``"injected_failure_type"``, ``"dataset"``), then scores each diagnoser
    independently per slice and depth — no LLM re-calls needed.

    Returns ``{slice_value: {diagnoser: {depth: accuracy}}}``.
    """
    raw = result.get("raw_by_depth", {})
    names = result.get("diagnosers", [])

    slices: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for depth_str, entry in raw.items():
        depth = int(depth_str)
        metadata = entry.get("metadata", [])
        truth = entry.get("truth", [])
        preds_by_name = entry.get("predictions", {})

        for i, meta in enumerate(metadata):
            val = str(meta.get(slice_by, "unknown"))
            slices.setdefault(val, {})
            slices[val].setdefault(depth, {"truth": [], "predictions": {n: [] for n in names}})
            if i < len(truth):
                slices[val][depth]["truth"].append(truth[i])
            for name in names:
                preds = preds_by_name.get(name, [])
                if i < len(preds):
                    slices[val][depth]["predictions"][name].append(preds[i])

    out: Dict[str, Dict[str, Dict[int, float]]] = {}
    for val, by_depth in slices.items():
        out[val] = {n: {} for n in names}
        for depth, entry in by_depth.items():
            t = entry["truth"]
            for name in names:
                p = entry["predictions"].get(name, [])
                n_cases = min(len(p), len(t))
                if n_cases == 0:
                    out[val][name][depth] = 0.0
                    continue
                pred_objs = [
                    type("_D", (), {"stage": _as_stage(x[0]), "predicted_hop": int(x[1])})()
                    for x in p[:n_cases]
                ]
                truth_pairs = [(_as_stage(x[0]), int(x[1])) for x in t[:n_cases]]
                out[val][name][depth] = localization_accuracy(
                    pred_objs, truth_pairs,
                    criterion=criterion, hop_tolerance=hop_tolerance,
                )
    return out


def cost_per_correct_diagnosis(
    diagnoses: Sequence[Any],
    truths: Sequence[Any],
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> Dict[str, float]:
    """Deployability metric: token cost spent per correctly-localized diagnosis.

    Returns ``{"total_cost", "n_correct", "n", "accuracy", "cost_per_correct"}``.
    ``cost_per_correct`` is ``inf`` when no diagnosis is correct (all cost, no
    yield) — the right behaviour for a diagnoser that spends tokens but never
    localizes. Lets the paper trade a method's accuracy against its token budget.
    """
    if len(diagnoses) != len(truths):
        raise ValueError("diagnoses and truths must have the same length")
    total_cost = float(sum(getattr(d, "cost_tokens", 0) for d in diagnoses))
    n = len(diagnoses)
    n_correct = sum(
        1
        for d, t in zip(diagnoses, truths)
        if diagnosis_correct(d, t, criterion=criterion, hop_tolerance=hop_tolerance)
    )
    return {
        "total_cost": total_cost,
        "n_correct": float(n_correct),
        "n": float(n),
        "accuracy": (n_correct / n) if n else 0.0,
        "cost_per_correct": (total_cost / n_correct) if n_correct else float("inf"),
    }


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
