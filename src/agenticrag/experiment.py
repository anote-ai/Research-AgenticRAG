"""Ablation experiment runner for AgenticRAG failure-propagation studies.

``run_ablation`` drives the full (injection_method × hop) grid on a fixed
set of pipeline traces, diagnoses every injected variant, and returns an
``AblationResult`` whose convenience methods produce:

- Paper benchmark table: method × metric dict (sensitivity, root-cause
  accuracy, severity, stage failure rates)
- Per-method ``records_by_hop`` dict for downstream use with PropagationGraph

Typical usage::

    from agenticrag.core import DiagnosticBenchmark
    from agenticrag.injection import FailureInjector
    from agenticrag.experiment import run_ablation

    result = run_ablation(traces, references, FailureInjector(), DiagnosticBenchmark())
    print(result.metrics_table())
    print(result.sensitivity_table())
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .core import (
    DiagnosticBenchmark,
    FailureRecord,
    FailureStage,
    FailureType,
    PipelineTrace,
    _answer_correct,
    _token_recall,
)
from .corruption import CorruptionRecord
from .corruption import classify_absorption
from .evaluate import (
    cost_per_correct_diagnosis,
    end_to_end_accuracy,
    localization_accuracy,
    root_cause_accuracy,
    severity_weighted_failure_rate,
    stage_attribution_rate,
)
from .injection import FailureInjector, InjectionResult, LiveFailureInjector
from .propagation import counterfactual_recovery_rate


@dataclass
class _CachedDiagnosis:
    stage: FailureStage
    predicted_hop: int
    cost_tokens: int = 0


# ---------------------------------------------------------------------------
# Default noise documents used when no noise_docs are supplied to
# inject_irrelevant_docs.  Chosen to have near-zero token overlap with any
# realistic QA corpus.
# ---------------------------------------------------------------------------

_DEFAULT_NOISE_DOCS: List[str] = [
    "Xylophones are percussion instruments with wooden bars.",
    "The migration patterns of Arctic terns span both polar regions.",
    "Fermentation converts sugars into ethanol under anaerobic conditions.",
]


# ---------------------------------------------------------------------------
# Injection method registry
# ---------------------------------------------------------------------------

# Methods that accept a hop argument (1-based hop index).
HOP_METHODS: List[str] = [
    "inject_empty_retrieval",
    "inject_irrelevant_docs",
]

# Methods that operate at the answer / tool-call level (no hop argument).
ANSWER_METHODS: List[str] = [
    "inject_no_tool_calls",
    "inject_empty_answer",
    "inject_hallucinated_answer",
]

ALL_METHODS: List[str] = HOP_METHODS + ANSWER_METHODS


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class AblationCell:
    """Results for a single (injection_method, hop) cell in the ablation grid.

    Attributes
    ----------
    injection_method:
        Name of the FailureInjector method applied.
    hop:
        1-based hop index for hop-specific injections; 0 for answer-level.
    injected_stage:
        The pipeline stage corrupted by the injection — used as ground-truth
        root cause when computing ``root_cause_accuracy``.
    records:
        Diagnostic records for every injected trace in this cell.
    sensitivity:
        Fraction of injected failures that were detected as non-NONE.
    root_cause_accuracy_score:
        Fraction of records where the diagnosed root-cause stage matches
        ``injected_stage``.
    severity_rate:
        Mean severity of non-NONE records in this cell.
    stage_rates:
        Dict mapping stage.value → fraction of records attributed there.
    """

    injection_method: str
    hop: int
    injected_stage: FailureStage
    records: List[FailureRecord]
    sensitivity: float
    root_cause_accuracy_score: float
    severity_rate: float
    stage_rates: Dict[str, float]


@dataclass
class AblationResult:
    """Full results from running the ablation grid on a fixed trace set.

    Attributes
    ----------
    n_samples:
        Number of traces used.
    baseline_records:
        Diagnostic records from the unmodified (clean) traces.
    cells:
        One ``AblationCell`` per (method, hop) pair actually run.
    """

    n_samples: int
    baseline_records: List[FailureRecord]
    cells: List[AblationCell]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def cell(self, method: str, hop: int = 0) -> Optional[AblationCell]:
        """Return the cell for *method* × *hop*, or None if not present."""
        for c in self.cells:
            if c.injection_method == method and c.hop == hop:
                return c
        return None

    def cells_for_method(self, method: str) -> List[AblationCell]:
        """All cells for *method*, ordered by hop index."""
        return sorted(
            [c for c in self.cells if c.injection_method == method],
            key=lambda c: c.hop,
        )

    # ------------------------------------------------------------------
    # Paper-ready outputs
    # ------------------------------------------------------------------

    def sensitivity_table(self) -> Dict[str, float]:
        """Detection rate per (method, hop) key.

        Keys are formatted as ``"method"`` for answer-level injections and
        ``"method@hop{N}"`` for hop-specific ones.
        """
        table: Dict[str, float] = {}
        for c in self.cells:
            key = c.injection_method if c.hop == 0 else f"{c.injection_method}@hop{c.hop}"
            table[key] = c.sensitivity
        return table

    def metrics_table(self) -> Dict[str, Dict[str, float]]:
        """Full metrics per (method, hop) cell.

        Returns a dict keyed by the same string as ``sensitivity_table``.
        Each value is a dict with keys: ``sensitivity``,
        ``root_cause_accuracy``, ``severity_rate``, and one key per stage
        named ``stage_<stage_value>``.
        """
        table: Dict[str, Dict[str, float]] = {}
        for c in self.cells:
            key = c.injection_method if c.hop == 0 else f"{c.injection_method}@hop{c.hop}"
            row: Dict[str, float] = {
                "sensitivity": c.sensitivity,
                "root_cause_accuracy": c.root_cause_accuracy_score,
                "severity_rate": c.severity_rate,
            }
            for stage_name, rate in c.stage_rates.items():
                row[f"stage_{stage_name}"] = rate
            table[key] = row
        return table

    def records_by_hop(self, method: str) -> Dict[int, List[FailureRecord]]:
        """Return ``{hop: records}`` for *method* — ready for PropagationGraph.

        Only hop-specific methods (HOP_METHODS) produce multiple hops.
        Answer-level methods return a single-entry dict with key 0.
        """
        return {c.hop: c.records for c in self.cells_for_method(method)}

    def baseline_accuracy(self) -> float:
        """End-to-end accuracy of the unmodified pipeline."""
        return end_to_end_accuracy(self.baseline_records)

    def baseline_severity(self) -> float:
        """Severity-weighted failure rate of the unmodified pipeline."""
        return severity_weighted_failure_rate(self.baseline_records)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_ablation(
    traces: List[PipelineTrace],
    references: List[Dict[str, Any]],
    injector: FailureInjector,
    benchmark: DiagnosticBenchmark,
    methods: Optional[List[str]] = None,
    hops: Optional[List[int]] = None,
    noise_docs: Optional[List[str]] = None,
) -> AblationResult:
    """Run the (method × hop) ablation grid and return a structured result.

    Parameters
    ----------
    traces:
        Clean (unmodified) pipeline traces to use as the baseline.
    references:
        Corresponding reference dicts — same length as *traces*.
        Each must have at least an ``"answer"`` key.
    injector:
        ``FailureInjector`` instance.
    benchmark:
        ``DiagnosticBenchmark`` used to diagnose every injected trace.
    methods:
        Injection methods to run.  Defaults to ``ALL_METHODS`` (all five).
    hops:
        Hop indices for ``HOP_METHODS``.  Defaults to ``[1, 2, 3]``.
        Values larger than the number of hops in any trace are silently
        skipped for that trace (the injector handles out-of-range hops
        gracefully — it no-ops on missing hop indices).
    noise_docs:
        Documents used as irrelevant replacements for
        ``inject_irrelevant_docs``.  Defaults to ``_DEFAULT_NOISE_DOCS``.

    Returns
    -------
    ``AblationResult`` with baseline records and one ``AblationCell`` per
    (method, hop) pair.

    Raises
    ------
    ValueError
        If ``traces`` and ``references`` have different lengths, or if
        ``traces`` is empty.
    """
    if len(traces) != len(references):
        raise ValueError(
            f"traces and references must have the same length "
            f"(got {len(traces)} and {len(references)})"
        )
    if not traces:
        raise ValueError("traces must be non-empty")

    if methods is None:
        methods = list(ALL_METHODS)
    if hops is None:
        hops = [1, 2, 3]
    if noise_docs is None:
        noise_docs = list(_DEFAULT_NOISE_DOCS)

    # Baseline diagnosis (no injection)
    baseline_records = benchmark.batch_diagnose(traces, references)

    cells: List[AblationCell] = []

    for method in methods:
        if method in HOP_METHODS:
            for hop in hops:
                cell = _run_cell(
                    traces, references, injector, benchmark,
                    method=method, hop=hop, noise_docs=noise_docs,
                )
                cells.append(cell)
        elif method in ANSWER_METHODS:
            cell = _run_cell(
                traces, references, injector, benchmark,
                method=method, hop=0, noise_docs=noise_docs,
            )
            cells.append(cell)
        else:
            raise ValueError(
                f"Unknown injection method '{method}'. "
                f"Choose from: {ALL_METHODS}"
            )

    return AblationResult(
        n_samples=len(traces),
        baseline_records=baseline_records,
        cells=cells,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_cell(
    traces: List[PipelineTrace],
    references: List[Dict[str, Any]],
    injector: FailureInjector,
    benchmark: DiagnosticBenchmark,
    method: str,
    hop: int,
    noise_docs: List[str],
) -> AblationCell:
    """Run one (method, hop) cell and return an ``AblationCell``."""
    inject_fn = getattr(injector, method)

    injection_results: List[InjectionResult] = []
    for trace in traces:
        kwargs: Dict[str, Any] = {}
        if hop > 0:
            kwargs["hop"] = hop
        if method == "inject_irrelevant_docs":
            kwargs["noise_docs"] = noise_docs
        injection_results.append(inject_fn(trace, **kwargs))

    injected_traces = [r.injected_trace for r in injection_results]
    records = benchmark.batch_diagnose(injected_traces, references)

    # Ground-truth root cause: the stage that was injected
    # (uniform across all traces for a given method × hop cell)
    injected_stage = injection_results[0].injected_stage
    true_stages = [injected_stage] * len(records)

    n = len(records)
    detected = sum(1 for r in records if r.stage != FailureStage.NONE)
    sensitivity = detected / n if n > 0 else 0.0
    rca = root_cause_accuracy(records, true_stages)
    severity = severity_weighted_failure_rate(records)
    stage_rates = {
        stage.value: stage_attribution_rate(records, stage)
        for stage in FailureStage
        if stage != FailureStage.NONE
    }

    return AblationCell(
        injection_method=method,
        hop=hop,
        injected_stage=injected_stage,
        records=records,
        sensitivity=sensitivity,
        root_cause_accuracy_score=rca,
        severity_rate=severity,
        stage_rates=stage_rates,
    )


# ---------------------------------------------------------------------------
# Identifiability experiment (C2 headline + C3 method comparison)
# ---------------------------------------------------------------------------

# Live interventions used for the depth curve by default. Each corrupts retrieval
# at the chosen hop; the agent then reacts for real (it may self-correct).
_DEFAULT_LIVE_METHODS: List[str] = [
    "inject_irrelevant_docs",
    "inject_empty_retrieval",
    "inject_false_premise",
    "inject_stale_evidence",
]


@dataclass
class IdentifiabilityResult:
    """Per-diagnoser root-cause-attribution accuracy vs injection depth.

    ``accuracy[diagnoser][depth]`` is the localization accuracy over the *failed*
    injected traces at that depth (you can only attribute a failure that
    occurred). ``recovery_rate_by_depth`` reports the complementary counterfactual
    recovery (faults the agent absorbed). ``cost[diagnoser][depth]`` carries the
    cost-per-correct-diagnosis breakdown for the deployability angle.
    """

    hops: List[int]
    diagnoser_names: List[str]
    accuracy: Dict[str, Dict[int, float]]
    cost: Dict[str, Dict[int, Dict[str, float]]]
    recovery_rate_by_depth: Dict[int, float]
    n_total_by_depth: Dict[int, int]
    n_failed_by_depth: Dict[int, int]
    n_eligible_by_depth: Dict[int, int] = field(default_factory=dict)
    n_skipped_short_trace_by_depth: Dict[int, int] = field(default_factory=dict)
    n_skipped_base_incorrect_by_depth: Dict[int, int] = field(default_factory=dict)
    # (method, depth) pairs skipped because no certifiable corruption span was
    # found in the hop's docs (only inject_corrupted_evidence can skip this way).
    n_skipped_no_span_by_depth: Dict[int, int] = field(default_factory=dict)
    # Per-depth raw diagnoses + ground truth, so accuracy can be re-scored under
    # a different criterion (stage / hop_tolerance) without re-spending tokens.
    # raw_by_depth[depth] = {"truth": [[stage, hop], ...],
    #                        "predictions": {name: [[stage, hop, cost], ...]},
    #                        "metadata": [{case metadata}, ...]}
    raw_by_depth: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def accuracy_table(self) -> Dict[str, Dict[str, float]]:
        """``{diagnoser: {"hop{d}": acc}}`` — ready for a paper table."""
        return {
            name: {f"hop{d}": self.accuracy[name].get(d, 0.0) for d in self.hops}
            for name in self.diagnoser_names
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hops": self.hops,
            "diagnosers": self.diagnoser_names,
            "accuracy": {n: {str(k): v for k, v in self.accuracy[n].items()} for n in self.diagnoser_names},
            "cost": {n: {str(k): v for k, v in self.cost[n].items()} for n in self.diagnoser_names},
            "recovery_rate_by_depth": {str(k): v for k, v in self.recovery_rate_by_depth.items()},
            "n_total_by_depth": {str(k): v for k, v in self.n_total_by_depth.items()},
            "n_failed_by_depth": {str(k): v for k, v in self.n_failed_by_depth.items()},
            "n_eligible_by_depth": {str(k): v for k, v in self.n_eligible_by_depth.items()},
            "n_skipped_short_trace_by_depth": {
                str(k): v for k, v in self.n_skipped_short_trace_by_depth.items()
            },
            "n_skipped_base_incorrect_by_depth": {
                str(k): v for k, v in self.n_skipped_base_incorrect_by_depth.items()
            },
            "n_skipped_no_span_by_depth": {
                str(k): v for k, v in self.n_skipped_no_span_by_depth.items()
            },
            "raw_by_depth": {str(k): v for k, v in self.raw_by_depth.items()},
        }


def _cache_paths(checkpoint_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not checkpoint_path:
        return None, None
    return checkpoint_path + ".cases.jsonl", checkpoint_path + ".diagnoses.jsonl"


def _json_signature(extra: Optional[Dict[str, Any]], exclude: Sequence[str] = ()) -> str:
    payload = dict(extra or {})
    for key in exclude:
        payload.pop(key, None)
    return json.dumps(payload, sort_keys=True, default=str)


def _case_cache_signature(extra: Optional[Dict[str, Any]]) -> str:
    return _json_signature(
        extra,
        exclude=("diagnosers_run", "judge", "injection_methods"),
    )


def _diagnosis_cache_signature(extra: Optional[Dict[str, Any]]) -> str:
    return _json_signature(extra, exclude=("diagnosers_run",))


def _append_jsonl(path: Optional[str], record: Dict[str, Any]) -> None:
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a") as f:
        json.dump(record, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _load_jsonl(path: Optional[str], signature: str) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("cache_signature") == signature:
                    records.append(record)
    except OSError:
        return []
    return records


def _serialize_injection_result(result: InjectionResult) -> Dict[str, Any]:
    return {
        "original_trace_id": result.original_trace_id,
        "injected_trace": result.injected_trace.model_dump(),
        "injected_stage": result.injected_stage.value,
        "injected_failure_type": result.injected_failure_type.value,
        "injected_at_hop": result.injected_at_hop,
        "corruption": result.corruption.to_dict() if result.corruption is not None else None,
    }


def _deserialize_injection_result(data: Dict[str, Any]) -> InjectionResult:
    corruption_data = data.get("corruption")
    corruption = CorruptionRecord(**corruption_data) if corruption_data else None
    return InjectionResult(
        original_trace_id=data["original_trace_id"],
        injected_trace=PipelineTrace(**data["injected_trace"]),
        injected_stage=FailureStage(data["injected_stage"]),
        injected_failure_type=FailureType(data["injected_failure_type"]),
        injected_at_hop=int(data["injected_at_hop"]),
        corruption=corruption,
    )


def _load_case_cache(
    path: Optional[str],
    signature: str,
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int, str], Dict[str, Any]]]:
    statuses: Dict[Tuple[int, int], Dict[str, Any]] = {}
    cases: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for record in _load_jsonl(path, signature):
        record_type = record.get("record_type")
        depth = int(record.get("depth", 0))
        sample_idx = int(record.get("sample_idx", -1))
        if record_type == "sample_status":
            statuses[(depth, sample_idx)] = record
        elif record_type == "case":
            method = str(record.get("method", ""))
            cases[(depth, sample_idx, method)] = record
    return statuses, cases


def _load_diagnosis_cache(
    path: Optional[str],
    signature: str,
) -> Dict[Tuple[int, int, str, str], List[Any]]:
    out: Dict[Tuple[int, int, str, str], List[Any]] = {}
    for record in _load_jsonl(path, signature):
        if record.get("record_type") != "diagnosis":
            continue
        key = (
            int(record.get("depth", 0)),
            int(record.get("sample_idx", -1)),
            str(record.get("method", "")),
            str(record.get("diagnoser", "")),
        )
        prediction = record.get("prediction")
        if isinstance(prediction, list) and len(prediction) >= 3:
            out[key] = prediction
    return out


def _diagnosis_from_prediction(prediction: List[Any]) -> _CachedDiagnosis:
    return _CachedDiagnosis(
        stage=FailureStage(prediction[0]),
        predicted_hop=int(prediction[1]),
        cost_tokens=int(prediction[2]),
    )


def run_identifiability(
    agent: Any,
    samples: Sequence[Any],
    diagnosers: Dict[str, Any],
    hops: Sequence[int] = (1, 2, 3),
    injection_methods: Optional[List[str]] = None,
    injector: Optional[LiveFailureInjector] = None,
    criterion: str = "hop",
    hop_tolerance: int = 0,
    min_corpus: int = 1,
    strict_depth: bool = True,
    require_base_correct: bool = True,
    checkpoint_path: Optional[str] = None,
    checkpoint_extra: Optional[Dict[str, Any]] = None,
    resume: bool = False,
) -> IdentifiabilityResult:
    """Run the attribution-identifiability experiment (C2 curve + C3 comparison).

    For each injection ``depth`` and each :class:`QASample`, the agent produces a
    base trajectory; each live injection method corrupts retrieval at ``depth``
    and the agent re-runs the suffix (live re-execution). Every diagnoser then
    attempts to localize the certified root cause. The headline output is
    ``accuracy[diagnoser][depth]`` — post-hoc baselines decay with depth; the
    propagation-aware diagnoser should hold up.

    Parameters
    ----------
    agent:
        A resumable :class:`~agenticrag.agents.LLMAgent`.
    samples:
        :class:`QASample` objects (``question``, ``answer``, ``supporting_docs``).
    diagnosers:
        ``{name: diagnoser}``. The propagation-aware diagnoser reads the corpus
        from ``reference["corpus"]``, which this driver supplies.
    hops:
        Injection depths to sweep.
    injection_methods:
        Live ``LiveFailureInjector`` method names. Defaults to the retrieval
        corruption family.
    criterion / hop_tolerance:
        Forwarded to the localization metric ('hop', 'stage', or 'both').
    min_corpus:
        Skip samples whose corpus is smaller than this (need docs to retrieve).
    strict_depth:
        When True, a requested injection depth is eligible only if the base trace
        actually reached that many hops. This prevents depth-2/3 buckets from
        silently containing clamped hop-1 interventions on shallow traces.
    require_base_correct:
        When True, only inject into base traces that answered correctly. Otherwise
        a natural base failure can be miscounted as a causal effect of injection.
    checkpoint_path:
        If set, the partial result is written here as JSON after *each depth*
        completes, so an API error (e.g. credit exhaustion) mid-sweep never
        discards already-computed depths.
    checkpoint_extra:
        Extra keys merged into the checkpoint JSON (e.g. provider / dataset tags).
    """
    injector = injector or LiveFailureInjector(agent)
    methods = injection_methods or list(_DEFAULT_LIVE_METHODS)
    names = list(diagnosers.keys())

    accuracy: Dict[str, Dict[int, float]] = {n: {} for n in names}
    cost: Dict[str, Dict[int, Dict[str, float]]] = {n: {} for n in names}
    recovery: Dict[int, float] = {}
    n_total: Dict[int, int] = {}
    n_failed: Dict[int, int] = {}
    n_eligible: Dict[int, int] = {}
    n_skipped_short_trace: Dict[int, int] = {}
    n_skipped_base_incorrect: Dict[int, int] = {}
    n_skipped_no_span: Dict[int, int] = {}
    raw: Dict[int, Dict[str, Any]] = {}

    # Per-depth resume: preload depths already computed in a matching checkpoint,
    # so an interrupted / retried run skips them instead of re-spending tokens.
    done_depths = _preload_checkpoint(
        checkpoint_path if resume else None, checkpoint_extra, names,
        accuracy, cost, recovery, n_total, n_failed, raw,
        n_eligible, n_skipped_short_trace, n_skipped_base_incorrect,
        n_skipped_no_span,
    )
    case_cache_path, diagnosis_cache_path = _cache_paths(checkpoint_path)
    case_sig = _case_cache_signature(checkpoint_extra)
    diagnosis_sig = _diagnosis_cache_signature(checkpoint_extra)
    cached_statuses: Dict[Tuple[int, int], Dict[str, Any]] = {}
    cached_cases: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    cached_diagnoses: Dict[Tuple[int, int, str, str], List[Any]] = {}
    if resume:
        cached_statuses, cached_cases = _load_case_cache(case_cache_path, case_sig)
        cached_diagnoses = _load_diagnosis_cache(diagnosis_cache_path, diagnosis_sig)

    def _snapshot() -> IdentifiabilityResult:
        return IdentifiabilityResult(
            hops=list(hops),
            diagnoser_names=names,
            accuracy=accuracy,
            cost=cost,
            recovery_rate_by_depth=recovery,
            n_total_by_depth=n_total,
            n_failed_by_depth=n_failed,
            n_eligible_by_depth=n_eligible,
            n_skipped_short_trace_by_depth=n_skipped_short_trace,
            n_skipped_base_incorrect_by_depth=n_skipped_base_incorrect,
            n_skipped_no_span_by_depth=n_skipped_no_span,
            raw_by_depth=raw,
        )

    for depth in hops:
        if depth in done_depths:
            continue  # already computed in a matching checkpoint (resume)
        injected: List[InjectionResult] = []
        refs: List[Dict[str, Any]] = []
        metadata: List[Dict[str, Any]] = []
        eligible_samples = 0
        skipped_short = 0
        skipped_base_wrong = 0
        skipped_no_span = 0
        for sample_idx, s in enumerate(samples):
            corpus = list(s.supporting_docs)
            if len(corpus) < min_corpus:
                continue
            status_record = cached_statuses.get((depth, sample_idx))
            base: Optional[PipelineTrace] = None
            if status_record is not None:
                status = status_record.get("status")
                if status == "skipped_short":
                    skipped_short += 1
                    continue
                if status == "skipped_base_incorrect":
                    skipped_base_wrong += 1
                    continue
                if status == "eligible":
                    eligible_samples += 1
                    base_dump = status_record.get("base_trace")
                    base = PipelineTrace(**base_dump) if base_dump else None
                    base_hops = int(status_record.get("base_hop_count", 0))
                    base_correct = bool(status_record.get("base_correct", True))
                else:
                    status_record = None

            if status_record is None:
                base = agent.run(s.question, corpus, reference_answer=s.answer)
                base_hops = len(base.hop_docs)
                if strict_depth and base_hops < depth:
                    skipped_short += 1
                    _append_jsonl(case_cache_path, {
                        "cache_signature": case_sig,
                        "record_type": "sample_status",
                        "depth": depth,
                        "sample_idx": sample_idx,
                        "sample_id": getattr(s, "sample_id", ""),
                        "status": "skipped_short",
                        "base_hop_count": base_hops,
                    })
                    continue
                base_correct = _answer_correct(base.final_answer, s.answer)
                if require_base_correct and not base_correct:
                    skipped_base_wrong += 1
                    _append_jsonl(case_cache_path, {
                        "cache_signature": case_sig,
                        "record_type": "sample_status",
                        "depth": depth,
                        "sample_idx": sample_idx,
                        "sample_id": getattr(s, "sample_id", ""),
                        "status": "skipped_base_incorrect",
                        "base_hop_count": base_hops,
                        "base_answer": base.final_answer,
                    })
                    continue
                eligible_samples += 1
                _append_jsonl(case_cache_path, {
                    "cache_signature": case_sig,
                    "record_type": "sample_status",
                    "depth": depth,
                    "sample_idx": sample_idx,
                    "sample_id": getattr(s, "sample_id", ""),
                    "status": "eligible",
                    "base_hop_count": base_hops,
                    "base_correct": base_correct,
                    "base_answer": base.final_answer,
                    "base_trace": base.model_dump(),
                })

            for method in methods:
                case_record = cached_cases.get((depth, sample_idx, method))
                if case_record is not None:
                    if case_record.get("skipped_no_span"):
                        skipped_no_span += 1
                        continue
                    res = _deserialize_injection_result(case_record["injection_result"])
                    ref = case_record["ref"]
                    case_meta = dict(case_record["metadata"])
                else:
                    if base is None:
                        base_dump = status_record.get("base_trace") if status_record else None
                        base = PipelineTrace(**base_dump) if base_dump else agent.run(
                            s.question, corpus, reference_answer=s.answer
                        )
                        base_hops = len(base.hop_docs)
                        base_correct = _answer_correct(base.final_answer, s.answer)
                    res = getattr(injector, method)(base, corpus, hop=depth)
                    if res is None:
                        # No certifiable corruption span at this hop — skip rather
                        # than inject an unverifiable fault.
                        skipped_no_span += 1
                        _append_jsonl(case_cache_path, {
                            "cache_signature": case_sig,
                            "record_type": "case",
                            "depth": depth,
                            "sample_idx": sample_idx,
                            "sample_id": getattr(s, "sample_id", ""),
                            "method": method,
                            "skipped_no_span": True,
                        })
                        continue
                    ref = {"answer": s.answer, "corpus": corpus}
                    final_correct = _answer_correct(res.injected_trace.final_answer, s.answer)
                    case_meta = {
                        "sample_idx": sample_idx,
                        "sample_id": getattr(s, "sample_id", ""),
                        "dataset": getattr(s, "dataset", ""),
                        "question": getattr(s, "question", ""),
                        "reference_answer": getattr(s, "answer", ""),
                        "requested_depth": depth,
                        "actual_hop": res.injected_at_hop,
                        "base_hop_count": base_hops,
                        "declared_hop_count": getattr(s, "hop_count", None),
                        "base_correct": base_correct,
                        "base_answer": base.final_answer,
                        "intervention_method": method,
                        "injected_failure_type": res.injected_failure_type.value,
                        "injected_stage": res.injected_stage.value,
                        "final_correct": final_correct,
                        "recovered": final_correct,
                        "final_answer": res.injected_trace.final_answer,
                        "iterations_used": res.injected_trace.iterations_used,
                    }
                    if res.corruption is not None:
                        c = res.corruption
                        corrupted_span = c.corrupted_span
                        case_meta.update(c.to_dict())
                        # Deterministic generation-level evaluation against the
                        # certified spans — no LLM judge involved.
                        case_meta["absorption"] = classify_absorption(
                            res.injected_trace.final_answer, s.answer, corrupted_span
                        )
                        # Did the corrupted value leak into later sub-queries?
                        # (Direct evidence of the propagation mechanism.)
                        case_meta["query_contaminated"] = any(
                            _token_recall(q, corrupted_span) >= 1.0
                            for q in res.injected_trace.hop_queries[depth:]
                        )
                    _append_jsonl(case_cache_path, {
                        "cache_signature": case_sig,
                        "record_type": "case",
                        "depth": depth,
                        "sample_idx": sample_idx,
                        "sample_id": getattr(s, "sample_id", ""),
                        "method": method,
                        "skipped_no_span": False,
                        "ref": ref,
                        "injection_result": _serialize_injection_result(res),
                        "metadata": case_meta,
                    })
                injected.append(res)
                refs.append(ref)
                metadata.append(case_meta)

        n_eligible[depth] = eligible_samples
        n_skipped_short_trace[depth] = skipped_short
        n_skipped_base_incorrect[depth] = skipped_base_wrong
        n_skipped_no_span[depth] = skipped_no_span
        n_total[depth] = len(injected)
        recovery[depth] = counterfactual_recovery_rate(injected, refs) if injected else 0.0

        # Restrict RCA to traces where the injected fault actually caused a wrong
        # answer — a recovered trace has no failure to attribute.
        failed = [
            (r, ref, meta)
            for r, ref, meta in zip(injected, refs, metadata)
            if not _answer_correct(r.injected_trace.final_answer, ref["answer"])
        ]
        n_failed[depth] = len(failed)
        f_results = [r for r, _, _ in failed]
        f_refs = [ref for _, ref, _ in failed]
        f_metadata = [meta for _, _, meta in failed]

        truth = [[r.injected_stage.value, r.injected_at_hop] for r in f_results]
        predictions: Dict[str, List[List[Any]]] = {}
        for name, diag in diagnosers.items():
            diags: List[Any] = []
            for r, ref, meta in zip(f_results, f_refs, f_metadata):
                cache_key = (
                    depth,
                    int(meta.get("sample_idx", -1)),
                    str(meta.get("intervention_method", "")),
                    name,
                )
                cached_prediction = cached_diagnoses.get(cache_key)
                if cached_prediction is not None:
                    diags.append(_diagnosis_from_prediction(cached_prediction))
                    continue
                d = diag.diagnose(r.injected_trace, ref)
                prediction = [d.stage.value, d.predicted_hop, d.cost_tokens]
                _append_jsonl(diagnosis_cache_path, {
                    "cache_signature": diagnosis_sig,
                    "record_type": "diagnosis",
                    "depth": depth,
                    "sample_idx": int(meta.get("sample_idx", -1)),
                    "sample_id": meta.get("sample_id", ""),
                    "method": meta.get("intervention_method", ""),
                    "diagnoser": name,
                    "prediction": prediction,
                })
                cached_diagnoses[cache_key] = prediction
                diags.append(d)
            accuracy[name][depth] = localization_accuracy(
                diags, f_results, criterion=criterion, hop_tolerance=hop_tolerance
            )
            cost[name][depth] = cost_per_correct_diagnosis(
                diags, f_results, criterion=criterion, hop_tolerance=hop_tolerance
            )
            predictions[name] = [[d.stage.value, d.predicted_hop, d.cost_tokens] for d in diags]
        # Recovered (injected-but-still-correct) cases carry no diagnosis but do
        # carry the absorption label, so persist their metadata separately —
        # absorption rates are then recomputable from the JSON at zero cost.
        recovered_metadata = [
            meta
            for r, ref, meta in zip(injected, refs, metadata)
            if _answer_correct(r.injected_trace.final_answer, ref["answer"])
        ]
        raw[depth] = {
            "truth": truth,
            "predictions": predictions,
            "metadata": f_metadata,
            "recovered_metadata": recovered_metadata,
            "requested_depth": depth,
            "n_eligible": eligible_samples,
            "n_skipped_short_trace": skipped_short,
            "n_skipped_base_incorrect": skipped_base_wrong,
            "n_skipped_no_span": skipped_no_span,
            "n_recovered": len(injected) - len(failed),
        }

        if checkpoint_path:
            _write_checkpoint(checkpoint_path, _snapshot(), checkpoint_extra)

    return _snapshot()


def _preload_checkpoint(
    path: Optional[str],
    extra: Optional[Dict[str, Any]],
    names: List[str],
    accuracy: Dict[str, Dict[int, float]],
    cost: Dict[str, Dict[int, Dict[str, float]]],
    recovery: Dict[int, float],
    n_total: Dict[int, int],
    n_failed: Dict[int, int],
    raw: Dict[int, Dict[str, Any]],
    n_eligible: Optional[Dict[int, int]] = None,
    n_skipped_short_trace: Optional[Dict[int, int]] = None,
    n_skipped_base_incorrect: Optional[Dict[int, int]] = None,
    n_skipped_no_span: Optional[Dict[int, int]] = None,
) -> set:
    """Load already-computed depths from a *matching* checkpoint; return their set.

    Only reuses depths when the checkpoint's run config (``max_samples`` /
    ``propagation_budget`` / ``retriever`` — whatever keys are in ``extra``)
    matches the current run, so a config change safely recomputes from scratch.
    """
    if not path or not os.path.exists(path):
        return set()
    try:
        with open(path) as f:
            prior = json.load(f)
    except (OSError, json.JSONDecodeError):
        return set()

    for key in (
        "max_samples",
        "propagation_budget",
        "retriever",
        "strict_depth",
        "require_base_correct",
        "injection_methods",
    ):
        if extra is not None and key in extra and prior.get(key) != extra.get(key):
            return set()  # config differs — do not reuse

    def _iload(d: Dict[str, Any]) -> Dict[int, Any]:
        return {int(k): v for k, v in d.items()}

    for name in names:
        accuracy[name].update(_iload(prior.get("accuracy", {}).get(name, {})))
        cost[name].update(_iload(prior.get("cost", {}).get(name, {})))
    recovery.update(_iload(prior.get("recovery_rate_by_depth", {})))
    n_total.update(_iload(prior.get("n_total_by_depth", {})))
    n_failed.update(_iload(prior.get("n_failed_by_depth", {})))
    raw.update(_iload(prior.get("raw_by_depth", {})))
    if n_eligible is not None:
        n_eligible.update(_iload(prior.get("n_eligible_by_depth", {})))
    if n_skipped_short_trace is not None:
        n_skipped_short_trace.update(
            _iload(prior.get("n_skipped_short_trace_by_depth", {}))
        )
    if n_skipped_base_incorrect is not None:
        n_skipped_base_incorrect.update(
            _iload(prior.get("n_skipped_base_incorrect_by_depth", {}))
        )
    if n_skipped_no_span is not None:
        n_skipped_no_span.update(_iload(prior.get("n_skipped_no_span_by_depth", {})))
    return set(raw.keys())


def _write_checkpoint(
    path: str, result: IdentifiabilityResult, extra: Optional[Dict[str, Any]]
) -> None:
    payload = dict(extra or {})
    payload.update(result.to_dict())
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)  # atomic; a crash never leaves a truncated file
