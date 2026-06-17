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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .core import DiagnosticBenchmark, FailureRecord, FailureStage, PipelineTrace
from .evaluate import (
    end_to_end_accuracy,
    root_cause_accuracy,
    severity_weighted_failure_rate,
    stage_attribution_rate,
)
from .injection import FailureInjector, InjectionResult


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
