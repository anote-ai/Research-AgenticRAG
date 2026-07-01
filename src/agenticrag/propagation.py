"""PropagationGraph: causal model of failure propagation in agentic RAG pipelines.

Builds a directed graph from hop-grouped FailureRecords (output of
DiagnosticBenchmark.batch_diagnose run per hop) and exposes metrics that
quantify *how* failures cascade across pipeline stages.

Typical usage::

    from agenticrag.propagation import PropagationGraph
    from agenticrag.core import DiagnosticBenchmark, FailureStage

    bench = DiagnosticBenchmark()
    records_by_hop = {
        1: bench.batch_diagnose(hop1_traces, refs),
        2: bench.batch_diagnose(hop2_traces, refs),
        3: bench.batch_diagnose(hop3_traces, refs),
    }

    graph = PropagationGraph()
    graph.infer_from_hops(records_by_hop)

    print(graph.stage_coupling(FailureStage.RETRIEVAL, FailureStage.ANSWER_GENERATION))
    print(graph.summary())
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .core import FailureRecord, FailureStage, _answer_correct


@dataclass(frozen=True)
class PropagationEdge:
    """Directed causal edge from one failure stage to another.

    ``weight`` is the empirical conditional probability P(target | source),
    estimated from the ingested hop records.  ``count`` is the raw
    co-occurrence tally.
    """

    source_stage: FailureStage
    target_stage: FailureStage
    weight: float
    count: int


class PropagationGraph:
    """Directed graph of failure propagation across pipeline stages.

    Nodes are pipeline stages (FailureStage values).  A directed edge
    source → target with weight *w* means: in fraction *w* of traces where
    a failure at ``source`` was observed at hop H, a failure (at any stage)
    was also observed at hop H+1, attributed to ``target``.

    The graph is built incrementally via :meth:`infer_from_hops` and can be
    queried with :meth:`stage_coupling`, :meth:`propagation_depth`,
    :meth:`critical_path`, :meth:`failure_rate_by_stage`, and
    :meth:`summary`.
    """

    def __init__(self) -> None:
        # _edge_counts[src][tgt] = number of consecutive-hop (src→tgt) transitions
        self._edge_counts: Dict[FailureStage, Dict[FailureStage, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # _failure_counts[stage] = total (hop, trace) cells with a failure at stage
        self._failure_counts: Dict[FailureStage, int] = defaultdict(int)
        # _propagatable_counts[stage] = failures at stage that were NOT at the final
        # hop of their call — only these can be the source of a propagation edge.
        # Used as the denominator for stage_coupling so that self-correcting traces
        # (failure at hop H, success at hop H+1) correctly reduce the coupling score.
        self._propagatable_counts: Dict[FailureStage, int] = defaultdict(int)
        # Per-trace ordered list of (hop, stage) failure events
        self._trace_paths: List[List[Tuple[int, FailureStage]]] = []
        # Number of traces ingested (across all infer_from_hops calls)
        self._total_traces: int = 0
        # Highest hop index seen across all infer_from_hops calls; used by
        # self_correction_rate to determine whether a trace resolved before the end.
        self._max_hop_seen: int = 0

    # ------------------------------------------------------------------ #
    # Graph construction                                                    #
    # ------------------------------------------------------------------ #

    def infer_from_hops(
        self, records_by_hop: Dict[int, List[FailureRecord]]
    ) -> None:
        """Build propagation edges from hop-grouped diagnostic records.

        For each trace at index *i*, walks through hops in ascending order.
        Whenever a non-NONE failure at hop H is followed by another non-NONE
        failure at hop H+1, an edge source_stage → target_stage is recorded.

        Parameters
        ----------
        records_by_hop:
            Mapping from 1-based hop index to a list of FailureRecords.
            All lists must have the same length (one record per trace per hop).
            Hops need not be contiguous — only consecutive pairs in the
            sorted hop order are considered.
        """
        if not records_by_hop:
            return

        hops = sorted(records_by_hop.keys())
        n_traces = min(len(records_by_hop[h]) for h in hops)
        self._total_traces += n_traces
        self._max_hop_seen = max(self._max_hop_seen, hops[-1])

        for idx in range(n_traces):
            path: List[Tuple[int, FailureStage]] = []

            for hop_pos, hop in enumerate(hops):
                record = records_by_hop[hop][idx]
                is_last_hop = hop_pos == len(hops) - 1
                if record.stage != FailureStage.NONE:
                    path.append((hop, record.stage))
                    self._failure_counts[record.stage] += 1
                    if not is_last_hop:
                        # This failure occurred before the final hop in this call,
                        # so it had an opportunity to propagate (or self-correct).
                        self._propagatable_counts[record.stage] += 1

            self._trace_paths.append(path)

            # Emit an edge for each consecutive failing pair
            for i in range(len(path) - 1):
                src_stage = path[i][1]
                tgt_stage = path[i + 1][1]
                self._edge_counts[src_stage][tgt_stage] += 1

    # ------------------------------------------------------------------ #
    # Graph queries                                                         #
    # ------------------------------------------------------------------ #

    def edges(self) -> List[PropagationEdge]:
        """All inferred propagation edges with empirical weights.

        Returns an empty list if no propagation has been observed.
        """
        result: List[PropagationEdge] = []
        for src, targets in self._edge_counts.items():
            src_total = self._propagatable_counts.get(src, 0)
            for tgt, count in targets.items():
                weight = count / src_total if src_total > 0 else 0.0
                result.append(
                    PropagationEdge(
                        source_stage=src,
                        target_stage=tgt,
                        weight=weight,
                        count=count,
                    )
                )
        return result

    def stage_coupling(
        self,
        source: FailureStage,
        target: FailureStage,
    ) -> float:
        """P(target stage fails at hop H+1 | source stage failed at hop H).

        The denominator is the number of source-stage failures that were NOT at
        the final hop of their call — i.e., failures that had an opportunity to
        either propagate or self-correct.  This ensures that a trace which
        self-corrects (failure at H, success at H+1) correctly reduces the
        coupling score.

        Ranges from 0.0 (no propagation observed) to 1.0 (every non-terminal
        source failure propagated to target).

        Returns 0.0 when no propagatable failure at *source* has been ingested.
        """
        src_count = self._propagatable_counts.get(source, 0)
        if src_count == 0:
            return 0.0
        edge_count = self._edge_counts.get(source, {}).get(target, 0)
        return edge_count / src_count

    def coupling_matrix(
        self,
    ) -> Dict[str, Dict[str, float]]:
        """Full stage-coupling matrix as nested dicts keyed by stage value strings.

        ``matrix[src][tgt]`` equals ``stage_coupling(src, tgt)``.  Only stages
        with at least one observed failure appear as row keys; all columns for
        those rows are included (zero if unobserved).

        Suitable for rendering as a paper table or seaborn heatmap.
        """
        stages = list(FailureStage)
        matrix: Dict[str, Dict[str, float]] = {}
        for src in stages:
            if self._failure_counts.get(src, 0) == 0:
                continue
            row: Dict[str, float] = {}
            for tgt in stages:
                row[tgt.value] = self.stage_coupling(src, tgt)
            matrix[src.value] = row
        return matrix

    def propagation_depth(
        self,
        stage: Optional[FailureStage] = None,
    ) -> float:
        """Mean number of consecutive hops a failure propagates before resolving.

        A depth of 1 means failures were isolated to a single hop; a depth of
        3 means the average failing trace had failures at 3 consecutive hops.

        Parameters
        ----------
        stage:
            If given, restrict to traces whose *first* failure was at *stage*.
            If None, average over all traces that had at least one failure.

        Returns 0.0 when no matching failing traces exist.
        """
        depths: List[int] = []
        for path in self._trace_paths:
            if not path:
                continue
            if stage is not None and path[0][1] != stage:
                continue
            depths.append(len(path))
        return sum(depths) / len(depths) if depths else 0.0

    def propagation_depth_distribution(
        self,
        stage: Optional[FailureStage] = None,
    ) -> Dict[int, int]:
        """Histogram of propagation depths.

        Returns a mapping from depth → count.  Useful for plotting hop-depth
        failure curves in the paper.
        """
        dist: Dict[int, int] = defaultdict(int)
        for path in self._trace_paths:
            if not path:
                continue
            if stage is not None and path[0][1] != stage:
                continue
            dist[len(path)] += 1
        return dict(dist)

    def critical_path(self) -> List[FailureStage]:
        """Stages ordered from highest to lowest outgoing propagation count.

        The first stage is the single largest source of cascading failures in
        the ingested traces.  Ties are broken by stage enum value for
        determinism.
        """
        out_counts: Dict[FailureStage, int] = defaultdict(int)
        for src, targets in self._edge_counts.items():
            out_counts[src] += sum(targets.values())
        return sorted(
            out_counts,
            key=lambda s: (-out_counts[s], s.value),
        )

    def failure_rate_by_stage(self) -> Dict[str, float]:
        """Fraction of ingested traces where each stage had at least one failure.

        Returns an empty dict when no traces have been ingested.
        """
        if self._total_traces == 0:
            return {}
        return {
            stage.value: count / self._total_traces
            for stage, count in self._failure_counts.items()
        }

    def self_correction_rate(self) -> float:
        """Fraction of traces that started failing but ended with no failure at the final hop.

        A trace is counted as self-correcting when its path is non-empty
        (at least one failure) but the last recorded failure is not at the
        last ingested hop — i.e., the final hop record was NONE and therefore
        not added to the path.

        Returns 0.0 when no failing traces have been ingested.
        """
        failing = [p for p in self._trace_paths if p]
        if not failing:
            return 0.0
        if self._max_hop_seen == 0:
            return 0.0

        # A trace is self-correcting when its last recorded failure occurred
        # before the globally highest hop seen — meaning the pipeline had at
        # least one more hop where it produced a NONE (success) result.
        corrected = sum(
            1 for path in failing if path[-1][0] < self._max_hop_seen
        )
        return corrected / len(failing)

    def summary(self) -> Dict[str, Any]:
        """High-level summary dict suitable for a paper table or logging.

        Keys
        ----
        total_traces : int
        failure_rate_by_stage : Dict[str, float]
        coupling_matrix : Dict[str, Dict[str, float]]
        mean_propagation_depth : float
        critical_path : List[str]
        self_correction_rate : float
        n_edges : int
        """
        return {
            "total_traces": self._total_traces,
            "failure_rate_by_stage": self.failure_rate_by_stage(),
            "coupling_matrix": self.coupling_matrix(),
            "mean_propagation_depth": self.propagation_depth(),
            "critical_path": [s.value for s in self.critical_path()],
            "self_correction_rate": self.self_correction_rate(),
            "n_edges": len(self.edges()),
        }


# --------------------------------------------------------------------------- #
# Counterfactual recovery (Pearl rung 3) — extends recovery_rate / self_correction
# --------------------------------------------------------------------------- #

def counterfactual_recovery_rate(
    injection_results: Sequence[Any],
    references: Sequence[Dict[str, Any]],
) -> float:
    """Fraction of live interventions the agent *absorbed* (Pearl rung 3).

    Given live ``InjectionResult`` objects — where the agent re-ran the suffix
    after a ``do(failure)`` intervention — this is the fraction whose injected
    trace *still* produced a correct answer. It quantifies how often the agent
    counterfactually recovers from an injected fault, complementing the
    observational ``self_correction_rate`` / ``recovery_rate`` with an
    interventional measurement.

    Parameters
    ----------
    injection_results:
        Objects exposing ``injected_trace`` (a ``PipelineTrace``), or bare
        ``PipelineTrace`` objects.
    references:
        Aligned reference dicts, each with an ``"answer"`` key.
    """
    if len(injection_results) != len(references):
        raise ValueError("injection_results and references must have the same length")
    if not injection_results:
        return 0.0
    recovered = 0
    for res, ref in zip(injection_results, references):
        trace = getattr(res, "injected_trace", res)
        if _answer_correct(trace.final_answer, ref.get("answer", "")):
            recovered += 1
    return recovered / len(injection_results)
