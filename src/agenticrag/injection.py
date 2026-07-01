from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .agents import HopState, LLMAgent
from .core import DiagnosticBenchmark, FailureStage, FailureType, PipelineTrace


@dataclass
class InjectionResult:
    """An injected trace alongside metadata describing the injected fault."""

    original_trace_id: str
    injected_trace: PipelineTrace
    injected_stage: FailureStage
    injected_failure_type: FailureType
    injected_at_hop: int  # 1-based; 0 = not hop-specific (answer-level)


class FailureInjector:
    """Injects controlled failures into PipelineTraces for ablation experiments.

    Every method returns a fresh InjectionResult — originals are never mutated.

    Downstream propagation rules applied deterministically:
    - ``inject_empty_retrieval`` at hop N clears all hop_docs at hop N and later,
      then rebuilds ``retrieved_docs`` from surviving hops.  When no docs survive,
      the final answer is also cleared (nothing left to generate from).
    - ``inject_irrelevant_docs`` replaces docs at one hop without touching later
      hops or the answer, letting callers measure whether the pipeline self-corrects.
    - Answer-level injections (``inject_empty_answer``, ``inject_hallucinated_answer``)
      leave retrieval state intact so the grounding check in DiagnosticBenchmark can
      distinguish an answer-stage failure from a retrieval-stage failure.
    """

    # ------------------------------------------------------------------ #
    # Retrieval-stage injections                                           #
    # ------------------------------------------------------------------ #

    def inject_empty_retrieval(
        self, trace: PipelineTrace, hop: int = 1
    ) -> InjectionResult:
        """Clear retrieved docs at *hop* and all subsequent hops.

        Parameters
        ----------
        trace:
            Source trace.  Must not be mutated by callers afterward.
        hop:
            1-based hop index at which to start the empty-retrieval fault.
        """
        data = trace.model_dump()
        hop_idx = hop - 1

        new_hop_docs: List[List[str]] = list(data.get("hop_docs", []))
        for i in range(hop_idx, len(new_hop_docs)):
            new_hop_docs[i] = []
        data["hop_docs"] = new_hop_docs

        surviving: List[str] = []
        for i, hop_d in enumerate(new_hop_docs):
            if i < hop_idx:
                surviving.extend(hop_d)
        data["retrieved_docs"] = list(dict.fromkeys(surviving))

        if not data["retrieved_docs"]:
            data["final_answer"] = ""

        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=PipelineTrace(**data),
            injected_stage=FailureStage.RETRIEVAL,
            injected_failure_type=FailureType.EMPTY_RETRIEVAL,
            injected_at_hop=hop,
        )

    def inject_irrelevant_docs(
        self, trace: PipelineTrace, noise_docs: List[str], hop: int = 1
    ) -> InjectionResult:
        """Replace docs retrieved at *hop* with off-topic *noise_docs*.

        Later hops and the final answer are untouched so callers can measure
        whether the pipeline corrects itself on subsequent iterations.

        Parameters
        ----------
        noise_docs:
            Documents that are irrelevant to the trace query.
        hop:
            1-based hop index to corrupt.
        """
        data = trace.model_dump()
        hop_idx = hop - 1

        new_hop_docs: List[List[str]] = list(data.get("hop_docs", []))
        if hop_idx < len(new_hop_docs):
            new_hop_docs[hop_idx] = list(noise_docs)
        data["hop_docs"] = new_hop_docs

        flat: List[str] = []
        for hop_d in new_hop_docs:
            flat.extend(hop_d)
        data["retrieved_docs"] = list(dict.fromkeys(flat))

        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=PipelineTrace(**data),
            injected_stage=FailureStage.RETRIEVAL,
            injected_failure_type=FailureType.IRRELEVANT_RETRIEVAL,
            injected_at_hop=hop,
        )

    # ------------------------------------------------------------------ #
    # Tool-call-stage injections                                           #
    # ------------------------------------------------------------------ #

    def inject_no_tool_calls(self, trace: PipelineTrace) -> InjectionResult:
        """Clear all tool calls, simulating a tool-routing failure."""
        data = trace.model_dump()
        data["tool_calls"] = []
        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=PipelineTrace(**data),
            injected_stage=FailureStage.TOOL_CALL,
            injected_failure_type=FailureType.NO_TOOL_CALLS,
            injected_at_hop=0,
        )

    # ------------------------------------------------------------------ #
    # Answer-generation-stage injections                                   #
    # ------------------------------------------------------------------ #

    def inject_empty_answer(self, trace: PipelineTrace) -> InjectionResult:
        """Clear the final answer, simulating an answer-generation crash."""
        data = trace.model_dump()
        data["final_answer"] = ""
        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=PipelineTrace(**data),
            injected_stage=FailureStage.ANSWER_GENERATION,
            injected_failure_type=FailureType.EMPTY_ANSWER,
            injected_at_hop=0,
        )

    def inject_hallucinated_answer(
        self,
        trace: PipelineTrace,
        fabricated: str = "The answer is 42.",
    ) -> InjectionResult:
        """Replace the final answer with *fabricated* — a string ungrounded in retrieved docs.

        The default fabrication is chosen to have zero token overlap with any
        realistic retrieved corpus so DiagnosticBenchmark._is_hallucination
        reliably fires.

        Parameters
        ----------
        fabricated:
            Replacement answer.  Pass a domain-specific nonsense string for
            more realistic hallucination simulation.
        """
        data = trace.model_dump()
        data["final_answer"] = fabricated
        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=PipelineTrace(**data),
            injected_stage=FailureStage.ANSWER_GENERATION,
            injected_failure_type=FailureType.HALLUCINATION,
            injected_at_hop=0,
        )


# --------------------------------------------------------------------------- #
# Live (interventional) injection — W2                                          #
# --------------------------------------------------------------------------- #

# Off-topic documents used as irrelevant replacements.  Chosen to have near-zero
# token overlap with any realistic QA corpus.
_LIVE_NOISE_DOCS: List[str] = [
    "Xylophones are percussion instruments with wooden bars.",
    "The migration patterns of Arctic terns span both polar regions.",
    "Fermentation converts sugars into ethanol under anaerobic conditions.",
]
# A confidently-stated but wrong premise (CRAG-style false-premise trigger).
_DEFAULT_FALSE_PREMISE = (
    "Established fact: the entity in question does not exist and the premise of "
    "the query is false, so no specific answer applies."
)
# Outdated evidence that contradicts the current ground truth (temporal trigger).
_DEFAULT_STALE_EVIDENCE = (
    "According to an outdated record that has since been superseded, the "
    "previously reported value no longer reflects the current state."
)


class LiveFailureInjector:
    """Injection as causal intervention: inject-then-re-run-the-suffix (W2).

    Where :class:`FailureInjector` statically edits a finished trace, this class
    corrupts the trajectory *prefix* at a chosen hop and then lets a real
    :class:`~agenticrag.agents.LLMAgent` continue from there — so the downstream
    trace is the agent's genuine reaction (it may self-correct, drift further, or
    collapse), not a deterministic edit.  This is the ``do(failure = f at stage
    s, hop h)`` operator from contribution C1.

    Every method returns an :class:`InjectionResult` with the same ground-truth
    schema (``injected_stage`` / ``injected_failure_type`` / ``injected_at_hop``)
    used by the static injector, so the certified-label dataset and the RCA
    metrics are computed identically across the static control and the live arm.

    Parameters
    ----------
    agent:
        A resumable :class:`~agenticrag.agents.LLMAgent`.
    noise_docs / false_premise / stale_evidence:
        Defaults for the corresponding interventions; override per-call.
    """

    def __init__(
        self,
        agent: LLMAgent,
        noise_docs: Optional[List[str]] = None,
        false_premise: Optional[str] = None,
        stale_evidence: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.noise_docs = list(noise_docs) if noise_docs else list(_LIVE_NOISE_DOCS)
        self.false_premise = false_premise or _DEFAULT_FALSE_PREMISE
        self.stale_evidence = stale_evidence or _DEFAULT_STALE_EVIDENCE

    # -- helpers ------------------------------------------------------------ #

    @staticmethod
    def _hops(trace: PipelineTrace) -> List[HopState]:
        return [
            HopState(query=q, docs=list(d))
            for q, d in zip(trace.hop_queries, trace.hop_docs)
        ]

    @staticmethod
    def _clamp(hop: int, n_hops: int) -> int:
        return max(1, min(hop, max(1, n_hops)))

    def _hop_query(self, hops: List[HopState], hop: int, trace: PipelineTrace) -> str:
        idx = hop - 1
        if 0 <= idx < len(hops):
            return hops[idx].query
        return hops[-1].query if hops else trace.query

    def _resume(
        self,
        trace: PipelineTrace,
        corpus: List[str],
        prefix: List[HopState],
        hop: int,
        failure_type: FailureType,
        stage: FailureStage = FailureStage.RETRIEVAL,
    ) -> InjectionResult:
        injected = self.agent.resume_from_hops(
            trace.query,
            corpus,
            prefix=prefix,
            reference_answer=trace.reference_answer,
            start_hop=hop + 1,
        )
        injected.trace_id = trace.trace_id  # preserve identity for paired analysis
        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=injected,
            injected_stage=stage,
            injected_failure_type=failure_type,
            injected_at_hop=hop,
        )

    # -- retrieval-stage interventions ------------------------------------- #

    def inject_empty_retrieval(
        self, trace: PipelineTrace, corpus: List[str], hop: int = 1
    ) -> InjectionResult:
        """Retrieval returns nothing at *hop*; the agent must react and continue."""
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        q = self._hop_query(hops, hop, trace)
        prefix = hops[: hop - 1] + [HopState(query=q, docs=[])]
        return self._resume(trace, corpus, prefix, hop, FailureType.EMPTY_RETRIEVAL)

    def inject_irrelevant_docs(
        self,
        trace: PipelineTrace,
        corpus: List[str],
        hop: int = 1,
        noise_docs: Optional[List[str]] = None,
    ) -> InjectionResult:
        """Retrieval returns off-topic docs at *hop* (retrieval drift)."""
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        q = self._hop_query(hops, hop, trace)
        prefix = hops[: hop - 1] + [HopState(query=q, docs=list(noise_docs or self.noise_docs))]
        return self._resume(trace, corpus, prefix, hop, FailureType.IRRELEVANT_RETRIEVAL)

    def inject_query_drift(
        self,
        trace: PipelineTrace,
        corpus: List[str],
        hop: int = 1,
        drift_query: Optional[str] = None,
    ) -> InjectionResult:
        """Corrupt the *hop* sub-query (~ over-extension), then re-retrieve and continue.

        The agent's reformulation is replaced with a drifted query; retrieval
        runs on the corrupted query, and the agent reasons over whatever that
        returns — modelling a self-inflicted reformulation error.
        """
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        base_q = self._hop_query(hops, hop, trace)
        if drift_query is None:
            drift_query = f"{base_q} unrelated tangent xylophone arctic fermentation"
        drifted_docs = self.agent._retrieve(drift_query, corpus)
        prefix = hops[: hop - 1] + [HopState(query=drift_query, docs=drifted_docs)]
        return self._resume(trace, corpus, prefix, hop, FailureType.QUERY_DRIFT)

    def inject_false_premise(
        self,
        trace: PipelineTrace,
        corpus: List[str],
        hop: int = 1,
        premise: Optional[str] = None,
    ) -> InjectionResult:
        """Inject a confidently-wrong premise into *hop* evidence (CRAG false-premise)."""
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        q = self._hop_query(hops, hop, trace)
        base_docs = list(hops[hop - 1].docs) if hop - 1 < len(hops) else []
        prefix = hops[: hop - 1] + [HopState(query=q, docs=[premise or self.false_premise] + base_docs)]
        return self._resume(trace, corpus, prefix, hop, FailureType.FALSE_PREMISE)

    def inject_stale_evidence(
        self,
        trace: PipelineTrace,
        corpus: List[str],
        hop: int = 1,
        stale: Optional[str] = None,
    ) -> InjectionResult:
        """Inject outdated/temporally-wrong evidence into *hop* (CRAG temporal trigger)."""
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        q = self._hop_query(hops, hop, trace)
        base_docs = list(hops[hop - 1].docs) if hop - 1 < len(hops) else []
        prefix = hops[: hop - 1] + [HopState(query=q, docs=[stale or self.stale_evidence] + base_docs)]
        return self._resume(trace, corpus, prefix, hop, FailureType.STALE_EVIDENCE)

    def inject_early_termination(
        self, trace: PipelineTrace, corpus: List[str], hop: int = 1
    ) -> InjectionResult:
        """Force the agent to answer from evidence up to *hop*-1 (premature collapse).

        ``corpus`` is unused (no further retrieval happens) but kept in the
        signature so all live interventions share one call shape.
        """
        hops = self._hops(trace)
        hop = self._clamp(hop, len(hops))
        prefix = hops[: hop - 1]
        injected = self.agent.force_answer(
            trace.query, prefix=prefix, reference_answer=trace.reference_answer
        )
        injected.trace_id = trace.trace_id
        return InjectionResult(
            original_trace_id=trace.trace_id,
            injected_trace=injected,
            injected_stage=FailureStage.RETRIEVAL,
            injected_failure_type=FailureType.EARLY_TERMINATION,
            injected_at_hop=hop,
        )


# Live interventions keyed by short name — for experiment grids.
LIVE_INJECTIONS: List[str] = [
    "inject_empty_retrieval",
    "inject_irrelevant_docs",
    "inject_query_drift",
    "inject_false_premise",
    "inject_stale_evidence",
    "inject_early_termination",
]


# --------------------------------------------------------------------------- #
# Sensitivity metric                                                            #
# --------------------------------------------------------------------------- #

def injection_sensitivity(
    clean_traces: List[PipelineTrace],
    references: List[Dict[str, Any]],
    injector: FailureInjector,
    benchmark: DiagnosticBenchmark,
    method: str = "inject_empty_retrieval",
    **kwargs: Any,
) -> float:
    """Fraction of injected failures correctly detected by *benchmark*.

    Applies *method* on every trace in *clean_traces*, diagnoses each injected
    trace, and returns the fraction flagged as non-NONE.  A score of 1.0 means
    the benchmark catches every instance of the injected fault mode; 0.0 means
    it misses all of them.

    This metric is the primary way to validate that DiagnosticBenchmark is
    sensitive to each failure type before running full ablation experiments.

    Parameters
    ----------
    clean_traces:
        Successfully completed pipeline traces (baseline with no failures).
    references:
        Corresponding reference dicts — same length as *clean_traces*.
    injector:
        FailureInjector instance.
    benchmark:
        DiagnosticBenchmark used to diagnose the injected traces.
    method:
        Name of the FailureInjector method to call, e.g.
        ``"inject_empty_retrieval"``, ``"inject_empty_answer"``,
        ``"inject_hallucinated_answer"``, ``"inject_no_tool_calls"``.
    **kwargs:
        Forwarded verbatim to *method* (e.g. ``hop=2``, ``noise_docs=[...]``).

    Returns
    -------
    float in [0, 1].  Returns 0.0 for an empty input list.
    """
    if not clean_traces:
        return 0.0
    if len(clean_traces) != len(references):
        raise ValueError(
            "clean_traces and references must have the same length"
        )

    inject_fn = getattr(injector, method)
    injected_traces = [inject_fn(t, **kwargs).injected_trace for t in clean_traces]
    records = benchmark.batch_diagnose(injected_traces, references)
    detected = sum(1 for r in records if r.stage != FailureStage.NONE)
    return detected / len(records)
