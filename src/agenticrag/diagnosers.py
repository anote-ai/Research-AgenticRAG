"""Failure-diagnosis methods (W3): baselines + the propagation-aware diagnoser (C3).

Every diagnoser maps an (injected or natural) :class:`PipelineTrace` to a
:class:`Diagnosis` — a predicted root-cause *stage* and the earliest failing
*hop* (Doctor-RAG's ``k†``), plus the token cost spent producing it. Scoring
``Diagnosis.predicted_hop`` against ``InjectionResult.injected_at_hop`` is what
yields the attribution-identifiability curve (C2), and ``cost_tokens`` feeds the
cost-per-correct-diagnosis metric (deployability angle).

Four diagnosers:

- :class:`RuleBasedDiagnoser` — wraps the existing rule-based
  ``DiagnosticBenchmark``; purely post-hoc (final trace only).
- :class:`DoctorRAGDiagnoser` — coverage-gated localization of the earliest
  low-support hop; post-hoc, and therefore bounded once a failure is masked by a
  later hop (the 50–70% ceiling the paper explains).
- :class:`LLMJudgeDiagnoser` — an LLM reads the trajectory and names the
  root-cause stage/hop; post-hoc but reasoning-driven, with real token cost.
- :class:`PropagationAwareDiagnoser` (**C3**) — active, counterfactual
  diagnosis: it re-executes single-hop *repairs* in causal order and returns the
  earliest hop whose repair flips the outcome. Because it intervenes rather than
  reads, it recovers root causes that surface-level diagnosis cannot — the lift
  that addresses Doctor-RAG's open problem, at the cost of re-execution tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .agents import HopState, LLMAgent
from .core import (
    DiagnosticBenchmark,
    FailureRecord,
    FailureStage,
    FailureType,
    PipelineTrace,
    _answer_correct,
    _token_overlap,
)


# --------------------------------------------------------------------------- #
# Diagnosis output                                                              #
# --------------------------------------------------------------------------- #

@dataclass
class Diagnosis:
    """A diagnoser's prediction for one trace.

    Attributes
    ----------
    stage:
        Predicted root-cause stage.
    failure_type:
        Predicted failure type (free string).
    predicted_hop:
        Earliest hop believed to carry the root cause (1-based); 0 when the
        diagnoser localizes to the answer level or abstains.
    confidence:
        Diagnoser confidence in [0, 1].
    cost_tokens:
        LLM/re-execution tokens spent producing this diagnosis.
    """

    trace_id: str
    stage: FailureStage
    failure_type: str
    predicted_hop: int = 0
    confidence: float = 0.5
    cost_tokens: int = 0
    # Which probe succeeded: "local_repair" | "suffix_regen" | "none" | "".
    # Set by SuffixRegenerationDiagnoser; empty for all other diagnosers.
    probe_type: str = ""

    def to_record(self) -> FailureRecord:
        """Adapt to a ``FailureRecord`` so existing RCA metrics apply unchanged.

        ``root_cause`` is set to the canonical stage value, which is what
        ``root_cause_stage`` / ``root_cause_accuracy`` read.
        """
        return FailureRecord(
            trace_id=self.trace_id,
            stage=self.stage,
            failure_type=self.failure_type,
            propagated=self.stage != FailureStage.NONE and self.predicted_hop >= 1,
            root_cause=self.stage.value,
            severity=0.5 if self.stage != FailureStage.NONE else 0.0,
        )


class Diagnoser(Protocol):
    name: str

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        ...


def _earliest_unsupported_hop(trace: PipelineTrace, target: str) -> int:
    """Return the 1-based index of the earliest hop whose docs don't support *target*.

    "Support" is any token overlap between the hop's docs and *target*. Returns 0
    when every hop supports the target (no localizable retrieval gap).
    """
    for i, docs in enumerate(trace.hop_docs):
        combined = " ".join(docs)
        if not docs or _token_overlap(target, combined) == 0.0:
            return i + 1
    return 0


# --------------------------------------------------------------------------- #
# Baseline 1 — rule-based (post-hoc)                                            #
# --------------------------------------------------------------------------- #

class RuleBasedDiagnoser:
    """Post-hoc rule-based diagnoser wrapping ``DiagnosticBenchmark``.

    Sees only the final trace, so it attributes to the *observed* failure stage.
    Hop localization is a best-effort earliest-empty-hop heuristic.
    """

    name = "rule_based"

    def __init__(self, benchmark: Optional[DiagnosticBenchmark] = None) -> None:
        self._bench = benchmark or DiagnosticBenchmark()

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        record = self._bench.diagnose_trace(trace, reference)
        predicted_hop = 0
        if record.stage == FailureStage.RETRIEVAL:
            # Earliest empty hop, else first hop.
            predicted_hop = next(
                (i + 1 for i, d in enumerate(trace.hop_docs) if not d), 1
            )
        return Diagnosis(
            trace_id=trace.trace_id,
            stage=record.stage,
            failure_type=str(record.failure_type),
            predicted_hop=predicted_hop,
            confidence=0.5,
            cost_tokens=0,
        )


# --------------------------------------------------------------------------- #
# Baseline 2 — Doctor-RAG-style coverage-gated localization (post-hoc)          #
# --------------------------------------------------------------------------- #

class DoctorRAGDiagnoser:
    """Coverage-gated earliest-failure localization (Doctor-RAG-style baseline).

    Estimates per-hop *coverage* (token support of each hop's docs for the gold
    answer) and localizes the root cause to the earliest hop whose coverage falls
    below ``coverage_threshold``. It is purely post-hoc: when an early failure is
    masked by a later hop that coincidentally restores overlap, it mislocalizes —
    reproducing the empirical 50–70% diagnosis ceiling the paper explains.
    """

    name = "doctor_rag"

    def __init__(self, coverage_threshold: float = 0.05) -> None:
        self.coverage_threshold = coverage_threshold

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        gold = reference.get("answer", "")
        if _answer_correct(trace.final_answer, gold):
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=FailureStage.NONE,
                failure_type=FailureType.SUCCESS,
                predicted_hop=0,
                confidence=0.6,
            )

        # Earliest hop whose coverage of the gold answer is below threshold.
        for i, docs in enumerate(trace.hop_docs):
            coverage = _token_overlap(gold, " ".join(docs)) if docs else 0.0
            if coverage < self.coverage_threshold:
                return Diagnosis(
                    trace_id=trace.trace_id,
                    stage=FailureStage.RETRIEVAL,
                    failure_type=FailureType.IRRELEVANT_RETRIEVAL,
                    predicted_hop=i + 1,
                    confidence=0.5,
                )

        # All hops look covered yet the answer is wrong -> attribute to answer gen.
        return Diagnosis(
            trace_id=trace.trace_id,
            stage=FailureStage.ANSWER_GENERATION,
            failure_type=FailureType.INCORRECT_ANSWER,
            predicted_hop=0,
            confidence=0.4,
        )


# --------------------------------------------------------------------------- #
# Baseline 3 — LLM-as-judge (post-hoc, reasoning-driven)                         #
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = (
    "You are a diagnostic judge for a multi-hop retrieval agent. Given the "
    "question, the per-hop retrieved evidence, the final answer, and the gold "
    "answer, identify the earliest hop where the agent went wrong. Respond with a "
    'single JSON object: {"stage": "retrieval"|"answer_generation"|"none", '
    '"hop": <1-based hop number, or 0 if none>}.'
)


class LLMJudgeDiagnoser:
    """LLM-as-judge diagnoser: an LLM reads the trajectory and localizes the failure.

    Post-hoc (reads, does not intervene) but reasoning-driven; carries real token
    cost. With the offline :class:`~agenticrag.agents.MockProvider` the judge
    falls back to a coverage heuristic so the suite runs without an API key.
    """

    name = "llm_judge"

    def __init__(self, provider: Any = None, max_tokens: int = 256) -> None:
        from .agents import MockProvider

        self.provider = provider or MockProvider()
        self.max_tokens = max_tokens

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        from .agents import MockProvider, parse_decision

        gold = reference.get("answer", "")
        # The MockProvider cannot truly judge; use a transparent coverage heuristic
        # so offline runs are deterministic. Real providers get the judge prompt.
        if isinstance(self.provider, MockProvider):
            if _answer_correct(trace.final_answer, gold):
                return Diagnosis(
                    trace_id=trace.trace_id,
                    stage=FailureStage.NONE,
                    failure_type=FailureType.SUCCESS,
                    predicted_hop=0,
                    confidence=0.55,
                    cost_tokens=16,
                )
            hop = _earliest_unsupported_hop(trace, gold)
            stage = FailureStage.RETRIEVAL if hop else FailureStage.ANSWER_GENERATION
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=stage,
                failure_type=FailureType.IRRELEVANT_RETRIEVAL if hop else FailureType.INCORRECT_ANSWER,
                predicted_hop=hop,
                confidence=0.5,
                cost_tokens=32,
            )

        prompt = self._build_prompt(trace, gold)
        resp = self.provider.generate(_JUDGE_SYSTEM, prompt, max_tokens=self.max_tokens)
        decision = parse_decision(resp.text)
        # Reuse the JSON parser opportunistically; otherwise parse stage/hop directly.
        stage, hop = self._parse_judge(resp.text)
        return Diagnosis(
            trace_id=trace.trace_id,
            stage=stage,
            failure_type=FailureType.IRRELEVANT_RETRIEVAL
            if stage == FailureStage.RETRIEVAL
            else (FailureType.INCORRECT_ANSWER if stage == FailureStage.ANSWER_GENERATION else FailureType.SUCCESS),
            predicted_hop=hop,
            confidence=0.6,
            cost_tokens=resp.total_tokens,
        )

    @staticmethod
    def _build_prompt(trace: PipelineTrace, gold: str) -> str:
        lines = [f"QUESTION: {trace.query}", "", "HOPS:"]
        for i, (q, docs) in enumerate(zip(trace.hop_queries, trace.hop_docs)):
            lines.append(f"  Hop {i + 1} query: {q}")
            for d in docs:
                lines.append(f"    - {d}")
            if not docs:
                lines.append("    - (no documents retrieved)")
        lines += ["", f"FINAL ANSWER: {trace.final_answer}", f"GOLD ANSWER: {gold}"]
        return "\n".join(lines)

    @staticmethod
    def _parse_judge(text: str) -> "tuple[FailureStage, int]":
        import json
        import re

        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                stage_str = str(data.get("stage", "none")).lower()
                hop = int(data.get("hop", 0))
                stage = {
                    "retrieval": FailureStage.RETRIEVAL,
                    "answer_generation": FailureStage.ANSWER_GENERATION,
                    "tool_call": FailureStage.TOOL_CALL,
                    "none": FailureStage.NONE,
                }.get(stage_str, FailureStage.ANSWER_GENERATION)
                return stage, max(0, hop)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        return FailureStage.ANSWER_GENERATION, 0


# --------------------------------------------------------------------------- #
# C3 — propagation-aware diagnoser (active / counterfactual)                     #
# --------------------------------------------------------------------------- #

class PropagationAwareDiagnoser:
    """Counterfactual, propagation-aware diagnosis (the C3 method).

    Rather than reading the final trace, it intervenes: for each hop in causal
    order it re-executes a *repaired* counterfactual (that hop's evidence
    re-retrieved from the corpus against the top-level question, all other hops
    held fixed) and re-runs the suffix with the agent. The earliest hop whose
    repair flips the outcome from wrong to correct is the certified root cause —
    Pearl rung-3 reasoning that recovers causes masked from surface diagnosis. A
    final "continue retrieving" probe catches premature-termination faults.

    This is the method that exploits propagation structure to beat post-hoc
    baselines at depth ≥ 2, at the price of re-execution tokens (reported in
    ``cost_tokens`` for the cost-per-correct-diagnosis metric).

    The diagnoser needs the agent and corpus. ``corpus`` is read from
    ``reference["corpus"]`` (preferred) or passed at construction.
    """

    name = "propagation_aware"

    def __init__(
        self,
        agent: LLMAgent,
        corpus: Optional[List[str]] = None,
        max_probes: Optional[int] = None,
    ) -> None:
        self.agent = agent
        self._corpus = corpus
        self.max_probes = max_probes  # cap re-executions for cost control

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        gold = reference.get("answer", "")
        corpus = reference.get("corpus", self._corpus) or []
        cost = 0

        if _answer_correct(trace.final_answer, gold):
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=FailureStage.NONE,
                failure_type=FailureType.SUCCESS,
                predicted_hop=0,
                confidence=0.7,
                cost_tokens=cost,
            )

        hops = [HopState(query=q, docs=list(d)) for q, d in zip(trace.hop_queries, trace.hop_docs)]
        n = len(hops)
        probe_budget = self.max_probes if self.max_probes is not None else n

        # Probe single-hop repairs in causal (earliest-first) order. Crucially the
        # counterfactual holds every *other* hop fixed (corruptions included) and
        # only swaps hop h's evidence + re-answers — re-running the suffix would
        # silently re-execute and un-corrupt later hops, destroying the signal.
        for h in range(1, min(n, probe_budget) + 1):
            # Repair hop h with a clean retrieval using the hop's own (sub-)query —
            # which, for the answer-bearing hop, targets the gold fact. Fall back to
            # the top-level question when the hop query yields nothing.
            repaired = self.agent._retrieve(hops[h - 1].query, corpus)
            if not repaired:
                repaired = self.agent._retrieve(trace.query, corpus)
            cf_hops = [HopState(query=hh.query, docs=list(hh.docs)) for hh in hops]
            cf_hops[h - 1] = HopState(query=hops[h - 1].query, docs=repaired)
            cf = self.agent.force_answer(trace.query, prefix=cf_hops, reference_answer=gold)
            cost += cf.tokens_used
            if _answer_correct(cf.final_answer, gold):
                return Diagnosis(
                    trace_id=trace.trace_id,
                    stage=FailureStage.RETRIEVAL,
                    failure_type=FailureType.IRRELEVANT_RETRIEVAL,
                    predicted_hop=h,
                    confidence=0.85,
                    cost_tokens=cost,
                )

        # "Continue retrieving" probe — catches premature termination.
        cont = self.agent.resume_from_hops(
            trace.query, corpus, prefix=hops, reference_answer=gold, start_hop=n + 1
        )
        cost += cont.tokens_used
        if _answer_correct(cont.final_answer, gold):
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.EARLY_TERMINATION,
                predicted_hop=n + 1,
                confidence=0.75,
                cost_tokens=cost,
            )

        # No single-hop repair flipped the outcome: fall back to coverage gating,
        # flagged low-confidence (the failure is genuinely hard to localize).
        hop = _earliest_unsupported_hop(trace, gold)
        return Diagnosis(
            trace_id=trace.trace_id,
            stage=FailureStage.RETRIEVAL if hop else FailureStage.ANSWER_GENERATION,
            failure_type=FailureType.IRRELEVANT_RETRIEVAL if hop else FailureType.INCORRECT_ANSWER,
            predicted_hop=hop,
            confidence=0.4,
            cost_tokens=cost,
        )


# --------------------------------------------------------------------------- #
# C3 variant — suffix-regeneration diagnoser                                    #
# --------------------------------------------------------------------------- #

class SuffixRegenerationDiagnoser:
    """Suffix-regeneration variant of the propagation-aware diagnoser.

    Unlike :class:`PropagationAwareDiagnoser`, which holds every hop outside
    the repaired one fixed (including corrupted later hops), this diagnoser
    **regenerates the suffix** after repairing hop *h*:

    1. Preserve hops 1 … h-1 exactly.
    2. Repair hop *h* by clean retrieval from the corpus.
    3. Call ``agent.resume_from_hops(start_hop=h+1)`` — the agent re-executes
       from hop h+1 in the repaired context rather than seeing frozen corrupt
       later hops.

    This recovers root causes whose downstream hops were generated from the
    corrupted prefix (e.g. hop 2's sub-query was derived from hop 1's wrong
    answer). Local-only repair misses these because force_answer sees all
    the corrupted later-hop docs alongside the one repaired doc, confusing
    the outcome.

    ``Diagnosis.probe_type`` records the localization path:
    - ``"suffix_regen"`` — suffix regeneration found the flip.
    - ``"none"`` — no single-hop repair flipped the outcome; coverage heuristic
      used as fallback.
    """

    name = "suffix_regen"

    def __init__(
        self,
        agent: LLMAgent,
        corpus: Optional[List[str]] = None,
        max_probes: Optional[int] = None,
    ) -> None:
        self.agent = agent
        self._corpus = corpus
        self.max_probes = max_probes

    def diagnose(self, trace: PipelineTrace, reference: Dict[str, Any]) -> Diagnosis:
        gold = reference.get("answer", "")
        corpus = reference.get("corpus", self._corpus) or []
        cost = 0

        if _answer_correct(trace.final_answer, gold):
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=FailureStage.NONE,
                failure_type=FailureType.SUCCESS,
                predicted_hop=0,
                confidence=0.7,
                cost_tokens=cost,
                probe_type="",
            )

        hops = [HopState(query=q, docs=list(d)) for q, d in zip(trace.hop_queries, trace.hop_docs)]
        n = len(hops)
        probe_budget = self.max_probes if self.max_probes is not None else n

        for h in range(1, min(n, probe_budget) + 1):
            # Repair hop h; keep hops 1..h-1 intact as the prefix.
            repaired_docs = self.agent._retrieve(hops[h - 1].query, corpus)
            if not repaired_docs:
                repaired_docs = self.agent._retrieve(trace.query, corpus)
            prefix = [HopState(query=hh.query, docs=list(hh.docs)) for hh in hops[: h - 1]]
            prefix.append(HopState(query=hops[h - 1].query, docs=repaired_docs))

            # Regenerate suffix from h+1 — this is the distinguishing step.
            regen = self.agent.resume_from_hops(
                trace.query, corpus, prefix=prefix,
                reference_answer=gold, start_hop=h + 1,
            )
            cost += regen.tokens_used
            if _answer_correct(regen.final_answer, gold):
                return Diagnosis(
                    trace_id=trace.trace_id,
                    stage=FailureStage.RETRIEVAL,
                    failure_type=FailureType.IRRELEVANT_RETRIEVAL,
                    predicted_hop=h,
                    confidence=0.85,
                    cost_tokens=cost,
                    probe_type="suffix_regen",
                )

        # "Continue retrieving" probe — catches premature termination.
        cont = self.agent.resume_from_hops(
            trace.query, corpus, prefix=hops, reference_answer=gold, start_hop=n + 1
        )
        cost += cont.tokens_used
        if _answer_correct(cont.final_answer, gold):
            return Diagnosis(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.EARLY_TERMINATION,
                predicted_hop=n + 1,
                confidence=0.75,
                cost_tokens=cost,
                probe_type="suffix_regen",
            )

        # No flip found — coverage heuristic fallback.
        hop = _earliest_unsupported_hop(trace, gold)
        return Diagnosis(
            trace_id=trace.trace_id,
            stage=FailureStage.RETRIEVAL if hop else FailureStage.ANSWER_GENERATION,
            failure_type=FailureType.IRRELEVANT_RETRIEVAL if hop else FailureType.INCORRECT_ANSWER,
            predicted_hop=hop,
            confidence=0.4,
            cost_tokens=cost,
            probe_type="none",
        )


# --------------------------------------------------------------------------- #
# Convenience                                                                   #
# --------------------------------------------------------------------------- #

def batch_diagnose(
    diagnoser: Diagnoser,
    traces: List[PipelineTrace],
    references: List[Dict[str, Any]],
) -> List[Diagnosis]:
    """Run *diagnoser* over paired traces/references."""
    if len(traces) != len(references):
        raise ValueError("traces and references must have the same length")
    return [diagnoser.diagnose(t, r) for t, r in zip(traces, references)]
