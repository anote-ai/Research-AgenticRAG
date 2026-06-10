from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, field_validator

from .core import FailureRecord, FailureStage, FailureType, PipelineTrace


class InjectionSpec(BaseModel):
    """Controlled failure to inject into a PipelineTrace."""

    stage: FailureStage
    failure_type: FailureType
    hop: int = 1
    severity: float = 0.8

    @field_validator("hop")
    @classmethod
    def hop_is_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"hop must be >= 1, got {value}")
        return value

    @field_validator("severity")
    @classmethod
    def severity_in_range(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"severity must be in [0, 1], got {value}")
        return value


class InjectionLabel(BaseModel):
    """Ground-truth propagation annotation for an injected trace."""

    trace_id: str
    root_cause_stage: FailureStage
    observed_stage: FailureStage
    failure_type: FailureType
    injection_hop: int
    propagated: bool
    severity: float

    def to_failure_record(self) -> FailureRecord:
        return FailureRecord(
            trace_id=self.trace_id,
            stage=self.observed_stage,
            failure_type=self.failure_type.value,
            propagated=self.propagated,
            root_cause=self.root_cause_stage.value,
            severity=self.severity,
        )


class InjectedTrace(BaseModel):
    """A trace plus its ground-truth injection label."""

    trace: PipelineTrace
    label: InjectionLabel

    @property
    def record(self) -> FailureRecord:
        return self.label.to_failure_record()


def inject_failure(trace: PipelineTrace, spec: InjectionSpec) -> InjectedTrace:
    """Return a copy of trace with a controlled failure injected.

    The injected trace is intentionally deterministic so generated benchmark
    labels can be reproduced exactly across experiment runs.
    """
    if spec.stage == FailureStage.NONE:
        raise ValueError("Cannot inject a failure at FailureStage.NONE")

    mutated = trace.model_copy(deep=True)
    _ensure_hop_capacity(mutated, spec.hop)

    if spec.stage == FailureStage.RETRIEVAL:
        _inject_retrieval_failure(mutated, spec.hop)
    elif spec.stage == FailureStage.TOOL_CALL:
        _inject_tool_failure(mutated, spec.hop)
    elif spec.stage == FailureStage.ANSWER_GENERATION:
        _inject_answer_failure(mutated, spec.failure_type)
    else:
        raise ValueError(f"Unsupported failure stage: {spec.stage}")

    observed_stage = _observed_stage(spec.stage)
    label = InjectionLabel(
        trace_id=mutated.trace_id,
        root_cause_stage=spec.stage,
        observed_stage=observed_stage,
        failure_type=spec.failure_type,
        injection_hop=spec.hop,
        propagated=observed_stage != spec.stage,
        severity=spec.severity,
    )
    return InjectedTrace(trace=mutated, label=label)


def inject_failures(
    trace: PipelineTrace,
    specs: List[InjectionSpec],
) -> List[InjectedTrace]:
    """Apply a list of injection specs independently to the same base trace."""
    return [inject_failure(trace, spec) for spec in specs]


def make_injection_grid(
    max_hops: int,
    stages: List[FailureStage] | None = None,
) -> List[InjectionSpec]:
    """Build a stage x hop injection grid for amplification experiments."""
    if max_hops < 1:
        return []

    selected_stages = stages or [
        FailureStage.RETRIEVAL,
        FailureStage.TOOL_CALL,
        FailureStage.ANSWER_GENERATION,
    ]
    specs: List[InjectionSpec] = []
    for hop in range(1, max_hops + 1):
        for stage in selected_stages:
            specs.append(
                InjectionSpec(
                    stage=stage,
                    failure_type=_default_failure_type(stage),
                    hop=hop,
                    severity=_default_severity(stage),
                )
            )
    return specs


def group_records_by_hop(injected: List[InjectedTrace]) -> Dict[int, List[FailureRecord]]:
    """Convert injected traces into the format used by hop-level metrics."""
    grouped: Dict[int, List[FailureRecord]] = {}
    for item in injected:
        grouped.setdefault(item.label.injection_hop, []).append(item.record)
    return grouped


def _ensure_hop_capacity(trace: PipelineTrace, hop: int) -> None:
    while len(trace.hop_queries) < hop:
        trace.hop_queries.append(trace.query)
    while len(trace.hop_docs) < hop:
        trace.hop_docs.append([])
    trace.iterations_used = max(trace.iterations_used, hop)


def _inject_retrieval_failure(trace: PipelineTrace, hop: int) -> None:
    idx = hop - 1
    trace.hop_docs[idx] = []
    trace.retrieved_docs = _dedupe_docs(trace.hop_docs)
    if not trace.retrieved_docs:
        trace.tool_calls = []
    trace.final_answer = ""


def _inject_tool_failure(trace: PipelineTrace, hop: int) -> None:
    trace.tool_calls = [
        call for call in trace.tool_calls if call.get("iteration") != hop
    ]
    if len(trace.tool_calls) == len(trace.hop_queries):
        trace.tool_calls = trace.tool_calls[: max(0, hop - 1)]
    trace.final_answer = ""


def _inject_answer_failure(trace: PipelineTrace, failure_type: FailureType) -> None:
    if failure_type == FailureType.HALLUCINATION:
        trace.final_answer = "Unsupported claim not present in retrieved evidence."
    elif failure_type == FailureType.INCORRECT_ANSWER:
        trace.final_answer = "Incorrect answer."
    else:
        trace.final_answer = ""


def _observed_stage(root_stage: FailureStage) -> FailureStage:
    if root_stage == FailureStage.ANSWER_GENERATION:
        return root_stage
    return FailureStage.ANSWER_GENERATION


def _default_failure_type(stage: FailureStage) -> FailureType:
    if stage == FailureStage.RETRIEVAL:
        return FailureType.EMPTY_RETRIEVAL
    if stage == FailureStage.TOOL_CALL:
        return FailureType.NO_TOOL_CALLS
    if stage == FailureStage.ANSWER_GENERATION:
        return FailureType.EMPTY_ANSWER
    raise ValueError(f"No default failure type for stage: {stage}")


def _default_severity(stage: FailureStage) -> float:
    if stage == FailureStage.RETRIEVAL:
        return 0.9
    if stage == FailureStage.TOOL_CALL:
        return 0.7
    if stage == FailureStage.ANSWER_GENERATION:
        return 0.8
    return 0.5


def _dedupe_docs(hop_docs: List[List[str]]) -> List[str]:
    docs: List[str] = []
    for hop in hop_docs:
        docs.extend(hop)
    return list(dict.fromkeys(docs))
