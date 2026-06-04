from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator, Field


class FailureStage(str, Enum):
    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    ANSWER_GENERATION = "answer_generation"
    NONE = "none"


class PipelineTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    retrieved_docs: List[str]
    tool_calls: List[Dict[str, Any]]
    final_answer: str
    reference_answer: str


class FailureRecord(BaseModel):
    trace_id: str
    stage: FailureStage
    failure_type: str
    propagated: bool = False
    root_cause: str = ""
    severity: float = 0.5

    @field_validator("severity")
    @classmethod
    def severity_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"severity must be in [0, 1], got {v}")
        return v


class DiagnosticBenchmark:
    """Diagnose pipeline traces and attribute failures."""

    def diagnose_trace(
        self, trace: PipelineTrace, reference: Dict[str, Any]
    ) -> FailureRecord:
        if not trace.retrieved_docs:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type="empty_retrieval",
                propagated=True,
                root_cause="No documents retrieved",
                severity=0.9,
            )
        elif not trace.tool_calls:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.TOOL_CALL,
                failure_type="no_tool_calls",
                propagated=True,
                root_cause="No tool calls made",
                severity=0.7,
            )
        elif trace.final_answer.strip() == "":
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type="empty_answer",
                propagated=False,
                root_cause="Answer generation produced empty string",
                severity=0.8,
            )
        elif trace.final_answer != reference.get("answer", ""):
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type="incorrect_answer",
                propagated=False,
                root_cause="Answer does not match reference",
                severity=0.5,
            )
        else:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.NONE,
                failure_type="success",
                propagated=False,
                root_cause="",
                severity=0.0,
            )

    def batch_diagnose(
        self,
        traces: List[PipelineTrace],
        references: List[Dict[str, Any]],
    ) -> List[FailureRecord]:
        return [
            self.diagnose_trace(trace, ref)
            for trace, ref in zip(traces, references)
        ]

    def attribute_failures(self, records: List[FailureRecord]) -> Dict[str, Any]:
        by_stage: Dict[str, int] = {stage.value: 0 for stage in FailureStage}
        total_failures = 0
        total_propagated = 0

        for record in records:
            by_stage[record.stage.value] += 1
            if record.stage != FailureStage.NONE:
                total_failures += 1
            if record.propagated:
                total_propagated += 1

        propagation_rate = (
            total_propagated / len(records) if records else 0.0
        )

        return {
            "by_stage": by_stage,
            "total_failures": total_failures,
            "propagation_rate": propagation_rate,
        }
