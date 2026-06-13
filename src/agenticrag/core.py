"""Core data structures and logic for the agentic RAG diagnostic benchmark."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class FailureStage(str, Enum):
    """Stage in the agentic RAG pipeline where a failure occurs."""

    RETRIEVAL = "retrieval"
    TOOL_CALL = "tool_call"
    ANSWER_GENERATION = "answer_generation"
    NONE = "none"


class PipelineTrace(BaseModel):
    """A full trace through an agentic RAG pipeline."""

    trace_id: str
    query: str
    retrieved_docs: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    final_answer: str
    reference_answer: str


class FailureRecord(BaseModel):
    """Record of a failure detected in a pipeline trace."""

    trace_id: str
    stage: FailureStage
    failure_type: str
    propagated: bool
    root_cause: str


class DiagnosticBenchmark:
    """Benchmark harness for diagnosing failure propagation in agentic RAG."""

    def load_traces(self, path: str) -> list[PipelineTrace]:
        """Load pipeline traces from a JSON file.

        Each line is expected to be a JSON object conforming to PipelineTrace.
        """
        traces: list[PipelineTrace] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    traces.append(PipelineTrace.model_validate(json.loads(line)))
        return traces

    def diagnose_trace(
        self, trace: PipelineTrace, reference: dict[str, Any]
    ) -> FailureRecord:
        """Diagnose a single trace and return a FailureRecord stub.

        In a full implementation this would run LLM-based attribution.
        Here we apply simple heuristics.
        """
        # Heuristic: if no docs retrieved, failure is at retrieval
        if not trace.retrieved_docs:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type="empty_retrieval",
                propagated=True,
                root_cause="No documents retrieved for query.",
            )
        # Heuristic: if tool_calls present but answer is empty, tool failure
        if trace.tool_calls and not trace.final_answer.strip():
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.TOOL_CALL,
                failure_type="tool_no_output",
                propagated=True,
                root_cause="Tool call produced no output.",
            )
        # Heuristic: if answer does not match reference
        if trace.final_answer.strip().lower() != trace.reference_answer.strip().lower():
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type="wrong_answer",
                propagated=False,
                root_cause="Final answer does not match reference.",
            )
        return FailureRecord(
            trace_id=trace.trace_id,
            stage=FailureStage.NONE,
            failure_type="none",
            propagated=False,
            root_cause="",
        )

    def attribute_failures(self, records: list[FailureRecord]) -> dict[str, int]:
        """Count failures by stage."""
        counts: dict[str, int] = {stage.value: 0 for stage in FailureStage}
        for rec in records:
            counts[rec.stage.value] += 1
        return counts
