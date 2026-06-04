from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .core import PipelineTrace, FailureStage


def make_trace(
    query: str = "What is revenue?",
    success: bool = True,
    failure_stage: str | None = None,
    seed: int = 42,
) -> Tuple[PipelineTrace, Dict]:
    """Create a (PipelineTrace, reference_dict) pair."""
    rng = random.Random(seed)
    reference_answer = f"The revenue is ${rng.randint(100, 999)} million."
    trace_id = f"trace-{rng.randint(1000, 9999)}"

    if success:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A: Financial report Q3.", "Doc B: Revenue breakdown."],
            tool_calls=[{"name": "search", "args": {"q": query}}],
            final_answer=reference_answer,
            reference_answer=reference_answer,
        )
        return trace, {"answer": reference_answer}

    stage = failure_stage or FailureStage.RETRIEVAL.value

    if stage == FailureStage.RETRIEVAL.value:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=[],
            tool_calls=[],
            final_answer="",
            reference_answer=reference_answer,
        )
    elif stage == FailureStage.TOOL_CALL.value:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[],
            final_answer="",
            reference_answer=reference_answer,
        )
    elif stage == FailureStage.ANSWER_GENERATION.value:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[{"name": "search", "args": {}}],
            final_answer="Wrong answer entirely.",
            reference_answer=reference_answer,
        )
    else:
        # Default: empty answer
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[{"name": "search", "args": {}}],
            final_answer="",
            reference_answer=reference_answer,
        )

    return trace, {"answer": reference_answer}


def make_dataset(
    n_success: int = 10,
    n_retrieval_fail: int = 5,
    n_tool_fail: int = 3,
    n_answer_fail: int = 7,
    seed: int = 42,
) -> Tuple[List[PipelineTrace], List[Dict]]:
    """Generate a dataset of traces and references."""
    rng = random.Random(seed)
    traces: List[PipelineTrace] = []
    refs: List[Dict] = []

    for i in range(n_success):
        t, r = make_trace(success=True, seed=rng.randint(0, 99999))
        traces.append(t)
        refs.append(r)

    for i in range(n_retrieval_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.RETRIEVAL.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    for i in range(n_tool_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.TOOL_CALL.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    for i in range(n_answer_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.ANSWER_GENERATION.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    return traces, refs
