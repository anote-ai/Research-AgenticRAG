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


class FailureType(str, Enum):
    # Retrieval stage
    EMPTY_RETRIEVAL = "empty_retrieval"
    # Tool call stage
    NO_TOOL_CALLS = "no_tool_calls"
    # Answer generation stage
    EMPTY_ANSWER = "empty_answer"
    INCORRECT_ANSWER = "incorrect_answer"
    HALLUCINATION = "hallucination"       # answer not grounded in retrieved docs
    # Multi-hop / loop failures
    OVER_RETRIEVAL = "over_retrieval"     # exhausted iteration budget without satisfying query
    CONTEXT_OVERFLOW = "context_overflow" # retrieved docs exceed context window
    # No failure
    SUCCESS = "success"


class PipelineTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    retrieved_docs: List[str]
    tool_calls: List[Dict[str, Any]]
    final_answer: str
    reference_answer: str
    # Multi-hop fields
    hop_queries: List[str] = Field(default_factory=list)
    hop_docs: List[List[str]] = Field(default_factory=list)
    iterations_used: int = 1


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


class KnowledgeNode(BaseModel):
    """A node in a synthetic knowledge graph."""

    node_id: str
    entity: str
    facts: List[str] = Field(default_factory=list)
    neighbors: List[str] = Field(default_factory=list)  # node_ids


class KnowledgeGraph(BaseModel):
    """Directed knowledge graph used for multi-hop question generation."""

    nodes: Dict[str, KnowledgeNode] = Field(default_factory=dict)

    def add_node(self, node: KnowledgeNode) -> None:
        self.nodes[node.node_id] = node

    def neighbors_of(self, node_id: str) -> List[KnowledgeNode]:
        node = self.nodes.get(node_id)
        if node is None:
            return []
        return [self.nodes[n] for n in node.neighbors if n in self.nodes]

    def facts_for_path(self, path: List[str]) -> List[str]:
        """Collect all facts along a node path."""
        facts: List[str] = []
        for nid in path:
            node = self.nodes.get(nid)
            if node:
                facts.extend(node.facts)
        return facts


def _token_overlap(a: str, b: str) -> float:
    """Jaccard token overlap between two strings."""
    tok_a = set(a.lower().split())
    tok_b = set(b.lower().split())
    if not tok_a and not tok_b:
        return 1.0
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / len(tok_a | tok_b)


def reformulate_query(query: str, retrieved_docs: List[str]) -> str:
    """Produce a follow-up query by appending context keywords from docs."""
    if not retrieved_docs:
        return query
    # Extract the most frequent non-stop words across docs
    stop = {"the", "a", "an", "is", "in", "of", "and", "to", "for"}
    word_counts: Dict[str, int] = {}
    for doc in retrieved_docs:
        for word in doc.lower().split():
            if word not in stop and len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
    top_words = sorted(word_counts, key=lambda w: -word_counts[w])[:3]
    if top_words:
        return f"{query} [context: {' '.join(top_words)}]"
    return query


class AgenticRAGPipeline:
    """Agent loop that performs multi-hop retrieval with query reformulation."""

    def __init__(
        self,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        max_iterations: int = 3,
    ) -> None:
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.max_iterations = max_iterations

    def _retrieve(self, query: str, docs: List[str]) -> List[str]:
        """Return docs with token overlap > 0 against query; fallback to all."""
        ranked = [(doc, _token_overlap(query, doc)) for doc in docs]
        relevant = [doc for doc, score in ranked if score > 0]
        return relevant if relevant else docs[:2]

    def run(
        self,
        query: str,
        corpus: List[str],
        reference_answer: str = "",
    ) -> PipelineTrace:
        """Execute the agent loop and return a PipelineTrace."""
        trace_id = str(uuid.uuid4())
        hop_queries: List[str] = []
        hop_docs: List[List[str]] = []
        tool_calls: List[Dict[str, Any]] = []
        current_query = query
        all_retrieved: List[str] = []

        for iteration in range(1, self.max_iterations + 1):
            hop_queries.append(current_query)
            retrieved = self._retrieve(current_query, corpus)
            hop_docs.append(retrieved)
            all_retrieved.extend(retrieved)
            tool_calls.append({"name": "retrieve", "args": {"q": current_query}, "iteration": iteration})

            # Check if answer is satisfiable from retrieved docs
            combined = " ".join(retrieved)
            if _token_overlap(reference_answer, combined) > 0.3:
                break

            # Reformulate for next hop
            current_query = reformulate_query(current_query, retrieved)

        # Generate a simple extractive answer: first sentence from top doc
        final_answer = ""
        if all_retrieved:
            first_sentence = all_retrieved[0].split(".")[0].strip()
            final_answer = first_sentence if first_sentence else all_retrieved[0]

        return PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=list(dict.fromkeys(all_retrieved)),  # deduplicate
            tool_calls=tool_calls,
            final_answer=final_answer,
            reference_answer=reference_answer,
            hop_queries=hop_queries,
            hop_docs=hop_docs,
            iterations_used=len(hop_queries),
        )


class DiagnosticBenchmark:
    """Diagnose pipeline traces and attribute failures."""

    # Tokens that must appear in retrieved docs for an answer to be grounded.
    # An answer is considered hallucinated when none of its content words appear
    # in any retrieved document.
    _MIN_GROUNDING_OVERLAP = 0.05

    def diagnose_trace(
        self, trace: PipelineTrace, reference: Dict[str, Any]
    ) -> FailureRecord:
        max_iter = reference.get("max_iterations", 3)

        if not trace.retrieved_docs:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.EMPTY_RETRIEVAL,
                propagated=True,
                root_cause="No documents retrieved",
                severity=0.9,
            )
        elif not trace.tool_calls:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.TOOL_CALL,
                failure_type=FailureType.NO_TOOL_CALLS,
                propagated=True,
                root_cause="No tool calls made",
                severity=0.7,
            )
        elif trace.iterations_used >= max_iter and trace.final_answer.strip() == "":
            # Exhausted the hop budget without producing an answer
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.RETRIEVAL,
                failure_type=FailureType.OVER_RETRIEVAL,
                propagated=True,
                root_cause="Iteration budget exhausted without a satisfying answer",
                severity=0.75,
            )
        elif trace.final_answer.strip() == "":
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type=FailureType.EMPTY_ANSWER,
                propagated=False,
                root_cause="Answer generation produced empty string",
                severity=0.8,
            )
        elif self._is_hallucination(trace):
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type=FailureType.HALLUCINATION,
                propagated=False,
                root_cause="Answer not grounded in retrieved documents",
                severity=0.65,
            )
        elif trace.final_answer != reference.get("answer", ""):
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.ANSWER_GENERATION,
                failure_type=FailureType.INCORRECT_ANSWER,
                propagated=False,
                root_cause="Answer does not match reference",
                severity=0.5,
            )
        else:
            return FailureRecord(
                trace_id=trace.trace_id,
                stage=FailureStage.NONE,
                failure_type=FailureType.SUCCESS,
                propagated=False,
                root_cause="",
                severity=0.0,
            )

    def _is_hallucination(self, trace: PipelineTrace) -> bool:
        """Return True when the answer has near-zero overlap with retrieved docs."""
        if not trace.final_answer.strip() or not trace.retrieved_docs:
            return False
        corpus_text = " ".join(trace.retrieved_docs)
        return _token_overlap(trace.final_answer, corpus_text) < self._MIN_GROUNDING_OVERLAP

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
