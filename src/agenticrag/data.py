from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .core import FailureStage, KnowledgeGraph, KnowledgeNode, PipelineTrace


# ---------------------------------------------------------------------------
# Knowledge graph construction
# ---------------------------------------------------------------------------

ENTITIES = [
    ("CompanyA", "tech company founded in 2005"),
    ("CompanyB", "acquired by CompanyA in 2018"),
    ("ProductX", "flagship product of CompanyA launched in 2015"),
    ("CEOJohn", "CEO of CompanyA since 2010"),
    ("InvestorFund", "major investor in CompanyA and CompanyB"),
    ("CountryUS", "headquarter country of CompanyA"),
    ("TechSector", "industry sector of CompanyA and ProductX"),
]

_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (1, 4), (3, 5), (0, 6), (2, 6),
]


def build_knowledge_graph() -> KnowledgeGraph:
    """Build a small synthetic knowledge graph of companies and products."""
    kg = KnowledgeGraph()
    nodes = []
    for i, (entity, desc) in enumerate(ENTITIES):
        node = KnowledgeNode(
            node_id=f"N{i}",
            entity=entity,
            facts=[f"{entity}: {desc}."],
        )
        nodes.append(node)
        kg.add_node(node)
    for src, dst in _EDGES:
        nodes[src].neighbors.append(f"N{dst}")
    return kg


# ---------------------------------------------------------------------------
# Multi-hop question generation
# ---------------------------------------------------------------------------

MULTI_HOP_QUESTIONS: List[Dict] = [
    {
        "question": "Who is the CEO of the company that acquired CompanyB?",
        "path": ["N1", "N0", "N3"],
        "answer": "CEOJohn is the CEO of CompanyA since 2010.",
        "hops": 2,
    },
    {
        "question": "In which sector does the flagship product of CompanyA belong?",
        "path": ["N0", "N2", "N6"],
        "answer": "ProductX belongs to the TechSector industry sector.",
        "hops": 2,
    },
    {
        "question": "Which fund invested in the company that launched ProductX?",
        "path": ["N2", "N0", "N1", "N4"],
        "answer": "InvestorFund is a major investor in CompanyA and CompanyB.",
        "hops": 3,
    },
    {
        "question": "What country is home to the CEO of the company in TechSector?",
        "path": ["N6", "N0", "N3", "N5"],
        "answer": "CEOJohn is headquartered in CountryUS.",
        "hops": 3,
    },
]


def make_multi_hop_corpus(kg: KnowledgeGraph) -> List[str]:
    """Build a flat document corpus from all graph node facts."""
    corpus: List[str] = []
    for node in kg.nodes.values():
        corpus.extend(node.facts)
        # Add relational sentences for neighbors
        for neighbor_id in node.neighbors:
            nb = kg.nodes.get(neighbor_id)
            if nb:
                corpus.append(f"{node.entity} is connected to {nb.entity}.")
    return corpus


# ---------------------------------------------------------------------------
# Original trace helpers (kept for backwards compatibility)
# ---------------------------------------------------------------------------


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
            hop_queries=[query],
            hop_docs=[["Doc A: Financial report Q3.", "Doc B: Revenue breakdown."]],
            iterations_used=1,
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
            hop_queries=[query],
            hop_docs=[[]],
            iterations_used=1,
        )
    elif stage == FailureStage.TOOL_CALL.value:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[],
            final_answer="",
            reference_answer=reference_answer,
            hop_queries=[query],
            hop_docs=[["Doc A"]],
            iterations_used=1,
        )
    elif stage == FailureStage.ANSWER_GENERATION.value:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[{"name": "search", "args": {}}],
            final_answer="Wrong answer entirely.",
            reference_answer=reference_answer,
            hop_queries=[query],
            hop_docs=[["Doc A"]],
            iterations_used=1,
        )
    else:
        trace = PipelineTrace(
            trace_id=trace_id,
            query=query,
            retrieved_docs=["Doc A"],
            tool_calls=[{"name": "search", "args": {}}],
            final_answer="",
            reference_answer=reference_answer,
            hop_queries=[query],
            hop_docs=[["Doc A"]],
            iterations_used=1,
        )

    return trace, {"answer": reference_answer}


def make_multi_hop_trace(
    question_idx: int = 0,
    kg: KnowledgeGraph | None = None,
    seed: int = 42,
) -> Tuple[PipelineTrace, Dict]:
    """Create a multi-hop PipelineTrace using a knowledge graph question."""
    if kg is None:
        kg = build_knowledge_graph()
    rng = random.Random(seed)
    q_data = MULTI_HOP_QUESTIONS[question_idx % len(MULTI_HOP_QUESTIONS)]
    corpus = make_multi_hop_corpus(kg)

    hop_queries: List[str] = []
    hop_docs_list: List[List[str]] = []
    all_docs: List[str] = []
    query = q_data["question"]
    current_query = query
    for _ in range(q_data["hops"]):
        hop_queries.append(current_query)
        sample = rng.sample(corpus, min(3, len(corpus)))
        hop_docs_list.append(sample)
        all_docs.extend(sample)
        current_query = f"{current_query} [follow-up]"

    return PipelineTrace(
        trace_id=f"mh-{rng.randint(1000, 9999)}",
        query=query,
        retrieved_docs=list(dict.fromkeys(all_docs)),
        tool_calls=[{"name": "retrieve", "args": {"q": hq}} for hq in hop_queries],
        final_answer=q_data["answer"],
        reference_answer=q_data["answer"],
        hop_queries=hop_queries,
        hop_docs=hop_docs_list,
        iterations_used=q_data["hops"],
    ), {"answer": q_data["answer"]}


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

    for _ in range(n_success):
        t, r = make_trace(success=True, seed=rng.randint(0, 99999))
        traces.append(t)
        refs.append(r)

    for _ in range(n_retrieval_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.RETRIEVAL.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    for _ in range(n_tool_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.TOOL_CALL.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    for _ in range(n_answer_fail):
        t, r = make_trace(
            success=False,
            failure_stage=FailureStage.ANSWER_GENERATION.value,
            seed=rng.randint(0, 99999),
        )
        traces.append(t)
        refs.append(r)

    return traces, refs
