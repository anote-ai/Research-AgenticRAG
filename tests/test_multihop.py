from __future__ import annotations

import pytest

from agenticrag.core import AgenticRAGPipeline, KnowledgeGraph, KnowledgeNode, reformulate_query
from agenticrag.data import (
    build_knowledge_graph,
    make_multi_hop_corpus,
    make_multi_hop_trace,
)
from agenticrag.evaluate import (
    hop_doc_coverage,
    mean_hops_per_trace,
    multi_hop_accuracy,
    retrieval_loop_efficiency,
)


# ---------------------------------------------------------------------------
# Knowledge graph tests
# ---------------------------------------------------------------------------


def test_knowledge_graph_build() -> None:
    kg = build_knowledge_graph()
    assert len(kg.nodes) == 7
    # Every node should have at least one fact
    for node in kg.nodes.values():
        assert len(node.facts) >= 1


def test_knowledge_graph_neighbors() -> None:
    kg = build_knowledge_graph()
    # N0 (CompanyA) should have multiple neighbors
    neighbors = kg.neighbors_of("N0")
    assert len(neighbors) >= 2


def test_facts_for_path() -> None:
    kg = build_knowledge_graph()
    facts = kg.facts_for_path(["N0", "N1"])
    assert len(facts) >= 2


# ---------------------------------------------------------------------------
# Corpus tests
# ---------------------------------------------------------------------------


def test_corpus_non_empty() -> None:
    kg = build_knowledge_graph()
    corpus = make_multi_hop_corpus(kg)
    assert len(corpus) > 0
    assert all(isinstance(d, str) and len(d) > 0 for d in corpus)


# ---------------------------------------------------------------------------
# AgenticRAGPipeline tests
# ---------------------------------------------------------------------------


def test_pipeline_single_hop() -> None:
    corpus = ["Revenue for Q3 was 500 million.", "Operating income grew by 10%."]
    pipeline = AgenticRAGPipeline(max_iterations=3)
    trace = pipeline.run("What is revenue?", corpus, reference_answer="Revenue for Q3 was 500 million.")
    assert trace.iterations_used >= 1
    assert len(trace.retrieved_docs) > 0
    assert trace.final_answer != ""


def test_pipeline_multi_hop_iterations() -> None:
    corpus = [
        "CompanyA acquired CompanyB in 2018.",
        "CompanyB was founded by InvestorFund.",
        "InvestorFund manages $10B in assets.",
    ]
    pipeline = AgenticRAGPipeline(max_iterations=3)
    trace = pipeline.run(
        "Who founded the company acquired by CompanyA?",
        corpus,
        reference_answer="InvestorFund manages 10B",
    )
    assert trace.iterations_used <= 3
    assert len(trace.hop_queries) == trace.iterations_used
    assert len(trace.hop_docs) == trace.iterations_used


def test_pipeline_empty_corpus() -> None:
    pipeline = AgenticRAGPipeline(max_iterations=2)
    trace = pipeline.run("anything?", [], reference_answer="answer")
    assert trace.final_answer == ""
    assert trace.retrieved_docs == []


def test_pipeline_max_iterations_respected() -> None:
    corpus = ["Unrelated doc about weather."]
    pipeline = AgenticRAGPipeline(max_iterations=2)
    trace = pipeline.run("What is the capital of France?", corpus, reference_answer="Paris")
    assert trace.iterations_used <= 2


# ---------------------------------------------------------------------------
# Reformulate query tests
# ---------------------------------------------------------------------------


def test_reformulate_query_adds_context() -> None:
    docs = ["revenue report financial data analysis"]
    result = reformulate_query("original query", docs)
    assert "original query" in result
    assert len(result) > len("original query")


def test_reformulate_query_empty_docs() -> None:
    result = reformulate_query("my query", [])
    assert result == "my query"


# ---------------------------------------------------------------------------
# New evaluation metric tests
# ---------------------------------------------------------------------------


def test_multi_hop_accuracy_all_success() -> None:
    kg = build_knowledge_graph()
    traces = [make_multi_hop_trace(i, kg)[0] for i in range(4)]
    acc = multi_hop_accuracy(traces)
    assert 0.0 <= acc <= 1.0
    assert acc == 1.0  # all have non-empty answers


def test_multi_hop_accuracy_no_multihop() -> None:
    from agenticrag.data import make_trace
    traces = [make_trace(success=True, seed=i)[0] for i in range(5)]
    # single-hop traces → should return 0
    acc = multi_hop_accuracy(traces)
    assert acc == 0.0


def test_retrieval_loop_efficiency() -> None:
    kg = build_knowledge_graph()
    traces = [make_multi_hop_trace(i, kg)[0] for i in range(2)]
    eff = retrieval_loop_efficiency(traces, max_iterations=3)
    assert 0.0 <= eff <= 1.0


def test_retrieval_loop_efficiency_empty() -> None:
    assert retrieval_loop_efficiency([], max_iterations=3) == 0.0


def test_mean_hops_per_trace() -> None:
    kg = build_knowledge_graph()
    traces = [make_multi_hop_trace(i, kg)[0] for i in range(4)]
    mean = mean_hops_per_trace(traces)
    assert mean >= 1.0


def test_hop_doc_coverage_full() -> None:
    kg = build_knowledge_graph()
    traces = [make_multi_hop_trace(i, kg)[0] for i in range(3)]
    cov = hop_doc_coverage(traces)
    assert cov == 1.0  # all hops have docs


def test_hop_doc_coverage_empty_list() -> None:
    assert hop_doc_coverage([]) == 0.0
