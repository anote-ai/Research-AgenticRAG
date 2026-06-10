from .core import (
    FailureStage,
    FailureType,
    PipelineTrace,
    FailureRecord,
    DiagnosticBenchmark,
    KnowledgeNode,
    KnowledgeGraph,
    AgenticRAGPipeline,
    reformulate_query,
)
from .evaluate import (
    stage_attribution_rate,
    propagation_rate,
    failure_confusion_matrix,
    end_to_end_accuracy,
    severity_weighted_failure_rate,
    multi_hop_accuracy,
    retrieval_loop_efficiency,
    mean_hops_per_trace,
    hop_doc_coverage,
    failure_amplification_rate,
    recovery_rate,
)
from .data import make_trace, make_dataset, make_multi_hop_trace, build_knowledge_graph, make_multi_hop_corpus
from .retrievers import BM25Retriever, TokenOverlapRetriever
from .datasets import QASample, load_hotpotqa, load_musique, load_dataset, iter_batches
from .injection import (
    InjectionSpec,
    InjectionLabel,
    InjectedTrace,
    inject_failure,
    inject_failures,
    make_injection_grid,
    group_records_by_hop,
)
from .propagation import PropagationEdge, PropagationGraph

__all__ = [
    # Core models
    "FailureStage",
    "FailureType",
    "PipelineTrace",
    "FailureRecord",
    "DiagnosticBenchmark",
    "KnowledgeNode",
    "KnowledgeGraph",
    "AgenticRAGPipeline",
    "reformulate_query",
    # Evaluation metrics
    "stage_attribution_rate",
    "propagation_rate",
    "failure_confusion_matrix",
    "end_to_end_accuracy",
    "severity_weighted_failure_rate",
    "multi_hop_accuracy",
    "retrieval_loop_efficiency",
    "mean_hops_per_trace",
    "hop_doc_coverage",
    "failure_amplification_rate",
    "recovery_rate",
    # Synthetic data
    "make_trace",
    "make_dataset",
    "make_multi_hop_trace",
    "build_knowledge_graph",
    "make_multi_hop_corpus",
    # Retrievers
    "BM25Retriever",
    "TokenOverlapRetriever",
    # Dataset adapters
    "QASample",
    "load_hotpotqa",
    "load_musique",
    "load_dataset",
    "iter_batches",
    # Failure injection
    "InjectionSpec",
    "InjectionLabel",
    "InjectedTrace",
    "inject_failure",
    "inject_failures",
    "make_injection_grid",
    "group_records_by_hop",
    # Propagation graph
    "PropagationEdge",
    "PropagationGraph",
]
