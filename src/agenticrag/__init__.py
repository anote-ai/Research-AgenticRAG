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
    root_cause_stage,
    root_cause_accuracy,
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
from .injection import FailureInjector, InjectionResult, injection_sensitivity

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
    "root_cause_stage",
    "root_cause_accuracy",
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
    "FailureInjector",
    "InjectionResult",
    "injection_sensitivity",
]
