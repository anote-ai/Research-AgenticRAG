#!/usr/bin/env python3
"""Experiments 1–5: full ablation study across fault types, retrievers, and datasets.

Runs run_ablation() over all (injection method × hop) pairs for BM25 and
TokenOverlap retrievers on HotpotQA and MuSiQue.  For each condition:
  - Prints the paper benchmark table (sensitivity, root-cause accuracy, severity)
  - Builds a PropagationGraph from hop-based injection records
  - Computes failure amplification curves and recovery rates
  - Saves full results to results/ablation_{retriever}_{dataset}.json

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --datasets hotpotqa --retrievers bm25 --max-samples 50
    python scripts/run_ablation.py --all-conditions --max-samples 100 --hops 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _env import load_dotenv

load_dotenv()  # populate os.environ from .env (e.g. HF_TOKEN) before loaders run

from agenticrag import (
    AgenticRAGPipeline,
    BM25Retriever,
    DiagnosticBenchmark,
    FailureInjector,
    FailureStage,
    PipelineTrace,
    PropagationGraph,
    TokenOverlapRetriever,
    failure_amplification_rate,
    load_dataset,
    recovery_rate,
    run_ablation,
)
from agenticrag.datasets import QASample

ALL_DATASETS = ["hotpotqa", "musique"]
ALL_RETRIEVERS = ["bm25", "token_overlap"]
HOP_METHODS = ["inject_empty_retrieval", "inject_irrelevant_docs"]


def _make_retriever(name: str) -> Any:
    if name == "bm25":
        return BM25Retriever()
    if name in ("token_overlap", "overlap"):
        return TokenOverlapRetriever()
    raise ValueError(f"Unknown retriever '{name}'.")


def _build_traces(
    samples: List[QASample],
    retriever: Any,
    max_iterations: int = 3,
) -> Tuple[List[PipelineTrace], List[Dict]]:
    pipeline = AgenticRAGPipeline(max_iterations=max_iterations, retriever=retriever)
    traces: List[PipelineTrace] = []
    refs: List[Dict] = []
    for sample in samples:
        trace = pipeline.run(
            sample.question,
            sample.supporting_docs,
            reference_answer=sample.answer,
        )
        traces.append(trace)
        refs.append({"answer": sample.answer, "max_iterations": max_iterations})
    return traces, refs


def _build_propagation_data(result, hops: List[int]) -> Dict:
    """Extract PropagationGraph summaries, amplification curves, and recovery rates."""
    prop_summaries: Dict[str, Any] = {}
    amplification: Dict[str, Dict[str, float]] = {}
    recovery_rates: Dict[str, float] = {}

    for method in HOP_METHODS:
        rbh = result.records_by_hop(method)
        if not rbh:
            continue

        graph = PropagationGraph()
        graph.infer_from_hops(rbh)
        prop_summaries[method] = graph.summary()

        amp = failure_amplification_rate(rbh)
        amplification[method] = {str(k): v for k, v in amp.items()}

        recovery_rates[method] = recovery_rate(rbh)

    return {
        "propagation_graphs": prop_summaries,
        "failure_amplification": amplification,
        "recovery_rates": recovery_rates,
    }


def run_condition(
    dataset_name: str,
    retriever_name: str,
    max_samples: int,
    hops: List[int],
    max_iterations: int = 3,
) -> Dict:
    samples = load_dataset(dataset_name, split="validation", max_samples=max_samples)
    retriever = _make_retriever(retriever_name)
    traces, refs = _build_traces(samples, retriever, max_iterations=max_iterations)

    injector = FailureInjector()
    bench = DiagnosticBenchmark()
    ablation_result = run_ablation(traces, refs, injector, bench, hops=hops)

    prop_data = _build_propagation_data(ablation_result, hops)

    return {
        "dataset": dataset_name,
        "retriever": retriever_name,
        "n_samples": ablation_result.n_samples,
        "baseline_accuracy": ablation_result.baseline_accuracy(),
        "baseline_severity": ablation_result.baseline_severity(),
        "metrics_table": ablation_result.metrics_table(),
        "sensitivity_table": ablation_result.sensitivity_table(),
        **prop_data,
    }


def _print_metrics_table(cond: Dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        title = (
            f"Benchmark — {cond['retriever']} on {cond['dataset']}  "
            f"(baseline acc={cond['baseline_accuracy']:.3f})"
        )
        table = Table(title=title)
        table.add_column("Method@Hop", style="bold")
        table.add_column("Sensitivity", justify="right")
        table.add_column("Root-Cause Acc", justify="right")
        table.add_column("Severity", justify="right")

        for key, metrics in cond["metrics_table"].items():
            table.add_row(
                key,
                f"{metrics['sensitivity']:.3f}",
                f"{metrics['root_cause_accuracy']:.3f}",
                f"{metrics['severity_rate']:.3f}",
            )

        console.print(table)

        # Print amplification summary
        amp = cond.get("failure_amplification", {})
        if amp:
            console.print("\n[bold]Failure amplification by injection hop:[/bold]")
            for method, rates in amp.items():
                console.print(
                    f"  {method}: "
                    + "  ".join(f"hop{h}={v:.3f}" for h, v in sorted(rates.items()))
                )

        rr = cond.get("recovery_rates", {})
        if rr:
            console.print("\n[bold]Recovery rates:[/bold]")
            for method, rate in rr.items():
                console.print(f"  {method}: {rate:.3f}")

    except ImportError:
        print(f"\n=== {cond['retriever']} on {cond['dataset']} ===")
        for key, metrics in cond["metrics_table"].items():
            print(
                f"  {key}: sens={metrics['sensitivity']:.3f}"
                f"  rca={metrics['root_cause_accuracy']:.3f}"
                f"  sev={metrics['severity_rate']:.3f}"
            )
        for method, rates in cond.get("failure_amplification", {}).items():
            print(f"  amp/{method}: {rates}")
        for method, rate in cond.get("recovery_rates", {}).items():
            print(f"  recovery/{method}: {rate:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full ablation study (Experiments 1–5)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa"],
        choices=ALL_DATASETS,
    )
    parser.add_argument(
        "--retrievers",
        nargs="+",
        default=["bm25"],
        choices=ALL_RETRIEVERS,
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all 4 retriever × dataset conditions",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.all_conditions:
        conditions = [
            (d, r) for d in ALL_DATASETS for r in ALL_RETRIEVERS
        ]
    else:
        conditions = [(d, r) for d in args.datasets for r in args.retrievers]

    all_results = []
    for dataset, retriever in conditions:
        print(
            f"\nRunning ablation: {retriever} on {dataset} "
            f"({args.max_samples} samples, hops={args.hops})"
        )
        cond = run_condition(
            dataset, retriever, args.max_samples, args.hops, args.max_iterations
        )
        _print_metrics_table(cond)

        out_path = os.path.join(
            args.output_dir, f"ablation_{retriever}_{dataset}.json"
        )
        with open(out_path, "w") as f:
            json.dump(cond, f, indent=2)
        print(f"Saved to {out_path}")
        all_results.append(cond)

    consolidated = os.path.join(args.output_dir, "ablation_all.json")
    with open(consolidated, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nConsolidated results saved to {consolidated}")


if __name__ == "__main__":
    main()
