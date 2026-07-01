#!/usr/bin/env python3
"""Phase 0 baseline: natural failure rates without fault injection.

Runs AgenticRAGPipeline on HotpotQA and/or MuSiQue samples with a chosen
retriever, diagnoses all traces, and prints baseline metrics.

Results are saved to results/baseline_{retriever}_{dataset}.json for
downstream use by plot_amplification.py.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --dataset hotpotqa --retriever bm25 --max-samples 100
    python scripts/run_baseline.py --all-conditions --max-samples 50
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
    FailureStage,
    PipelineTrace,
    TokenOverlapRetriever,
    end_to_end_accuracy,
    load_dataset,
    multi_hop_accuracy,
    propagation_rate,
    retrieval_loop_efficiency,
    severity_weighted_failure_rate,
    stage_attribution_rate,
)
from agenticrag.datasets import QASample


def _make_retriever(name: str) -> Any:
    if name == "bm25":
        return BM25Retriever()
    if name in ("token_overlap", "overlap"):
        return TokenOverlapRetriever()
    raise ValueError(f"Unknown retriever '{name}'. Choose 'bm25' or 'token_overlap'.")


def build_traces(
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


def compute_baseline(
    dataset_name: str,
    retriever_name: str,
    max_samples: int,
    max_iterations: int = 3,
) -> Dict:
    samples = load_dataset(dataset_name, split="validation", max_samples=max_samples)
    retriever = _make_retriever(retriever_name)
    traces, refs = build_traces(samples, retriever, max_iterations=max_iterations)

    bench = DiagnosticBenchmark()
    records = bench.batch_diagnose(traces, refs)
    attribution = bench.attribute_failures(records)

    return {
        "dataset": dataset_name,
        "retriever": retriever_name,
        "n_samples": len(traces),
        "end_to_end_accuracy": end_to_end_accuracy(records),
        "propagation_rate": propagation_rate(records),
        "severity_weighted_failure_rate": severity_weighted_failure_rate(records),
        "multi_hop_accuracy": multi_hop_accuracy(traces),
        "retrieval_loop_efficiency": retrieval_loop_efficiency(
            traces, max_iterations=max_iterations
        ),
        "total_failures": attribution["total_failures"],
        "stage_rates": {
            stage.value: stage_attribution_rate(records, stage)
            for stage in FailureStage
            if stage != FailureStage.NONE
        },
    }


def _print_result(result: Dict) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"Baseline — {result['retriever']} on {result['dataset']}"
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        table.add_row("n_samples", str(result["n_samples"]))
        table.add_row("end_to_end_accuracy", f"{result['end_to_end_accuracy']:.4f}")
        table.add_row("propagation_rate", f"{result['propagation_rate']:.4f}")
        table.add_row(
            "severity_weighted_failure_rate",
            f"{result['severity_weighted_failure_rate']:.4f}",
        )
        table.add_row("multi_hop_accuracy", f"{result['multi_hop_accuracy']:.4f}")
        table.add_row(
            "retrieval_loop_efficiency",
            f"{result['retrieval_loop_efficiency']:.4f}",
        )
        table.add_row("total_failures", str(result["total_failures"]))
        for stage, rate in result["stage_rates"].items():
            table.add_row(f"  stage_{stage}", f"{rate:.4f}")

        console.print(table)
    except ImportError:
        print(f"\n=== {result['retriever']} on {result['dataset']} ===")
        for k, v in result.items():
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0 baseline experiment"
    )
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=["hotpotqa", "musique"],
    )
    parser.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "token_overlap"],
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument(
        "--all-conditions",
        action="store_true",
        help="Run all 4 retriever × dataset conditions",
    )
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    conditions = (
        [("hotpotqa", "bm25"), ("hotpotqa", "token_overlap"),
         ("musique", "bm25"),  ("musique", "token_overlap")]
        if args.all_conditions
        else [(args.dataset, args.retriever)]
    )

    all_results = []
    for dataset, retriever in conditions:
        print(f"\nRunning baseline: {retriever} on {dataset} ({args.max_samples} samples)")
        result = compute_baseline(
            dataset, retriever, args.max_samples, args.max_iterations
        )
        _print_result(result)

        out_path = os.path.join(args.output_dir, f"baseline_{retriever}_{dataset}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to {out_path}")
        all_results.append(result)

    if len(all_results) > 1:
        consolidated = os.path.join(args.output_dir, "baseline_all.json")
        with open(consolidated, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nConsolidated results saved to {consolidated}")


if __name__ == "__main__":
    main()
