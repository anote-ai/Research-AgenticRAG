#!/usr/bin/env python3
"""Demo script for agenticrag package."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.core import DiagnosticBenchmark
from agenticrag.data import make_dataset
from agenticrag.evaluate import (
    end_to_end_accuracy,
    propagation_rate,
    severity_weighted_failure_rate,
    stage_attribution_rate,
)
from agenticrag.core import FailureStage


def main() -> None:
    try:
        from rich.console import Console
        from rich.table import Table
        console = Console()
    except ImportError:
        console = None

    traces, refs = make_dataset(n_success=10, n_retrieval_fail=5, n_tool_fail=3, n_answer_fail=7)
    bench = DiagnosticBenchmark()
    records = bench.batch_diagnose(traces, refs)
    attribution = bench.attribute_failures(records)

    print(f"\nTotal traces: {len(traces)}")
    print(f"End-to-end accuracy: {end_to_end_accuracy(records):.4f}")
    print(f"Propagation rate: {propagation_rate(records):.4f}")
    print(f"Severity-weighted failure rate: {severity_weighted_failure_rate(records):.4f}")

    print("\nFailure attribution by stage:")
    for stage_val, count in attribution["by_stage"].items():
        rate = stage_attribution_rate(records, FailureStage(stage_val)) if stage_val != "none" else 0.0
        print(f"  {stage_val}: {count} ({rate:.2%} of failures)")

    print(f"\nTotal failures: {attribution['total_failures']}")
    print(f"Propagation rate (attr): {attribution['propagation_rate']:.4f}")


if __name__ == "__main__":
    main()
