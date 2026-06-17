#!/usr/bin/env python3
"""Phase 4: generate paper figures from saved ablation results.

Requires matplotlib and seaborn:
    pip install matplotlib seaborn

Produces (in --output-dir, default figures/):
  propagation_heatmap_{retriever}_{dataset}_{method}.{pdf,png}
      — Coupling matrix from PropagationGraph (source→target stage heatmap)
  amplification_curves_{retriever}_{dataset}.{pdf,png}
      — Failure rate vs. injection hop depth (Experiment 3)
  recovery_rates_{retriever}_{dataset}.{pdf,png}
      — Recovery rate bar chart per failure type (Experiment 4)
  benchmark_table_{metric}.{pdf,png}
      — Retriever × dataset heatmap table for sensitivity / RCA / severity (Experiment 5)

Usage:
    python scripts/plot_amplification.py                         # needs results/ablation_all.json
    python scripts/plot_amplification.py --input results/ablation_bm25_hotpotqa.json
    python scripts/plot_amplification.py --figures amplification recovery
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _check_deps() -> None:
    missing = []
    for pkg in ("matplotlib", "seaborn", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing plotting dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def _save(fig: Any, output_dir: str, fname: str) -> None:
    import matplotlib.pyplot as plt

    pdf_path = os.path.join(output_dir, fname)
    png_path = pdf_path.replace(".pdf", ".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {pdf_path} + {png_path}")


# ---------------------------------------------------------------------------
# Figure 1: Propagation heatmap (coupling matrix)
# ---------------------------------------------------------------------------

def plot_propagation_heatmaps(results: List[Dict], output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    stage_labels = ["retrieval", "tool_call", "answer_generation"]
    display_labels = ["Retrieval", "Tool Call", "Answer Gen"]

    for result in results:
        for method, prop_summary in result.get("propagation_graphs", {}).items():
            coupling = prop_summary.get("coupling_matrix", {})
            if not coupling:
                continue

            matrix = np.zeros((len(stage_labels), len(stage_labels)))
            for i, src in enumerate(stage_labels):
                for j, tgt in enumerate(stage_labels):
                    matrix[i, j] = coupling.get(src, {}).get(tgt, 0.0)

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                xticklabels=display_labels,
                yticklabels=display_labels,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                ax=ax,
                linewidths=0.5,
            )
            ax.set_xlabel("Target Stage (H+1)", fontsize=11)
            ax.set_ylabel("Source Stage (H)", fontsize=11)
            short_method = method.replace("inject_", "")
            ax.set_title(
                f"Failure Propagation Coupling Matrix\n"
                f"{result['retriever']} · {result['dataset']} · {short_method}",
                fontsize=11,
            )
            plt.tight_layout()

            fname = (
                f"propagation_heatmap_{result['retriever']}_"
                f"{result['dataset']}_{short_method}.pdf"
            )
            _save(fig, output_dir, fname)


# ---------------------------------------------------------------------------
# Figure 2: Failure amplification curves
# ---------------------------------------------------------------------------

def plot_amplification_curves(results: List[Dict], output_dir: str) -> None:
    import matplotlib.pyplot as plt

    method_styles = {
        "inject_empty_retrieval": dict(marker="o", linestyle="-", label="Empty Retrieval"),
        "inject_irrelevant_docs": dict(marker="s", linestyle="--", label="Irrelevant Docs"),
    }

    for result in results:
        amp = result.get("failure_amplification", {})
        if not amp:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        any_plotted = False
        for method, style in method_styles.items():
            if method not in amp:
                continue
            hop_data: Dict[str, float] = amp[method]
            hops = sorted(int(h) for h in hop_data.keys())
            rates = [hop_data[str(h)] for h in hops]
            ax.plot(hops, rates, **style)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue

        ax.set_xlabel("Injection Hop Depth", fontsize=11)
        ax.set_ylabel("Downstream Failure Rate", fontsize=11)
        ax.set_title(
            f"Failure Amplification by Hop Depth\n"
            f"{result['retriever']} · {result['dataset']}",
            fontsize=11,
        )
        ax.set_ylim(0.0, 1.05)
        all_hops = sorted({int(h) for m in amp.values() for h in m.keys()})
        ax.set_xticks(all_hops)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = f"amplification_curves_{result['retriever']}_{result['dataset']}.pdf"
        _save(fig, output_dir, fname)


# ---------------------------------------------------------------------------
# Figure 3: Recovery rate bar chart
# ---------------------------------------------------------------------------

def plot_recovery_rates(results: List[Dict], output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    method_labels = {
        "inject_empty_retrieval": "Empty\nRetrieval",
        "inject_irrelevant_docs": "Irrelevant\nDocs",
    }
    colors = ["#2196F3", "#FF9800"]

    for result in results:
        rr: Dict[str, float] = result.get("recovery_rates", {})
        if not rr:
            continue

        labels = [method_labels.get(m, m.replace("inject_", "")) for m in rr]
        values = list(rr.values())

        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            labels, values,
            color=colors[: len(values)],
            edgecolor="white",
            linewidth=1.5,
        )
        ax.set_ylabel("Recovery Rate", fontsize=11)
        ax.set_title(
            f"Recovery Rate by Fault Type\n"
            f"{result['retriever']} · {result['dataset']}",
            fontsize=11,
        )
        ax.set_ylim(0.0, 1.15)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        fname = f"recovery_rates_{result['retriever']}_{result['dataset']}.pdf"
        _save(fig, output_dir, fname)


# ---------------------------------------------------------------------------
# Figure 4: Retriever × dataset benchmark table
# ---------------------------------------------------------------------------

def plot_benchmark_table(results: List[Dict], output_dir: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    if not results:
        return

    # Collect all method × hop keys (union across all conditions)
    all_keys: List[str] = []
    for result in results:
        for key in result.get("metrics_table", {}):
            if key not in all_keys:
                all_keys.append(key)
    if not all_keys:
        return

    conditions = [f"{r['retriever']}\n{r['dataset']}" for r in results]
    metrics_cfg = {
        "sensitivity": "Sensitivity",
        "root_cause_accuracy": "Root-Cause Accuracy",
        "severity_rate": "Severity",
    }

    for metric_key, metric_label in metrics_cfg.items():
        matrix = np.zeros((len(results), len(all_keys)))
        for i, result in enumerate(results):
            mt = result.get("metrics_table", {})
            for j, key in enumerate(all_keys):
                matrix[i, j] = mt.get(key, {}).get(metric_key, 0.0)

        fig_w = max(8, len(all_keys) * 1.1)
        fig_h = max(3, len(results) * 0.9 + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)

        ax.set_xticks(range(len(all_keys)))
        ax.set_xticklabels(all_keys, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(conditions)))
        ax.set_yticklabels(conditions, fontsize=9)

        for i in range(len(results)):
            for j in range(len(all_keys)):
                val = matrix[i, j]
                color = "white" if val > 0.65 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=7, color=color,
                )

        ax.set_title(
            f"Benchmark Table — {metric_label}\n(retriever × dataset × injection method)",
            fontsize=11,
        )
        plt.tight_layout()

        fname = f"benchmark_table_{metric_key}.pdf"
        _save(fig, output_dir, fname)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FIGURE_REGISTRY = {
    "heatmap": plot_propagation_heatmaps,
    "amplification": plot_amplification_curves,
    "recovery": plot_recovery_rates,
    "benchmark": plot_benchmark_table,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper figures from ablation results"
    )
    parser.add_argument(
        "--input",
        default="results/ablation_all.json",
        help="Path to ablation JSON (single result dict or list of results)",
    )
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument(
        "--figures",
        nargs="+",
        default=list(FIGURE_REGISTRY.keys()),
        choices=list(FIGURE_REGISTRY.keys()),
        help="Which figures to generate",
    )
    args = parser.parse_args()

    _check_deps()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        print("Run scripts/run_ablation.py first to generate ablation results.")
        sys.exit(1)

    with open(args.input) as f:
        data = json.load(f)

    results: List[Dict] = data if isinstance(data, list) else [data]
    print(f"Loaded {len(results)} condition(s) from {args.input}")

    for fig_name in args.figures:
        print(f"\n--- Generating: {fig_name} ---")
        FIGURE_REGISTRY[fig_name](results, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
