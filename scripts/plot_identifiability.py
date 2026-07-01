#!/usr/bin/env python3
"""Plot the attribution-identifiability curves (C2 headline + C3 comparison).

Reads results/identifiability_*.json (written by run_identifiability.py) and, per
(provider, dataset), produces a root-cause-attribution-accuracy vs injection-depth
line plot — one line per diagnoser. The collapse of the post-hoc baselines with
depth is the C2 finding; where the propagation-aware line sits above them is C3.

Usage:
    python scripts/plot_identifiability.py                 # all results/identifiability_*.json
    python scripts/plot_identifiability.py --glob 'results/identifiability_claude_*.json'
    python scripts/plot_identifiability.py --include-mock
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def _check_deps() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Missing matplotlib. Install with: pip install matplotlib")
        sys.exit(1)


_STYLES = {
    "rule_based": dict(marker="x", linestyle=":", color="#9e9e9e", label="Rule-based"),
    "doctor_rag": dict(marker="s", linestyle="--", color="#1f77b4", label="Doctor-RAG (coverage)"),
    "llm_judge": dict(marker="^", linestyle="-.", color="#ff7f0e", label="LLM-as-judge"),
    "propagation_aware": dict(marker="o", linestyle="-", color="#d62728", label="Propagation-aware (ours)"),
}


def _safe(s: str) -> str:
    return s.replace(":", "_").replace(".", "_").replace("/", "_")


def plot_one(result: dict, output_dir: str) -> None:
    import matplotlib.pyplot as plt

    prov = result.get("provider", "?")
    ds = result.get("dataset", "?")
    hops = result.get("hops", [])
    acc = result.get("accuracy", {})
    rec = result.get("recovery_rate_by_depth", {})
    nfail = result.get("n_failed_by_depth", {})
    if not hops or not acc:
        return

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for name, style in _STYLES.items():
        if name not in acc:
            continue
        ys = [acc[name].get(str(h), 0.0) for h in hops]
        ax.plot(hops, ys, **style)

    ax.set_xlabel("Injection depth (hop)", fontsize=11)
    ax.set_ylabel("Root-cause attribution accuracy", fontsize=11)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(hops)
    n_failed = [nfail.get(str(h), 0) for h in hops]
    rec_line = [round(rec.get(str(h), 0.0), 2) for h in hops]
    ax.set_title(
        f"Attribution identifiability — {prov} · {ds}\n"
        f"(n_failed/depth={n_failed}; recovery={rec_line})",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    base = os.path.join(output_dir, f"identifiability_curve_{_safe(prov)}_{ds}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {base}.pdf + .png")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot identifiability curves from result JSONs")
    p.add_argument("--glob", default="results/identifiability_*.json")
    p.add_argument("--output-dir", default="figures")
    p.add_argument("--include-mock", action="store_true",
                   help="Also plot the mock-backbone control curves")
    args = p.parse_args()

    _check_deps()
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No result files matched {args.glob}. Run scripts/run_identifiability.py first.")
        sys.exit(1)

    n = 0
    for fp in files:
        result = json.load(open(fp))
        if not args.include_mock and str(result.get("provider", "")).startswith("mock"):
            continue
        plot_one(result, args.output_dir)
        n += 1
    print(f"\nGenerated {n} identifiability figure(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
