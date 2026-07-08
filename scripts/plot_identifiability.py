#!/usr/bin/env python3
"""Plot the attribution-identifiability curves (C2 headline + C3 comparison).

Reads results/identifiability_*.json (written by run_identifiability.py) and, per
(provider, dataset), produces a root-cause-attribution-accuracy vs injection-depth
line plot — one line per diagnoser plus a "best post-hoc" envelope.

LLM-judge is treated as the main post-hoc comparator (equal visual weight to
Doctor-RAG). The "best post-hoc" line shows the element-wise max of the three
post-hoc methods, giving the tightest baseline the propagation-aware method
must beat.

Usage:
    python scripts/plot_identifiability.py                 # all results/identifiability_*.json
    python scripts/plot_identifiability.py --glob 'results/identifiability_claude_*.json'
    python scripts/plot_identifiability.py --include-mock
    python scripts/plot_identifiability.py --no-best-posthoc   # omit the envelope line
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
    # LLM-judge is promoted to an equal visual weight as Doctor-RAG (solid line, named marker).
    "llm_judge": dict(marker="^", linestyle="-", color="#ff7f0e", label="LLM-as-judge"),
    "propagation_aware": dict(marker="o", linestyle="-", color="#d62728", label="Propagation-aware (ours)"),
    "suffix_regen": dict(marker="D", linestyle="-", color="#2ca02c", label="Suffix-regeneration (ours)"),
}

# Best post-hoc envelope line (element-wise max of doctor_rag, rule_based, llm_judge).
_BEST_POSTHOC_STYLE = dict(
    linestyle="--", color="#7f7f7f", linewidth=1.5, label="Best post-hoc", zorder=1
)
_POST_HOC_NAMES = ["doctor_rag", "rule_based", "llm_judge"]


def _safe(s: str) -> str:
    return s.replace(":", "_").replace(".", "_").replace("/", "_")


def _compute_best_posthoc(acc: dict, hops: list) -> list:
    """Element-wise max of post-hoc baselines per hop depth."""
    result = []
    for h in hops:
        vals = [acc.get(n, {}).get(str(h), 0.0) for n in _POST_HOC_NAMES if n in acc]
        result.append(max(vals) if vals else 0.0)
    return result


def _pa_vs_llm_delta(acc: dict, hops: list) -> list:
    """PA accuracy minus LLM-judge accuracy per depth (rounded to 3dp)."""
    pa = acc.get("propagation_aware", {})
    llm = acc.get("llm_judge", {})
    return [round(pa.get(str(h), 0.0) - llm.get(str(h), 0.0), 3) for h in hops]


def _strict_accuracy(result: dict) -> dict:
    """Rescore accuracies from raw diagnoses with actual-depth filtering.

    Pre-strict-fix result files have depth buckets contaminated by clamped
    shallow interventions; the stored ``accuracy`` field reflects that
    contamination. When raw diagnoses are persisted, recompute accuracy
    keeping only cases whose actual injected hop equals the requested depth.
    Returns {} when no raw data is available (caller falls back to stored).
    """
    if not result.get("raw_by_depth"):
        return {}
    from agenticrag.evaluate import rescore_identifiability

    strict = rescore_identifiability(result, criterion="hop", require_actual_depth=True)
    # Match the stored-accuracy shape: {diagnoser: {str(depth): acc}}, dropping
    # depths whose strict bucket is empty (no cases at the requested depth).
    ns = {
        int(d): sum(1 for t in e.get("truth", []) if len(t) >= 2 and int(t[1]) == int(d))
        for d, e in result["raw_by_depth"].items()
    }
    return {
        name: {str(d): a for d, a in by_depth.items() if ns.get(int(d), 0) > 0}
        for name, by_depth in strict.items()
    }


def plot_one(
    result: dict, output_dir: str, show_best_posthoc: bool = True,
    source_path: str = "", strict_rescore: bool = True,
) -> None:
    import matplotlib.pyplot as plt

    prov = result.get("provider", "?")
    ds = result.get("dataset", "?")
    hops = result.get("hops", [])
    acc = (_strict_accuracy(result) if strict_rescore else {}) or result.get("accuracy", {})
    rec = result.get("recovery_rate_by_depth", {})
    nfail = result.get("n_failed_by_depth", {})
    if not hops or not acc:
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, style in _STYLES.items():
        if name not in acc:
            continue
        # Under strict rescoring a missing depth means the bucket is empty
        # (no failed cases) — omit the point rather than plotting a false 0.
        pts = [(h, acc[name][str(h)]) for h in hops if str(h) in acc[name]]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, **style)

    posthoc_hops = [
        h for h in hops
        if any(str(h) in acc.get(n, {}) for n in _POST_HOC_NAMES)
    ]
    if show_best_posthoc and posthoc_hops:
        best_ys = _compute_best_posthoc(acc, posthoc_hops)
        ax.plot(posthoc_hops, best_ys, **_BEST_POSTHOC_STYLE)

    ax.set_xlabel("Injection depth (hop)", fontsize=11)
    ax.set_ylabel("Root-cause attribution accuracy", fontsize=11)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(hops)
    n_failed = [nfail.get(str(h), 0) for h in hops]
    rec_line = [round(rec.get(str(h), 0.0), 2) for h in hops]
    delta = _pa_vs_llm_delta(acc, hops)
    ax.set_title(
        f"Attribution identifiability — {prov} · {ds}\n"
        f"(n_failed={n_failed}; recovery={rec_line})\n"
        f"PA−LLM-judge Δ by depth: {delta}",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Name the figure after the result file's stem so distinct runs on the same
    # (provider, dataset) — bm25 vs dense_b4 vs bm25_corruption, or old-format
    # files — never overwrite each other's figures.
    if source_path:
        stem = os.path.splitext(os.path.basename(source_path))[0]
        stem = stem.replace("identifiability_", "", 1)
    else:
        stem = f"{_safe(prov)}_{ds}"
    base = os.path.join(output_dir, f"identifiability_curve_{stem}")
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
    p.add_argument("--no-best-posthoc", action="store_true",
                   help="Omit the best-post-hoc envelope line")
    p.add_argument("--no-strict-rescore", action="store_true",
                   help="Plot the stored (loose) accuracies instead of strict "
                        "actual-depth rescoring from raw diagnoses")
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
        plot_one(result, args.output_dir, show_best_posthoc=not args.no_best_posthoc,
                 source_path=fp, strict_rescore=not args.no_strict_rescore)
        n += 1
    print(f"\nGenerated {n} identifiability figure(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
