#!/usr/bin/env python3
"""Summarize attribution-identifiability results with LLM-judge as main comparator.

Reads results/identifiability_*.json files and for each condition produces:

  - best_posthoc_by_depth: element-wise max of doctor_rag, rule_based, llm_judge
  - pa_vs_llm_judge_by_depth: propagation_aware accuracy minus llm_judge
  - pa_vs_best_posthoc_by_depth: propagation_aware accuracy minus best post-hoc
  - Bootstrap 95% CIs (from raw_by_depth)
  - Sliced accuracy by intervention_method and injected_failure_type

Answers the paper question "Does PA beat LLM-judge?" directly, using LLM-judge
as the serious baseline rather than Doctor-RAG.

Usage:
    python scripts/summarize_identifiability.py
    python scripts/summarize_identifiability.py --glob 'results/identifiability_claude_*.json'
    python scripts/summarize_identifiability.py --criterion stage --hop-tolerance 1
    python scripts/summarize_identifiability.py --write-md results/RESULTS_SUMMARY.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agenticrag.evaluate import bootstrap_localization_ci, slice_identifiability
from agenticrag.core import FailureStage


_POST_HOC_NAMES = ["doctor_rag", "rule_based", "llm_judge"]
_POSTHOC_LABEL = {
    "doctor_rag": "Doctor-RAG",
    "rule_based": "Rule-based",
    "llm_judge": "LLM-judge",
}


# --------------------------------------------------------------------------- #
# Core computations                                                              #
# --------------------------------------------------------------------------- #

def _best_posthoc_by_depth(acc: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Element-wise max over doctor_rag, rule_based, llm_judge per depth."""
    depths = set()
    for name in _POST_HOC_NAMES:
        depths |= set(acc.get(name, {}).keys())
    out: Dict[str, float] = {}
    for d in sorted(depths, key=int):
        vals = [acc.get(name, {}).get(d, 0.0) for name in _POST_HOC_NAMES]
        out[d] = max(vals)
    return out


def _delta_by_depth(
    acc: Dict[str, Dict[str, float]], baseline_name: str, pa_name: str = "propagation_aware"
) -> Dict[str, float]:
    """PA accuracy minus baseline accuracy per depth."""
    pa_acc = acc.get(pa_name, {})
    base_acc = acc.get(baseline_name, {})
    depths = set(pa_acc.keys()) | set(base_acc.keys())
    return {
        d: pa_acc.get(d, 0.0) - base_acc.get(d, 0.0)
        for d in sorted(depths, key=int)
    }


def _ci_table(
    raw: Dict[str, Any],
    names: List[str],
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Bootstrap CIs per diagnoser per depth from raw_by_depth."""
    out: Dict[str, Dict[int, Dict[str, Any]]] = {n: {} for n in names}
    for depth_str, entry in raw.items():
        depth = int(depth_str)
        truth = entry.get("truth", [])
        preds_by_name = entry.get("predictions", {})
        for name in names:
            preds = preds_by_name.get(name, [])
            n_cases = min(len(preds), len(truth))
            if n_cases == 0:
                out[name][depth] = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
                continue
            diag_objs = [
                type("_D", (), {"stage": FailureStage(p[0]), "predicted_hop": int(p[1])})()
                for p in preds[:n_cases]
            ]
            truth_pairs = [(FailureStage(t[0]), int(t[1])) for t in truth[:n_cases]]
            out[name][depth] = bootstrap_localization_ci(
                diag_objs, truth_pairs, criterion=criterion, hop_tolerance=hop_tolerance
            )
    return out


def summarize_one(
    result: Dict[str, Any],
    criterion: str = "hop",
    hop_tolerance: int = 0,
) -> Dict[str, Any]:
    """Build a full summary dict for one result JSON."""
    acc = result.get("accuracy", {})
    raw = result.get("raw_by_depth", {})
    names = result.get("diagnosers", list(acc.keys()))
    hops = result.get("hops", [])

    best_ph = _best_posthoc_by_depth(acc)
    pa_vs_llm = _delta_by_depth(acc, "llm_judge")
    pa_vs_best = {d: acc.get("propagation_aware", {}).get(d, 0.0) - v for d, v in best_ph.items()}

    # Which diagnoser is the best post-hoc at each depth?
    best_ph_name: Dict[str, str] = {}
    depths_acc = {str(h): {n: acc.get(n, {}).get(str(h), 0.0) for n in _POST_HOC_NAMES} for h in hops}
    for d_str, vals in depths_acc.items():
        best_ph_name[d_str] = max(vals, key=lambda n: vals[n])

    ci_table = _ci_table(raw, names, criterion=criterion, hop_tolerance=hop_tolerance) if raw else {}

    slices_by_method = slice_identifiability(
        result, slice_by="intervention_method", criterion=criterion, hop_tolerance=hop_tolerance
    ) if raw else {}
    slices_by_type = slice_identifiability(
        result, slice_by="injected_failure_type", criterion=criterion, hop_tolerance=hop_tolerance
    ) if raw else {}

    # Flag CRAG as low-n
    n_failed = result.get("n_failed_by_depth", {})
    low_n_depths = [d for d, v in n_failed.items() if v < 10]

    return {
        "provider": result.get("provider", "?"),
        "dataset": result.get("dataset", "?"),
        "retriever": result.get("retriever", "?"),
        "criterion": criterion,
        "hop_tolerance": hop_tolerance,
        "hops": hops,
        "diagnosers": names,
        "accuracy_by_diagnoser": {n: acc.get(n, {}) for n in names},
        "best_posthoc_by_depth": best_ph,
        "best_posthoc_name_by_depth": best_ph_name,
        "pa_vs_llm_judge_by_depth": pa_vs_llm,
        "pa_vs_best_posthoc_by_depth": pa_vs_best,
        "bootstrap_ci": {n: {str(k): v for k, v in ci_table.get(n, {}).items()} for n in names},
        "sliced_by_intervention_method": slices_by_method,
        "sliced_by_failure_type": slices_by_type,
        "n_failed_by_depth": n_failed,
        "recovery_rate_by_depth": result.get("recovery_rate_by_depth", {}),
        "low_n_depths": low_n_depths,
    }


# --------------------------------------------------------------------------- #
# Rendering                                                                      #
# --------------------------------------------------------------------------- #

def _fmt(v: float) -> str:
    return f"{v:+.3f}" if v >= 0 else f"{v:.3f}"


def _render_summary(s: Dict[str, Any]) -> str:
    lines = [
        f"## {s['provider']} · {s['dataset']} · retriever={s['retriever']}",
        f"   criterion={s['criterion']}, hop_tolerance={s['hop_tolerance']}",
        "",
    ]

    hops = s["hops"]
    names = s["diagnosers"]
    acc = s["accuracy_by_diagnoser"]
    ci = s["bootstrap_ci"]
    n_failed = s["n_failed_by_depth"]
    low_n = set(s.get("low_n_depths", []))

    # Main accuracy table with CIs
    header = f"{'Diagnoser':<22} " + "  ".join(f"hop{h}" for h in hops)
    lines.append(header)
    lines.append("-" * len(header))
    for name in names:
        row = f"{name:<22}"
        for h in hops:
            d = str(h)
            a = acc.get(name, {}).get(d, 0.0)
            ci_entry = ci.get(name, {}).get(d, {})
            lo = ci_entry.get("ci_low", a)
            hi = ci_entry.get("ci_high", a)
            flag = "†" if d in low_n else " "
            row += f"  {a:.3f}[{lo:.2f},{hi:.2f}]{flag}"
        lines.append(row)

    lines += ["", "† low-n (< 10 failed cases) — CIs are wide by design", ""]

    # Best post-hoc vs PA
    lines.append("PA vs. best post-hoc baseline by depth:")
    bph = s["best_posthoc_by_depth"]
    pa_acc = acc.get("propagation_aware", {})
    pa_vs_bph = s["pa_vs_best_posthoc_by_depth"]
    pa_vs_llm = s["pa_vs_llm_judge_by_depth"]
    bph_name = s["best_posthoc_name_by_depth"]
    for h in hops:
        d = str(h)
        lines.append(
            f"  hop{h}: PA={pa_acc.get(d, 0.0):.3f}  "
            f"best_posthoc={bph.get(d, 0.0):.3f} ({_POSTHOC_LABEL.get(bph_name.get(d, '?'), bph_name.get(d, '?'))})  "
            f"Δ(PA−best)={_fmt(pa_vs_bph.get(d, 0.0))}  "
            f"Δ(PA−LLM-judge)={_fmt(pa_vs_llm.get(d, 0.0))}"
        )

    lines += [""]

    # Sliced by intervention type
    slices = s["sliced_by_failure_type"]
    if slices:
        lines.append("PA accuracy by failure type:")
        pa_slices = {ft: slices[ft].get("propagation_aware", {}) for ft in slices}
        llm_slices = {ft: slices[ft].get("llm_judge", {}) for ft in slices}
        for ft in sorted(slices.keys()):
            pa_vals = "  ".join(f"h{h}={pa_slices[ft].get(str(h), 0.0):.2f}" for h in hops)
            llm_vals = "  ".join(f"h{h}={llm_slices[ft].get(str(h), 0.0):.2f}" for h in hops)
            lines.append(f"  {ft:<30} PA: {pa_vals}  |  LLM-judge: {llm_vals}")
        lines.append("")

    slices_m = s["sliced_by_intervention_method"]
    if slices_m:
        lines.append("PA accuracy by intervention method:")
        for method in sorted(slices_m.keys()):
            pa_m = slices_m[method].get("propagation_aware", {})
            llm_m = slices_m[method].get("llm_judge", {})
            pa_vals = "  ".join(f"h{h}={pa_m.get(str(h), 0.0):.2f}" for h in hops)
            llm_vals = "  ".join(f"h{h}={llm_m.get(str(h), 0.0):.2f}" for h in hops)
            lines.append(f"  {method:<35} PA: {pa_vals}  |  LLM-judge: {llm_vals}")
        lines.append("")

    n_fail_str = "  ".join(f"h{h}={n_failed.get(str(h), 0)}" for h in hops)
    rec = s["recovery_rate_by_depth"]
    rec_str = "  ".join(f"h{h}={rec.get(str(h), 0.0):.2f}" for h in hops)
    lines += [f"n_failed:  {n_fail_str}", f"recovery:  {rec_str}", ""]

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point                                                                    #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Summarize identifiability results")
    p.add_argument("--glob", default="results/identifiability_*.json")
    p.add_argument("--include-mock", action="store_true")
    p.add_argument("--criterion", default="hop", choices=["hop", "stage", "both"])
    p.add_argument("--hop-tolerance", type=int, default=0)
    p.add_argument("--write-md", default=None, help="Write markdown summary to this path")
    p.add_argument("--write-json", default=None, help="Write summary JSON to this path")
    args = p.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        print(f"No result files matched {args.glob!r}. Run scripts/run_identifiability.py first.")
        sys.exit(1)

    summaries: List[Dict[str, Any]] = []
    md_parts: List[str] = ["# Identifiability Results Summary\n"]

    for fp in files:
        with open(fp) as f:
            result = json.load(f)
        if not args.include_mock and str(result.get("provider", "")).startswith("mock"):
            continue
        s = summarize_one(result, criterion=args.criterion, hop_tolerance=args.hop_tolerance)
        summaries.append(s)
        md_parts.append(_render_summary(s))
        print(_render_summary(s))

    if args.write_md:
        os.makedirs(os.path.dirname(args.write_md) or ".", exist_ok=True)
        with open(args.write_md, "w") as f:
            f.write("\n".join(md_parts))
        print(f"\nWrote markdown to {args.write_md}")

    if args.write_json:
        os.makedirs(os.path.dirname(args.write_json) or ".", exist_ok=True)
        with open(args.write_json, "w") as f:
            json.dump(summaries, f, indent=2, default=str)
        print(f"Wrote JSON to {args.write_json}")

    print(f"\nSummarized {len(summaries)} result file(s).")


if __name__ == "__main__":
    main()
