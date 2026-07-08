#!/usr/bin/env python3
"""
Audit all existing identifiability result files for depth-bucket contamination.

For each non-empty result file, rescores with require_actual_depth=True and
compares against the loose (no filtering) rescore. Prints a table showing how
many cases were filtered per depth bucket, and writes results/rescore_audit.json.
"""
import json
import glob
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agenticrag.evaluate import rescore_identifiability


def audit_file(path: str) -> dict | None:
    with open(path) as f:
        result = json.load(f)
    raw = result.get("raw_by_depth", {})
    if not raw:
        return None

    loose = rescore_identifiability(result, criterion="hop")
    strict = rescore_identifiability(result, criterion="hop", require_actual_depth=True)

    diagnosers = list(result.get("diagnosers", loose.keys()))
    depth_audit = {}
    for depth_str, entry in raw.items():
        depth = int(depth_str)
        truth = entry.get("truth", [])
        total = len(truth)
        kept = sum(1 for t in truth if len(t) >= 2 and int(t[1]) == depth)
        filtered = total - kept
        depth_audit[depth] = {
            "total_failed": total,
            "kept_actual_depth": kept,
            "filtered_clamped": filtered,
            "pct_contaminated": round(filtered / total * 100, 1) if total else 0,
        }
        for d in diagnosers:
            loose_acc = loose.get(d, {}).get(depth, None)
            strict_acc = strict.get(d, {}).get(depth, None)
            depth_audit[depth][f"{d}_loose"] = loose_acc
            depth_audit[depth][f"{d}_strict"] = strict_acc

    return {
        "file": path,
        "provider": result.get("provider"),
        "dataset": result.get("dataset"),
        "retriever": result.get("retriever"),
        "strict_depth_flag": result.get("strict_depth", "MISSING"),
        "depth_audit": depth_audit,
    }


def main():
    files = sorted(glob.glob("results/identifiability_*.json"))
    audits = []
    for path in files:
        a = audit_file(path)
        if a is None:
            print(f"SKIP (no data): {path}")
            continue
        audits.append(a)
        print(f"\n=== {path} ===")
        print(f"  provider={a['provider']}  dataset={a['dataset']}  retriever={a['retriever']}")
        print(f"  strict_depth flag in file: {a['strict_depth_flag']}")
        for depth, info in sorted(a["depth_audit"].items()):
            pct = info["pct_contaminated"]
            flag = " *** CONTAMINATED" if pct > 0 else ""
            print(
                f"  hop{depth}: {info['total_failed']} failed cases, "
                f"{info['filtered_clamped']} clamped ({pct}%){flag}"
            )
            for key in sorted(info):
                if key.endswith("_loose") or key.endswith("_strict"):
                    print(f"    {key}: {info[key]}")

    out = "results/rescore_audit.json"
    with open(out, "w") as f:
        json.dump(audits, f, indent=2)
    print(f"\nAudit written to {out}")


if __name__ == "__main__":
    main()
