#!/usr/bin/env python3
"""Headline experiment: attribution-identifiability vs propagation depth (C2 + C3).

Runs a real (or offline-mock) LLM agent over a dataset, injects live retrieval
faults at each hop depth (the agent re-runs the suffix), and compares diagnosers:
rule-based, Doctor-RAG-style, LLM-as-judge, and the propagation-aware method.
Produces the RCA-vs-depth curve and the cost-per-correct-diagnosis table.

Backbones (``--provider``):
    mock    deterministic, no API key  (default; for offline smoke + the
            heuristic control condition)
    claude  Anthropic (set ANTHROPIC_API_KEY); ``--model claude-opus-4-8`` etc.
    openai  OpenAI (set OPENAI_API_KEY)

Usage:
    python scripts/run_identifiability.py                       # offline smoke (frames fallback)
    python scripts/run_identifiability.py --dataset frames --provider claude --max-samples 100 --hops 1 2 3
    python scripts/run_identifiability.py --provider claude --model claude-sonnet-4-6 --retriever dense
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from _env import load_dotenv

load_dotenv()  # populate os.environ from .env before SDK clients read keys

from agenticrag import (
    BM25Retriever,
    DenseRetriever,
    DoctorRAGDiagnoser,
    LLMAgent,
    LLMJudgeDiagnoser,
    PropagationAwareDiagnoser,
    RuleBasedDiagnoser,
    TokenOverlapRetriever,
    load_dataset,
    make_provider,
    run_identifiability,
)
from agenticrag.datasets import frames_corpus_stats


def _already_complete(out_path: str, hops, expected: dict | None = None) -> bool:
    """True if *out_path* exists and its matching raw_by_depth covers all *hops*."""
    if not os.path.exists(out_path):
        return False
    try:
        with open(out_path) as f:
            prev = json.load(f)
        for key, value in (expected or {}).items():
            if prev.get(key) != value:
                return False
        have = set(prev.get("raw_by_depth", {}).keys())
        return {str(h) for h in hops}.issubset(have)
    except Exception:
        return False


def _make_retriever(name: str):
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever()
    if name in ("token_overlap", "overlap"):
        return TokenOverlapRetriever()
    raise ValueError(f"Unknown retriever '{name}'.")


def main() -> None:
    p = argparse.ArgumentParser(description="Attribution-identifiability headline experiment")
    p.add_argument("--dataset", default="frames",
                   choices=["hotpotqa", "musique", "frames", "crag"])
    p.add_argument("--provider", default="mock", choices=["mock", "claude", "openai"])
    p.add_argument("--model", default=None, help="Backbone model id (provider default if unset)")
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "token_overlap"])
    p.add_argument("--max-samples", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--hops", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--frames-fetch-passages", action="store_true",
                   help="Build a Wikipedia passage corpus for FRAMES (real retrieval)")
    p.add_argument("--frames-max-passages", type=int, default=40,
                   help="Cap passages per question when fetching FRAMES corpus")
    p.add_argument("--allow-link-corpus", action="store_true",
                   help="Suppress the warning when FRAMES is loaded without --frames-fetch-passages")
    p.add_argument("--criterion", default="hop", choices=["hop", "stage", "both"])
    p.add_argument("--hop-tolerance", type=int, default=0)
    p.add_argument("--allow-short-depth-clamp", action="store_true",
                   help="Allow old behavior where requested depths beyond a base trace's hops "
                        "are clamped to the final available hop")
    p.add_argument("--include-base-failures", action="store_true",
                   help="Inject into base traces even when the original answer is already wrong")
    p.add_argument("--propagation-budget", type=int, default=None,
                   help="Cap re-execution probes for the propagation-aware diagnoser")
    p.add_argument("--tag", default="",
                   help="Suffix appended to the output filename (distinguish runs, "
                        "e.g. 'dense_b4' so a dense run doesn't overwrite a bm25 one)")
    p.add_argument("--resume", action="store_true",
                   help="Skip if the output JSON already covers all requested hops")
    p.add_argument("--output-dir", default="results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Construct the provider first (cheap, no network) so we can compute the
    # output path and honour --resume BEFORE any expensive dataset load / fetch.
    provider = make_provider(args.provider, model=args.model)
    tag = f"_{args.tag}" if args.tag else ""
    # Include the sanitized backbone (provider:model) in the filename so distinct
    # models never collide — e.g. gpt-4o-mini vs an Ollama model, both of which
    # run under --provider openai.
    prov_slug = provider.name.replace(":", "_").replace(".", "_").replace("/", "_")
    out_path = os.path.join(
        args.output_dir, f"identifiability_{prov_slug}_{args.dataset}{tag}.json"
    )
    meta = {
        "dataset": args.dataset,
        "provider": provider.name,
        "retriever": args.retriever,
        "criterion": args.criterion,
        "max_samples": args.max_samples,
        "propagation_budget": args.propagation_budget,
        "strict_depth": not args.allow_short_depth_clamp,
        "require_base_correct": not args.include_base_failures,
    }
    if args.resume and _already_complete(out_path, args.hops, meta):
        print(f"[resume] {out_path} already covers hops {args.hops} — skipping.")
        return

    print(f"Loading {args.dataset} ({args.max_samples} samples)...")
    load_kwargs = {}
    if args.dataset == "frames" and args.frames_fetch_passages:
        load_kwargs = {"fetch_passages": True, "max_passages_per_q": args.frames_max_passages}
        print("  fetching FRAMES Wikipedia passages (cached under .agenticrag_cache/)...")
    samples = load_dataset(args.dataset, max_samples=args.max_samples, **load_kwargs)
    n_docs = sum(len(s.supporting_docs) for s in samples)
    print(f"  loaded {len(samples)} samples ({n_docs} passages total)")

    corpus_quality: dict = {}
    if args.dataset == "frames":
        corpus_quality = frames_corpus_stats(samples)
        link_frac = corpus_quality.get("link_only_fraction", 0.0)
        if link_frac > 0.5 and not args.frames_fetch_passages and not args.allow_link_corpus:
            print(
                f"\nWARNING: {link_frac:.0%} of FRAMES samples appear to use bare Wikipedia links "
                "instead of passage text. Results on link-only FRAMES are hard to interpret.\n"
                "Re-run with --frames-fetch-passages for real retrieval, or pass "
                "--allow-link-corpus to suppress this warning.\n"
            )
        print(
            f"  FRAMES corpus stats: mean_docs={corpus_quality['mean_docs_per_sample']:.1f}, "
            f"link_only={corpus_quality['link_only_fraction']:.0%}, "
            f"answer_recall={corpus_quality['mean_answer_recall_in_corpus']:.2f}, "
            f"mean_passage_len={corpus_quality['mean_passage_length']:.0f}ch"
        )
        meta["corpus_quality"] = corpus_quality

    retriever = _make_retriever(args.retriever)
    agent = LLMAgent(provider=provider, retriever=retriever, max_iterations=args.max_iterations)
    print(f"Agent backbone: {provider.name} · retriever: {args.retriever}")

    # Build a separate judge provider so the judge doesn't share the agent's loop.
    judge_provider = make_provider(args.provider, model=args.model)
    diagnosers = {
        "rule_based": RuleBasedDiagnoser(),
        "doctor_rag": DoctorRAGDiagnoser(),
        "llm_judge": LLMJudgeDiagnoser(provider=judge_provider),
        "propagation_aware": PropagationAwareDiagnoser(agent, max_probes=args.propagation_budget),
    }

    print(f"Running identifiability sweep over hops={args.hops} "
          f"(checkpointing to {out_path} after each depth) ...")
    result = run_identifiability(
        agent, samples, diagnosers,
        hops=args.hops, criterion=args.criterion, hop_tolerance=args.hop_tolerance,
        strict_depth=not args.allow_short_depth_clamp,
        require_base_correct=not args.include_base_failures,
        checkpoint_path=out_path, checkpoint_extra=meta, resume=args.resume,
    )

    _print_result(result, args)

    with open(out_path, "w") as f:
        json.dump({**meta, **result.to_dict()}, f, indent=2)
    print(f"\nSaved to {out_path}")


def _print_result(result, args) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Root-cause-attribution accuracy vs injection depth")
        table.add_column("Diagnoser", style="bold")
        for d in result.hops:
            table.add_column(f"hop{d}", justify="right")
        table.add_column("cost/correct@hop1", justify="right")
        for name in result.diagnoser_names:
            row = [name]
            for d in result.hops:
                row.append(f"{result.accuracy[name].get(d, 0.0):.3f}")
            cpc = result.cost[name].get(result.hops[0], {}).get("cost_per_correct", float("inf"))
            row.append("inf" if cpc == float("inf") else f"{cpc:.0f}")
            table.add_row(*row)
        console.print(table)
        console.print(f"[bold]Counterfactual recovery by depth:[/bold] {result.recovery_rate_by_depth}")
        console.print(
            f"[dim]n (total / failed) by depth: "
            f"{result.n_total_by_depth} / {result.n_failed_by_depth}[/dim]"
        )
        console.print(
            f"[dim]eligible / skipped-short / skipped-base-wrong: "
            f"{result.n_eligible_by_depth} / "
            f"{result.n_skipped_short_trace_by_depth} / "
            f"{result.n_skipped_base_incorrect_by_depth}[/dim]"
        )
    except ImportError:
        print("\n=== RCA vs depth ===")
        for name in result.diagnoser_names:
            cells = "  ".join(f"hop{d}={result.accuracy[name].get(d, 0.0):.3f}" for d in result.hops)
            print(f"  {name:20s} {cells}")
        print(f"recovery_by_depth: {result.recovery_rate_by_depth}")
        print(f"n_total/n_failed: {result.n_total_by_depth} / {result.n_failed_by_depth}")
        print(
            "eligible/skipped_short/skipped_base_wrong: "
            f"{result.n_eligible_by_depth} / "
            f"{result.n_skipped_short_trace_by_depth} / "
            f"{result.n_skipped_base_incorrect_by_depth}"
        )


if __name__ == "__main__":
    main()
