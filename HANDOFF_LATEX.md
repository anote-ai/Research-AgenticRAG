# Handoff: LaTeX Sections Update (Non-Results, Non-Conclusions)

Starting point: `main` after commit `0a93450`.
Paper file: `paper/main.tex`.
Do NOT touch §7 Results or §9 Conclusion — those sections depend on freshly
re-run eval numbers that are not yet final.

---

## What Changed in This Session (Code Side)

The following features were added and need to be reflected in the paper's method,
metrics, and appendix sections:

### 1. SuffixRegenerationDiagnoser (new, `src/agenticrag/diagnosers.py`)
A fifth diagnoser called `suffix_regen`. Unlike the existing
`PropagationAwareDiagnoser` (which repairs hop `h` and forces an answer from the
frozen remaining hops), this one:
- Preserves hops 1..h-1 unchanged
- Repairs hop `h` by clean retrieval
- **Regenerates the suffix from hop h+1** via `agent.resume_from_hops(start_hop=h+1)`

This addresses exactly the gap identified in the current Discussion
(§9 "Implications for repair"): "structural counterfactuals that regenerate
downstream hops after each repair." The suffix regen diagnoser IS that fix.
`Diagnosis.probe_type` records which path succeeded: `"suffix_regen"` or `"none"`.

### 2. Bootstrap CIs + Softer Localization Metrics (`src/agenticrag/evaluate.py`)
New metrics added:
- `bootstrap_localization_ci(diagnoses, truths, criterion, hop_tolerance, n_boot=1000, seed=42)` → `{mean, ci_low, ci_high, n}`. Seeded, deterministic. Small-n conditions (CRAG) naturally yield wide CIs.
- `ancestor_hit(diag, truth)` → bool. True when `pred_hop ≤ true_hop` AND stages match. Captures partial causal credit for propagation paths.
- `ancestor_hit_rate(diagnoses, truths)` → float.
- `slice_identifiability(result, slice_by, criterion, hop_tolerance)` → sliced accuracy dict, grouping cases by metadata field (intervention method, failure type, dataset) without re-running LLM calls.

### 3. FRAMES Corpus Validation (`src/agenticrag/datasets.py`, `scripts/run_identifiability.py`)
- `frames_corpus_stats(samples)` reports: mean_docs/sample, n_empty_corpus, mean_passage_length, mean_answer_recall_in_corpus, link_only_count/fraction.
- `run_identifiability.py` now warns when FRAMES appears to use bare Wikipedia links (link_only_fraction > 50%) and persists `corpus_quality` stats in the result JSON.
- New flag: `--allow-link-corpus` suppresses the warning for intentional link-only runs.

### 4. Summary Script + Plot Updates
- New script: `scripts/summarize_identifiability.py` computes `best_posthoc_by_depth`, `pa_vs_llm_judge_by_depth`, `pa_vs_best_posthoc_by_depth`, bootstrap CIs, and per-slice tables from existing JSON files without re-running LLM calls.
- `scripts/plot_identifiability.py`: LLM-judge promoted to a solid line (equal visual weight to Doctor-RAG). Best-posthoc envelope added. Title now shows `PA−LLM-judge Δ` per depth.

---

## Sections to Update in `paper/main.tex`

### §5 Diagnosers (currently lists 4, now 5)

After the `\pa` paragraph, add a fifth paragraph for `SuffixRegenerationDiagnoser`. Suggested wording:

```latex
\paragraph{\sr.}
A suffix-regeneration variant of \pa.  For each candidate hop $h$ in causal
order, it preserves hops $1{,}\ldots,h{-}1$ from the original trace, repairs hop
$h$ by re-retrieving from the corpus, and then \emph{resumes the agent from hop
$h{+}1$} — regenerating the downstream suffix in the repaired context rather than
forcing an answer from frozen later hops.  This recovers root causes whose
downstream hops were themselves generated from the corrupted prefix, the failure
mode identified in~\S\ref{sec:discussion} as the main gap of local-only repair.
\sr{} records the localization path in a \texttt{probe\_type} field
(\texttt{"suffix\_regen"} or \texttt{"none"} for the coverage-heuristic
fallback).
```

Add a macro at the top of the file alongside the existing ones:
```latex
\newcommand{\sr}{\textnormal{\textsc{Suf-Regen}}}
```

### §6 Metrics — Paragraph expansions

**After the `\paragraph{Attribution identifiability.}` block**, add:

```latex
\paragraph{Softer localization metrics.}
Exact-hop accuracy is brittle when failures are causally entangled across hops.
We additionally report (i) \emph{stage accuracy}: whether the predicted stage
matches the injected stage; (ii) \emph{hop-tolerance accuracy}: whether
$|\hat{h} - h| \leq 1$; and (iii) \emph{ancestor-hit rate}: the fraction of
diagnoses where $\hat{h} \leq h$ and $\hat{s} = s$, capturing partial causal
credit when an early-hop prediction is a valid causal ancestor of the true fault.
Mean absolute hop error is also reported.

\paragraph{Bootstrap confidence intervals.}
All accuracy figures are accompanied by bootstrap 95\% confidence intervals
($B{=}1{,}000$ resamples, fixed seed).  Low-$n$ conditions (notably CRAG, which
has $n < 10$ failed traces per depth in many configurations) are flagged; their
wide CIs are the correct signal rather than a formatting artifact.
```

### §6 Experimental Setup — Add FRAMES corpus paragraph

After the `\paragraph{Datasets.}` block, expand with:

```latex
FRAMES is loaded with \texttt{--frames-fetch-passages} for all headline runs,
fetching Wikipedia passage text via the MediaWiki API and caching it under
\texttt{.agenticrag\_cache/}.  Runs without passage fetching use bare Wikipedia
links as a stand-in corpus; this substantially reduces retrieval fidelity and
is flagged by the \texttt{corpus\_quality.link\_only\_fraction} field persisted
in each result JSON.  All result files discussed in §\ref{sec:results} have
\texttt{link\_only\_fraction} $< 0.1$ unless noted.
```

*(Check with the eval worker whether this is actually true for the re-run files
before including it. If FRAMES runs were not re-run with `--frames-fetch-passages`,
soften this to say "should be" or drop the claim.)*

### §8 Discussion — "Implications for repair" paragraph

The existing paragraph already mentions "structural counterfactuals that regenerate
downstream hops after each repair" as a candidate next step. Now that
`SuffixRegenerationDiagnoser` exists and has been run, update this to:

```latex
\paragraph{Implications for repair.}
The negative C3 result is useful.  It suggests that deployable repair should not
only ask ``which hop flips if repaired?'' but should model paths of downstream
dependence.  Our \sr{} diagnoser (§\ref{sec:diagnosers}) implements one such
path: it regenerates the downstream suffix after repairing hop $h$, rather than
forcing an answer from the frozen corrupted later hops.  Preliminary results
suggest \sr{} recovers root causes that \pa{} misses when later hops were
generated from the corrupted prefix.  Other candidate improvements include
multi-hop repair sets, explicit query-drift detection, and confidence-calibrated
abstention when multiple hops are causally entangled.
```

*(If full suffix_regen eval numbers are available by the time this lands, replace
"preliminary results suggest" with the actual Δ over PA.)*

### §11 Limitations — FRAMES and CI notes

Add at the end of the paragraph about corpus construction:

```latex
The FRAMES corpus relies on Wikipedia passage fetching; runs without
\texttt{--frames-fetch-passages} use bare link text as the retrieval corpus,
which inflates both recovery and identifiability scores relative to real
passage retrieval.  This is now detected automatically via
\texttt{corpus\_quality.link\_only\_fraction} and warned at run time.
Exact-hop accuracy CIs reported in this paper use bootstrap resampling ($B =
1{,}000$, seeded); low-$n$ conditions (CRAG, $n < 10$) have wide CIs that
should be read cautiously.
```

### §A Implementation Notes (Appendix)

Update the bullet list to add:
```latex
  \item \texttt{src/agenticrag/diagnosers.py}: rule-based, coverage-gated,
  LLM-judge, propagation-aware, and suffix-regeneration diagnosers.
  \item \texttt{src/agenticrag/datasets.py}: dataset adapters plus
  \texttt{frames\_corpus\_stats} for FRAMES corpus quality validation.
  \item \texttt{scripts/summarize\_identifiability.py}: post-hoc summary of
  result JSONs, computing best-post-hoc baselines, PA vs.\ LLM-judge deltas,
  bootstrap CIs, and per-intervention-type slices.
```

### §B Reproduction Commands (Appendix)

Add after the existing block:
```latex
# Generate summary tables (best post-hoc, PA vs LLM-judge, slices, CIs):
python scripts/summarize_identifiability.py \
  --write-md results/RESULTS_SUMMARY.md --criterion hop

# Plots now show LLM-judge as primary comparator + best-posthoc envelope:
python scripts/plot_identifiability.py

# Run suffix-regen diagnoser (add to --diagnosers when supported in CLI):
# See HANDOFF_EVAL_WORKER.md for how to include suffix_regen in a run.
```

---

## What NOT to Change

- §7 Results: numbers will change once the eval worker re-runs and regenerates
  `RESULTS_SUMMARY.md`. Do not edit those tables manually.
- §9 Conclusion: depends on whether suffix_regen results are strong enough to
  change the claim in the final paragraph. Wait for eval numbers.
- The abstract: same — numbers will shift slightly once bootstrap CIs and
  suffix_regen conditions are included.
- References: no new citations needed for this round unless you want to cite
  a suffix-regeneration or do-calculus paper.
