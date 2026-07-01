# Handoff: Eval Worker — Run Evals + Generate Final Summaries

Starting point: `main` after commit `0a93450`.
All tests pass (278/278). Read this in full before starting any API runs.

---

## What Was Done in the Last Session (Context)

Five improvements were implemented and committed:

1. **SuffixRegenerationDiagnoser** (`src/agenticrag/diagnosers.py`): a new 5th
   diagnoser (`name = "suffix_regen"`) that repairs hop h, preserves prefix
   hops 1..h-1, and regenerates the suffix via `agent.resume_from_hops(start_hop=h+1)`.
   This is the "structural counterfactual" the Discussion section called for.
   It is NOT yet wired into `scripts/run_identifiability.py` — that's your job.

2. **Bootstrap CIs + softer metrics** (`src/agenticrag/evaluate.py`): `bootstrap_localization_ci`,
   `ancestor_hit_rate`, `slice_identifiability`.

3. **FRAMES corpus validation** (`datasets.py`, `run_identifiability.py`): warns
   when FRAMES is loaded without `--frames-fetch-passages`; persists
   `corpus_quality` stats in result JSON.

4. **`scripts/summarize_identifiability.py`** (new): reads existing result JSONs
   and outputs `best_posthoc_by_depth`, `pa_vs_llm_judge_by_depth`,
   `pa_vs_best_posthoc_by_depth`, bootstrap CIs, and per-intervention-type
   sliced tables — NO LLM calls needed.

5. **Plot update**: LLM-judge is now a solid line equal in weight to Doctor-RAG;
   best-posthoc envelope added; title shows PA−LLM-judge Δ.

---

## Step 0: Check Which Existing Results Have Metadata

Before running anything expensive, check whether existing result JSONs have
`raw_by_depth` with `metadata` entries (older runs pre-dating the metadata
feature will have empty slices).

```bash
python3 - <<'EOF'
import glob, json
for fp in sorted(glob.glob("results/identifiability_*.json")):
    r = json.load(open(fp))
    raw = r.get("raw_by_depth", {})
    has_meta = any(entry.get("metadata") for entry in raw.values())
    first_depth = next(iter(raw.values()), {}) if raw else {}
    n_meta = len(first_depth.get("metadata", []))
    print(f"{fp.split('/')[-1]:60s}  raw_depths={len(raw)}  has_meta={has_meta}  n_meta_d1={n_meta}")
EOF
```

Files **without** metadata will produce empty slice tables in `summarize_identifiability.py`.
Those need to be regenerated (or you can accept point-estimate-only summaries for older runs).

---

## Step 1: Wire SuffixRegenerationDiagnoser into the Run Script

Edit `scripts/run_identifiability.py` — add `suffix_regen` to the diagnosers dict.
Find the block around line 148:

```python
diagnosers = {
    "rule_based": RuleBasedDiagnoser(),
    "doctor_rag": DoctorRAGDiagnoser(),
    "llm_judge": LLMJudgeDiagnoser(provider=judge_provider),
    "propagation_aware": PropagationAwareDiagnoser(agent, max_probes=args.propagation_budget),
}
```

Change to:

```python
from agenticrag import SuffixRegenerationDiagnoser   # add at top of file

diagnosers = {
    "rule_based": RuleBasedDiagnoser(),
    "doctor_rag": DoctorRAGDiagnoser(),
    "llm_judge": LLMJudgeDiagnoser(provider=judge_provider),
    "propagation_aware": PropagationAwareDiagnoser(agent, max_probes=args.propagation_budget),
    "suffix_regen": SuffixRegenerationDiagnoser(agent, max_probes=args.propagation_budget),
}
```

`SuffixRegenerationDiagnoser` calls `resume_from_hops` (not `force_answer`), so
its token cost is similar to `PropagationAwareDiagnoser` but potentially higher
because the suffix re-executes freely rather than stopping at one step.
Set `--propagation-budget 3` or `4` to cap re-execution per sample.

**Important**: Because suffix_regen is expensive, run it on the same
`--tag` so output goes to a new file rather than overwriting existing ones.
Suggested tag: `--tag _sr_b4` for "suffix regen, budget 4".

---

## Step 2: Verify FRAMES Corpus Quality

Before any real FRAMES run, check the existing FRAMES result files:

```bash
python3 - <<'EOF'
import glob, json
for fp in sorted(glob.glob("results/identifiability_*frames*.json")):
    r = json.load(open(fp))
    cq = r.get("corpus_quality", {})
    print(f"{fp.split('/')[-1]}")
    if cq:
        print(f"  link_only_fraction={cq.get('link_only_fraction', '?'):.2f}  "
              f"mean_passage_len={cq.get('mean_passage_length', 0):.0f}ch  "
              f"answer_recall={cq.get('mean_answer_recall_in_corpus', 0):.2f}")
    else:
        print("  NO corpus_quality field (pre-validation run)")
EOF
```

Any FRAMES file with `corpus_quality` absent OR `link_only_fraction > 0.5` should
be re-run with `--frames-fetch-passages` before it is cited in the paper.
The Wikipedia cache lives in `.agenticrag_cache/frames_wiki.json` and is reused
across runs — the first fetch is slow but subsequent runs are fast.

---

## Step 3: Prioritized Eval Runs

Run these in order. Each is resumable — pass `--resume` after any interruption.

### 3a. Complete the partial dense Claude runs (cheapest fix first)

Dense Claude HotpotQA is hop-1-only; dense Claude MuSiQue is hops 1–2 only.
These need hops 2–3 and 3 respectively.

```bash
# Dense Claude HotpotQA — add hops 2 and 3
python scripts/run_identifiability.py \
  --dataset hotpotqa --provider claude --model claude-haiku-4-5 \
  --retriever dense --max-samples 60 --hops 1 2 3 \
  --propagation-budget 4 --tag dense_b4 --resume

# Dense Claude MuSiQue — add hop 3
python scripts/run_identifiability.py \
  --dataset musique --provider claude --model claude-haiku-4-5 \
  --retriever dense --max-samples 60 --hops 1 2 3 \
  --propagation-budget 4 --tag dense_b4 --resume
```

### 3b. Re-run FRAMES with passage fetching (if existing runs are link-only)

```bash
python scripts/run_identifiability.py \
  --dataset frames --provider claude --model claude-haiku-4-5 \
  --retriever bm25 --max-samples 60 --hops 1 2 3 \
  --frames-fetch-passages --frames-max-passages 40 \
  --propagation-budget 4 --tag passages --resume

python scripts/run_identifiability.py \
  --dataset frames --provider openai --model gpt-4o-mini \
  --retriever dense --max-samples 60 --hops 1 2 3 \
  --frames-fetch-passages --frames-max-passages 40 \
  --propagation-budget 4 --tag passages_dense --resume
```

### 3c. SuffixRegenerationDiagnoser runs (add after Step 1 edit is done)

Run suffix_regen on the conditions where PA most clearly struggles:
FRAMES (big gap between PA and best post-hoc) and deep MuSiQue.

```bash
# Claude FRAMES BM25 + suffix regen
python scripts/run_identifiability.py \
  --dataset frames --provider claude --model claude-haiku-4-5 \
  --retriever bm25 --max-samples 60 --hops 1 2 3 \
  --frames-fetch-passages --propagation-budget 3 \
  --tag sr_b3 --resume

# OpenAI MuSiQue dense + suffix regen
python scripts/run_identifiability.py \
  --dataset musique --provider openai --model gpt-4o-mini \
  --retriever dense --max-samples 60 --hops 1 2 3 \
  --propagation-budget 3 --tag sr_b3 --resume
```

---

## Step 4: Run summarize_identifiability.py After Each Batch

After every batch of runs, regenerate the summary. This is fast (no LLM calls):

```bash
python scripts/summarize_identifiability.py \
  --write-md results/RESULTS_SUMMARY.md \
  --write-json results/summary_all.json \
  --criterion hop
```

To check stage-criterion rescoring (often more favorable to PA):
```bash
python scripts/summarize_identifiability.py \
  --criterion stage \
  --write-md results/RESULTS_SUMMARY_stage.md
```

The key numbers to extract for the paper LaTeX writer:
- `best_posthoc_by_depth` per condition (to update Table 3)
- `pa_vs_llm_judge_by_depth` per condition (new column needed in results tables)
- `pa_vs_best_posthoc_by_depth` per condition (headline C3 verdict)
- Slice tables by `intervention_method` and `injected_failure_type` (for discussion)
- Bootstrap CI widths for CRAG (should be wide — sanity check)
- `suffix_regen` accuracy vs `propagation_aware` at depth ≥ 2 (new C3 result)

---

## Step 5: Regenerate Figures

```bash
python scripts/plot_identifiability.py
```

The updated plot now shows:
- LLM-judge as a solid line (equal visual weight to Doctor-RAG)
- A "best post-hoc" envelope (dashed gray)
- PA−LLM-judge Δ in the plot title

For suffix_regen, `plot_identifiability.py` will pick up `suffix_regen` automatically
because it iterates over all keys in `acc`. The style will default to matplotlib's
next color since it's not in `_STYLES`. You may want to add it:

```python
# In scripts/plot_identifiability.py, add to _STYLES:
"suffix_regen": dict(marker="D", linestyle="-.", color="#2ca02c", label="Suffix-regen (ours)"),
```

---

## Step 6: Sanity Checks Before Handing Off to the LaTeX Writer

Run these checks and fix anything that fails before sending numbers to the paper:

```bash
# 1. Confirm all target runs cover hops 1-2-3
python3 - <<'EOF'
import glob, json
for fp in sorted(glob.glob("results/identifiability_*.json")):
    r = json.load(open(fp))
    if r.get("provider", "").startswith("mock"):
        continue
    have = sorted(r.get("raw_by_depth", {}).keys(), key=int)
    missing = [h for h in [1,2,3] if str(h) not in r.get("raw_by_depth", {})]
    status = "COMPLETE" if not missing else f"MISSING hops {missing}"
    print(f"{fp.split('/')[-1]:60s}  {status}")
EOF

# 2. Confirm FRAMES runs have real passages
python3 - <<'EOF'
import glob, json
for fp in sorted(glob.glob("results/identifiability_*frames*.json")):
    r = json.load(open(fp))
    cq = r.get("corpus_quality", {})
    lof = cq.get("link_only_fraction", 1.0)
    flag = "OK" if lof < 0.1 else f"WARN link_only={lof:.0%}"
    print(f"{fp.split('/')[-1]:60s}  {flag}")
EOF

# 3. Check n_failed is non-trivial
python3 - <<'EOF'
import glob, json
for fp in sorted(glob.glob("results/identifiability_*.json")):
    r = json.load(open(fp))
    if r.get("provider","").startswith("mock"): continue
    nf = r.get("n_failed_by_depth", {})
    low = [f"h{k}:n={v}" for k,v in nf.items() if int(v) < 10]
    if low:
        print(f"LOW-N WARNING {fp.split('/')[-1]:40s}  {low}")
EOF
```

---

## Known Gaps and Caveats to Communicate to the LaTeX Writer

1. **CRAG always has low n** (small dataset, few failed traces). CIs will be wide
   by design. The paper should flag CRAG as a low-confidence condition.

2. **Dense Claude runs are currently partial** (HotpotQA hop 1 only, MuSiQue
   hops 1–2 only). Do not cite them as complete in Tables until Step 3a is done.

3. **Suffix regen is new** — there are no existing result files with it. It will
   only appear in tables after Step 3c runs complete. If time/budget runs out,
   the paper can mention it as a contribution under evaluation and report
   whatever partial numbers exist.

4. **stage-criterion results are consistently more favorable to PA** than exact-hop.
   The rescoring is free (uses `rescore_identifiability` on existing raw_by_depth).
   Run `summarize_identifiability.py --criterion stage` and include those numbers
   in the paper — they don't require any new API calls.

5. **`bootstrap_localization_ci` uses the Diagnosis objects reconstructed from
   raw_by_depth**, so CIs are only available for result files that have
   `raw_by_depth`. Files from before the `raw_by_depth` feature was added will
   have missing CIs in the summary — mark those tables with `†` in the paper.

---

## Output Package for the LaTeX Writer

Once evals are done, hand off:
- `results/RESULTS_SUMMARY.md` (auto-generated by `summarize_identifiability.py`)
- `results/summary_all.json` (machine-readable for table extraction)
- `results/RESULTS_SUMMARY_stage.md` (stage-criterion rescoring)
- Regenerated figures in `figures/` (especially updated `identifiability_curve_*.pdf`)
- A short note on which FRAMES runs used `--frames-fetch-passages` and which did not
- `suffix_regen` vs `propagation_aware` Δ at depth ≥ 2, per condition
