# results/

This directory holds the JSON/figure outputs of the experiment scripts in `scripts/`.

**As of this audit (added by the research-readiness review on branch `claude/relaxed-albattani-FA8RR`), this directory is empty of actual experiment output.** The experiment code (`scripts/run_baseline.py`, `scripts/run_ablation.py`, `scripts/plot_amplification.py`) is implemented and appears runnable, but no one has yet executed it and committed the resulting JSON/PNG artifacts here.

## How to populate this directory

```bash
pip install -r requirements.txt

# Phase 0: baseline failure rates
python scripts/run_baseline.py --all-conditions --max-samples 50
# -> results/baseline_{retriever}_{dataset}.json, results/baseline_all.json

# Experiments 1-5: injection sensitivity + ablation grid
python scripts/run_ablation.py
# -> results/ablation_*.json (see script --help for exact naming)

# Experiment 3: amplification curve figures
python scripts/plot_amplification.py
# -> results/*.png
```

## Why this matters

The DESIGN_DOC.md "Expected Results Summary" table and the README.md "Benchmark Stats Template" table contain **projected/hypothesized numbers**, not measured ones. Until real output lands in this directory, any number quoted in the README, blog post, or paper draft outside of this disclaimer should be treated as a placeholder, not a finding. See the linked GitHub issue ("Research Readiness Audit") for the full list of numbers that could not be traced to an actual run at audit time.
