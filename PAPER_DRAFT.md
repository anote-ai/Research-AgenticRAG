# Failure Attribution in Agentic RAG: A Diagnostic Benchmark

**Status: DRAFT SKELETON -- experiments not yet run.** This document scaffolds the paper structure described in `DESIGN_DOC.md` (target venue: EMNLP ORACLE 2026). All numeric values below are either (a) carried over verbatim from the design doc's *expected/hypothesized* results, explicitly marked as such, or (b) left as `TBD` pending an actual experiment run. Nothing in this file should be cited as a measured result.

## Abstract (draft)

Agentic retrieval-augmented generation (RAG) pipelines chain retrieval, tool calls, and generation across multiple reasoning hops. When the final answer is wrong, existing evaluation methodology can only say *that* the pipeline failed, not *where* the failure originated or whether it was amplified, absorbed, or self-corrected along the way. We introduce a controlled fault-injection methodology for agentic RAG: five injector types (`empty_retrieval`, `irrelevant_docs`, `no_tool_calls`, `empty_answer`, `hallucinated_answer`) applied at chosen pipeline hops, paired with a `DiagnosticBenchmark` that attributes each failure to its root-cause stage. We define root-cause accuracy, failure amplification curves, and recovery rate as metrics that are invisible to end-to-end accuracy alone, and report results on HotpotQA and MuSiQue with BM25 and token-overlap retrievers. *(Results section pending -- see Section 5.)*

## 1. Introduction

See `DESIGN_DOC.md` "Problem Statement & Novelty" and "Why This Matters" sections -- reproduced and expanded here in the full paper draft, not duplicated in this skeleton.

## 2. Related Work

Draft bullets (full related-work prose and `related_work.tex`/`related_work.bib` referenced in `DESIGN_DOC.md` were not found in the current repository tree as of this audit -- see the audit issue for follow-up):

- **Self-RAG** (Asai et al., 2023) -- self-reflective RAG with per-token retrieval decisions; no failure-propagation analysis.
- **FLARE** (Jiang et al., 2023) -- forward-looking active retrieval, uncertainty-triggered; no injection methodology.
- **ReAct** (Yao et al., 2022) -- tool-use agent loop; evaluates task completion, not stage-level attribution.
- **IRCoT** (Trivedi et al., 2022) -- iterative retrieval with CoT; multi-hop but no controlled fault study.

## 3. Method

### 3.1 Pipeline trace model
Implemented in `src/agenticrag/core.py`: `PipelineTrace`, `FailureRecord`, `AgenticRAGPipeline`, `DiagnosticBenchmark`.

### 3.2 Failure injection
Implemented in `src/agenticrag/injection.py`: `FailureInjector` with five methods, plus `injection_sensitivity` as a validation metric.

### 3.3 Propagation graph
Implemented in `src/agenticrag/propagation.py` (DESIGN_DOC.md lists this as "Phase 2 -- In progress"; code review during this audit found the module already present and covered by `tests/test_propagation.py`, so this phase appears further along than the design doc's status table reflects).

### 3.4 Ablation runner
Implemented in `src/agenticrag/experiment.py`: `run_ablation` sweeps the (injection method x hop) grid and produces `AblationResult` with `sensitivity_table()` and `metrics_table()`.

### 3.5 Metrics
Implemented in `src/agenticrag/evaluate.py`: root-cause accuracy, failure amplification rate, recovery rate, severity-weighted failure rate, stage attribution rate, propagation rate, end-to-end accuracy, multi-hop accuracy, retrieval loop efficiency.

## 4. Experimental Setup

- **Datasets**: HotpotQA, MuSiQue (via `src/agenticrag/datasets.py`, with built-in small fallback sets when the `datasets` library/network is unavailable).
- **Retrievers**: `BM25Retriever`, `TokenOverlapRetriever` (`src/agenticrag/retrievers.py`).
- **Scripts**: `scripts/run_baseline.py` (Phase 0), `scripts/run_ablation.py` (Experiments 1-5), `scripts/plot_amplification.py` (Experiment 3 figures).

**To reproduce once run:**
```
python scripts/run_baseline.py --all-conditions --max-samples 50
python scripts/run_ablation.py
python scripts/plot_amplification.py
```
Results will be written to `results/*.json`; see `results/README.md`.

## 5. Results

### 5.1 Phase 0 baseline
| Metric | Value |
|---|---|
| End-to-end accuracy (BM25, HotpotQA) | TBD (projected, pending full experiment run: ~0.62 per DESIGN_DOC.md) |
| Retrieval-stage failure share | TBD (projected: ~60% of failures) |
| Propagation rate | TBD (projected: ~50%) |

### 5.2 Experiment 1 -- Injection sensitivity
| Injection Method | Sensitivity |
|---|---|
| inject_empty_retrieval | TBD (projected >= 0.95) |
| inject_irrelevant_docs | TBD (projected >= 0.80) |
| inject_no_tool_calls | TBD (projected >= 0.90) |
| inject_empty_answer | TBD (projected >= 0.98) |
| inject_hallucinated_answer | TBD (projected >= 0.85) |

### 5.3 Experiment 2 -- Root-cause accuracy
TBD -- see DESIGN_DOC.md "Expected results" table for the per-cell projections (all marked projected, pending full experiment run).

### 5.4 Experiment 3 -- Failure amplification curves
TBD -- requires `scripts/plot_amplification.py` to be run against real ablation output.

### 5.5 Experiment 4 -- Recovery rate by failure type
TBD -- projected values in DESIGN_DOC.md (~0.28-0.35 for retrieval-stage failures, ~0.0 for answer-stage failures) are hypotheses, not measurements.

### 5.6 Experiment 5 -- Retriever x dataset benchmark
TBD -- requires running all 4 conditions (`--all-conditions` flag in `run_baseline.py`).

## 6. Discussion / Limitations (draft)

- Injection methodology produces "clean" faults (e.g., fully empty retrieval); DESIGN_DOC.md's own risk register flags that real-world failures are noisier (partial retrieval, partial hallucination) and proposes noisy variants as future work.
- Scale: design doc targets 500 baseline traces and 200 traces per injection sensitivity check -- modest by NLP-benchmark standards; results should be framed as a methodology contribution validated at small-to-moderate scale, with full-split HotpotQA/MuSiQue runs as a stated mitigation for generalization risk.
- `related_work.tex`/`related_work.bib` referenced in DESIGN_DOC.md's file structure were not present in the repository at audit time; full literature positioning still needs drafting before submission.

## 7. Reproducibility checklist

- [x] Core pipeline, injection, ablation, and metric code implemented and unit-tested (`tests/`).
- [x] CLI scripts exist for baseline and ablation runs.
- [ ] Experiments actually executed and results saved under `results/`.
- [ ] Figures generated by `scripts/plot_amplification.py`.
- [ ] Related work prose / bibliography file present in repo.
- [ ] Paper expanded from this skeleton into full prose with real numbers.
