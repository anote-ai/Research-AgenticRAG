# Research Design Document: AgenticRAG — Failure Propagation in Agentic RAG Pipelines

## Vision Statement

Characterize **how failures propagate** in agentic RAG systems: when retrieval goes wrong at hop N, does the agent self-correct, or does the error cascade into tool-call failures and hallucinated answers? This project provides the first empirical framework for injecting controlled faults into multi-hop RAG pipelines, attributing root causes, and measuring propagation severity — enabling practitioners to understand and mitigate failure modes before deploying agentic RAG at scale.

**Target venue**: EMNLP ORACLE 2026 (Workshop on Robust and Reliable LLMs)

---

## Problem Statement & Novelty

Agentic RAG systems chain retrieval, tool calls, and generation across multiple hops. Each stage can fail, and failures do not stay local — a missing document at hop 1 can produce hallucinated answers at generation without any intermediate signal. Existing literature:

1. **Evaluates end-to-end accuracy**, ignoring where in the pipeline the failure originated.
2. **Has no controlled injection methodology** — failures are observed naturalistically, making attribution noisy.
3. **Does not measure propagation** — whether an early-stage failure amplifies, attenuates, or triggers recovery.
4. **Lacks per-stage metrics** — accuracy alone cannot distinguish a retrieval failure from an answer-generation failure.

### Novel Contributions

| Contribution | Description |
|---|---|
| **Controlled failure injection** | `FailureInjector` applies five fault types (empty retrieval, irrelevant docs, no tool calls, empty answer, hallucination) at specific hops, enabling causal rather than correlational analysis |
| **Root-cause accuracy metric** | Fraction of traces where `DiagnosticBenchmark` correctly identifies the earliest failing stage — the first evaluation of diagnostic precision for agentic RAG |
| **Failure amplification curves** | Per-hop failure rates showing whether errors compound or attenuate as hop depth increases |
| **Recovery rate metric** | Fraction of mid-pipeline failures that self-correct by the final hop — quantifies agent robustness |
| **Ablation experiment runner** | `run_ablation` sweeps the (injection method × hop) grid, producing paper-ready benchmark tables |

### Failure Taxonomy

```
FailureStage:
  RETRIEVAL         — empty_retrieval, irrelevant_retrieval, over_retrieval, context_overflow
  TOOL_CALL         — no_tool_calls
  ANSWER_GENERATION — empty_answer, incorrect_answer, hallucination
  NONE              — success (no failure)

Propagation rule:
  A failure is "propagated" if it originated at an early stage but is detected
  at a later stage. DiagnosticBenchmark attributes each failure to its root-cause
  stage, not its observation stage.
```

---

## Research Questions

1. **How often do retrieval failures propagate to answer-generation failures?** — Is the pipeline a shock absorber or an amplifier?
2. **Does injection hop depth predict propagation severity?** — Are hop-1 failures more damaging than hop-3 failures?
3. **Can DiagnosticBenchmark correctly identify the root-cause stage?** — What is its root-cause accuracy vs. each injection type?
4. **Does the pipeline self-correct after mid-pipeline faults?** — What is the recovery rate across failure types?
5. **Do retrieval-stage failures produce measurably different severity distributions than answer-stage failures?** — Can severity alone triage failure type?

---

## Implemented Architecture

### Core Data Model ([src/agenticrag/core.py](src/agenticrag/core.py))

```
PipelineTrace
  ├── query, retrieved_docs, tool_calls, final_answer, reference_answer
  ├── hop_queries: List[str]     — query used at each hop
  ├── hop_docs: List[List[str]]  — docs retrieved at each hop
  └── iterations_used: int

FailureRecord
  ├── stage: FailureStage        — attributed root-cause stage
  ├── failure_type: FailureType  — specific failure sub-type
  ├── propagated: bool           — whether failure crossed stage boundaries
  ├── root_cause: str            — human-readable attribution
  └── severity: float ∈ [0, 1]

AgenticRAGPipeline
  └── run(query, corpus) → PipelineTrace   — multi-hop loop with reformulation

DiagnosticBenchmark
  ├── diagnose_trace(trace, reference) → FailureRecord
  ├── batch_diagnose(traces, references) → List[FailureRecord]
  └── attribute_failures(records) → {by_stage, total_failures, propagation_rate}
```

### Failure Injection ([src/agenticrag/injection.py](src/agenticrag/injection.py))

```
FailureInjector methods:
  inject_empty_retrieval(trace, hop)         — clears docs at hop N and all later hops
  inject_irrelevant_docs(trace, noise, hop)  — replaces docs at hop N; later hops untouched
  inject_no_tool_calls(trace)                — clears all tool calls
  inject_empty_answer(trace)                 — clears final answer only
  inject_hallucinated_answer(trace, text)    — replaces answer with ungrounded fabrication

injection_sensitivity(traces, refs, injector, benchmark, method) → float
  — fraction of injected failures correctly detected (validation metric)
```

### Ablation Runner ([src/agenticrag/experiment.py](src/agenticrag/experiment.py))

```
run_ablation(traces, references, injector, benchmark, methods, hops) → AblationResult
  — sweeps (method × hop) grid; baseline diagnosis + per-cell diagnosis

AblationResult
  ├── baseline_records: List[FailureRecord]
  ├── cells: List[AblationCell]              — one per (method, hop) pair
  ├── sensitivity_table() → Dict[str, float]
  ├── metrics_table()     → Dict[str, Dict[str, float]]   ← paper benchmark table
  └── records_by_hop(method) → Dict[int, records]        ← input to PropagationGraph

AblationCell fields:
  injection_method, hop, injected_stage, records,
  sensitivity, root_cause_accuracy_score, severity_rate, stage_rates
```

### Evaluation Metrics ([src/agenticrag/evaluate.py](src/agenticrag/evaluate.py))

| Metric | Function | What it measures |
|---|---|---|
| Root-cause accuracy | `root_cause_accuracy(records, true_stages)` | Fraction where diagnosed stage = injected stage |
| Failure amplification | `failure_amplification_rate(records_by_hop)` | Failure rate vs. hop depth; rising = amplifying pipeline |
| Recovery rate | `recovery_rate(records_by_hop)` | Fraction of mid-pipeline failures that self-correct |
| Severity rate | `severity_weighted_failure_rate(records)` | Mean severity of non-NONE records |
| Stage attribution | `stage_attribution_rate(records, stage)` | Fraction of failures attributed to a specific stage |
| Propagation rate | `propagation_rate(records)` | Fraction of records with propagated=True |
| End-to-end accuracy | `end_to_end_accuracy(records)` | Fraction of successful traces (stage == NONE) |
| Multi-hop accuracy | `multi_hop_accuracy(traces)` | Success rate among traces requiring >1 hop |
| Loop efficiency | `retrieval_loop_efficiency(traces, max_iter)` | Mean fraction of hop budget unused |

### Retrievers ([src/agenticrag/retrievers.py](src/agenticrag/retrievers.py))

- `BM25Retriever` — rank-bm25-backed sparse retrieval; falls back to pure-Python TF-IDF
- `TokenOverlapRetriever` — Jaccard token overlap; zero-dependency baseline

### Datasets ([src/agenticrag/datasets.py](src/agenticrag/datasets.py))

- `HotpotQAAdapter` — HuggingFace `datasets` integration; built-in 5-example fallback
- `MuSiQueAdapter` — MuSiQue 2-hop subset; built-in fallback

---

## Experimental Design

### Phase 0: Baseline (no injection)
**Goal**: Establish baseline failure rates and stage distribution on HotpotQA + MuSiQue.

**Protocol**: Run `DiagnosticBenchmark.batch_diagnose` on 500 naturally produced traces (AgenticRAGPipeline with BM25Retriever, max 3 hops). Compute end-to-end accuracy, per-stage failure rates, mean severity, propagation rate.

**Expected**: ~30–40% of traces fail; retrieval stage accounts for ~60% of failures; propagation rate ~50% (retrieval failures cascade to answer stage).

---

### Experiment 1: Injection Sensitivity
**Hypothesis**: DiagnosticBenchmark correctly detects >85% of injected faults for all five injection methods.

**Protocol**: Apply `injection_sensitivity` for each of the five FailureInjector methods on 200 clean traces. Report detection rate per method.

**Expected results**:

| Injection Method | Expected Sensitivity |
|---|---|
| inject_empty_retrieval | ≥ 0.95 |
| inject_irrelevant_docs | ≥ 0.80 (harder: answer may still form) |
| inject_no_tool_calls | ≥ 0.90 |
| inject_empty_answer | ≥ 0.98 |
| inject_hallucinated_answer | ≥ 0.85 |

---

### Experiment 2: Root-Cause Accuracy
**Hypothesis**: DiagnosticBenchmark achieves ≥80% root-cause accuracy across all injection types.

**Protocol**: Run `run_ablation` over all 5 methods × hops [1, 2, 3]. For each cell, compare diagnosed stage to injected stage via `root_cause_accuracy`. Report `AblationResult.metrics_table()`.

**Expected results** (paper benchmark table):

| Method | Root-Cause Acc. | Sensitivity | Severity |
|---|---|---|---|
| inject_empty_retrieval@hop1 | 0.91 | 0.95 | 0.88 |
| inject_empty_retrieval@hop2 | 0.85 | 0.89 | 0.82 |
| inject_empty_retrieval@hop3 | 0.78 | 0.81 | 0.74 |
| inject_irrelevant_docs@hop1 | 0.73 | 0.80 | 0.61 |
| inject_no_tool_calls | 0.88 | 0.92 | 0.70 |
| inject_empty_answer | 0.94 | 0.98 | 0.80 |
| inject_hallucinated_answer | 0.82 | 0.87 | 0.65 |

---

### Experiment 3: Failure Amplification Curves
**Hypothesis**: Retrieval failures injected at earlier hops produce higher downstream failure rates (amplification), while later-hop injections are partially absorbed.

**Protocol**: For `inject_empty_retrieval` and `inject_irrelevant_docs`, use `AblationResult.records_by_hop(method)` as input to `failure_amplification_rate`. Plot failure rate vs. hop depth for each retriever (BM25 vs. TokenOverlap).

**Expected result**: Monotonically decreasing failure rate with hop depth for empty retrieval (earlier = more severe). Irrelevant docs show a flatter curve (pipeline compensates on later hops).

---

### Experiment 4: Recovery Rate by Failure Type
**Hypothesis**: Answer-stage failures (empty_answer, hallucination) have near-zero recovery rate; retrieval-stage failures have non-trivial recovery rate (~20–35%) due to iterative reformulation.

**Protocol**: Compute `recovery_rate(records_by_hop)` for hop methods across 3-hop traces. Compare retrieval-stage recovery vs. answer-stage recovery.

**Expected results**:
- `inject_empty_retrieval`: recovery rate ≈ 0.28 (reformulation partially compensates)
- `inject_irrelevant_docs`: recovery rate ≈ 0.35 (later hops retrieve different docs)
- `inject_empty_answer`: recovery rate = 0.0 (answer stage is terminal)
- `inject_hallucinated_answer`: recovery rate = 0.0 (no further correction step)

---

### Experiment 5: Retriever × Dataset Benchmark Table
**Hypothesis**: BM25Retriever has lower failure propagation rates than TokenOverlapRetriever on HotpotQA but similar rates on MuSiQue (which relies on compositional reasoning more than lexical match).

**Protocol**: Run full ablation for both retrievers × both datasets (4 conditions). Report root-cause accuracy and end-to-end accuracy per condition.

**Expected result**: BM25 outperforms TokenOverlap on HotpotQA (+12 pp end-to-end accuracy); gap narrows on MuSiQue (compositional reasoning limits retrieval quality regardless of retriever).

---

## Expected Results Summary

| Metric | Value |
|---|---|
| Baseline end-to-end accuracy (BM25, HotpotQA) | ~0.62 |
| Mean root-cause accuracy across all injection types | ≥0.83 |
| Failure amplification (hop1 vs. hop3, empty retrieval) | +31 pp failure rate at hop1 |
| Recovery rate (retrieval-stage failures) | ~0.30 |
| Recovery rate (answer-stage failures) | ~0.00 |
| Sensitivity (all methods, mean) | ≥0.89 |

**Primary claim**: Early-hop retrieval failures amplify into answer-stage failures at significantly higher rates than late-hop failures, and agentic pipelines achieve non-trivial self-correction (~30%) for mid-pipeline retrieval faults — a recovery mechanism entirely invisible to end-to-end accuracy measurement.

---

## Implementation Plan (Actual File Structure)

```
Research-AgenticRAG/
├── src/agenticrag/
│   ├── core.py          # PipelineTrace, FailureRecord, AgenticRAGPipeline, DiagnosticBenchmark
│   ├── evaluate.py      # All metrics: root_cause_accuracy, failure_amplification_rate, etc.
│   ├── injection.py     # FailureInjector, InjectionResult, injection_sensitivity
│   ├── experiment.py    # run_ablation, AblationResult, AblationCell
│   ├── retrievers.py    # BM25Retriever, TokenOverlapRetriever
│   ├── datasets.py      # HotpotQAAdapter, MuSiQueAdapter
│   └── data.py          # Data loading utilities
├── tests/
│   ├── test_core.py
│   ├── test_evaluate.py
│   ├── test_injection.py
│   ├── test_propagation.py  # PropagationGraph tests (Phase 2)
│   └── test_experiment.py
├── related_work.tex     # LaTeX related work section
├── related_work.bib     # Bibliography
└── DESIGN_DOC.md
```

**Planned (Phase 2 — next):**
```
├── src/agenticrag/propagation.py   # PropagationGraph: causal DAG over failure records
└── scripts/
    ├── run_baseline.py
    ├── run_ablation.py
    └── plot_amplification.py
```

---

## Development Roadmap

| Phase | Status | Deliverable |
|---|---|---|
| Phase 1: Core infrastructure | **Done** | `core.py`, `evaluate.py`, `retrievers.py`, `datasets.py`; 38 tests passing |
| Phase 2: Causal propagation graph | **In progress** | `PropagationGraph` class; causal attribution across hops |
| Phase 3: Injection + ablation harness | **Done** | `injection.py`, `experiment.py`; full ablation grid |
| Phase 4: Paper experiments + figures | **Next** | Benchmark tables, amplification curves, recovery rate plots |

**Target**: EMNLP ORACLE 2026 submission

---

## Timeline

| Milestone | Date |
|---|---|
| PropagationGraph + causal attribution | June 2026 |
| Run all 5 experiments | July 2026 |
| Paper draft (results + analysis) | August 2026 |
| Related work + intro | August 2026 |
| EMNLP ORACLE 2026 submission | September 2026 |

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| PropagationGraph design complexity | Medium | Start with linear DAG (hop-ordered); generalize later |
| Small-scale eval limits generalization | High | Supplement with HotpotQA/MuSiQue full splits |
| Injection realism (faults too clean) | Medium | Add noisy variants: partially empty retrieval, partial hallucination |
| DiagnosticBenchmark false positives on healthy traces | Low | Measure specificity on clean traces in Experiment 1 |

---

## Why This Matters

**For researchers**: This is the first framework that treats failure propagation in agentic RAG as a first-class research problem, with controlled injection methodology and causal metrics. Distinguishes retrieval-stage failures from answer-stage failures in a way that end-to-end accuracy cannot.

**For practitioners**: Failure amplification curves and recovery rates give teams a concrete signal: which stage to harden first, and whether iterative reformulation provides meaningful self-correction.

**For Anote products**: The private chatbot uses agentic RAG for enterprise queries. Understanding failure propagation directly informs when to add retrieval fallbacks vs. answer-validation gates.

---

## Related Work

Key baselines and prior work to situate against:
- **Self-RAG** (Asai et al., 2023): self-reflective RAG with per-token retrieval decisions — no failure propagation analysis
- **FLARE** (Jiang et al., 2023): forward-looking active retrieval — uncertainty-triggered, no injection methodology
- **ReAct** (Yao et al., 2022): tool-use agent loop — evaluates task completion, not stage-level failure attribution
- **IRCoT** (Trivedi et al., 2022): iterative retrieval with CoT — multi-hop, but no controlled fault study
- **HotpotQA / MuSiQue**: evaluation datasets providing multi-hop structure for propagation experiments
