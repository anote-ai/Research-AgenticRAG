# Research Plan: Failure Propagation in Agentic RAG Systems

**Working title:** *Cascading Failures in Agentic RAG: A Causal Framework for Attribution and Hop-Depth Amplification*

**Target venues:** AAAI 2027 (primary) → SIGIR 2027 (secondary)
**Estimated deadlines:** AAAI abstract ~Aug 2026 · SIGIR full paper ~Oct 2026

---

## Core Thesis

Agentic RAG pipelines fail in structured, predictable ways: failures originate at a root-cause stage (most often retrieval), propagate downstream, and amplify with hop depth. We present a causal attribution framework, a labeled benchmark, and empirical evidence that retriever quality is the single strongest predictor of end-to-end failure — with implications for where practitioners should invest in robustness.

**One-sentence version:** *Failures in agentic RAG are not random — they cascade causally from retrieval, compound across hops, and can be diagnosed with a lightweight attribution model.*

---

## Paper Structure

### §1 Introduction
- Motivate agentic RAG as an emerging paradigm with understudied failure modes
- Distinguish from standard RAG: the agent loop introduces *propagation* — a failure at hop k can corrupt all subsequent hops
- State contributions explicitly (see below)

### §2 Background & Related Work
- RAG failure modes (prior work on retrieval quality, hallucination)
- Multi-hop QA (HotpotQA, MuSiQue)
- Causal attribution in ML pipelines
- Gap: no prior work models *inter-stage propagation* in agentic loops

### §3 Failure Taxonomy
- 3 pipeline stages: Retrieval · Tool Use · Answer Generation
- 8 failure types: `empty_retrieval`, `no_tool_calls`, `empty_answer`, `incorrect_answer`, `hallucination`, `over_retrieval`, `context_overflow`, `success`
- Define **propagated** failures (root cause is upstream of observed symptom) vs. **local** failures
- Diagram: failure propagation DAG (Retrieval → Tool Use → Generation)

### §4 Benchmark Construction
- Datasets: HotpotQA (distractor, 2-hop) + MuSiQue (2–4 hop)
- Controlled failure injection harness: inject failures at specific stage × hop combinations
- Label each trace with: observed failure stage, root-cause stage, hop of injection, severity
- Statistics table: N traces, failure type distribution, hop depth distribution
- **Artifact:** publicly released benchmark with ground-truth propagation labels

### §5 Experiments

#### 5.1 Hop-Depth Failure Amplification
- Hypothesis: failure rate at the final answer increases monotonically with injection hop k
- Method: inject `empty_retrieval` at hop k ∈ {1, 2, 3}, measure end-to-end accuracy
- Expected result: failure curve rising with k, showing compounding
- Metric: `failure_amplification_rate(records_by_hop)`

#### 5.2 Recovery Rate
- Question: how often does the agent self-correct from a mid-hop failure?
- Method: observe natural recovery across multi-hop traces with early retrieval failures
- Metric: `recovery_rate(records_by_hop)`
- Expected finding: recovery is low (<20%), motivating the need for explicit robustness mechanisms

#### 5.3 Retriever as a Risk Factor
- Compare: token-overlap baseline · BM25 · dense retriever (sentence-transformers)
- Show: retriever quality predicts downstream failure rate better than any other single factor
- Benchmark table: retriever × dataset × {end_to_end_accuracy, propagation_rate, severity_weighted_failure_rate}

### §6 Causal Propagation Model
- Represent each trace as a directed graph: nodes = pipeline stages, edges = propagation events
- Estimate edge weights (propagation probabilities) from the benchmark dataset
- Define **Root-Cause Accuracy**: does the classifier identify the *earliest* failing stage, not just the final symptom?
- Baseline: rule-based `DiagnosticBenchmark` (already implemented)
- Proposed: probability-weighted causal graph traversal
- Show: causal model improves root-cause accuracy over symptom-only detection

### §7 Analysis & Discussion
- Propagation heatmap: stage × stage matrix of observed propagation frequencies
- Hop-depth failure curve (Figure 1 — the hero figure)
- When does recovery happen? Conditions under which the agent self-corrects
- Limitations: synthetic injection may not capture all natural failure modes

### §8 Conclusion
- Summarise: causal framework + benchmark + finding (retrieval is the dominant root cause)
- Call to action: future work on retrieval robustness, recovery mechanisms, learned severity models

---

## Contributions (bullet form for §1)

1. A **failure taxonomy** for agentic RAG with 8 typed failure modes across 3 pipeline stages
2. A **labeled benchmark** of traces with ground-truth propagation annotations (HotpotQA + MuSiQue)
3. A **causal attribution model** that identifies root-cause stage rather than symptom stage, with a new Root-Cause Accuracy metric
4. Empirical evidence that **failures amplify with hop depth** and that **retriever quality is the dominant predictor** of end-to-end failure

---

## What Is Already Built

| Component | Status | File |
|---|---|---|
| Failure taxonomy (5 types) | Done | `core.py` |
| FailureType enum (8 types) | Done | `core.py` |
| DiagnosticBenchmark (rule-based) | Done | `core.py` |
| BM25Retriever | Done | `retrievers.py` |
| HotpotQA + MuSiQue adapters | Done | `datasets.py` |
| `failure_amplification_rate()` | Done | `evaluate.py` |
| `recovery_rate()` | Done | `evaluate.py` |
| Synthetic benchmark (4 questions) | Done | `data.py` |
| Failure injection harness | Done | `injection.py` |
| PropagationGraph skeleton | Done | `propagation.py` |
| 46 passing tests | Done | `tests/` |

## What Needs to Be Built

| Component | Maps to | Priority |
|---|---|---|
| Root-cause accuracy metric | §6 | High |
| Causal model prototype | §6 | High |
| Dense retriever (sentence-transformers) | §5.3 | Medium |
| Benchmark annotation pipeline (scale to 500+ traces) | §4 | High |
| Propagation heatmap figure | §7 | Medium |
| Hop-depth failure curve figure | §7 | Medium |
| Experiment runner scripts | §5 | High |

---

## Timeline

| Week | Goal |
|---|---|
| Jun 9–15 | Done: Failure injection harness (`injection.py`); PropagationGraph skeleton |
| Jun 16–22 | Root-cause accuracy metric; causal model prototype |
| Jun 23–29 | Scale benchmark to 500 traces (HotpotQA); run §5.1 amplification experiment |
| Jun 30–Jul 6 | §5.2 recovery rate experiment; §5.3 retriever comparison |
| Jul 7–20 | Analysis: heatmap, hop-depth curve; dense retriever integration |
| Jul 21–Aug 3 | Paper draft (§3–6 first) |
| Aug 4–17 | Full draft → revisions → AAAI submission |
| Oct | SIGIR revision if needed |

---

## Open Questions

- **Scale:** AAAI will want 500–1000 labeled traces minimum. Is the annotation fully automated (injection-based) or does any human labeling happen?
- **LLM reader:** Do we plug in a real LLM (claude-haiku) for answer generation, or keep the extractive stub? A real LLM produces more naturalistic hallucination failures but adds cost and non-determinism.
- **Dense retriever:** sentence-transformers adds a heavy dependency. Alternative: use an API-based embedding endpoint to keep the package lightweight.
- **Causal model complexity:** start with a simple Bayesian network or logistic regression on propagation features — no need for a learned neural model at this scope.
