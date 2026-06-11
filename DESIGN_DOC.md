# Research Design Document: AgenticRAG

## Vision Statement

Pioneer **AgenticRAG**: a framework where AI agents dynamically decide *when* to retrieve, *what* to retrieve, and *how many times* to iterate — moving beyond single-shot RAG to autonomous, multi-step information gathering that matches or exceeds human research assistant quality on complex, multi-hop questions, at production-feasible latency and cost.

---

## Problem Statement & Novelty

Standard RAG (retrieve-then-generate) has a fundamental architectural mismatch with complex information needs:

1. **Static retrieval**: Retrieve once, generate once. Fails on multi-hop questions requiring iterative evidence gathering.
2. **Always-retrieve**: Systems retrieve even for questions answerable from parametric memory, wasting latency and cost.
3. **No retrieval gating**: No mechanism to decide when evidence is sufficient vs. when more retrieval is needed.
4. **Fixed query formulation**: Uses the user's literal question as the retrieval query, ignoring that reformulation often dramatically improves recall.

Existing approaches (Self-RAG, FLARE, ReAct) address parts of this problem but lack:
- A principled **retrieval gating model** with calibrated thresholds
- Empirical characterization of the **efficiency-accuracy tradeoff curve**
- Evaluation on **adversarial queries** designed to trigger unnecessary retrieval
- A **cost model** for comparing agentic vs. standard RAG in production

### Novel Contributions

| Contribution | Description |
|---|---|
| **Retrieval gating model** | Binary classifier deciding retrieve vs. generate-from-memory, with calibrated confidence |
| **RGAR metric** | Retrieval-Gated Accuracy Rate: accuracy weighted by retrieval efficiency |
| **Multi-hop orchestration** | Iterative sub-query decomposition with stopping criterion |
| **Adversarial query set** | 500 queries designed to test over-retrieval and under-retrieval failure modes |
| **Production cost model** | First empirical cost comparison: agentic RAG vs. standard RAG at scale |

### RGAR Definition

```
RGAR = Accuracy × Retrieval_Efficiency

where:
  Retrieval_Efficiency = 1 - (unnecessary_retrievals / total_retrievals)
  unnecessary_retrieval = retrieval_call where answer_was_in_parametric_memory

Interpretation:
  RGAR = 1.0: perfect accuracy + zero unnecessary retrievals
  RGAR < 0.5: either poor accuracy or excessive retrieval waste
```

---

## Research Objectives

1. Build a **retrieval gating model** that achieves >90% accuracy in deciding when to retrieve vs. generate from memory.
2. Show that **query reformulation** improves multi-hop recall by ≥20 pp vs. literal query pass-through.
3. Characterize the **efficiency-accuracy Pareto curve** for agentic RAG with variable iteration budgets.
4. Demonstrate that AgenticRAG achieves >85% of maximum accuracy at <50% of the retrieval calls of "always retrieve" baselines.
5. Evaluate on **adversarial queries** to expose failure modes specific to agentic architectures.

---

## Dataset Construction

### Question Types

| Type | Count | Description | Retrieval Required |
|---|---|---|---|
| Single-hop factual | 500 | Direct fact lookup | Yes |
| Parametric memory | 500 | Answerable from model knowledge | No |
| Multi-hop (2-hop) | 500 | Two retrieval steps needed | Yes (×2) |
| Multi-hop (3+ hop) | 300 | Three or more retrieval steps | Yes (×3+) |
| Comparative | 300 | Compare entities across documents | Yes (×2) |
| Temporal reasoning | 300 | Time-sensitive, requires recent retrieval | Yes |
| Adversarial over-retrieve | 250 | Designed to trigger unnecessary retrieval | No |
| Adversarial under-retrieve | 250 | Designed to cause insufficient retrieval | Yes |

**Total: 2,900 questions**

### Corpus
- Wikipedia (2024 snapshot): general knowledge
- PubMed abstracts: biomedical domain
- ArXiv (CS/AI): technical domain
- News (2023–2024): time-sensitive facts

---

## Systems Under Evaluation

| System | Architecture | Gating | Iterations | Notes |
|---|---|---|---|---|
| Standard RAG (BM25) | Retrieve-once | None | 1 | Baseline |
| Standard RAG (dense) | Retrieve-once | None | 1 | Baseline |
| Self-RAG | Self-reflective | LLM token | Variable | Prior work |
| FLARE | Forward-look | Confidence | Variable | Prior work |
| ReAct | Tool-use | LLM decision | Variable | Prior work |
| AgenticRAG-Basic | Gating model | Classifier | ≤3 | Ours (v1) |
| AgenticRAG-Full | Gating + reformulation | Classifier | ≤5 | Ours (v2) |
| Oracle RAG | Always correct retrieval | — | Optimal | Upper bound |

---

## Experimental Design

### Baseline Experiment (Experiment 0)
**Protocol**: Run Standard RAG (BM25 + GPT-4o reader) on all 2,900 questions. Compute Accuracy, RGAR, mean retrieval calls per question.

**Expected result**: Accuracy ≈ 0.67 overall; RGAR ≈ 0.48 (accuracy hurt by unnecessary retrievals on parametric memory questions); 1.0 retrieval calls/question.

---

### Experiment 1: Retrieval Gating Model
**Hypothesis**: A fine-tuned gating classifier achieves >90% accuracy on the retrieve vs. generate decision, outperforming LLM self-assessment.

**Protocol**:
1. Collect training data: (question, context) → {retrieve, generate} labels from human annotators (n=1,500).
2. Fine-tune DeBERTa-v3 as binary gating classifier.
3. Compare: (a) fine-tuned classifier, (b) GPT-4o prompted "Do you need to retrieve?", (c) confidence-threshold baseline.
4. Evaluate on held-out 400 questions.

**Expected results**:

| Gating Method | Accuracy | False Positive (unnecessary retrieve) | False Negative (missed retrieve) |
|---|---|---|---|
| Always retrieve (baseline) | — | 100% | 0% |
| Confidence threshold | 0.71 | 0.35 | 0.22 |
| GPT-4o self-assess | 0.83 | 0.18 | 0.14 |
| Fine-tuned classifier | 0.91 | 0.09 | 0.08 |

- Fine-tuned classifier is both more accurate and cheaper than LLM self-assessment.

---

### Experiment 2: Query Reformulation
**Hypothesis**: LLM-based query reformulation (decomposing multi-hop questions into sub-queries) improves Recall@10 by ≥20 pp vs. literal query.

**Protocol**:
1. For multi-hop questions (n=800), compare: (a) literal query, (b) LLM-decomposed sub-queries, (c) iterative reformulation (reformulate after each retrieval step).
2. Compute Recall@10 for each method.
3. Measure mean sub-queries generated and total retrieval calls.

**Expected results**:
- Literal query: Recall@10 ≈ 0.51 on multi-hop
- LLM-decomposed: Recall@10 ≈ 0.72 (+21 pp)
- Iterative reformulation: Recall@10 ≈ 0.79 (+28 pp)
- Mean sub-queries: 1.0 vs. 2.1 vs. 2.8 retrieval calls
- Cost-recall tradeoff: iterative is optimal above 0.75 recall target; decomposition is optimal for 0.65–0.75

---

### Experiment 3: Efficiency-Accuracy Pareto Curve
**Hypothesis**: AgenticRAG-Full achieves ≥85% of Oracle RAG accuracy at ≤50% of Oracle's retrieval calls.

**Protocol**:
1. Sweep the iteration budget (1 to 5 retrieval calls allowed).
2. For each budget: compute Accuracy and mean retrieval calls for AgenticRAG-Full.
3. Compare to Oracle RAG (unlimited, optimal retrieval).
4. Plot efficiency-accuracy Pareto curve.

**Expected results**:

| Budget | Accuracy | Mean Calls | % of Oracle Accuracy |
|---|---|---|---|
| 1 (standard) | 0.67 | 1.0 | 72% |
| 2 | 0.76 | 1.6 | 82% |
| 3 (AgenticRAG-Basic) | 0.82 | 2.1 | 88% |
| 5 (AgenticRAG-Full) | 0.86 | 2.9 | 93% |
| Oracle | 0.93 | 4.8 | 100% |

- Key finding: AgenticRAG-Full achieves 93% of Oracle accuracy at 60% of Oracle retrieval calls.

---

### Experiment 4: Adversarial Evaluation
**Hypothesis**: Agentic systems are vulnerable to adversarial over-retrieval (>2× retrieval calls on parametric questions) and under-retrieval (fails to retrieve when needed); RGAR exposes this where accuracy does not.

**Protocol**:
1. Run all systems on 500 adversarial questions (250 over-retrieve, 250 under-retrieve).
2. Compute accuracy, retrieval calls, and RGAR for adversarial subsets.
3. Compare systems' vulnerability profiles.

**Expected results**:
- Standard RAG (over-retrieve subset): accuracy artificially inflated; RGAR ≈ 0.41 (many unnecessary calls)
- Self-RAG (under-retrieve subset): accuracy ≈ 0.58; misses critical retrieval in 31% of cases
- AgenticRAG-Full with gating: RGAR ≈ 0.76 on adversarial set (best among agentic systems)
- Key finding: RGAR reveals adversarial vulnerabilities invisible to accuracy alone

---

### Experiment 5: Production Cost Model
**Hypothesis**: AgenticRAG-Full costs ≤2× standard RAG per query while improving accuracy by ≥15 pp, making it cost-justified for enterprise deployments.

**Protocol**:
1. Measure cost per query: LLM API calls (gating + generation + reformulation) + retrieval infrastructure.
2. Compute cost-accuracy tradeoff ratio: (accuracy_improvement / baseline_accuracy) / (cost_increase / baseline_cost).
3. Compare against Self-RAG, FLARE, ReAct.

**Expected results**:
- Standard RAG: $0.003/query, 0.67 accuracy
- AgenticRAG-Full: $0.005/query (+67% cost), 0.86 accuracy (+28% accuracy)
- Self-RAG: $0.008/query (+167% cost), 0.79 accuracy (+18% accuracy)
- AgenticRAG efficiency ratio: 0.28/0.67 = 0.42 (best among all agentic systems)

---

## Expected Results Summary

| Metric | Standard RAG | AgenticRAG-Full | Improvement |
|---|---|---|---|
| Accuracy (overall) | 0.67 | 0.86 | +28% |
| Accuracy (multi-hop) | 0.51 | 0.79 | +55% |
| RGAR | 0.48 | 0.76 | +58% |
| Mean retrieval calls | 1.0 | 2.9 | +190% |
| Cost per query | $0.003 | $0.005 | +67% |

**Primary claim**: A calibrated retrieval gating model + iterative query reformulation closes 80% of the gap between standard RAG and oracle RAG, with only 67% cost increase — making agentic RAG economically viable for production deployment.

---

## Why This Matters

**For researchers**: AgenticRAG provides the first principled framework for agentic retrieval, with metrics (RGAR) and experiments that distinguish genuine reasoning improvement from retrieval over-spending.

**For practitioners**: The cost model and Pareto curve give engineering teams concrete guidance for when to deploy agentic vs. standard RAG.

**For Anote products**: The private chatbot product can directly benefit from AgenticRAG — particularly for complex enterprise queries requiring multi-hop reasoning.

**RSI connection**: AgenticRAG is a building block for recursive self-improvement: an agent that can retrieve its own prior results and improve iteratively is a core RSI primitive.

---

## Implementation Plan

```
research-agenticrag/
├── data/
│   ├── questions/       # 2,900 questions with labels
│   ├── corpus/          # Wikipedia, PubMed, ArXiv, News
│   └── adversarial/     # 500 adversarial queries
├── gating/
│   ├── train_gating.py  # DeBERTa fine-tuning
│   └── gating_model/    # Saved weights
├── retrieval/
│   ├── bm25.py
│   ├── dense.py
│   └── hybrid.py
├── agentic/
│   ├── orchestrator.py  # Main agentic loop
│   ├── reformulator.py  # Query reformulation
│   └── stopping.py      # Iteration stopping criterion
├── metrics/
│   ├── rgar.py
│   └── efficiency.py
├── experiments/
│   ├── exp0_baseline.py
│   ├── exp1_gating.py
│   ├── exp2_reformulation.py
│   ├── exp3_pareto.py
│   ├── exp4_adversarial.py
│   └── exp5_cost.py
```

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| Dataset construction | 6 weeks | 2,900 questions labeled |
| Gating model training | 3 weeks | Fine-tuned DeBERTa classifier |
| Agentic system implementation | 4 weeks | AgenticRAG-Basic + Full |
| Experiments | 5 weeks | All results |
| Paper writing | 4 weeks | ACL 2026 submission |

**Target venue**: ACL 2026 or NAACL 2026

---

## Open Questions & Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Gating model training data cost | Medium | Start with 500 examples; scale if needed |
| LLM API cost for large-scale eval | High | Use open-source models for most experiments |
| Adversarial query design subjectivity | Medium | Two-annotator design + adjudication |
| ReAct / Self-RAG reproducibility | Medium | Use official implementations |

---

## Related Issues

- Product integration: private chatbot (AgenticRAG deployment)
- RSI connection: retrieval as RSI primitive
- Reproducibility package
- Related work audit: Self-RAG, FLARE, ReAct, IRCoT
