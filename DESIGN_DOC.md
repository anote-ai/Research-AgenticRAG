# AgenticRAG — Research Design Document

## Goal

Demonstrate that adaptive multi-hop retrieval — where the system decides when to retrieve, what to retrieve, and when to stop — outperforms static single-pass RAG on complex multi-hop questions, while characterizing the efficiency-accuracy tradeoff to make the approach deployable.

## Objective

1. Build an agentic retrieval system with explicit retrieval gating, query reformulation, and stopping criteria
2. Benchmark against single-pass RAG baselines on multi-hop QA datasets
3. Characterize the API call reduction (efficiency gain) vs. accuracy improvement tradeoff

## Background / Motivation

Single-pass RAG fails systematically on multi-hop questions because a single query can't retrieve all necessary passages. Prior work (Self-RAG, FLARE, IRCoT) has shown iterative retrieval improves accuracy. But none of these papers characterize the cost: how many extra API calls are needed? When is the extra cost justified? This is the gap between academic research and production deployment that we close.

## Experimental Design

### Baseline Experiment

**Evaluate single-pass RAG (E5-large + GPT-4o) on MuSiQue, HotpotQA, and 2WikiMultiHopQA**

- Metric: Exact Match (EM) and F1; number of retrieval API calls per question
- Purpose: establish baseline accuracy and efficiency for multi-hop questions
- Expected result: EM of 25–35% on MuSiQue (single-pass consistently misses later hops)

### Test Experiment 1: Gating Model for Retrieval Decisions

Train a lightweight gating model that predicts whether additional retrieval is needed at each generation step.

- Compare vs. baselines: always retrieve, never retrieve after first round, retrieve every k tokens (FLARE-style)

**Expected result:** learned gating reduces unnecessary retrievals by 40% vs. FLARE while matching or exceeding FLARE accuracy (+3–5 EM points vs. single-pass)

### Test Experiment 2: Query Reformulation for Hop 2+

Compare query strategies for subsequent hops: original query, rule-based reformulation, LLM-based reformulation. Measure retrieval recall at each hop and token cost per strategy.

**Expected result:** LLM-based reformulation improves recall at hop 2 by 15–20% vs. original query; adds ~500 tokens per question

### Test Experiment 3: Efficiency-Accuracy Tradeoff Curve

Sweep stopping criteria aggressiveness and plot: X-axis = average API calls per question; Y-axis = EM on MuSiQue. Find the Pareto-optimal operating point.

**Expected result:** clear elbow in the tradeoff curve at ~2.3 average API calls per question — beyond this, additional retrievals add cost but not accuracy

## Expected Results

1. A trained agentic retrieval system with learned gating + query reformulation
2. Benchmark results on MuSiQue, HotpotQA, 2WikiMultiHopQA showing +5–10 EM vs. single-pass
3. The efficiency-accuracy tradeoff curve: first systematic characterization of when additional hops stop being worth it
4. **Key finding:** "Adaptive retrieval improves multi-hop accuracy by 8 EM points while adding only 1.3 API calls per question on average"

## Why This Matters / Why People Would Care

- **RAG practitioners:** need to know whether adaptive retrieval is worth the extra API cost
- **LlamaIndex and LangChain:** implement retrieval pipelines; this paper is immediately actionable for their product teams
- **LLM API providers:** adaptive retrieval increases API usage; they want to show practitioners the accuracy gains justify the cost
- **AI researchers:** the gating model and reformulation methodology are novel and generalizable

## Timeline

| Month | Milestone |
|---|---|
| 1 | System implementation (gating model, query reformulation, stopping criteria) |
| 2 | Baseline evaluation on MuSiQue, HotpotQA, 2WikiMultiHopQA |
| 3 | Gating model training and evaluation |
| 4 | Tradeoff curve experiments |
| 5 | Paper writing |
| 6 | Submission to ACL 2026 |

## Related Issues

- Design doc GitHub issue: #22
- Target conferences: see issues labeled `conference-prep`
- Reproducibility package: see issues labeled `artifact-release`
