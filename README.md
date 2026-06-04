# research-agenticrag

**Failure Propagation in Agentic RAG Pipelines: A Diagnostic Benchmark**

## Overview

This repository contains the research code for studying how retrieval errors cascade into
agent tool-calls and final answers in agentic RAG systems. We introduce a diagnostic benchmark
that attributes end-to-end failures to one of three pipeline stages.

## Failure Taxonomy

| Stage             | Failure Types                              | Examples                                         |
|-------------------|--------------------------------------------|--------------------------------------------------|
| Retrieval         | empty_retrieval, low_relevance, truncation | No docs returned; top-k irrelevant docs          |
| Tool Call         | tool_no_output, tool_error, timeout        | API error; malformed arguments; rate limit       |
| Answer Generation | wrong_answer, hallucination, refusal       | Off-topic response; fabricated citations         |
| None              | —                                          | Correct end-to-end trace                         |

## Pipeline Diagram

```
Query
  │
  ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Retrieval  │────▶│  Tool Calls  │────▶│ Answer Generation │
└─────────────┘     └──────────────┘     └───────────────────┘
       │                    │                      │
  [Stage 1 fail]      [Stage 2 fail]         [Stage 3 fail]
  propagates ──────────────▶                 does not propagate
```

## Diagnostic Methodology

1. **Trace collection** — Run a target agentic RAG system on benchmark queries; record all
   retrieved documents, tool calls, and final answers.
2. **Stage attribution** — Apply heuristics and LLM-based classifiers to attribute each
   failure to its root stage.
3. **Propagation analysis** — Flag failures where the stage-1 error causally leads to
   stage-2 or stage-3 errors.
4. **Confusion matrix** — Compare attributed stages to human-labeled ground truth.

## Benchmark Statistics Template

| Metric                     | Value  |
|----------------------------|--------|
| Total traces               | TBD    |
| Retrieval failure rate     | TBD    |
| Tool-call failure rate     | TBD    |
| Answer generation fail rate| TBD    |
| Propagation rate           | TBD    |
| End-to-end accuracy        | TBD    |

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```python
from agenticrag.core import DiagnosticBenchmark, PipelineTrace
from agenticrag.evaluate import stage_attribution_rate, propagation_rate, end_to_end_accuracy
from agenticrag.core import FailureStage

benchmark = DiagnosticBenchmark()
traces = benchmark.load_traces("data/traces.jsonl")
records = [benchmark.diagnose_trace(t, {}) for t in traces]

print("Retrieval failure rate:", stage_attribution_rate(records, FailureStage.RETRIEVAL))
print("Propagation rate:", propagation_rate(records))
print("E2E accuracy:", end_to_end_accuracy(traces))
```

## Citation

```bibtex
@misc{anote2024agenticrag,
  title  = {Failure Propagation in Agentic RAG Pipelines: A Diagnostic Benchmark},
  author = {Anote AI Research},
  year   = {2024},
  url    = {https://github.com/anote-ai/research-agenticrag},
}
```
