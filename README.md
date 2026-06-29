# research-agenticrag

**Failure diagnosis and attribution for agentic RAG pipelines.**

## Failure Taxonomy

| Stage              | Failure Type      | Propagated | Severity | Description                                |
|--------------------|-------------------|------------|----------|-----------------------------------------|
| Retrieval          | empty_retrieval   | Yes        | 0.9      | No documents retrieved from corpus      |
| Tool Call          | no_tool_calls     | Yes        | 0.7      | Agent skipped tool invocation           |
| Answer Generation  | empty_answer      | No         | 0.8      | LLM produced empty response             |
| Answer Generation  | incorrect_answer  | No         | 0.5      | Answer does not match reference         |
| None               | success           | No         | 0.0      | Pipeline completed correctly            |

## Pipeline Architecture

```
Query
  │
  ▼
[Retrieval] ──fail──► empty_retrieval (propagated)
  │
  ▼
[Tool Call] ──fail──► no_tool_calls (propagated)
  │
  ▼
[Answer Generation] ──fail──► empty_answer / incorrect_answer
  │
  ▼
[Success]
```

## Diagnostic Method

1. Run `DiagnosticBenchmark.diagnose_trace` on each pipeline trace.
2. Aggregate with `attribute_failures` to get per-stage counts and propagation rates.
3. Evaluate with `end_to_end_accuracy`, `severity_weighted_failure_rate`, and `stage_attribution_rate`.

## Benchmark Stats Template

> **Note:** the table below is a *template* showing the metrics this benchmark
> reports and their expected shape -- the values are illustrative placeholders,
> not measured results from a real experiment run. See `results/README.md` for
> how to generate real numbers and `PAPER_DRAFT.md` for which figures in
> `DESIGN_DOC.md` are still projections vs. confirmed.

| Metric                      | Value  |
|-----------------------------|--------|
| End-to-end accuracy         | 0.40   |
| Retrieval failure rate      | 0.20   |
| Tool call failure rate      | 0.12   |
| Answer generation fail rate | 0.28   |
| Propagation rate            | 0.32   |

## More resources

- [`DESIGN_DOC.md`](DESIGN_DOC.md) -- full research design, experiments, and roadmap.
- [`PAPER_DRAFT.md`](PAPER_DRAFT.md) -- paper draft skeleton (sections implemented, results marked TBD/projected pending real experiment runs).
- [`BLOG.md`](BLOG.md) -- plain-language summary of the project for a non-academic audience.
- [`results/`](results/) -- where experiment script output lands once `scripts/run_baseline.py` / `run_ablation.py` are actually executed.

## Venue

Submitted to **EMNLP ORACLE 2026** — Workshop on Observations, Reasoning, and Causal Analysis in Language Evaluation.

## Citation

```bibtex
@inproceedings{anote2026agenticrag,
  title     = {Failure Attribution in Agentic RAG: A Diagnostic Benchmark},
  author    = {Anote AI},
  booktitle = {EMNLP ORACLE 2026},
  year      = {2026},
}
```
