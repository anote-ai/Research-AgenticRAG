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

## Real LLM Agent + Interventional Diagnosis

Beyond the heuristic control pipeline, the package ships a real, provider-agnostic
agent and an *interventional* failure-propagation framework.

- **`LLMAgent` ([agents.py](src/agenticrag/agents.py))** — ReAct-style iterative
  retrieval (decompose → retrieve → reason → re-retrieve → answer) behind the same
  `retriever` seam as the heuristic pipeline. Backbones via thin adapters:
  `ClaudeProvider` (Opus 4.8 / Sonnet 4.6, primary), `OpenAIProvider`, and a
  deterministic `MockProvider` that needs no API key (offline smoke + the
  heuristic control). The agent is **resumable** (`resume_from_hops`,
  `force_answer`), which is what makes injection interventional.
- **`LiveFailureInjector` ([injection.py](src/agenticrag/injection.py))** —
  `do(failure = f at hop h)`: corrupt the trajectory prefix at a hop, then let the
  agent re-run the suffix, so downstream propagation is the agent's *real reaction*
  (it may self-correct). Interventions: empty / irrelevant retrieval, query drift,
  CRAG-style false-premise and stale-evidence, and early termination. Ground-truth
  labels (`injected_stage`, `injected_failure_type`, `injected_at_hop`) are
  preserved for certified-root-cause evaluation.
- **Diagnosers ([diagnosers.py](src/agenticrag/diagnosers.py))** — three post-hoc
  baselines (`RuleBasedDiagnoser`, `DoctorRAGDiagnoser`, `LLMJudgeDiagnoser`) and
  the **`PropagationAwareDiagnoser`** (C3): counterfactual single-hop repair probes
  that localize the earliest causally-responsible hop, recovering root causes that
  surface-level diagnosis misses — at a re-execution token cost.
- **Metrics ([evaluate.py](src/agenticrag/evaluate.py),
  [propagation.py](src/agenticrag/propagation.py))** — `attribution_identifiability`
  (RCA vs propagation depth, the C2 curve), `cost_per_correct_diagnosis`
  (deployability), and `counterfactual_recovery_rate` (Pearl rung 3).
- **Datasets ([datasets.py](src/agenticrag/datasets.py))** — anchor sets
  (HotpotQA, MuSiQue) plus the richness sets **FRAMES** (variable 2–15-hop depth
  substrate) and **CRAG** (multi-domain; false-premise / long-tail / temporal).

### Headline experiment

```bash
# Offline smoke (mock backbone, no API key, FRAMES fallback):
python scripts/run_identifiability.py --dataset frames --provider mock --hops 1 2 3

# Real backbone (set ANTHROPIC_API_KEY), variable-depth substrate:
python scripts/run_identifiability.py --dataset frames --provider claude \
    --model claude-opus-4-8 --retriever dense --max-samples 100 --hops 1 2 3
```

Produces the root-cause-attribution-accuracy-vs-injection-depth table per diagnoser
plus cost-per-correct-diagnosis and counterfactual recovery, saved to
`results/identifiability_{provider}_{dataset}.json`.

By default, the identifiability runner now keeps depth buckets strict: a hop-2
or hop-3 intervention is only applied to base traces that actually reached that
many hops, and injections are skipped when the unmodified base trace already
answers incorrectly. The output JSON records `n_eligible_by_depth`,
`n_skipped_short_trace_by_depth`, `n_skipped_base_incorrect_by_depth`, and
per-failed-case metadata (`sample_id`, requested depth, actual injected hop,
intervention type, base/final answers). Use `--allow-short-depth-clamp` or
`--include-base-failures` only when intentionally reproducing older exploratory
runs.

Long real-backbone runs are resumable. In addition to the final result JSON,
`scripts/run_identifiability.py --resume` writes per-case caches next to the
output file:

- `<result>.cases.jsonl` stores base eligibility decisions and injected traces.
- `<result>.diagnoses.jsonl` stores per-diagnoser predictions for failed cases.

If a run drops halfway through a depth, re-run the exact same command with
`--resume`; completed depths load from the JSON checkpoint, and partially
completed depths reuse these JSONL caches instead of regenerating finished
cases or diagnoses.

## Benchmark Stats Template

| Metric                      | Value  |
|-----------------------------|--------|
| End-to-end accuracy         | 0.40   |
| Retrieval failure rate      | 0.20   |
| Tool call failure rate      | 0.12   |
| Answer generation fail rate | 0.28   |
| Propagation rate            | 0.32   |

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
