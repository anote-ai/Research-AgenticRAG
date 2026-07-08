# Failure-Propagation Identifiability — Final Results

Real-backbone conditions: 14  (backbones: claude-haiku-4-5, gpt-4o-mini, qwen2.5:7b; retrievers: bm25 + dense-budget4; datasets: musique/hotpotqa/frames/crag)

## C2 — Root-cause attribution collapses with propagation depth

Doctor-RAG-style (coverage) baseline, accuracy by depth:

| backbone · dataset · retr | hop1 | hop2 | hop3 |
|---|---|---|---|
| claude:claude-haiku-4-5 · crag · bm25 | 0.50 | 0.50 | 0.50 |
| claude:claude-haiku-4-5 · frames · bm25 | 1.00 | 0.03 | 0.03 |
| claude:claude-haiku-4-5 · hotpotqa · dense | 1.00 | – | – |
| claude:claude-haiku-4-5 · musique · dense | 1.00 | 0.12 | – |
| claude:claude-haiku-4-5 · hotpotqa · bm25 | 1.00 | 0.33 | 0.18 |
| claude:claude-haiku-4-5 · musique · bm25 | 1.00 | 0.05 | 0.07 |
| openai:gpt-4o-mini · crag · bm25 | 0.50 | 0.50 | 0.50 |
| openai:gpt-4o-mini · frames · bm25 | 0.99 | 0.07 | 0.01 |
| openai:gpt-4o-mini · frames · dense | 0.98 | 0.03 | – |
| openai:gpt-4o-mini · hotpotqa · dense | 1.00 | 0.55 | 0.52 |
| openai:gpt-4o-mini · musique · bm25 | 1.00 | 0.39 | 0.34 |
| openai:gpt-4o-mini · musique · dense | 1.00 | 0.46 | 0.43 |
| openai:gpt-4o-mini · hotpotqa · bm25 | 1.00 | 0.71 | 0.58 |
| openai:qwen2.5:7b · musique · bm25 | 1.00 | 0.08 | 0.14 |

**Headline:** mean hop-1 = **0.93**  →  mean hop-2/3 = **0.30**  (n=14 conditions). Attribution is near-perfect at the injection hop and collapses once the failure propagates — the identifiability bound, on real LLM agents.

## C3 — Propagation-aware method vs best post-hoc baseline (depth≥2)

| backbone · dataset · retr | PA | best baseline | verdict |
|---|---|---|---|
| claude:claude-haiku-4-5 · crag · bm25 | 0.25 | 0.50 | lose |
| claude:claude-haiku-4-5 · frames · bm25 | 0.11 | 0.33 | lose |
| claude:claude-haiku-4-5 · musique · dense | 0.35 | 0.76 | lose |
| claude:claude-haiku-4-5 · hotpotqa · bm25 | 0.34 | 0.28 | **win** |
| claude:claude-haiku-4-5 · musique · bm25 | 0.32 | 0.48 | lose |
| openai:gpt-4o-mini · crag · bm25 | 0.25 | 0.75 | lose |
| openai:gpt-4o-mini · frames · bm25 | 0.16 | 0.52 | lose |
| openai:gpt-4o-mini · frames · dense | 0.20 | 0.64 | lose |
| openai:gpt-4o-mini · hotpotqa · dense | 0.50 | 0.54 | lose |
| openai:gpt-4o-mini · musique · bm25 | 0.40 | 0.39 | **win** |
| openai:gpt-4o-mini · musique · dense | 0.53 | 0.51 | **win** |
| openai:gpt-4o-mini · hotpotqa · bm25 | 0.54 | 0.64 | lose |
| openai:qwen2.5:7b · musique · bm25 | 0.33 | 0.46 | lose |

**Verdict:** propagation-aware beats the best baseline at depth≥2 in **3/13** conditions (only narrow musique wins). Dense + budget-4 + larger-n did **not** change this — C3 as designed is **not supported**; it needs a redesigned localizer, not more compute.

## Secondary — LLM-judge is depth-robust; coverage-gating collapses

| backbone·dataset·retr | doctor_rag hop1→deep drop | llm_judge hop1→deep drop |
|---|---|---|
| claude:claude-haiku-4-5·crag·bm25 | +0.00 | +0.12 |
| claude:claude-haiku-4-5·frames·bm25 | +0.97 | +0.03 |
| claude:claude-haiku-4-5·musique·dense | +0.88 | -0.60 |
| claude:claude-haiku-4-5·hotpotqa·bm25 | +0.75 | -0.03 |
| claude:claude-haiku-4-5·musique·bm25 | +0.94 | -0.14 |
| openai:gpt-4o-mini·crag·bm25 | +0.00 | +0.00 |
| openai:gpt-4o-mini·frames·bm25 | +0.96 | +0.17 |
| openai:gpt-4o-mini·frames·dense | +0.95 | +0.04 |
| openai:gpt-4o-mini·hotpotqa·dense | +0.46 | -0.06 |
| openai:gpt-4o-mini·musique·bm25 | +0.64 | +0.18 |
| openai:gpt-4o-mini·musique·dense | +0.55 | -0.01 |
| openai:gpt-4o-mini·hotpotqa·bm25 | +0.36 | -0.02 |
| openai:qwen2.5:7b·musique·bm25 | +0.89 | +0.07 |

Coverage-gating (doctor_rag) drops sharply with depth; the LLM-judge stays comparatively flat — reasoning-based diagnosis is more robust to propagation than structural coverage.

## Counterfactual recovery (agent absorbs the injected fault)

| backbone · dataset · retr | recovery by depth |
|---|---|---|
| claude:claude-haiku-4-5 · crag · bm25 | {'1': 0.67, '2': 0.67, '3': 0.67} |
| claude:claude-haiku-4-5 · frames · bm25 | {'1': 0.26, '2': 0.28, '3': 0.23} |
| claude:claude-haiku-4-5 · hotpotqa · dense | {'1': 0.81} |
| claude:claude-haiku-4-5 · musique · dense | {'1': 0.72, '2': 0.74} |
| claude:claude-haiku-4-5 · hotpotqa · bm25 | {'1': 0.78, '2': 0.81, '3': 0.79} |
| claude:claude-haiku-4-5 · musique · bm25 | {'1': 0.72, '2': 0.75, '3': 0.74} |
| openai:gpt-4o-mini · crag · bm25 | {'1': 0.67, '2': 0.67, '3': 0.67} |
| openai:gpt-4o-mini · frames · bm25 | {'1': 0.2, '2': 0.24, '3': 0.24} |
| openai:gpt-4o-mini · frames · dense | {'1': 0.2, '2': 0.18} |
| openai:gpt-4o-mini · hotpotqa · dense | {'1': 0.69, '2': 0.69, '3': 0.71} |
| openai:gpt-4o-mini · musique · bm25 | {'1': 0.38, '2': 0.34, '3': 0.39} |
| openai:gpt-4o-mini · musique · dense | {'1': 0.5, '2': 0.53, '3': 0.56} |
| openai:gpt-4o-mini · hotpotqa · bm25 | {'1': 0.68, '2': 0.74, '3': 0.73} |
| openai:qwen2.5:7b · musique · bm25 | {'1': 0.25, '2': 0.2, '3': 0.17} |

## Rescore (stage-criterion) — from persisted raw diagnoses

| backbone·dataset·retr | PA hop-exact (deep) | PA stage (deep) |
|---|---|---|
| claude:claude-haiku-4-5·crag·bm25 | 0.25 | 0.25 |
| claude:claude-haiku-4-5·frames·bm25 | 0.11 | 0.96 |
| claude:claude-haiku-4-5·musique·dense | 0.35 | 0.88 |
| openai:gpt-4o-mini·frames·dense | 0.20 | 0.87 |
| openai:gpt-4o-mini·hotpotqa·dense | 0.50 | 0.71 |
| openai:gpt-4o-mini·musique·bm25 | 0.40 | 0.81 |
| openai:gpt-4o-mini·musique·dense | 0.53 | 0.72 |

(Stage-criterion is looser than exact-hop; PA does better on stage but the depth≥2 gap to baselines persists.)

---

## 2026-07-07 — Content-corruption arm (certified spans, deterministic generation eval)

New intervention `inject_corrupted_evidence`: the key fact inside the injected hop's
docs is flipped to a certified-wrong value (gold-answer span → in-domain distractor,
or numeric perturbation; bridge-entity / salient-entity fallbacks). Docs stay
topically intact — the deployment-realistic fault (stale index, poisoned doc).
Conditions: {gpt-4o-mini, claude-haiku-4-5} × {hotpotqa, musique}, bm25, n=40,
hops 1–3, strict defaults, **cross-family judges** (A8 control), 5 diagnosers incl.
suffix_regen (first run — A1). Files: `identifiability_*_bm25_corruption.json`.

### Hop-exact accuracy on content faults (pooled over 4 conditions, failed cases, 95% bootstrap CI)

| diagnoser | hop1 | hop2 | hop3 |
|---|---|---|---|
| doctor_rag (coverage) | 1.00 [1.00,1.00] n=44 | 0.00 [0.00,0.00] n=18 | 0.00 n=3 |
| llm_judge (cross-family) | 0.59 [0.43,0.73] | 0.89 [0.72,1.00] | 0.33 n=3 |
| propagation_aware | 0.89 [0.80,0.98] | 0.67 [0.44,0.89] | 0.00 n=3 |
| suffix_regen | 0.91 [0.82,0.98] | 0.11 [0.00,0.28] | 0.00 n=3 |

- **Coverage collapse is total at depth ≥ 2 on content faults** (0.00, tight CI) —
  sharper than the structural arm (0.30 mean). Hop-1 = 1.00 remains partly gifted
  (answer_fact removes the gold string → coverage gate fires; defaults gift the rest).
- **PA decisively beats coverage at hop 2** (0.67 vs 0.00) — the C3 signal the
  structural arm lacked. But the best post-hoc *envelope* (judge included) still
  edges PA at depth ≥ 2 in discordant pairs (6 vs 1; n too small for significance).
- **llm_judge's hop-2 score is prior-inflated**: its predicted-hop distribution at
  depth 2 is almost all "hop 2" (12/14) — a mid-trace prior, not localization.
  Deflates the earlier "judge is depth-robust" bonus finding.
- **suffix_regen mechanism finding**: SR collapses at hop 2 (0.11) *because* suffix
  regeneration re-retrieves from the clean corpus, un-corrupting the deeper fault
  while probing hop 1 → systematic under-localization. PA (frozen other hops) holds.
  Conversely SR wins the bridge_entity slice (1.00 vs PA 0.73, n=11) where
  downstream hops derive from the corrupted bridge and must be regenerated.
  → The two probes are complementary; cascade diagnoser (B4) directly motivated.

### Deterministic absorption (all injected cases, zero-LLM eval)

| depth | absorbed | resisted | derailed | n |
|---|---|---|---|---|
| hop1 | 0.15 | 0.58 | 0.26 | 106 |
| hop2 | 0.09 | 0.74 | 0.18 | 68 |
| hop3 | 0.00 | 0.85 | 0.15 | 20 |

- **Verbatim absorption is answer-shaped**: 20/22 absorbed cases are `answer_fact`
  corruptions; bridge/salient corruptions derail or are resisted, never absorbed
  (0/40 salient). Backbone-stable (hop-1 absorbed: 0.18/0.19/0.11/0.11 across the
  4 conditions).
- **Query contamination ≈ 0** (3/129): corrupted facts propagate through reasoning
  /answer generation, not through the agent's re-query text.
- Recovery on content faults (0.54–0.85) is far higher than structural (0.00–0.10):
  the trajectory is corrupted but the *corpus stays clean*, so re-retrieval heals.
  Motivates a persistent-corruption variant (corrupt the corpus copy) as the
  stale-index model — expected to raise absorption and depress recovery.

Caveat: failed-n at depth ≥ 2 is small (18/3) because recovery is high — depth-2/3
content-fault comparisons are indicative, not significant. Scale n or add the
persistent variant before headline claims.
