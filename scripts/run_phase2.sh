#!/usr/bin/env bash
# Phase-2 orchestration (network-outage resilient):
#   A) Complete/repair the original BM25 matrix (Claude frames+crag lost to credit
#      exhaustion; gpt-4o-mini & llama3.1 musique lost to a since-fixed filename
#      collision).
#   B) Strengthened C3: dense retriever + higher probe budget + larger n, tagged
#      'dense_b4' so it doesn't overwrite the bm25 results.
#
# Resilience:
#   * Every network-dependent run WAITS for connectivity before starting, so a
#     travel/Wi-Fi outage PAUSES the matrix instead of fail-skipping queued runs.
#   * On failure it waits for the network and RETRIES once (covers a run that was
#     in flight when the network dropped).
#   * --resume + per-depth checkpointing => nothing already finished is redone.
#   * Ollama runs are local (no network wait).
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python}"
CLAUDE_N="${CLAUDE_N:-50}"
OPENAI_N="${OPENAI_N:-60}"
OLLAMA_N="${OLLAMA_N:-30}"
BUDGET="${BUDGET:-4}"
HOPS="${HOPS:-1 2 3}"

mkdir -p results/logs
MASTER=results/logs/run_phase2_master.log
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

net_up() {
  curl -s --max-time 8 -o /dev/null https://api.anthropic.com 2>/dev/null && return 0
  curl -s --max-time 8 -o /dev/null https://api.openai.com 2>/dev/null && return 0
  return 1
}

wait_for_network() {
  local waited=0
  while ! net_up; do
    log "network DOWN — pausing (${waited}s); will resume when reachable"
    sleep 30
    waited=$((waited + 30))
  done
  [ "$waited" -gt 0 ] && log "network back after ${waited}s — resuming"
  return 0
}

run() {  # run <logname> <cmd...>  — returns the command's exit code
  local name="$1"; shift
  local start=$SECONDS
  log "START $name"
  if "$@" >"results/logs/$name.log" 2>&1; then
    log "DONE  $name  ($((SECONDS-start))s)"
    return 0
  fi
  local rc=$?
  log "FAIL  $name  (exit $rc, see results/logs/$name.log)"
  return $rc
}

run_llm() {  # like run, but waits for network first + retries once after a wait
  local name="$1"; shift
  wait_for_network
  if ! run "$name" "$@"; then
    log "RETRY $name after network wait"
    wait_for_network
    run "$name" "$@" || true
  fi
}

log "================ run_phase2 START (network-resilient) ================"
log "PY=$PY CLAUDE_N=$CLAUDE_N OPENAI_N=$OPENAI_N OLLAMA_N=$OLLAMA_N BUDGET=$BUDGET HOPS='$HOPS'"

# Trimmed: OpenAI dense musique+hotpotqa are done; OpenAI frames dense is left at
# depths 1-2 already on disk (shows the C3-loses pattern). Claude dense on the two
# anchors gives the 2-backbone confirmation. Frames-dense (both) + Ollama dense are
# skipped — the C3 verdict is settled across 3 backbones and more compute won't move it.
C3="musique hotpotqa"          # OpenAI loop (both already complete -> --resume skips)
C3_CLAUDE="musique hotpotqa"   # Claude anchors — the only remaining new work

# ---------------------------------------------------------------------------
# Phase A — complete/repair the BM25 matrix (comparable to the first run)
# ---------------------------------------------------------------------------
run_llm A_claude_frames $PY scripts/run_identifiability.py --dataset frames \
  --frames-fetch-passages --provider claude --model claude-haiku-4-5 \
  --retriever bm25 --max-samples 40 --hops 1 2 3 --propagation-budget 2 --resume
run_llm A_claude_crag $PY scripts/run_identifiability.py --dataset crag \
  --provider claude --model claude-haiku-4-5 \
  --retriever bm25 --max-samples 40 --hops 1 2 3 --propagation-budget 2 --resume
run_llm A_openai_musique $PY scripts/run_identifiability.py --dataset musique \
  --provider openai --model gpt-4o-mini \
  --retriever bm25 --max-samples 40 --hops 1 2 3 --propagation-budget 2 --resume

# ---------------------------------------------------------------------------
# Phase B — strengthened C3 (dense, budget 4, larger n). OpenAI first (cheap),
# then Claude (credit-permitting).
# ---------------------------------------------------------------------------
for d in $C3; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run_llm B_openai_${d}_dense $PY scripts/run_identifiability.py --dataset "$d" $EXTRA \
    --provider openai --model gpt-4o-mini --retriever dense \
    --max-samples "$OPENAI_N" --hops $HOPS --propagation-budget "$BUDGET" --tag dense_b4 --resume
done
for d in $C3_CLAUDE; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run_llm B_claude_${d}_dense $PY scripts/run_identifiability.py --dataset "$d" $EXTRA \
    --provider claude --model claude-haiku-4-5 --retriever dense \
    --max-samples "$CLAUDE_N" --hops $HOPS --propagation-budget "$BUDGET" --tag dense_b4 --resume
done

# ---------------------------------------------------------------------------
# Ollama dense + Claude frames dense intentionally SKIPPED — the C3 verdict is
# settled across 3 backbones + dense; these are pure (expensive) confirmation.
# Re-enable later with the original scripts/run_phase2.sh git history if wanted.
# ---------------------------------------------------------------------------

log "================ run_phase2 COMPLETE ================"
