#!/usr/bin/env bash
# Re-run all Claude identifiability conditions with strict depth defaults.
# Runs sequentially to avoid Anthropic rate limits.
# Each run starts fresh (old pre-fix files don't match strict config).
set -u
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
N="${CLAUDE_N:-40}"
N_DENSE="${CLAUDE_N_DENSE:-50}"
BUDGET="${BUDGET:-2}"
BUDGET_DENSE="${BUDGET_DENSE:-4}"
HOPS="${HOPS:-1 2 3}"

mkdir -p results/logs
MASTER=results/logs/run_strict_claude_master.log
ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

net_up() {
  curl -s --max-time 8 -o /dev/null https://api.anthropic.com 2>/dev/null
}
wait_net() {
  local w=0
  while ! net_up; do
    log "network down — waiting (${w}s)"; sleep 30; w=$((w+30))
  done
  [ "$w" -gt 0 ] && log "network back after ${w}s"
}

run_llm() {
  local name="$1"; shift
  local start=$SECONDS
  wait_net
  log "START $name"
  if "$@" >"results/logs/${name}.log" 2>&1; then
    log "DONE  $name  ($((SECONDS-start))s)"
  else
    local rc=$?
    log "FAIL  $name (exit $rc) — retrying after network wait"
    wait_net
    if "$@" >>"results/logs/${name}.log" 2>&1; then
      log "DONE  $name (retry ok,  $((SECONDS-start))s)"
    else
      log "FAIL  $name (retry also failed)"
    fi
  fi
}

log "======== run_strict_claude START  N=$N N_DENSE=$N_DENSE BUDGET=$BUDGET BUDGET_DENSE=$BUDGET_DENSE HOPS='$HOPS' ========"

# BM25 matrix
for d in musique hotpotqa frames crag; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run_llm strict_claude_${d}_bm25 $PY scripts/run_identifiability.py \
    --dataset "$d" $EXTRA \
    --provider claude --model claude-haiku-4-5 \
    --retriever bm25 --max-samples "$N" --hops $HOPS \
    --propagation-budget "$BUDGET"
done

# Dense C3 anchors
for d in musique hotpotqa; do
  run_llm strict_claude_${d}_dense $PY scripts/run_identifiability.py \
    --dataset "$d" \
    --provider claude --model claude-haiku-4-5 \
    --retriever dense --max-samples "$N_DENSE" --hops $HOPS \
    --propagation-budget "$BUDGET_DENSE" --tag dense_b4
done

log "======== run_strict_claude COMPLETE ========"
