#!/usr/bin/env bash
# Unattended orchestration of the full experiment matrix.
#
# Continues past per-run failures (no `set -e`), logs every run to
# results/logs/<name>.log, and writes a master timeline to
# results/logs/run_all_master.log. Most-valuable runs go first (free controls,
# then Claude primary, then OpenAI, then the open model) so partial completion
# still yields usable results.
#
# Override the interpreter / sample sizes via env vars, e.g.:
#   PY=/path/to/python CLAUDE_N=60 bash scripts/run_all.sh
set -u

cd "$(dirname "$0")/.."

PY="${PY:-python}"
MOCK_N="${MOCK_N:-100}"
CLAUDE_N="${CLAUDE_N:-40}"
OPENAI_N="${OPENAI_N:-40}"
OLLAMA_N="${OLLAMA_N:-15}"
HOPS="${HOPS:-1 2 3}"
BUDGET="${BUDGET:-2}"

mkdir -p results/logs
MASTER=results/logs/run_all_master.log

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*" | tee -a "$MASTER"; }

run() {  # run <logname> <cmd...>
  local name="$1"; shift
  local start=$SECONDS
  log "START $name"
  if "$@" >"results/logs/$name.log" 2>&1; then
    log "DONE  $name  ($((SECONDS-start))s)"
  else
    local rc=$?
    log "FAIL  $name  (exit $rc, see results/logs/$name.log)"
  fi
}

log "================ run_all START ================"
log "PY=$PY  CLAUDE_N=$CLAUDE_N OPENAI_N=$OPENAI_N OLLAMA_N=$OLLAMA_N HOPS='$HOPS' BUDGET=$BUDGET"

DATASETS="musique hotpotqa frames crag"

# ---------------------------------------------------------------------------
# Phase 1 — warm the FRAMES Wikipedia passage cache (zero LLM tokens)
# ---------------------------------------------------------------------------
run warm_frames $PY scripts/run_identifiability.py --dataset frames \
  --frames-fetch-passages --provider mock --retriever bm25 \
  --max-samples "$MOCK_N" --hops 1

# ---------------------------------------------------------------------------
# Phase 2 — free heuristic control grid + figures
# ---------------------------------------------------------------------------
run baseline_all  $PY scripts/run_baseline.py --all-conditions --max-samples 500
run ablation_all  $PY scripts/run_ablation.py --all-conditions --max-samples 200 --hops 1 2 3
run figures       $PY scripts/plot_amplification.py --input results/ablation_all.json

# Phase 2d — mock identifiability control (free) across all datasets
for d in $DATASETS; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run ident_mock_$d $PY scripts/run_identifiability.py --dataset "$d" $EXTRA \
    --provider mock --retriever bm25 --max-samples "$MOCK_N" --hops $HOPS
done

# ---------------------------------------------------------------------------
# Phase 3 — Claude headline (primary backbone)
# ---------------------------------------------------------------------------
for d in $DATASETS; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run ident_claude_$d $PY scripts/run_identifiability.py --dataset "$d" $EXTRA \
    --provider claude --model claude-haiku-4-5 --retriever bm25 \
    --max-samples "$CLAUDE_N" --hops $HOPS --propagation-budget "$BUDGET"
done

# ---------------------------------------------------------------------------
# Phase 4 — OpenAI (second backbone)
# ---------------------------------------------------------------------------
for d in $DATASETS; do
  EXTRA=""; [ "$d" = frames ] && EXTRA="--frames-fetch-passages"
  run ident_openai_$d $PY scripts/run_identifiability.py --dataset "$d" $EXTRA \
    --provider openai --model gpt-4o-mini --retriever bm25 \
    --max-samples "$OPENAI_N" --hops $HOPS --propagation-budget "$BUDGET"
done

# ---------------------------------------------------------------------------
# Phase 5 — open model via Ollama (best-effort, small n)
# ---------------------------------------------------------------------------
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  export OPENAI_BASE_URL=http://localhost:11434/v1
  export OPENAI_API_KEY=ollama
  for m in llama3.1:8b qwen2.5:7b; do
    safe=$(echo "$m" | tr ':/' '__')
    run ident_ollama_${safe}_musique $PY scripts/run_identifiability.py \
      --dataset musique --provider openai --model "$m" --retriever bm25 \
      --max-samples "$OLLAMA_N" --hops $HOPS --propagation-budget "$BUDGET"
  done
  unset OPENAI_BASE_URL OPENAI_API_KEY
else
  log "SKIP ollama — server not reachable at localhost:11434"
fi

log "================ run_all COMPLETE ================"
