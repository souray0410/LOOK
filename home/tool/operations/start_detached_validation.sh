#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
SESSION="look-validation"
GPUS="0,1"
STUDY_MODE="baseline-selection"
CACHE_WORKERS=64
WARM_CACHE=1
PYTHON=""
LOG_DIR=""
LOG_FILE=""
WORKER=0
EXTRA_ARGS=()
RUNTIME_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --data-root|--dataset-root|--image-root|--cohort-root|--labels-csv|--natural-labels-csv|--preprocess-cache-root|--cache-root|--runs-root)
      RUNTIME_ARGS+=("$1" "$2"); shift 2 ;;
    --session) SESSION=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --mode) STUDY_MODE=$2; shift 2 ;;
    --cache-workers) CACHE_WORKERS=$2; shift 2 ;;
    --skip-cache-warm) WARM_CACHE=0; shift ;;
    --python) PYTHON=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --log-file) LOG_FILE=$2; shift 2 ;;
    --worker) WORKER=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help)
      echo "Usage: $0 [--project-root PATH] [--gpus 0,1] [--mode baseline-selection|full-study] [--cache-workers N] [--skip-cache-warm] [--session NAME] [-- EXTRA_STEP19_ARGS]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
CONFIG="$PROJECT_ROOT/project.json"
[[ -f "$CONFIG" ]]
if [[ -z "$PYTHON" ]]; then
  VENV_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["venv_path"])' "$CONFIG")"
  PYTHON="$VENV_PATH/bin/python"
fi
[[ -x "$PYTHON" ]]
if [[ -z "$LOG_DIR" ]]; then
  DATA_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["data_root"])' "$CONFIG")"
  LOG_DIR="$DATA_ROOT/runs/logs"
fi
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/detached_validation_status.env"

write_status() {
  local state=$1 exit_code=${2:-}
  local temporary="$STATUS_FILE.partial.$$"
  {
    printf 'state=%s\n' "$state"
    printf 'session=%s\n' "$SESSION"
    printf 'gpus=%s\n' "$GPUS"
    printf 'study_mode=%s\n' "$STUDY_MODE"
    printf 'project_root=%s\n' "$PROJECT_ROOT"
    printf 'log=%s\n' "$LOG_FILE"
    printf 'updated_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    [[ -n "$exit_code" ]] && printf 'exit_code=%s\n' "$exit_code"
  } > "$temporary"
  mv "$temporary" "$STATUS_FILE"
}

if [[ "$WORKER" -eq 1 ]]; then
  [[ -n "$LOG_FILE" ]]
  handle_signal() {
    local exit_code=$1
    trap - INT TERM
    write_status interrupted "$exit_code"
    exit "$exit_code"
  }
  trap 'handle_signal 130' INT
  trap 'handle_signal 143' TERM
  if [[ "$WARM_CACHE" -eq 1 ]]; then
    write_status warming_cache
    set +e
    set -o pipefail
    "$PYTHON" "$PROJECT_ROOT/tool/operations/warm_preprocess_cache.py" \
      --project-root "$PROJECT_ROOT" "${RUNTIME_ARGS[@]}" \
      --workers "$CACHE_WORKERS" 2>&1 | tee -a "$LOG_FILE"
    CACHE_EXIT_CODE=${PIPESTATUS[0]}
    set -e
    if [[ "$CACHE_EXIT_CODE" -ne 0 ]]; then
      write_status failed "$CACHE_EXIT_CODE"
      exit "$CACHE_EXIT_CODE"
    fi
  fi
  write_status running
  set +e
  set -o pipefail
  "$PYTHON" "$PROJECT_ROOT/pipeline/19_run_study_sweep.py" \
    --project-root "$PROJECT_ROOT" "${RUNTIME_ARGS[@]}" \
    --mode "$STUDY_MODE" --phase validation --gpus "$GPUS" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  EXIT_CODE=${PIPESTATUS[0]}
  set -e
  if [[ "$EXIT_CODE" -eq 0 ]]; then
    write_status complete "$EXIT_CODE"
  else
    write_status failed "$EXIT_CODE"
  fi
  exit "$EXIT_CODE"
fi

command -v tmux >/dev/null
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already running: $SESSION" >&2
  exit 1
fi
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/${SESSION}_${TIMESTAMP}.log"
COMMAND=(
  "$SCRIPT_PATH" --worker --project-root "$PROJECT_ROOT" --session "$SESSION"
  --gpus "$GPUS" --mode "$STUDY_MODE" --cache-workers "$CACHE_WORKERS" --python "$PYTHON"
  --log-dir "$LOG_DIR" --log-file "$LOG_FILE"
)
COMMAND+=("${RUNTIME_ARGS[@]}")
if [[ "$WARM_CACHE" -eq 0 ]]; then
  COMMAND+=(--skip-cache-warm)
fi
COMMAND+=(
  --
  "${EXTRA_ARGS[@]}"
)
printf -v ESCAPED_COMMAND '%q ' "${COMMAND[@]}"
tmux new-session -d -s "$SESSION" "$ESCAPED_COMMAND"
sleep 1
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Detached session failed to start; inspect $LOG_FILE" >&2
  exit 1
fi
echo "Started detached validation session: $SESSION"
echo "Log: $LOG_FILE"
echo "Attach: tmux attach -t $SESSION"
