#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION=look-overnight
PYTHON=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --session) SESSION=$2; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Already running: $SESSION"; exit 0
fi
if [[ -z "$PYTHON" ]]; then
  PYTHON="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["deployment_defaults"]["venv_path"]+"/bin/python")' "$PROJECT_ROOT/project.json")"
fi
RUNS_ROOT="$("$PYTHON" -c 'import argparse;from look_core.cli import add_runtime_arguments,resolve_runtime_arguments;p=argparse.ArgumentParser();add_runtime_arguments(p);a,_=p.parse_known_args();print(resolve_runtime_arguments(a).runs_root)' --project-root "$PROJECT_ROOT" "${ARGS[@]}")"
mkdir -p "$RUNS_ROOT/logs"
LOG="$RUNS_ROOT/logs/${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).log"
COMMAND=("$PYTHON" -u "$PROJECT_ROOT/pipeline/33_run_overnight_validation.py" --project-root "$PROJECT_ROOT" "${ARGS[@]}")
printf -v CMD '%q ' "${COMMAND[@]}"
printf -v LOG_QUOTED '%q' "$LOG"
tmux new-session -d -s "$SESSION" "exec $CMD > $LOG_QUOTED 2>&1"
sleep 1
tmux has-session -t "$SESSION"
echo "session=$SESSION log=$LOG"
