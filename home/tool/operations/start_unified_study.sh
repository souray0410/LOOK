#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPERATIONS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION=look-unified-20260904-191807
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
export PYTHONPATH="$PROJECT_ROOT/tool${PYTHONPATH:+:$PYTHONPATH}"
RUNS_ROOT="$("$PYTHON" -c 'import argparse;from look_core.cli import add_runtime_arguments,resolve_runtime_arguments;p=argparse.ArgumentParser();add_runtime_arguments(p);a,_=p.parse_known_args();print(resolve_runtime_arguments(a).runs_root)' --project-root "$PROJECT_ROOT" "${ARGS[@]}")"
mkdir -p "$RUNS_ROOT/logs"
LOG="$RUNS_ROOT/logs/${SESSION}_$(date -u +%Y%m%dT%H%M%SZ).log"
NOFILE_LIMIT="${LOOK_NOFILE_LIMIT:-65536}"
# Apply the limit inside the tmux child, including an existing server.
# Pin transport inside the tmux child rather than inheriting stale server settings.
# Keep watchdog monitoring and its production timeout unchanged.
COMMAND=(bash -c 'ulimit -Sn "$1" || exit; shift; exec "$@"' _ "$NOFILE_LIMIT" env "PYTHONPATH=$PYTHONPATH" "NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}" "NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0}" "NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}" "NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE:-0}" "TORCH_FR_BUFFER_SIZE=${TORCH_FR_BUFFER_SIZE:-2000}" "TORCH_NCCL_DUMP_ON_TIMEOUT=${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}" "OMP_NUM_THREADS=$OMP_NUM_THREADS" "MKL_NUM_THREADS=$MKL_NUM_THREADS" "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS" "$PYTHON" -u "$OPERATIONS_ROOT/run_unified_with_idle_graph_parking.py" --runtime-audit-root "$RUNS_ROOT/maintenance/idle_graph_parking" --project-root "$PROJECT_ROOT" "${ARGS[@]}")
printf -v CMD '%q ' "${COMMAND[@]}"
printf -v LOG_QUOTED '%q' "$LOG"
tmux new-session -d -s "$SESSION" "exec $CMD > $LOG_QUOTED 2>&1"
sleep 1
tmux has-session -t "$SESSION"
echo "session=$SESSION log=$LOG nofile_limit=$NOFILE_LIMIT"
