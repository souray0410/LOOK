#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESSION="look-validation"
LINES=25
MODE="summary"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --session) SESSION=$2; shift 2 ;;
    --lines) LINES=$2; shift 2 ;;
    --follow) MODE=follow; shift ;;
    --attach) MODE=attach; shift ;;
    -h|--help)
      echo "Usage: $0 [--session NAME] [--lines N] [--follow|--attach]"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
CONFIG="$PROJECT_ROOT/project.json"
DATA_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["data_root"])' "$CONFIG")"
RUNS_ROOT="$DATA_ROOT/runs"
STATUS_FILE="$RUNS_ROOT/logs/detached_validation_status.env"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session=running name=$SESSION"
else
  echo "session=not-running name=$SESSION"
fi
[[ -f "$STATUS_FILE" ]] && cat "$STATUS_FILE"
PREPROCESS_CACHE="$DATA_ROOT/cache/preprocessed_pairs"
if [[ -d "$PREPROCESS_CACHE" ]]; then
  echo "preprocess_cache_files=$(find "$PREPROCESS_CACHE" -type f -name '*.npy' | wc -l)"
  echo "preprocess_cache_size=$(du -sh "$PREPROCESS_CACHE" | awk '{print $1}')"
fi

"$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["venv_path"])' "$CONFIG")/bin/python" - "$RUNS_ROOT" <<'PY'
import json
import sys
from pathlib import Path

runs = Path(sys.argv[1])
plans = sorted((runs / "sweeps").glob("validation__*/study_plan.json"), key=lambda p: p.stat().st_mtime)
if plans:
    plan_path = plans[-1]
    plan = json.loads(plan_path.read_text())
    progress_path = plan_path.with_name("progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    print(f"plan={plan_path.parent.name}")
    print(f"configurations={progress.get('completed_count', 0)}/{plan['configuration_count']}")
histories = sorted((runs / "backbones").glob("*/history.json"), key=lambda p: p.stat().st_mtime)
if histories:
    history_path = histories[-1]
    history = json.loads(history_path.read_text())
    if history:
        record = history[-1]
        validation = record.get("validation", {})
        print(f"latest_backbone={history_path.parent.name}")
        print(f"latest_epoch={record.get('epoch')}")
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_auroc_ovr"):
            print(f"validation_{key}={validation.get(key)}")
PY

echo "gpu:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

LOG_FILE=""
if [[ -f "$STATUS_FILE" ]]; then
  LOG_FILE="$(sed -n 's/^log=//p' "$STATUS_FILE" | tail -1)"
fi
if [[ -n "$LOG_FILE" && -f "$LOG_FILE" ]]; then
  echo "log=$LOG_FILE"
  if [[ "$MODE" == "follow" ]]; then
    exec tail -n "$LINES" -f "$LOG_FILE"
  fi
  echo "latest_log_lines:"
  tr '\r' '\n' < "$LOG_FILE" | tail -n "$LINES"
fi
if [[ "$MODE" == "attach" ]]; then
  exec tmux attach-session -t "$SESSION"
fi
