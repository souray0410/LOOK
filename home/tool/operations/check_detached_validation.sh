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
    if progress.get("current_index"):
        print(f"current_configuration={progress['current_index']}/{plan['configuration_count']}")
        print(f"current_backbone={progress.get('current_backbone_id')}")
        print(f"current_parameters={json.dumps(progress.get('current_configuration', {}), sort_keys=True)}")
histories = sorted((runs / "backbones").glob("*/history.json"), key=lambda p: p.stat().st_mtime)
if histories:
    history_path = histories[-1]
    history = json.loads(history_path.read_text())
    if history:
        record = history[-1]
        validation = record.get("validation", {})
        print(f"latest_backbone={history_path.parent.name}")
        print(f"epochs_recorded={len(history)}")
        print(f"latest_epoch={record.get('epoch')}")
        print(f"train_loss={record.get('train_loss')}")
        print(f"train_batch_accuracy={record.get('train_batch_accuracy')}")
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "macro_auroc_ovr"):
            print(f"validation_{key}={validation.get(key)}")
        print(f"validation_cross_entropy={validation.get('cross_entropy')}")
        print(f"validation_ece_15={validation.get('ece_15')}")
        print(f"validation_f1_per_class={validation.get('f1_per_class')}")
        print(f"validation_sensitivity_per_class={validation.get('sensitivity_per_class')}")
        print(f"validation_confusion_matrix={validation.get('confusion_matrix')}")
        best = max(history, key=lambda item: float(item.get('validation', {}).get('macro_f1', -1.0)))
        print(f"best_epoch_so_far={best.get('epoch')}")
        print(f"best_macro_f1_so_far={best.get('validation', {}).get('macro_f1')}")
leaderboards = sorted((runs / "sweeps").glob("validation__*/baseline_search_results.json"), key=lambda p: p.stat().st_mtime)
if leaderboards:
    rows = json.loads(leaderboards[-1].read_text())
    print("leaderboard_top5:")
    for rank, row in enumerate(rows[:5], start=1):
        print(
            f"  {rank}. fusion={row['fusion_position']} beta={row['class_balance_beta']} "
            f"lr={row['pretrained_lr']}/{row['new_layer_lr']} "
            f"dropout={row['classifier_dropout']} smoothing={row['label_smoothing']} "
            f"macro_f1={row.get('macro_f1')} balanced_accuracy={row.get('balanced_accuracy')} "
            f"macro_auroc={row.get('macro_auroc_ovr')} f1_per_class={row.get('f1_per_class')}"
        )
diagnostics = sorted((runs / "sweeps").glob("validation__*/baseline_search_diagnostics.json"), key=lambda p: p.stat().st_mtime)
if diagnostics:
    summary = json.loads(diagnostics[-1].read_text())
    print(f"diagnostic_completed={summary.get('completed_configurations')}")
    for axis, groups in summary.get("groups", {}).items():
        ordered = sorted(
            groups.items(),
            key=lambda item: -float(item[1].get("mean_macro_f1", -1.0)),
        )
        compact = [
            {
                "value": value,
                "n": values.get("completed"),
                "mean_macro_f1": values.get("mean_macro_f1"),
                "best_macro_f1": values.get("best", {}).get("macro_f1"),
            }
            for value, values in ordered
        ]
        print(f"diagnostic_{axis}={json.dumps(compact)}")
PY

echo "gpu:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "disk:"
df -h "$RUNS_ROOT" | tail -1

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
