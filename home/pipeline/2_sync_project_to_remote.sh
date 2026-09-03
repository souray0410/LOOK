#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--local-root" ]]; then LOCAL_ROOT="${ARGS[$((i+1))]}"; fi
done
LOCAL_ROOT="$(cd "$LOCAL_ROOT" && pwd)"
HOST="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ssh_host"])' "$LOCAL_ROOT/project.json")"
REMOTE_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["project_root"])' "$LOCAL_ROOT/project.json")"
REMOTE_RUNS_ROOT="$(python3 -c 'import json,sys,os; print(os.path.join(json.load(open(sys.argv[1]))["deployment_defaults"]["data_root"], "runs"))' "$LOCAL_ROOT/project.json")"
MODE=push

usage() {
  echo "Usage: $0 --host HOST --remote-root PATH [--local-root PATH]" >&2
  echo "       $0 --pull-results --host HOST --remote-runs-root PATH [--local-root PATH]" >&2
}

if (( ${#ARGS[@]} )); then set -- "${ARGS[@]}"; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST=$2; shift 2 ;;
    --local-root) LOCAL_ROOT=$2; shift 2 ;;
    --remote-root) REMOTE_ROOT=$2; shift 2 ;;
    --remote-runs-root) REMOTE_RUNS_ROOT=$2; shift 2 ;;
    --pull-results) MODE=pull; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

test -f "$LOCAL_ROOT/project.json"
test -f "$LOCAL_ROOT/pipeline/27_UKB_LOOK_Glaucoma_ResNet50_MHD.ipynb"
test -f "$LOCAL_ROOT/tool/MHD_Project/MHD_Framework_V4.py"

if [[ "$MODE" == pull ]]; then
  mkdir -p "$LOCAL_ROOT/tool/research/results"
  rsync -av --prune-empty-dirs \
    --include '*/' --include '*.json' --include '*.csv' \
    --include '*.png' --include '*.pdf' --exclude '*' \
    "$HOST:$REMOTE_RUNS_ROOT/" "$LOCAL_ROOT/tool/research/results/"
  exit 0
fi

REMOTE_FAMILY_ROOT="$(dirname "$REMOTE_ROOT")"
LOCAL_FAMILY_ROOT="$(cd "$LOCAL_ROOT/.." && pwd)"
ssh "$HOST" "mkdir -p '$REMOTE_ROOT' '$REMOTE_FAMILY_ROOT'"
for RELEASE_DOCUMENT in README.md GENERAL_PROJECT_STANDARD.md; do
  if [[ -f "$LOCAL_FAMILY_ROOT/$RELEASE_DOCUMENT" ]]; then
    rsync -av \
      "$LOCAL_FAMILY_ROOT/$RELEASE_DOCUMENT" \
      "$HOST:$REMOTE_FAMILY_ROOT/$RELEASE_DOCUMENT"
  fi
done
rsync -av --delete \
  --exclude '.git/' --exclude 'tool/environment/.venv/' --exclude 'tool/*.egg-info/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.DS_Store' \
  "$LOCAL_ROOT/" "$HOST:$REMOTE_ROOT/"
