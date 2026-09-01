#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS=0,1
PYTHON=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --gpus) GPUS=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    -h|--help) echo "Usage: $0 [--project-root PATH] [--gpus 0,1] [--python BIN]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$PROJECT_ROOT"
if [[ -z "$PYTHON" ]]; then
  VENV_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["venv_path"])' "$PROJECT_ROOT/project.json")"
  PYTHON="$VENV_PATH/bin/python"
fi
"$PYTHON" -m pytest -q tool/tests
