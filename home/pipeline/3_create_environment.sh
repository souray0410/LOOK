#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--project-root" ]]; then PROJECT_ROOT="${ARGS[$((i+1))]}"; fi
done
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
TORCH_INDEX_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torch_index_url"])' "$PROJECT_ROOT/project.json")"
TORCH_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torch_version"])' "$PROJECT_ROOT/project.json")"
TORCHVISION_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torchvision_version"])' "$PROJECT_ROOT/project.json")"

if (( ${#ARGS[@]} )); then set -- "${ARGS[@]}"; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL=$2; shift 2 ;;
    --torch-version) TORCH_VERSION=$2; shift 2 ;;
    --torchvision-version) TORCHVISION_VERSION=$2; shift 2 ;;
    -h|--help)
      echo "Usage: $0 --torch-index-url URL [--project-root PATH] [--python BIN]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ENV_ROOT="$PROJECT_ROOT/tool/environment"
VENV="$ENV_ROOT/.venv"
cd "$PROJECT_ROOT"
test -f "$ENV_ROOT/requirements-no-torch.txt"
test -f tool/MHD_Project/MHD_Framework_V3.py
"$PYTHON" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(f"Python 3.10-3.13 is required, found {sys.version.split()[0]}")
PY
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV" || "$PYTHON" -m venv --without-pip "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py"
  "$VENV/bin/python" "$VENV/get-pip.py"
  rm "$VENV/get-pip.py"
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/pip" install --index-url "$TORCH_INDEX_URL" \
  "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
"$VENV/bin/pip" install -r "$ENV_ROOT/requirements-no-torch.txt"
"$VENV/bin/pip" install --no-deps -e "$PROJECT_ROOT"
"$VENV/bin/python" -m pip freeze > "$ENV_ROOT/environment-lock.txt"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
import numpy, pandas, scipy, sklearn, torch, torchvision, look_core, MHD_Project
print("Environment import check: PASS")
print("Python:", __import__("sys").version.split()[0])
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
PY
