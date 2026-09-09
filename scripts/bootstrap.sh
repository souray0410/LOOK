#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
"$PYTHON" -c 'import sys; assert sys.prefix != sys.base_prefix, "Activate a dedicated project virtual environment first"'
git -C "$ROOT" submodule update --init --recursive
"$PYTHON" "$ROOT/scripts/manage.py" check
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
"$PYTHON" -m pip install --no-deps -e "$ROOT/third_party/MHD_Framework"
"$PYTHON" -m pip install --no-deps -e "$ROOT"
"$PYTHON" -m pip check
"$PYTHON" "$ROOT/scripts/manage.py" check
"$PYTHON" "$ROOT/scripts/manage.py" verify-env
