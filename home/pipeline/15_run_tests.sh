#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    -h|--help) echo "Usage: $0 [--project-root PATH] [--gpu INDEX]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$PROJECT_ROOT"
tool/environment/.venv/bin/python -m pytest -q tool/tests
