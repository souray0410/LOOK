#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--project-root" ]]; then PROJECT_ROOT="${ARGS[$((i+1))]}"; fi
done
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
SOURCE_UUID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_source_uuid"])' "$PROJECT_ROOT/project.json")"
SOURCE_MOUNT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_source_mount"])' "$PROJECT_ROOT/project.json")"
SOURCE_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_source_root"])' "$PROJECT_ROOT/project.json")"
LABEL_UUID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_label_uuid"])' "$PROJECT_ROOT/project.json")"
LABEL_MOUNT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_label_mount"])' "$PROJECT_ROOT/project.json")"
LABEL_ROOT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["ukb_label_root"])' "$PROJECT_ROOT/project.json")"
DATASET_ROOT="$(python3 -c 'import json,sys,os; print(os.path.join(json.load(open(sys.argv[1]))["deployment_defaults"]["data_root"], "dataset"))' "$PROJECT_ROOT/project.json")"

if (( ${#ARGS[@]} )); then set -- "${ARGS[@]}"; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --source-uuid) SOURCE_UUID=$2; shift 2 ;;
    --source-mount) SOURCE_MOUNT=$2; shift 2 ;;
    --source-root) SOURCE_ROOT=$2; shift 2 ;;
    --label-uuid) LABEL_UUID=$2; shift 2 ;;
    --label-mount) LABEL_MOUNT=$2; shift 2 ;;
    --label-root) LABEL_ROOT=$2; shift 2 ;;
    --dataset-root) DATASET_ROOT=$2; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--project-root PATH] [--source-mount PATH] [--source-root PATH] [--label-mount PATH] [--label-root PATH] [--dataset-root PATH] [--source-uuid UUID] [--label-uuid UUID]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

for value in SOURCE_MOUNT SOURCE_ROOT LABEL_MOUNT LABEL_ROOT DATASET_ROOT; do
  [[ -n "${!value}" ]] || { echo "Missing required path: $value" >&2; exit 2; }
done

verify_read_only_mount() {
  local mount_point=$1
  local expected_uuid=$2
  findmnt -rn "$mount_point" >/dev/null || { echo "Not mounted: $mount_point" >&2; exit 1; }
  local options
  options=$(findmnt -no OPTIONS "$mount_point")
  [[ ",$options," == *,ro,* ]] || { echo "Source mount is not read-only: $mount_point" >&2; exit 1; }
  if [[ -n "$expected_uuid" ]]; then
    local observed_uuid
    observed_uuid=$(findmnt -no UUID "$mount_point")
    [[ "$observed_uuid" == "$expected_uuid" ]] || { echo "Unexpected UUID at $mount_point" >&2; exit 1; }
  fi
}

verify_read_only_mount "$SOURCE_MOUNT" "$SOURCE_UUID"
verify_read_only_mount "$LABEL_MOUNT" "$LABEL_UUID"
for required in 21015.zip 21016.zip 21017 21018; do
  [[ -e "$SOURCE_ROOT/$required" ]] || { echo "Missing source item: $SOURCE_ROOT/$required" >&2; exit 1; }
done
for required in ukb670300.csv ukb679947.csv; do
  [[ -f "$LABEL_ROOT/$required" ]] || { echo "Missing label table: $LABEL_ROOT/$required" >&2; exit 1; }
done
mkdir -p "$DATASET_ROOT"
[[ -w "$DATASET_ROOT" ]] || { echo "Dataset root is not writable: $DATASET_ROOT" >&2; exit 1; }
echo "Source: $SOURCE_ROOT ($(findmnt -no OPTIONS "$SOURCE_MOUNT"))"
echo "Labels: $LABEL_ROOT ($(findmnt -no OPTIONS "$LABEL_MOUNT"))"
echo "Dataset: $DATASET_ROOT"
df -h "$DATASET_ROOT"
