#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=python3
VENV_PATH=""
KERNEL_NAME="look"
KERNEL_DISPLAY_NAME="LOOK"
ARGS=("$@")
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--project-root" ]]; then PROJECT_ROOT="${ARGS[$((i+1))]}"; fi
done
PROJECT_ROOT="$(cd "$PROJECT_ROOT" && pwd)"
TORCH_INDEX_URL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torch_index_url"])' "$PROJECT_ROOT/project.json")"
TORCH_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torch_version"])' "$PROJECT_ROOT/project.json")"
TORCHVISION_VERSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"]["torchvision_version"])' "$PROJECT_ROOT/project.json")"
VENV_PATH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["deployment_defaults"].get("venv_path", ""))' "$PROJECT_ROOT/project.json")"

if (( ${#ARGS[@]} )); then set -- "${ARGS[@]}"; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) PROJECT_ROOT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --venv-path) VENV_PATH=$2; shift 2 ;;
    --kernel-name) KERNEL_NAME=$2; shift 2 ;;
    --kernel-display-name) KERNEL_DISPLAY_NAME=$2; shift 2 ;;
    --torch-index-url) TORCH_INDEX_URL=$2; shift 2 ;;
    --torch-version) TORCH_VERSION=$2; shift 2 ;;
    --torchvision-version) TORCHVISION_VERSION=$2; shift 2 ;;
    -h|--help)
      echo "Usage: $0 [--project-root PATH] [--python BIN] [--venv-path PATH] [--kernel-name NAME] [--kernel-display-name NAME] [--torch-index-url URL]"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

ENV_ROOT="$PROJECT_ROOT/tool/environment"
if [[ -z "$VENV_PATH" ]]; then
  VENV="$ENV_ROOT/.venv"
elif [[ "$VENV_PATH" = /* ]]; then
  VENV="$VENV_PATH"
else
  VENV="$PROJECT_ROOT/$VENV_PATH"
fi
cd "$PROJECT_ROOT"
test -f "$ENV_ROOT/requirements-no-torch.txt"
test -f tool/MHD_Project/MHD_Framework_V4.py
"$PYTHON" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 13)):
    raise SystemExit(f"Python 3.10-3.13 is required, found {sys.version.split()[0]}")
PY
if [[ -x "$VENV/bin/python" ]] && ! "$VENV/bin/python" - <<'PY'
import sys
raise SystemExit(0 if sys.prefix != sys.base_prefix else 1)
PY
then
  echo "Existing environment is invalid; rebuilding only the declared venv: $VENV" >&2
  "$PYTHON" -m venv --clear "$VENV" || "$PYTHON" -m venv --clear --without-pip "$VENV"
elif [[ ! -x "$VENV/bin/python" ]]; then
  echo "Creating or completing virtual environment: $VENV"
  "$PYTHON" -m venv "$VENV" || "$PYTHON" -m venv --without-pip "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  curl --fail --location --silent --show-error https://bootstrap.pypa.io/get-pip.py -o "$VENV/get-pip.py"
  "$VENV/bin/python" "$VENV/get-pip.py"
  rm "$VENV/get-pip.py"
fi
DEPENDENCIES_READY=0
if "$VENV/bin/python" - "$ENV_ROOT/requirements-no-torch.txt" "$TORCH_VERSION" "$TORCHVISION_VERSION" <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys

requirements, expected_torch, expected_torchvision = sys.argv[1:]
expected = {}
for line in Path(requirements).read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        package, package_version = line.split("==", 1)
        expected[package] = package_version
expected.update({"torch": expected_torch, "torchvision": expected_torchvision})
try:
    valid = all(version(package).split("+")[0] == wanted for package, wanted in expected.items())
except PackageNotFoundError:
    valid = False
raise SystemExit(0 if valid else 1)
PY
then
  DEPENDENCIES_READY=1
  echo "Reusing complete environment: $VENV"
fi
if [[ "$DEPENDENCIES_READY" -eq 0 ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$VENV/bin/pip" install --index-url "$TORCH_INDEX_URL" \
    "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION"
  "$VENV/bin/pip" install -r "$ENV_ROOT/requirements-no-torch.txt"
fi
"$VENV/bin/pip" install --no-deps -e "$PROJECT_ROOT"
"$VENV/bin/python" -m pip freeze > "$VENV/.look_environment_lock.txt"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - "$TORCH_VERSION" "$TORCHVISION_VERSION" <<'PY'
import sys
import numpy, pandas, scipy, sklearn, torch, torchvision, look_core, MHD_Project
expected_torch, expected_torchvision = sys.argv[1:]
if torch.__version__.split("+")[0] != expected_torch:
    raise SystemExit(f"Expected torch {expected_torch}, found {torch.__version__}")
if torchvision.__version__.split("+")[0] != expected_torchvision:
    raise SystemExit(f"Expected torchvision {expected_torchvision}, found {torchvision.__version__}")
print("Environment import check: PASS")
print("Executable:", sys.executable)
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
PY

"$VENV/bin/python" -m ipykernel install --user \
  --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"
"$VENV/bin/python" - "$KERNEL_NAME" "$VENV/bin/python" <<'PY'
import json
import sys
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager

kernel_name, expected_python = sys.argv[1:]
resource_dir = Path(KernelSpecManager().get_kernel_spec(kernel_name).resource_dir)
kernel_json = resource_dir / "kernel.json"
payload = json.loads(kernel_json.read_text(encoding="utf-8"))
observed = Path(payload["argv"][0]).absolute()
expected = Path(expected_python).absolute()
if observed != expected:
    raise SystemExit(f"Kernel uses {observed}, expected {expected}")
print(f"Kernel registration check: PASS ({kernel_name} -> {observed})")
PY

"$VENV/bin/python" - "$VENV/.look_environment.json" "$KERNEL_NAME" "$KERNEL_DISPLAY_NAME" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

destination = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "kernel_name": sys.argv[2],
    "kernel_display_name": sys.argv[3],
}
descriptor, temporary_name = tempfile.mkstemp(prefix=".look_environment.", suffix=".partial", dir=destination.parent)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    Path(temporary_name).replace(destination)
finally:
    Path(temporary_name).unlink(missing_ok=True)
PY
