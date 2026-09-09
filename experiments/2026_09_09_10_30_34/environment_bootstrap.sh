#!/bin/bash
set -euo pipefail
umask 077
module load python/3.11.0
unset PYTHONPATH
LOOK_ENV=/ibex/project/c2377/souray/home/mengh/environments/look_2026_09_09_10_30_34
LOOK_CODE=/ibex/project/c2377/souray/home/mengh/LOOK/2026_09_09_10_30_34
LOOK_RUN=/ibex/project/c2377/souray/data/mengh/LOOK/runs/2026_09_09_10_30_34
mkdir -p "$LOOK_RUN" "$(dirname "$LOOK_ENV")"
trap 'echo needs_attention > "$LOOK_RUN/environment_state.txt"' ERR
printf 'installing\n' > "$LOOK_RUN/environment_state.txt"
[ -x "$LOOK_ENV/bin/python" ] || python3 -m venv "$LOOK_ENV"
# Download cache can be shared; interpreter and installed packages cannot.
export PIP_CACHE_DIR=/ibex/project/c2377/souray/data/mengh/Radon_Bridge/package_cache
"$LOOK_ENV/bin/python" -m pip install --upgrade pip
"$LOOK_ENV/bin/python" -m pip install -r "$LOOK_CODE/tool/environment/requirements.txt"
"$LOOK_ENV/bin/python" -m pip install --no-deps -e "$LOOK_CODE"
"$LOOK_ENV/bin/python" -m pip check > "$LOOK_RUN/environment_dependency_check.txt"
"$LOOK_ENV/bin/python" -m pip freeze > "$LOOK_RUN/environment_versions.txt"
source "$LOOK_CODE/experiments/2026_09_09_10_30_34/ibex_environment.sh"
"$LOOK_ENV/bin/python" "$LOOK_CODE/workspace/check_framework.py" --project-root "$LOOK_CODE" > "$LOOK_RUN/framework_acceptance.json"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$LOOK_ENV/bin/python" -m pytest -q "$LOOK_CODE/tool/tests/test_v4_backward.py" "$LOOK_CODE/tool/tests/test_pruning.py" > "$LOOK_RUN/v4_cpu_acceptance.txt"
"$LOOK_ENV/bin/python" -c 'from look_core.paths import ProjectPaths; from dataclasses import asdict; import json; p=ProjectPaths.load(); print(json.dumps({k:str(v) for k,v in asdict(p).items()},indent=2)); assert all("/ibex/" in str(v) for v in asdict(p).values())' > "$LOOK_RUN/path_acceptance.json"
printf 'environment_and_v4_cpu_accepted_data_protocol_gpu_pending\n' > "$LOOK_RUN/environment_state.txt"
touch "$LOOK_ENV/environment_accepted"
