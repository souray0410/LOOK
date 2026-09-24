"""Build a runtime-pinned copy of a prepared LOOK V5 monitor package."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from look.runtime.v5_monitor_runtime_pin import file_sha256


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(file_sha256(path).encode() + b"\n")
    return digest.hexdigest()


def build(source: Path, destination: Path, verifier_source: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    verifier = destination / "runtime_verify.py"
    shutil.copy2(verifier_source.resolve(), verifier)

    package_manifest_path = destination / "package_manifest.json"
    package_manifest = json.loads(package_manifest_path.read_text())
    transformed = dict(package_manifest.get("transformed_files", {}))
    transformed.pop("monitor.sh", None)
    package_manifest.update(
        schema="look_v5_v25_runtime_pinned_package_manifest_v1",
        transformed_files=transformed,
        monitor_bootstrap_pinned_by_held_submission=True,
    )
    write_json(package_manifest_path, package_manifest)

    excluded = {"monitor.sh", "runtime_binding.json", "runtime_manifest.json", "runtime_verify.py"}
    files = {
        str(path.relative_to(destination)): file_sha256(path)
        for path in sorted(p for p in destination.rglob("*") if p.is_file())
        if path.name not in excluded and "__pycache__" not in path.parts
    }
    runtime_manifest = destination / "runtime_manifest.json"
    write_json(runtime_manifest, {
        "schema": "look_v5_monitor_runtime_manifest_v1", "files": files,
        "test_access": False,
    })
    binding = destination / "runtime_binding.json"
    write_json(binding, {
        "schema": "look_v5_monitor_runtime_binding_v1",
        "package_root": str(destination), "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": file_sha256(runtime_manifest),
        "parent_gpu_job_id": "52429877",
        "allowed_policy_sha256": [
            "28fe66b4c3bb29530dc8be22eb32b35e0d9b728b83f80a4c1165a834e20c4f62",
            "8582dcc6ee0fb6e897b6843b351a1e0700d34526a54451999fd7983db3d6f180",
        ],
        "test_access": False,
    })
    binding_sha, verifier_sha = file_sha256(binding), file_sha256(verifier)
    monitor = destination / "monitor.sh"
    monitor.write_text(f"""#!/bin/bash
set -euo pipefail
B={destination}
SIGNER=/ibex/user/mengh/MHD_Models/v5_rollout_20260922/ownership_overlays/bin/sign_legacy_native_overlay.py
test "$(sha256sum "$SIGNER" | awk '{{print $1}}')" = fe2cc4952244a638314b5eb048c2b0e9923d72c6ada74a311338ca3405cd28fb
test "$(sha256sum "$B/runtime_verify.py" | awk '{{print $1}}')" = {verifier_sha}
python3 "$B/runtime_verify.py" --binding "$B/runtime_binding.json" --binding-sha256 {binding_sha}
signed=$(python3 "$SIGNER")
export LOOK_LEGACY_NATIVE_OVERLAY=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["path"])' <<<"$signed")
export PYTHONPATH="$B/source/look/src:$B/source/framework/src:$B/source/models/src"
exec python3 "$B/chain_manager.py" "$@"
""", encoding="utf-8")
    monitor.chmod(0o755)
    return {
        "schema": "look_v5_runtime_pinned_package_receipt_v1",
        "package_root": str(destination), "tree_sha256": tree_sha256(destination),
        "monitor_sha256": file_sha256(monitor), "binding_sha256": binding_sha,
        "runtime_manifest_sha256": file_sha256(runtime_manifest),
        "runtime_verifier_sha256": verifier_sha, "test_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    verifier = Path(__file__).resolve().parents[1] / "src/look/runtime/v5_monitor_runtime_pin.py"
    receipt = build(args.source, args.destination, verifier)
    write_json(args.receipt, receipt)


if __name__ == "__main__":
    main()
