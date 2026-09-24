"""Fail-closed runtime verification for a long-lived V5 monitor package."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_runtime(binding_path: str | Path, expected_binding_sha256: str) -> dict[str, Any]:
    """Verify every runtime byte before importing the monitor implementation.

    The tiny Slurm-spooled bootstrap supplies ``expected_binding_sha256``.  The
    binding then pins the complete runtime manifest, and the manifest pins every
    source/config file used after the job has waited on its dependency.  The
    bootstrap itself is separately pinned by the held-job submission receipt.
    """
    binding_path = Path(binding_path).resolve()
    if file_sha256(binding_path) != expected_binding_sha256:
        raise ValueError("Runtime binding changed after admission")
    binding = _read(binding_path)
    required = {
        "schema", "package_root", "runtime_manifest", "runtime_manifest_sha256",
        "parent_gpu_job_id", "allowed_policy_sha256", "test_access",
    }
    if set(binding) != required or binding.get("schema") != "look_v5_monitor_runtime_binding_v1":
        raise ValueError("Unexpected runtime binding schema")
    if binding.get("test_access") is not False or str(binding.get("parent_gpu_job_id")) != "52429877":
        raise ValueError("Runtime safety identity changed")
    allowed = binding.get("allowed_policy_sha256")
    if not isinstance(allowed, list) or not allowed or any(
        not isinstance(value, str) or len(value) != 64 for value in allowed
    ):
        raise ValueError("Policy allowlist is malformed")

    root = Path(binding["package_root"]).resolve()
    manifest_path = Path(binding["runtime_manifest"]).resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("Runtime manifest escaped package root")
    if file_sha256(manifest_path) != binding["runtime_manifest_sha256"]:
        raise ValueError("Runtime manifest changed after admission")
    manifest = _read(manifest_path)
    if set(manifest) != {"schema", "files", "test_access"} or manifest.get("schema") != "look_v5_monitor_runtime_manifest_v1":
        raise ValueError("Unexpected runtime manifest schema")
    if manifest.get("test_access") is not False or not isinstance(manifest.get("files"), dict):
        raise ValueError("Runtime manifest safety fields changed")
    if not manifest["files"]:
        raise ValueError("Runtime manifest is empty")
    for relative, expected in sorted(manifest["files"].items()):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == binding_path or path == manifest_path:
            raise ValueError("Runtime manifest contains an invalid path")
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Runtime file changed after admission: {relative}")
    return binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--binding-sha256", required=True)
    args = parser.parse_args()
    verify_runtime(args.binding, args.binding_sha256)


if __name__ == "__main__":
    main()
