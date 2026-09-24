import hashlib
import json
from pathlib import Path

import pytest

from look.runtime.v5_monitor_runtime_pin import file_sha256, verify_runtime


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def fixture(tmp_path):
    root = tmp_path / "package"; root.mkdir()
    source = root / "source.py"; source.write_text("VALUE = 1\n")
    chain = root / "chain.py"; chain.write_text("print('ok')\n")
    manifest = root / "runtime_manifest.json"
    write_json(manifest, {
        "schema": "look_v5_monitor_runtime_manifest_v1", "test_access": False,
        "files": {"source.py": file_sha256(source), "chain.py": file_sha256(chain)},
    })
    binding = root / "binding.json"
    write_json(binding, {
        "schema": "look_v5_monitor_runtime_binding_v1", "package_root": str(root),
        "runtime_manifest": str(manifest), "runtime_manifest_sha256": file_sha256(manifest),
        "parent_gpu_job_id": "52429877", "allowed_policy_sha256": ["a" * 64],
        "test_access": False,
    })
    return root, binding, file_sha256(binding)


def test_complete_runtime_identity_is_accepted(tmp_path):
    root, binding, digest = fixture(tmp_path)
    assert verify_runtime(binding, digest)["package_root"] == str(root)


@pytest.mark.parametrize("target", ["source.py", "chain.py"])
def test_post_admission_runtime_drift_is_rejected(tmp_path, target):
    root, binding, digest = fixture(tmp_path)
    (root / target).write_text("tampered\n")
    with pytest.raises(ValueError, match="Runtime file changed"):
        verify_runtime(binding, digest)


def test_post_admission_manifest_drift_is_rejected(tmp_path):
    root, binding, digest = fixture(tmp_path)
    (root / "runtime_manifest.json").write_text("{}\n")
    with pytest.raises(ValueError, match="Runtime manifest changed"):
        verify_runtime(binding, digest)


def test_post_admission_binding_drift_is_rejected(tmp_path):
    _, binding, digest = fixture(tmp_path)
    value = json.loads(binding.read_text()); value["parent_gpu_job_id"] = "other"
    write_json(binding, value)
    with pytest.raises(ValueError, match="Runtime binding changed"):
        verify_runtime(binding, digest)


def test_manifest_cannot_escape_package_root(tmp_path):
    root, binding, _ = fixture(tmp_path)
    outside = tmp_path / "outside.json"; write_json(outside, {"schema": "x"})
    value = json.loads(binding.read_text()); value["runtime_manifest"] = str(outside)
    value["runtime_manifest_sha256"] = file_sha256(outside); write_json(binding, value)
    with pytest.raises(ValueError, match="escaped"):
        verify_runtime(binding, file_sha256(binding))
