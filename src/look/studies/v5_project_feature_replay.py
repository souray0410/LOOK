"""One-time feature-replay adapter for historical ``look_project_case_v1`` runs.

This module is intentionally separate from current V5 readers.  It verifies the
two selected-parent manifests and reconstructs only train/development datasets;
it never accepts a test split or silently interprets a current specification as
a historical one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from look.data.observed_pair import from_parent_specs
from look.runtime.state import file_sha256


ROLES = ("train", "development")


def _read(path):
    return json.loads(Path(path).read_text())


def _selected_parent(root, expected_manifest_sha256):
    root = Path(root).resolve()
    manifest_path = root / "selected_artifact.json"
    if file_sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("Historical selected-parent manifest changed")
    manifest = _read(manifest_path)
    if (manifest.get("schema") != "look_selected_native_v1"
            or manifest.get("test_access") is not False
            or manifest.get("complete_training_resume") is not False):
        raise ValueError("Historical selected-parent contract rejected")
    for name, digest in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != digest:
            raise ValueError("Historical selected-parent file changed")
    spec = _read(root / "spec.json")
    if spec.get("test_used") is not False:
        raise ValueError("Historical parent test provenance rejected")
    return spec


def datasets(source_run, inputs_factory, *, seed):
    """Return pinned train/development pairs for an offline migration only."""
    source_run = Path(source_run).resolve()
    spec = _read(source_run / "spec.json")
    if (spec.get("schema") != "look_project_case_v1"
            or spec.get("test_access") is not False or spec.get("seed") != seed):
        raise ValueError("Historical project-case identity rejected")
    parents = []
    for role in ("first", "second"):
        item = spec["parents"][role]
        parents.append(_selected_parent(item["path"], item["manifest_sha256"]))
    result = {role: from_parent_specs(*parents, role, inputs_factory,
                                      augment=False, seed=seed) for role in ROLES}
    identities = {}
    for role, dataset in result.items():
        rows = [{"participant_id": participant, "eyes": count}
                for participant, count in zip(dataset.participant_ids,
                                              dataset.counts, strict=True)]
        identities[role] = {
            "participants": len(rows),
            "eyes": sum(row["eyes"] for row in rows),
            "ordered_identity_sha256": hashlib.sha256(
                json.dumps(rows, separators=(",", ":")).encode()).hexdigest(),
        }
    return result, {"parent_manifests": [spec["parents"][r]["manifest_sha256"]
                                         for r in ("first", "second")],
                    "roles": identities, "test_access": False}
