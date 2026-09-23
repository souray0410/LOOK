"""Static integrity and current-identity gate for completed LOOK search caches."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from look.methods.affine_family import FamilyArtifact
from look.methods.linear_operator import fingerprint
from look.methods.positive_forward_tree import VERSION, prefix_identity
from look.runtime.state import file_sha256


def _read(path: Path):
    return json.loads(path.read_text())


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Search cache path escapes its root")
    return path


def _prediction(root: Path, evidence: dict) -> None:
    if evidence.get("role") != "development" or evidence.get("data_role") != "development":
        raise ValueError("Only development prediction evidence may be reused")
    source = Path(evidence["prediction"])
    local = _inside(root, f"predictions/{source.name}")
    if file_sha256(local) != evidence["sha256"]:
        raise ValueError("Cached prediction changed")


def audit_completed_tree(root: Path, *, expected_identity: dict | None = None,
                         expected_moment_identities: set[str] | None = None,
                         expected_artifact_pca_sources: set[str] | None = None) -> dict:
    """Verify an immutable completed tree without evaluating a model.

    ``expected_identity`` is mandatory for current reuse. Passing ``None`` is an
    integrity-only historical audit and never authorizes execution or resume.
    """
    root = Path(root)
    contract = _read(root / "contract.json")
    if contract.get("schema") != VERSION or contract.get("test_access") is not False:
        raise ValueError("Unsupported or test-bearing search contract")
    if expected_identity is not None:
        if contract.get("identity") != expected_identity:
            raise ValueError("Search cache identity differs from the current V5 inputs")
        if expected_moment_identities is None or expected_artifact_pca_sources is None:
            raise ValueError("Current reuse requires exact moment and PCA source identities")
    selection = _read(root / "selection.json")
    progress = _read(root / "tree_progress.json")
    if (selection.get("schema") != VERSION or selection.get("test_access") is not False
            or selection.get("contract_sha256") != file_sha256(root / "contract.json")
            or progress.get("schema") != VERSION or progress.get("state") != "completed"
            or progress.get("test_access") is not False):
        raise ValueError("Search completion contract changed")

    sites = contract["sites"]
    queue = [([], None)]
    verified_decisions = verified_candidates = verified_predictions = 0
    for path, inherited in queue:
        pid = prefix_identity(contract, path)
        folder = root / "prefixes" / ("root" if not path else pid)
        decision = _read(folder / "decision.json")
        baseline = _read(folder / "baseline.json")
        if (decision.get("identity") != pid or decision.get("path") != path
                or baseline.get("identity") != pid or baseline.get("evidence") != decision.get("baseline")):
            raise ValueError("Prefix decision identity changed")
        _prediction(root, decision["baseline"])
        if inherited is not None and decision["baseline"] != inherited:
            raise ValueError("Child baseline differs from its parent candidate")
        verified_predictions += 1
        descriptors = []
        for row in decision["candidates"]:
            index = row["index"]
            if (type(index) is not int or not 0 <= index < len(sites)
                    or row["node"] != sites[index] or row["score"] != row["evidence"]["score"]):
                raise ValueError("Candidate descriptor changed")
            artifact_path = _inside(root, row["artifact"])
            if file_sha256(artifact_path) != row["sha256"]:
                raise ValueError("Candidate artifact changed")
            record = torch.load(artifact_path, map_location="cpu", weights_only=False)
            artifact = FamilyArtifact.from_record(record)
            if (expected_artifact_pca_sources is not None
                    and artifact.base.pca_source_id not in expected_artifact_pca_sources):
                raise ValueError("Candidate artifact uses a different PCA source")
            _prediction(root, row["evidence"])
            descriptors.append({key: row[key] for key in ("index", "node", "key", "artifact", "sha256")})
            verified_candidates += 1
            verified_predictions += 1
        expected_children = []
        for child in decision["children"]:
            if (child["path"][:-1] != path or child["path"][-1] not in descriptors
                    or child["winner"] not in decision["candidates"]
                    or child["path"][-1] != {key: child["winner"][key]
                                             for key in ("index", "node", "key", "artifact", "sha256")}):
                raise ValueError("Child path is not a verified candidate")
            expected_children.append((child["path"], child["winner"]["evidence"]))
        queue.extend(expected_children)
        verified_decisions += 1

    paths = [path for path, _ in queue]
    if (selection.get("decisions") != [_read(root / "prefixes" / ("root" if not path else prefix_identity(contract, path)) / "decision.json") for path in paths]
            or selection.get("prefix_count") != len(queue)
            or progress.get("completed_prefixes") != len(queue)
            or selection.get("selected_path") not in paths
            or progress.get("best_path") != selection.get("selected_path")
            or progress.get("best_score") != selection["final"]["score"]
            or selection.get("candidate_evaluations") != verified_candidates):
        raise ValueError("Selection summary differs from committed prefixes")
    _prediction(root, selection["final"])
    verified_predictions += 1

    moment_files = sorted((root / "family_moments").glob("*.pt"))
    for moment in moment_files:
        saved = torch.load(moment, map_location="cpu", weights_only=False)
        payload = saved.get("payload")
        if (not isinstance(payload, dict) or fingerprint(payload) != saved.get("sha256")
                or moment.stem != payload.get("identity") or payload.get("complete") is not True):
            raise ValueError("Family moment resume state changed")
    if (expected_moment_identities is not None
            and {moment.stem for moment in moment_files} != expected_moment_identities):
        raise ValueError("Family moment identities differ from the current V5 inputs")
    return {
        "schema": "look_search_cache_static_audit_v1",
        "historical_integrity_only": expected_identity is None,
        "current_identity_matched": expected_identity is not None,
        "decisions": verified_decisions,
        "candidates": verified_candidates,
        "prediction_references": verified_predictions,
        "complete_family_moments": len(moment_files),
        "test_access": False,
    }
