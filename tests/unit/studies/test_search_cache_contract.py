import json
from pathlib import Path

import numpy as np
import pytest
import torch

from look.methods.affine_family import FamilyArtifact, FamilyMap
from look.methods.linear_operator import fingerprint
from look.methods.operator import LOOKArtifact
from look.methods.positive_forward_tree import tree_contract, prefix_identity
from look.runtime.state import file_sha256
from look.studies.search_cache_contract import audit_completed_tree


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def fixture(root: Path):
    identity = {"framework_api": "V5", "case": "current"}
    contract = tree_contract(identity, ["x"])
    write(root / "contract.json", contract)
    prediction = root / "predictions/p.npz"
    prediction.parent.mkdir(parents=True)
    np.savez(prediction, participant_ids=np.array([1]), labels=np.array([0]), logits=np.zeros((1, 2)))
    evidence = {"role": "development", "data_role": "development", "score": 0.5,
                "prediction": "/historical/root/predictions/p.npz", "sha256": file_sha256(prediction)}
    base = LOOKArtifact("x", "oct_missing", "normalized_mean", 1, 1, (1,), (1,),
        torch.zeros(1), torch.ones(1), torch.zeros(1), torch.ones(1, 1),
        torch.zeros(1, 1), torch.zeros(1), 1., 0., 0., pca_source_id="current/x")
    mapping = FamilyMap(torch.zeros(1), torch.zeros(1), torch.ones(1), torch.empty(0),
                        "residual_rrr", 1., 1, 2, {})
    artifact = root / "prefixes/root/artifacts/000_a.pt"
    artifact.parent.mkdir(parents=True)
    torch.save(FamilyArtifact(base, mapping).record(), artifact)
    row = {"index": 0, "node": "x", "key": "a", "score": 0.5, "evidence": evidence,
           "artifact": str(artifact.relative_to(root)), "sha256": file_sha256(artifact)}
    pid = prefix_identity(contract, [])
    decision = {"identity": pid, "path": [], "baseline": evidence, "candidates": [row],
                "children": [], "terminal": True}
    write(root / "prefixes/root/baseline.json", {"identity": pid, "evidence": evidence})
    write(root / "prefixes/root/decision.json", decision)
    selection = {"schema": contract["schema"], "contract_sha256": file_sha256(root / "contract.json"),
                 "decisions": [decision], "final": evidence, "prefix_count": 1,
                 "selected_path": [], "candidate_evaluations": 1, "test_access": False}
    write(root / "selection.json", selection)
    write(root / "tree_progress.json", {"schema": contract["schema"], "state": "completed",
          "completed_prefixes": 1, "best_path": [], "best_score": 0.5, "test_access": False})
    moments = root / "family_moments/m.pt"
    moments.parent.mkdir()
    payload = {"identity": "m", "complete": True, "sites": ["x"]}
    torch.save({"payload": payload, "sha256": fingerprint(payload)}, moments)
    return identity, prediction


def test_current_identity_and_all_resume_evidence_are_verified(tmp_path):
    identity, _ = fixture(tmp_path)
    receipt = audit_completed_tree(tmp_path, expected_identity=identity,
        expected_moment_identities={"m"}, expected_artifact_pca_sources={"current/x"})
    assert receipt == {"schema": "look_search_cache_static_audit_v1",
        "historical_integrity_only": False, "current_identity_matched": True,
        "decisions": 1, "candidates": 1, "prediction_references": 3,
        "complete_family_moments": 1, "test_access": False}


def test_old_identity_fails_closed_but_can_be_audited_as_history(tmp_path):
    identity, _ = fixture(tmp_path)
    assert audit_completed_tree(tmp_path)["historical_integrity_only"] is True
    with pytest.raises(ValueError, match="exact moment and PCA source"):
        audit_completed_tree(tmp_path, expected_identity=identity)
    with pytest.raises(ValueError, match="identity differs"):
        audit_completed_tree(tmp_path, expected_identity={"framework_api": "V5", "case": "new"},
            expected_moment_identities={"m"}, expected_artifact_pca_sources={"current/x"})


def test_changed_prediction_or_decision_is_rejected(tmp_path):
    identity, prediction = fixture(tmp_path)
    prediction.write_bytes(b"changed")
    with pytest.raises(ValueError, match="prediction changed"):
        audit_completed_tree(tmp_path, expected_identity=identity,
            expected_moment_identities={"m"}, expected_artifact_pca_sources={"current/x"})
