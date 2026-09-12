import hashlib
import json
from pathlib import Path

import pytest

from look.runtime.state import file_sha256
from look.models.native_materialization import materialize_selected, verify_selected


def fixture(tmp_path):
    source = tmp_path / "native"
    source.mkdir()
    spec = dict(test_used=False, training=dict(seed=3416), model=dict(name="fixture"))
    (source / "spec.json").write_text(json.dumps(spec))
    for name in ("best.pt", "history.json", "development_predictions.npz", "last.pt"):
        (source / name).write_bytes(name.encode())
    receipt = dict(status="accepted", test_used=False,
                   identity=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                   files={name: file_sha256(source / name) for name in
                          ("best.pt", "history.json", "development_predictions.npz", "last.pt", "spec.json")})
    (source / "accepted.json").write_text(json.dumps(receipt))
    def verifier(root, expected):
        r = json.loads((root / "accepted.json").read_text())
        assert expected == spec
        for n, h in r["files"].items():
            if file_sha256(root / n) != h:
                raise ValueError("Source acceptance invalid")
    return source, spec, verifier


def test_copy_is_independent_idempotent_and_not_resume_state(tmp_path):
    source, spec, verifier = fixture(tmp_path)
    target = materialize_selected(source, tmp_path / "project", spec, verifier)
    assert target == materialize_selected(source, tmp_path / "project", spec, verifier)
    assert (source / "best.pt").stat().st_ino != (target / "best.pt").stat().st_ino
    assert not (target / "last.pt").exists()
    m, s, _ = verify_selected(target, spec)
    assert s == spec and not m["complete_training_resume"]
    assert not m["selected_prediction_replay"]
    (source / "best.pt").write_bytes(b"later source corruption")
    verify_selected(target, spec)  # a verified independent project copy survives
    with pytest.raises(ValueError):
        materialize_selected(source, tmp_path / "project", spec, verifier)


def test_corrupt_project_copy_never_silently_overwritten(tmp_path):
    source, spec, verifier = fixture(tmp_path)
    target = materialize_selected(source, tmp_path / "project", spec, verifier)
    (target / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        materialize_selected(source, tmp_path / "project", spec, verifier)
    assert (target / "best.pt").read_bytes() == b"changed"


def test_test_exposure_and_wrong_parent_spec_refused(tmp_path):
    source, spec, verifier = fixture(tmp_path)
    target = materialize_selected(source, tmp_path / "project", spec, verifier)
    with pytest.raises(ValueError):
        verify_selected(target, dict(spec, training=dict(seed=3417)))
    receipt = json.loads((source / "accepted.json").read_text())
    receipt["test_used"] = True
    (source / "accepted.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        materialize_selected(source, tmp_path / "project", spec, verifier)


def test_actual_mhd_selected_state_load_and_node_mismatch(tmp_path, monkeypatch):
    import torch
    from mhd_framework.models import create_model
    from mhd_framework.models import artifacts
    from look.models.observed_participant import ObservedParticipantModel
    from look.models.native_materialization import load_selected
    torch.set_num_threads(1)
    source, spec, _ = fixture(tmp_path)
    spec.update(model=dict(name="resnet18", spatial_dims=2, views=1), framework={"fixture": True})
    # This test checks real MHD state loading; runtime-source audit is separately
    # tested by the framework and deliberately replaced by a checked fixture.
    seen = []
    monkeypatch.setattr(artifacts, "verify_runtime", lambda x: seen.append(x))
    native = ObservedParticipantModel(create_model(spec["model"], device="cpu")).eval()
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    torch.save(dict(model=native.state_dict(), identity=identity, epoch=8), source / "best.pt")
    (source / "spec.json").write_text(json.dumps(spec))
    receipt = json.loads((source / "accepted.json").read_text())
    receipt.update(identity=identity, best_epoch=8,
                   node_ids=[[n["id"], n["name"]] for n in native.graph.describe_nodes()])
    receipt["files"] = {n: file_sha256(source / n) for n in receipt["files"]}
    (source / "accepted.json").write_text(json.dumps(receipt))
    target = materialize_selected(source, tmp_path / "project", spec, lambda *a: None)
    restored = load_selected(target, create_model, spec)
    assert seen == [spec["framework"]]
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(native(x, [1, 2]), restored(x, [1, 2]))
    receipt["node_ids"][0][0] = -1
    (source / "accepted.json").write_text(json.dumps(receipt))
    wrong = materialize_selected(source, tmp_path / "project", spec, lambda *a: None)
    with pytest.raises(ValueError, match="node identity"):
        load_selected(wrong, create_model, spec)
