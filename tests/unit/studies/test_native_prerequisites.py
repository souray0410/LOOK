import copy
import json
from pathlib import Path

import pytest

from look.runtime.state import file_sha256
from look.studies.native_prerequisites import collect, audit


def candidate(tmp_path, name, score=None):
    root = tmp_path / name
    root.mkdir()
    spec = dict(model=dict(name="resnet50", spatial_dims=2), track="cfp_2d",
                disease="cataract", test_used=False, training=dict(seed=3416),
                train_manifest_sha256="train_fixture", development_manifest_sha256="dev_fixture",
                cache_receipt_sha256="cache_fixture", aggregation="valid_eye_mean_fixture")
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    if score is not None:
        (root / "accepted.json").write_text(json.dumps(dict(
            metrics=dict(macro_f1=score, auroc=.8), files={"best.pt": "fixture"})))
    return dict(spec=str(path), spec_sha256=file_sha256(path), run_dir=str(root))


def queue(tmp_path, tasks):
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(dict(tasks=tasks)))
    return path


def test_unfinished_competitor_prevents_early_best_lock(tmp_path):
    first = candidate(tmp_path, "first", .8)
    second = candidate(tmp_path, "second")
    catalog = collect([queue(tmp_path, [first, second])])
    result = audit(catalog, lambda root, spec: None)
    group = result["groups"]["cataract/resnet50/cfp_2d"]
    assert group["accepted_count"] == 1 and group["selected"] is None
    assert not result["project_dispatch_ready"]
    (Path(second["run_dir"]) / "accepted.json").write_text(json.dumps(dict(
        metrics=dict(macro_f1=.7, auroc=.9), files={"best.pt": "fixture"})))
    result = audit(catalog, lambda root, spec: None)
    assert result["groups"]["cataract/resnet50/cfp_2d"]["selected"]["run_dir"] == first["run_dir"]
    assert not result["project_dispatch_ready"]  # parent winner != complete LOOK gate


def test_bad_receipt_and_changed_spec_fail_closed(tmp_path):
    task = candidate(tmp_path, "first", .8)
    catalog = collect([queue(tmp_path, [task])])
    def reject(root, spec):
        raise ValueError("checkpoint hash mismatch")
    group = audit(catalog, reject)["groups"]["cataract/resnet50/cfp_2d"]
    assert group["selected"] is None and group["candidates"][0]["state"] == "needs_review"
    Path(task["spec"]).write_text("{}")
    with pytest.raises(ValueError):
        collect([tmp_path / "queue.json"])
    assert audit(catalog, lambda *a: None)["groups"]["cataract/resnet50/cfp_2d"]["selected"] is None


def test_aliases_deduplicated_and_invalid_score_rejected(tmp_path):
    task = candidate(tmp_path, "first", float("nan"))
    catalog = collect([queue(tmp_path, [task, copy.deepcopy(task)])])
    assert len(catalog["candidates"]) == 1
    assert audit(catalog, lambda *a: None)["groups"]["cataract/resnet50/cfp_2d"]["selected"] is None


def test_test_exposure_refused(tmp_path):
    task = candidate(tmp_path, "first")
    path = Path(task["spec"])
    spec = json.loads(path.read_text()); spec["test_used"] = True
    path.write_text(json.dumps(spec)); task["spec_sha256"] = file_sha256(path)
    with pytest.raises(ValueError):
        collect([queue(tmp_path, [task])])


def test_distinct_cohorts_cannot_be_ranked_together(tmp_path):
    first = candidate(tmp_path, "first", .8)
    second = candidate(tmp_path, "second", .9)
    path = Path(second["spec"]); spec = json.loads(path.read_text())
    spec["train_manifest_sha256"] = "other_cohort"
    path.write_text(json.dumps(spec)); second["spec_sha256"] = file_sha256(path)
    with pytest.raises(ValueError, match="Incompatible cohort"):
        collect([queue(tmp_path, [first, second])])
