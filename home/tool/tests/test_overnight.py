import copy
import json
import runpy
from pathlib import Path

import pytest

API = runpy.run_path(str(Path(__file__).resolve().parents[2] / "pipeline/33_run_overnight_validation.py"))


def test_six_bounded_profiles_preserve_other_parameters():
    base = dict(name="base", pretrained_lr=3e-4, new_layer_lr=3e-3, classifier_dropout=0.0,
                weight_decay=1e-4, label_smoothing=0.0, epochs=100)
    original = copy.deepcopy(base)
    profiles = API["regularization_profiles"](base)
    assert base == original
    assert len(profiles) == len({p["name"] for p in profiles}) == 6
    assert {(p["weight_decay"], p["label_smoothing"]) for p in profiles} == {(w, s) for w in (1e-3, 1e-2) for s in (0.0, .05, .1)}
    for p in profiles:
        assert (p["pretrained_lr"], p["new_layer_lr"], p["classifier_dropout"]) == (3e-4, 3e-3, 0.0)


def good_rows():
    rows = [dict(seed=s, fusion_position="feature", macro_auroc_ovr=.82, macro_f1=.74,
                 ece_15=.10, sensitivity_per_class=[.75, .75], specificity_per_class=[.75, .75]) for s in (3407,3408,3409)]
    refs = [dict(rows[0], fusion_position=f, macro_auroc_ovr=.77) for f in ("oct_only", "cfp_only")]
    return dict(mean_macro_auroc_ovr=.82, mean_macro_f1=.74), rows, refs


def test_gate_passes_only_complete_evidence():
    assert all(API["gate_checks"](*good_rows()).values())


@pytest.mark.parametrize("issue", ["low_auroc", "low_f1", "missing_seed", "nan", "bad_sensitivity", "better_unimodal"])
def test_gate_failure_cases(issue):
    winner, rows, refs = good_rows()
    if issue == "low_auroc": winner["mean_macro_auroc_ovr"] = .79
    if issue == "low_f1": winner["mean_macro_f1"] = .69
    if issue == "missing_seed": rows.pop()
    if issue == "nan": rows[0]["ece_15"] = float("nan")
    if issue == "bad_sensitivity": rows[0]["sensitivity_per_class"][1] = .1
    if issue == "better_unimodal": refs[0]["macro_auroc_ovr"] = .9
    assert not all(API["gate_checks"](winner, rows, refs).values())


def test_candidate_lookup_requires_exact_predecessor(tmp_path):
    with pytest.raises(RuntimeError): API["find_candidate"](tmp_path, "selected")
    (tmp_path / "baseline_candidate__a.json").write_text(json.dumps({"stage_b_plan_id": "other"}))
    with pytest.raises(RuntimeError): API["find_candidate"](tmp_path, "selected")
    (tmp_path / "baseline_candidate__b.json").write_text(json.dumps({"stage_b_plan_id": "selected"}))
    assert API["find_candidate"](tmp_path, "selected")[1]["stage_b_plan_id"] == "selected"


def test_followup_entry_has_no_sealed_test_or_gan_launch():
    text = (Path(__file__).resolve().parents[2] / "pipeline/33_run_overnight_validation.py").read_text()
    assert 'phase="validation"' in text
    assert 'phase="test"' not in text
    assert 'freeze_study_grid' not in text
    assert 'filling_strategies=["raw_zero", "normalized_mean"]' in text


def setup_execution(tmp_path, monkeypatch, predecessor="complete"):
    import sys
    from types import SimpleNamespace
    from look_core import cli, artifacts, task_selection, study_grid
    import torch

    labels = tmp_path / "labels.csv"
    labels.write_text("label_id\n0\n1\n")
    paths = SimpleNamespace(labels_csv=labels, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache")
    logs = paths.runs_root / "logs"
    logs.mkdir(parents=True)
    (logs / "detached_validation_status.env").write_text(f"session=look-glaucoma-baseline\nstate={predecessor}\n")
    monkeypatch.setattr(cli, "resolve_runtime_arguments", lambda args: paths)
    monkeypatch.setattr(sys, "argv", ["step33", "--confirmation-plan", "original", "--auto-look-validation", "--execute"])
    monkeypatch.setattr(API["subprocess"], "run", lambda *a, **k: SimpleNamespace(returncode=1))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(task_selection, "validate_selected_task", lambda p: None)
    monkeypatch.setattr(artifacts, "validate_file_manifest", lambda m: [])
    winner, rows, refs = good_rows()
    winner.update(fusion_position="feature", seeds=[3407, 3408, 3409])
    candidate = dict(winner=winner, ranking=[winner], stage_b_plan_id="original", calibration_plan_id="initial_calibration", unimodal_references=refs,
                     classifier_profile=dict(name="base", pretrained_lr=3e-4, new_layer_lr=3e-3, classifier_dropout=0.0, weight_decay=1e-4, label_smoothing=0.0), artifacts={"fixture": {}}, quality_gate_passed=True)
    folder = paths.runs_root / "baseline_selection/candidates"
    folder.mkdir(parents=True)
    (folder / "baseline_candidate__fixture.json").write_text(json.dumps(candidate))
    monkeypatch.setattr(study_grid, "_rows", lambda *a: rows)
    calls = []
    def run(grid, *args, **kwargs):
        calls.append((grid, kwargs))
        return {"plan_id": "pilot"}
    monkeypatch.setattr(study_grid, "run_study_grid", run)
    return paths, calls


def test_orchestrator_pass_starts_only_validation_pilot(tmp_path, monkeypatch):
    paths, calls = setup_execution(tmp_path, monkeypatch)
    API["main"]()
    assert len(calls) == 1
    grid, options = calls[0]
    assert options["phase"] == "validation"
    assert grid.seeds == [3407]
    assert grid.filling_strategies == ["raw_zero", "normalized_mean"]
    payload = json.loads(next((paths.runs_root / "overnight").glob("*/summary.json")).read_text())
    assert payload["status"] == "complete" and payload["test_access"] is False
    assert not list((paths.runs_root / "freezes").glob("*"))
    assert not (paths.cache_root / "pipeline_state/overnight_validation/stage.lock").exists()


def test_orchestrator_failed_predecessor_cannot_start_training(tmp_path, monkeypatch):
    paths, calls = setup_execution(tmp_path, monkeypatch, predecessor="failed")
    with pytest.raises(RuntimeError, match="Predecessor failed"):
        API["main"]()
    assert calls == []
    assert not (paths.cache_root / "pipeline_state/overnight_validation/stage.lock").exists()


def test_orchestrator_interrupt_releases_lock_and_can_resume(tmp_path, monkeypatch):
    from look_core import study_grid
    paths, calls = setup_execution(tmp_path, monkeypatch)
    original = study_grid.run_study_grid
    def interrupt(*a, **kw):
        raise KeyboardInterrupt()
    monkeypatch.setattr(study_grid, "run_study_grid", interrupt)
    with pytest.raises(KeyboardInterrupt): API["main"]()
    assert not (paths.cache_root / "pipeline_state/overnight_validation/stage.lock").exists()
    monkeypatch.setattr(study_grid, "run_study_grid", original)
    API["main"]()
    assert len(calls) == 1


@pytest.mark.parametrize("improves", [False, True])
def test_regularization_branch_preserves_selected_full_profile(tmp_path, monkeypatch, improves):
    from look_core import study_grid, artifacts
    paths, _ = setup_execution(tmp_path, monkeypatch)
    path = next((paths.runs_root / "baseline_selection/candidates").glob("*.json"))
    candidate = json.loads(path.read_text())
    candidate["winner"]["mean_macro_auroc_ovr"] = .77
    candidate["quality_gate_passed"] = False
    path.write_text(json.dumps(candidate))
    _, rows, _ = good_rows()
    storage = {"original": rows, "initial_calibration": [dict(rows[0], macro_auroc_ovr=.775)]}
    calls = []
    def run(grid, *a, **kw):
        calls.append(grid)
        generated = []
        for fusion in grid.fusion_positions:
            for seed in grid.seeds:
                for profile in grid.classifier_profiles:
                    score = .82 if improves else .70
                    if fusion in ("oct_only", "cfp_only"):
                        score = .75
                    generated.append(dict(rows[0], seed=seed, fusion_position=fusion,
                        macro_auroc_ovr=score, balanced_accuracy=.74,
                        classifier_profile=profile["name"],
                        experiment_id=f"e-{fusion}-{seed}", backbone_id=f"b-{fusion}-{seed}"))
        key = str(len(calls))
        storage[key] = generated
        return {"plan_id": key}
    monkeypatch.setattr(study_grid, "run_study_grid", run)
    monkeypatch.setattr(study_grid, "_rows", lambda p, r: storage[r["plan_id"]])
    monkeypatch.setattr(artifacts, "build_file_manifest", lambda p: {"fixture": {}})
    API["main"]()
    assert len(calls[0].classifier_profiles) == 6
    summary = json.loads(next((paths.runs_root / "overnight").glob("*/summary.json")).read_text())
    if improves:
        assert len(calls) == 5
        assert calls[2].fusion_positions == list(study_grid.FUSION_POSITIONS)
        assert calls[3].seeds == [3407,3408,3409]
        assert all(grid.classifier_profiles[0]["weight_decay"] == .001 for grid in calls[1:])
        assert calls[-1].filling_strategies == ["raw_zero", "normalized_mean"]
        assert summary["status"] == "complete"
    else:
        assert len(calls) == 1 and summary["status"] == "quality_gate_failed"
