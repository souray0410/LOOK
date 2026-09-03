import json
from pathlib import Path

import pytest
import torch

import look_core.study_grid as study_grid_module
from look_core.artifacts import build_file_manifest
from look_core.config import ExperimentConfig, ExperimentSelection
from look_core.distributed import parse_gpu_devices
from look_core.paths import ProjectPaths
from look_core.pipeline import ExperimentRunner, PipelineOptions
from look_core.study_grid import (
    StudyGrid,
    _aggregate_confirmation,
    _search_diagnostics,
    baseline_calibration_grid,
    baseline_confirmation_grid,
    baseline_search_grid,
    expand_study_grid,
    freeze_baseline_candidate,
    study_grid_from_baseline_selection,
    unimodal_reference_grid,
)


def _paths(tmp_path: Path) -> ProjectPaths:
    dataset = tmp_path / "dataset"
    cohort = dataset / "cohorts" / "four_class"
    balanced = cohort / "balanced" / "reference_labels.csv"
    natural = cohort / "natural" / "reference_labels.csv"
    balanced.parent.mkdir(parents=True)
    natural.parent.mkdir(parents=True)
    balanced.write_text("participant_id\n", encoding="utf-8")
    natural.write_text("participant_id\n", encoding="utf-8")
    return ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=dataset,
        image_root=dataset,
        cohort_root=cohort,
        labels_csv=balanced,
        natural_labels_csv=natural,
        preprocess_cache_root=tmp_path / "cache" / "preprocessed_pairs",
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )


def _config(tmp_path: Path, learning_rate: float = 1e-4) -> ExperimentConfig:
    paths = _paths(tmp_path)
    return ExperimentConfig(
        image_root=paths.image_root,
        labels_csv=paths.labels_csv,
        natural_labels_csv=paths.natural_labels_csv,
        preprocess_cache_root=paths.preprocess_cache_root,
        output_root=paths.runs_root,
        cache_root=paths.cache_root,
        pretrained_lr=learning_rate,
        num_workers=0,
    )


def test_experiment_fingerprint_changes_with_scientific_configuration(tmp_path):
    first = ExperimentRunner(
        _config(tmp_path / "a", 1e-4), ExperimentSelection(), PipelineOptions(),
        torch.device("cpu"),
    )
    second_config = ExperimentConfig.from_dict(first.config.as_dict())
    second_config.pretrained_lr = 2e-4
    second = ExperimentRunner(
        second_config, ExperimentSelection(), PipelineOptions(), torch.device("cpu")
    )
    assert first.experiment_id != second.experiment_id


def test_path_changes_do_not_change_scientific_identity(tmp_path):
    first = ExperimentRunner(
        _config(tmp_path / "a"), ExperimentSelection(), PipelineOptions(), torch.device("cpu")
    )
    second_config = _config(tmp_path / "b")
    second_config.labels_csv.write_bytes(first.config.labels_csv.read_bytes())
    second_config.natural_labels_csv.write_bytes(first.config.natural_labels_csv.read_bytes())
    second = ExperimentRunner(
        second_config, ExperimentSelection(), PipelineOptions(), torch.device("cpu")
    )
    assert first.experiment_id == second.experiment_id


def test_test_access_requires_frozen_manifest(tmp_path):
    with pytest.raises(ValueError, match="frozen configuration manifest"):
        ExperimentRunner(
            _config(tmp_path), ExperimentSelection(), PipelineOptions(phase="test"),
            torch.device("cpu"),
        )


def test_filling_strategy_changes_experiment_but_not_backbone_identity(tmp_path):
    config = _config(tmp_path)
    mean = ExperimentRunner(
        config, ExperimentSelection(filling_strategy="normalized_mean"),
        PipelineOptions(), torch.device("cpu"),
    )
    gan = ExperimentRunner(
        config, ExperimentSelection(filling_strategy="paired_cgan"),
        PipelineOptions(), torch.device("cpu"),
    )
    assert mean.experiment_id != gan.experiment_id
    assert mean._backbone_id() == gan._backbone_id()


def test_calibration_grid_has_eight_regularized_profiles(tmp_path):
    cases = expand_study_grid(
        baseline_calibration_grid(), _paths(tmp_path), gpu_devices=(0, 1)
    )
    assert len(cases) == 8
    assert {case.config.pretrained_lr for case in cases} == {3e-5, 1e-4}
    assert {case.config.new_layer_lr for case in cases} == {3e-4, 1e-3}
    assert {case.config.classifier_dropout for case in cases} == {0.0, 0.2}
    assert {case.config.label_smoothing for case in cases} == {0.0, 0.1}
    assert {case.config.effective_batch_size for case in cases} == {128}
    assert all(case.config.training_strategy == "end_to_end_finetuning" for case in cases)
    assert all(case.config.sampling_strategy == "natural_without_replacement" for case in cases)
    assert all(not case.options.fit_look for case in cases)


def test_stage_a_has_seven_linear_fusion_positions(tmp_path):
    cases = expand_study_grid(
        baseline_search_grid(), _paths(tmp_path), gpu_devices=(0, 1)
    )
    assert len(cases) == 7
    assert {case.selection.fusion_position for case in cases} == {
        "input", "stem", "layer1", "layer2", "layer3", "layer4", "feature"
    }


def test_stage_b_has_top_three_by_three_seeds(tmp_path):
    cases = expand_study_grid(
        baseline_confirmation_grid(["feature", "layer4", "layer3"]),
        _paths(tmp_path), gpu_devices=(0, 1),
    )
    assert len(cases) == 9
    assert {case.selection.seed for case in cases} == {3407, 3408, 3409}


def test_unimodal_reference_grid_is_not_missing_input_training(tmp_path):
    profile = baseline_search_grid().classifier_profiles[0]
    cases = expand_study_grid(unimodal_reference_grid(profile), _paths(tmp_path))
    assert {case.selection.fusion_position for case in cases} == {"oct_only", "cfp_only"}
    assert all(case.config.training_strategy == "end_to_end_finetuning" for case in cases)


def test_confirmation_ranking_uses_seed_mean_then_stability():
    rows = []
    for fusion, values in {"feature": [0.70, 0.70, 0.70], "layer4": [0.69, 0.74, 0.72]}.items():
        for seed, value in zip((3407, 3408, 3409), values):
            rows.append({
                "fusion_position": fusion, "seed": seed,
                "experiment_id": f"{fusion}-{seed}", "backbone_id": f"b-{fusion}-{seed}",
                "macro_f1": value, "balanced_accuracy": value,
                "macro_auroc_ovr": value + 0.2, "ece_15": 0.05,
            })
    ranking = _aggregate_confirmation(rows)
    assert ranking[0]["fusion_position"] == "layer4"
    assert ranking[0]["rank"] == 1


def test_below_gate_candidate_cannot_be_frozen(tmp_path):
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "status": "quality_gate_failed", "quality_gate_passed": False,
        "artifacts": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="quality gate"):
        freeze_baseline_candidate(candidate, reviewer_note="reviewed")


def test_approved_candidate_requires_explicit_note(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"candidate")
    candidate_dir = tmp_path / "baseline_selection" / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "candidate.json"
    candidate.write_text(json.dumps({
        "status": "candidate_selected", "quality_gate_passed": True,
        "selection_id": "abc", "protocol": "balanced_four_class_baseline_selection",
        "winner": {"fusion_position": "layer3", "seeds": [3407, 3408, 3409]},
        "classifier_profile": baseline_search_grid().classifier_profiles[0],
        "artifacts": build_file_manifest([checkpoint]),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer note"):
        freeze_baseline_candidate(candidate, reviewer_note="")


def test_frozen_baseline_defines_formal_look_grid(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    manifest = tmp_path / "baseline_selection.json"
    manifest.write_text(json.dumps({
        "status": "frozen", "protocol": "balanced_four_class_baseline_selection",
        "winner": {"fusion_position": "layer3", "seeds": [3407, 3408, 3409]},
        "classifier_profile": baseline_search_grid().classifier_profiles[0],
        "artifacts": build_file_manifest([checkpoint]),
    }), encoding="utf-8")
    grid = study_grid_from_baseline_selection(manifest)
    assert grid.fusion_positions == ["layer3"]
    assert grid.seeds == [3407, 3408, 3409]


def test_partial_search_diagnostics_keep_profile_and_fusion_evidence():
    rows = [
        {"experiment_id": "a", "fusion_position": "feature", "classifier_profile": "p1",
         "macro_f1": 0.41, "balanced_accuracy": 0.43, "macro_auroc_ovr": 0.71, "ece_15": 0.08},
        {"experiment_id": "b", "fusion_position": "layer4", "classifier_profile": "p2",
         "macro_f1": 0.47, "balanced_accuracy": 0.46, "macro_auroc_ovr": 0.74, "ece_15": 0.06},
    ]
    rows.sort(key=lambda row: -row["macro_f1"])
    diagnostics = _search_diagnostics(rows)
    assert diagnostics["best_overall"]["experiment_id"] == "b"
    assert diagnostics["groups"]["classifier_profile"]["p1"]["completed"] == 1


def test_gpu_list_is_only_compute_selector(tmp_path):
    assert parse_gpu_devices(["cuda:0"]) == (0,)
    assert parse_gpu_devices("0,1") == (0, 1)
    grid = StudyGrid(
        fusion_positions=["feature"], seeds=[3407],
        filling_strategies=["normalized_mean"],
    )
    single = expand_study_grid(grid, _paths(tmp_path / "single"), gpu_devices=(0,))[0]
    multi = expand_study_grid(grid, _paths(tmp_path / "multi"), gpu_devices=(0, 1))[0]
    assert single.config.world_size == 1
    assert multi.config.world_size == 2
    assert single.config.micro_batch_size == multi.config.micro_batch_size == 64


def test_sequential_sweep_releases_parent_cuda_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(study_grid_module.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(study_grid_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study_grid_module.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    study_grid_module._release_parent_cuda_cache()
    assert calls == ["gc", "cuda"]


def test_notebook_has_one_documented_configuration_cell():
    notebook = json.loads(
        (Path(__file__).resolve().parents[2] / "pipeline/18_UKB_LOOK_ResNet50_MHD.ipynb")
        .read_text(encoding="utf-8")
    )
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert index and notebook["cells"][index - 1]["cell_type"] == "markdown"
    assert notebook["metadata"]["kernelspec"]["display_name"] == "LOOK"
    source = "".join(next(cell for cell in notebook["cells"] if cell.get("id") == "configuration")["source"])
    assert 'PROJECT_ROOT = Path("/home/mengh/LOOK/2026_09_03_08_30_00")' in source
    assert 'EXECUTION_MODE = "baseline_selection"' in source
