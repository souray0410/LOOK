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
    _aggregate_baseline_confirmation,
    _search_diagnostics,
    baseline_confirmation_grid,
    baseline_search_grid,
    expand_study_grid,
    freeze_baseline_candidate,
    study_grid_from_baseline_selection,
)


def _config(tmp_path: Path, learning_rate: float = 1e-4) -> ExperimentConfig:
    (tmp_path / "reference_labels.csv").write_text("participant_id\n", encoding="utf-8")
    return ExperimentConfig(
        data_root=tmp_path,
        output_root=tmp_path / "outputs",
        pretrained_lr=learning_rate,
        num_workers=0,
    )


def test_experiment_fingerprint_changes_with_scientific_configuration(tmp_path):
    selection = ExperimentSelection()
    options = PipelineOptions(fit_look=True)
    first = ExperimentRunner(_config(tmp_path, 1e-4), selection, options, torch.device("cpu"))
    second = ExperimentRunner(_config(tmp_path, 2e-4), selection, options, torch.device("cpu"))
    assert first.experiment_id != second.experiment_id


def test_test_access_requires_frozen_manifest(tmp_path):
    with pytest.raises(ValueError, match="frozen configuration manifest"):
        ExperimentRunner(
            _config(tmp_path),
            ExperimentSelection(),
            PipelineOptions(phase="test"),
            torch.device("cpu"),
        )


def test_only_declared_filling_strategies_are_accepted(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="Unknown filling strategy"):
        ExperimentSelection(filling_strategy="missing_finetune").validate(config)


def test_filling_strategy_changes_experiment_but_not_backbone_identity(tmp_path):
    config = _config(tmp_path)
    options = PipelineOptions(fit_look=True)
    mean = ExperimentRunner(
        config, ExperimentSelection(filling_strategy="normalized_mean"), options, torch.device("cpu")
    )
    gan = ExperimentRunner(
        config, ExperimentSelection(filling_strategy="paired_cgan"), options, torch.device("cpu")
    )
    assert mean.experiment_id != gan.experiment_id
    assert mean._backbone_id() == gan._backbone_id()


def test_look_implementation_hash_does_not_change_backbone_identity(tmp_path):
    runner = ExperimentRunner(
        _config(tmp_path), ExperimentSelection(), PipelineOptions(), torch.device("cpu")
    )
    backbone_id = runner._backbone_id()
    runner.implementation_hash = "look-only-change"
    assert runner._backbone_id() == backbone_id
    runner.backbone_implementation_hash = "training-change"
    assert runner._backbone_id() != backbone_id


def test_every_notebook_code_cell_has_preceding_markdown():
    notebook_path = Path(__file__).resolve().parents[2] / "pipeline/18_UKB_LOOK_ResNet50_MHD.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert index > 0
            assert notebook["cells"][index - 1]["cell_type"] == "markdown"
    assert notebook["metadata"]["kernelspec"] == {
        "display_name": "LOOK",
        "language": "python",
        "name": "look",
    }
    configuration = next(cell for cell in notebook["cells"] if cell.get("id") == "configuration")
    source = "".join(configuration["source"])
    assert 'EXECUTION_MODE = "validation"' in source
    assert 'FUSION_POSITIONS = ["feature"]' in source
    assert "SEEDS = [3407]" in source
    assert 'FILLING_STRATEGIES = ["normalized_mean"]' in source


def test_parent_process_builds_only_a_lightweight_cpu_summary_graph():
    project_root = Path(__file__).resolve().parents[2]
    source = (project_root / "tool/look_core/pipeline.py").read_text(encoding="utf-8")
    summary_block = source.split("summary_graph = build_resnet50_mhd_graph(", 1)[1].split(
        "write_json_atomic(structure", 1
    )[0]
    assert 'torch.device("cpu")' in summary_block
    assert "pretrained=False" in summary_block
    assert "\n            1,\n" in summary_block
    assert "del summary_graph" in summary_block


def test_study_grid_cases_use_singleton_axes_and_keep_existing_case_stable(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "reference_labels.csv").write_text("participant_id\n", encoding="utf-8")
    paths = ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=dataset,
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )
    one = StudyGrid(fusion_positions=["feature"], seeds=[3407], filling_strategies=["normalized_mean"])
    two = StudyGrid(fusion_positions=["feature"], seeds=[3407, 3408], filling_strategies=["normalized_mean"])
    first = expand_study_grid(one, paths)[0]
    expanded = expand_study_grid(two, paths)
    assert first.config.seeds == [3407]
    assert first.config.fusion_positions == ["feature"]
    assert first.config.filling_strategies == ["normalized_mean"]
    assert expanded[0].config.as_dict() == first.config.as_dict()


def test_baseline_search_grid_has_seven_canonical_crt_cases(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "reference_labels.csv").write_text("participant_id\n", encoding="utf-8")
    paths = ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=dataset,
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )
    cases = expand_study_grid(baseline_search_grid(), paths, gpu_devices=(0, 1))
    assert len(cases) == 7
    assert len({case.selection.experiment_id for case in cases}) == 7
    assert all(not case.options.fit_look for case in cases)
    assert all(not case.options.evaluate_missing_baselines for case in cases)
    assert all(case.config.training_strategy == "classifier_retraining" for case in cases)
    assert all(case.config.loss_name == "cross_entropy" for case in cases)
    assert all(
        case.config.crt_sampling_strategy == "class_balanced_with_replacement"
        for case in cases
    )
    assert cases[0].config.pretrained_lr == 1e-4
    assert cases[0].config.new_layer_lr == 1e-3


def test_baseline_confirmation_uses_top_three_and_three_seeds(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "reference_labels.csv").write_text("participant_id\n", encoding="utf-8")
    paths = ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=dataset,
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )
    cases = expand_study_grid(
        baseline_confirmation_grid(["feature", "layer4", "layer3"]),
        paths,
        gpu_devices=(0, 1),
    )
    assert len(cases) == 9
    assert {case.selection.seed for case in cases} == {3407, 3408, 3409}
    assert {case.selection.fusion_position for case in cases} == {
        "feature", "layer4", "layer3",
    }


def test_baseline_confirmation_ranks_seed_means_before_variance():
    rows = []
    values_by_fusion = {
        "feature": [0.40, 0.42, 0.41],
        "layer4": [0.39, 0.45, 0.43],
    }
    for fusion, values in values_by_fusion.items():
        for seed, value in zip((3407, 3408, 3409), values):
            rows.append({
                "fusion_position": fusion,
                "seed": seed,
                "experiment_id": f"{fusion}-{seed}",
                "backbone_id": f"backbone-{fusion}-{seed}",
                "macro_f1": value,
                "balanced_accuracy": value,
                "macro_auroc_ovr": value + 0.3,
                "ece_15": 0.05,
            })
    ranking = _aggregate_baseline_confirmation(rows)
    assert ranking[0]["fusion_position"] == "layer4"
    assert ranking[0]["rank"] == 1
    assert ranking[1]["rank"] == 2


def test_frozen_baseline_manifest_defines_formal_look_grid(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"frozen")
    manifest = tmp_path / "baseline_selection.json"
    manifest.write_text(json.dumps({
        "status": "frozen",
        "protocol": "two_stage_complete_modality_baseline_selection",
        "winner": {
            "fusion_position": "layer3",
            "seeds": [3407, 3408, 3409],
        },
        "artifacts": build_file_manifest([checkpoint]),
    }), encoding="utf-8")
    grid = study_grid_from_baseline_selection(manifest)
    assert grid.fusion_positions == ["layer3"]
    assert grid.seeds == [3407, 3408, 3409]
    assert grid.filling_strategies == ["normalized_mean", "paired_cgan"]


def test_baseline_candidate_requires_explicit_review_before_freeze(tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"candidate")
    candidate_dir = tmp_path / "baseline_selection" / "candidates"
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "baseline_candidate__abc.json"
    candidate.write_text(json.dumps({
        "status": "candidate_selected",
        "decision": "scientific_review_required",
        "selection_id": "abc",
        "protocol": "two_stage_complete_modality_baseline_selection",
        "winner": {
            "fusion_position": "layer3",
            "seeds": [3407, 3408, 3409],
        },
        "artifacts": build_file_manifest([checkpoint]),
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer note"):
        freeze_baseline_candidate(candidate, reviewer_note="")
    frozen = freeze_baseline_candidate(
        candidate,
        reviewer_note="All classes and three-seed stability reviewed.",
    )
    assert frozen["status"] == "frozen"
    assert frozen["decision"] == "approved_for_look"
    assert Path(frozen["manifest_path"]).is_file()


def test_partial_search_diagnostics_retain_best_and_group_evidence():
    rows = [
        {
            "experiment_id": "a", "fusion_position": "feature",
            "training_strategy": "classifier_retraining", "loss_name": "cross_entropy",
            "pretrained_lr": 1e-4, "new_layer_lr": 1e-3,
            "crt_learning_rate": 1e-3, "macro_f1": 0.41,
            "balanced_accuracy": 0.43, "macro_auroc_ovr": 0.71, "ece_15": 0.08,
        },
        {
            "experiment_id": "b", "fusion_position": "layer4",
            "training_strategy": "classifier_retraining", "loss_name": "cross_entropy",
            "pretrained_lr": 1e-4, "new_layer_lr": 1e-3,
            "crt_learning_rate": 1e-3, "macro_f1": 0.47,
            "balanced_accuracy": 0.46, "macro_auroc_ovr": 0.74, "ece_15": 0.06,
        },
    ]
    rows.sort(key=lambda row: -row["macro_f1"])
    diagnostics = _search_diagnostics(rows)
    assert diagnostics["completed_configurations"] == 2
    assert diagnostics["best_overall"]["experiment_id"] == "b"
    assert diagnostics["groups"]["fusion_position"]["feature"]["completed"] == 1


def test_sequential_sweep_releases_parent_cuda_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(study_grid_module.gc, "collect", lambda: calls.append("gc"))
    monkeypatch.setattr(study_grid_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(study_grid_module.torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    study_grid_module._release_parent_cuda_cache()
    assert calls == ["gc", "cuda"]


def test_gpu_list_is_the_only_compute_selector_and_preserves_global_batches(tmp_path):
    assert parse_gpu_devices(["cuda:0"]) == (0,)
    assert parse_gpu_devices("0,1") == (0, 1)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "reference_labels.csv").write_text("participant_id\n", encoding="utf-8")
    paths = ProjectPaths(
        project_root=tmp_path,
        data_root=tmp_path,
        dataset_root=dataset,
        cache_root=tmp_path / "cache",
        runs_root=tmp_path / "runs",
        tool_root=tmp_path / "tool",
        pipeline_root=tmp_path / "pipeline",
    )
    grid = StudyGrid(fusion_positions=["feature"], seeds=[3407], filling_strategies=["paired_cgan"])
    single = expand_study_grid(grid, paths, gpu_devices=(0,))[0]
    multi = expand_study_grid(grid, paths, gpu_devices=(0, 1))[0]
    assert single.config.micro_batch_size == multi.config.micro_batch_size == 128
    assert single.config.per_device_micro_batch_size == 128
    assert multi.config.per_device_micro_batch_size == 128
    assert single.config.global_micro_batch_size == 128
    assert multi.config.global_micro_batch_size == 256
    assert single.config.gan_batch_size == multi.config.gan_batch_size == 224
    assert single.config.per_device_gan_batch_size == 224
    assert multi.config.per_device_gan_batch_size == 224
    assert single.config.gan_accumulation_steps == 2
    assert multi.config.gan_accumulation_steps == 1
    assert single.config.num_workers == multi.config.num_workers == 16
    assert single.config.gan_num_workers == multi.config.gan_num_workers == 16
    assert single.config.world_size == 1 and multi.config.world_size == 2
