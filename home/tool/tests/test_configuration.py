import json
from pathlib import Path

import pytest
import torch

from look_core.config import ExperimentConfig, ExperimentSelection
from look_core.distributed import parse_gpu_devices
from look_core.paths import ProjectPaths
from look_core.pipeline import ExperimentRunner, PipelineOptions
from look_core.study_grid import StudyGrid, expand_study_grid


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
