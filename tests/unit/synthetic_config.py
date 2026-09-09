from pathlib import Path
from look_core.paths import ProjectPaths
from look_core.config import ExperimentConfig

def _paths(tmp_path: Path) -> ProjectPaths:
    dataset = tmp_path / "dataset"
    cohort = dataset / "cohorts" / "glaucoma_binary"
    balanced = cohort / "primary" / "reference_labels.csv"
    natural = cohort / "natural" / "reference_labels.csv"
    balanced.parent.mkdir(parents=True)
    natural.parent.mkdir(parents=True)
    reference = (
        "participant_id,label_id,label_name,phenotype_profile\n"
        "1,0,normal,ukb_record_glaucoma_binary_bilateral\n"
        "2,1,glaucoma,ukb_record_glaucoma_binary_bilateral\n"
    )
    balanced.write_text(reference, encoding="utf-8")
    natural.write_text(reference, encoding="utf-8")
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
        tool_root=tmp_path / "src",
        pipeline_root=tmp_path / "scripts",
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
