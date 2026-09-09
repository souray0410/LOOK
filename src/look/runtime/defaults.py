from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from look.runtime.paths import discover_project_root, load_project_config, rooted_path, ProjectPaths


@dataclass(frozen=True)
class DeploymentDefaults:
    project_root: Path
    data_root: Path
    dataset_root: Path
    image_root: Path
    cohort_root: Path
    labels_csv: Path
    natural_labels_csv: Path
    preprocess_cache_root: Path
    ssh_host: str
    ukb_source_uuid: str
    ukb_source_mount: Path
    ukb_source_root: Path
    ukb_label_uuid: str
    ukb_label_mount: Path
    ukb_label_root: Path
    torch_index_url: str
    torch_version: str
    torchvision_version: str

    @classmethod
    def load(cls, project_root: Path | None = None) -> "DeploymentDefaults":
        root = (project_root or discover_project_root()).resolve()
        config = load_project_config(root)
        values = config["deployment_defaults"]
        paths = ProjectPaths.load(root)
        shared = paths.data_root.parent / "UKBiobank" / "ophthalmology"
        return cls(
            project_root=root,
            data_root=paths.data_root,
            dataset_root=paths.dataset_root,
            image_root=paths.image_root,
            cohort_root=paths.cohort_root,
            labels_csv=paths.labels_csv,
            natural_labels_csv=paths.natural_labels_csv,
            preprocess_cache_root=paths.preprocess_cache_root,
            ssh_host=values["ssh_host"],
            ukb_source_uuid=values["ukb_source_uuid"],
            ukb_source_mount=rooted_path(root, values.get("ukb_source_mount", shared)),
            ukb_source_root=rooted_path(root, values.get("ukb_source_root", shared / "raw")),
            ukb_label_uuid=values["ukb_label_uuid"],
            ukb_label_mount=rooted_path(root, values.get("ukb_label_mount", shared)),
            ukb_label_root=rooted_path(root, values.get("ukb_label_root", shared / "raw" / "labels")),
            torch_index_url=values["torch_index_url"],
            torch_version=values["torch_version"],
            torchvision_version=values["torchvision_version"],
        )
