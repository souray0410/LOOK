from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .paths import discover_project_root


@dataclass(frozen=True)
class DeploymentDefaults:
    project_root: Path
    data_root: Path
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
        config = json.loads((root / "project.json").read_text(encoding="utf-8"))
        values = config["deployment_defaults"]
        return cls(
            project_root=Path(values["project_root"]),
            data_root=Path(values["data_root"]),
            ssh_host=values["ssh_host"],
            ukb_source_uuid=values["ukb_source_uuid"],
            ukb_source_mount=Path(values["ukb_source_mount"]),
            ukb_source_root=Path(values["ukb_source_root"]),
            ukb_label_uuid=values["ukb_label_uuid"],
            ukb_label_mount=Path(values["ukb_label_mount"]),
            ukb_label_root=Path(values["ukb_label_root"]),
            torch_index_url=values["torch_index_url"],
            torch_version=values["torch_version"],
            torchvision_version=values["torchvision_version"],
        )
