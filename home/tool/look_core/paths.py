from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


PROJECT_FILE = "project.json"


def discover_project_root(start: Path | None = None) -> Path:
    override = os.environ.get("LOOK_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / PROJECT_FILE).is_file():
            raise FileNotFoundError(f"LOOK_PROJECT_ROOT has no {PROJECT_FILE}: {root}")
        return root
    origin = (start or Path(__file__)).resolve()
    for candidate in (origin, *origin.parents):
        directory = candidate if candidate.is_dir() else candidate.parent
        if (directory / PROJECT_FILE).is_file():
            return directory
    raise FileNotFoundError(f"Could not find {PROJECT_FILE} above {origin}")


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    dataset_root: Path
    cache_root: Path
    runs_root: Path
    tool_root: Path
    pipeline_root: Path

    @classmethod
    def load(
        cls,
        project_root: Path | None = None,
        data_root: Path | None = None,
        dataset_root: Path | None = None,
        cache_root: Path | None = None,
        runs_root: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ProjectPaths":
        env = os.environ if environ is None else environ
        root = (project_root or discover_project_root()).expanduser().resolve()
        config = json.loads((root / PROJECT_FILE).read_text(encoding="utf-8"))
        directories = config["directories"]
        configured_data = data_root or env.get("LOOK_DATA_ROOT") or config["deployment_defaults"]["data_root"]
        resolved_data = Path(configured_data).expanduser()
        configured_dataset = config["deployment_defaults"].get(
            "dataset_root", resolved_data / directories["dataset"]
        )
        dataset = Path(dataset_root or env.get("LOOK_DATASET_ROOT", configured_dataset))
        cache = Path(cache_root or env.get("LOOK_CACHE_ROOT", resolved_data / directories["cache"]))
        runs = Path(runs_root or env.get("LOOK_RUNS_ROOT", resolved_data / directories["runs"]))
        return cls(
            project_root=root,
            data_root=resolved_data.resolve(),
            dataset_root=dataset.resolve(),
            cache_root=cache.resolve(),
            runs_root=runs.resolve(),
            tool_root=(root / directories["tool"]).resolve(),
            pipeline_root=(root / directories["pipeline"]).resolve(),
        )

    def ensure_runtime_layout(self) -> None:
        for path in (
            self.dataset_root,
            self.cache_root / "pipeline_state",
            self.cache_root / "partial",
            self.cache_root / "quarantine",
            self.runs_root / "experiments",
            self.runs_root / "sweeps",
            self.runs_root / "smoke",
            self.runs_root / "backbones",
            self.runs_root / "generators",
            self.runs_root / "freezes",
        ):
            path.mkdir(parents=True, exist_ok=True)
