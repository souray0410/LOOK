from __future__ import annotations

import argparse
from pathlib import Path

from .defaults import DeploymentDefaults
from .paths import ProjectPaths, discover_project_root


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, help="Source root containing project.json.")
    parser.add_argument("--data-root", type=Path, help="Runtime root containing dataset/cache/runs.")
    parser.add_argument("--dataset-root", type=Path, help="Override only the validated dataset root.")
    parser.add_argument("--image-root", type=Path, help="Immutable root containing referenced CFP/OCT images.")
    parser.add_argument("--cohort-root", type=Path, help="Root containing balanced and natural cohort manifests.")
    parser.add_argument("--labels-csv", type=Path, help="Primary balanced four-class label table.")
    parser.add_argument("--natural-labels-csv", type=Path, help="Secondary natural-prevalence four-class label table.")
    parser.add_argument("--preprocess-cache-root", type=Path, help="Reusable image preprocessing cache.")
    parser.add_argument("--cache-root", type=Path, help="Override only the resumable cache root.")
    parser.add_argument("--runs-root", type=Path, help="Override only the experiment output root.")


def resolve_runtime_arguments(args: argparse.Namespace) -> ProjectPaths:
    project_root = (args.project_root or discover_project_root()).resolve()
    return ProjectPaths.load(
        project_root=project_root,
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        image_root=args.image_root,
        cohort_root=args.cohort_root,
        labels_csv=args.labels_csv,
        natural_labels_csv=args.natural_labels_csv,
        preprocess_cache_root=args.preprocess_cache_root,
        cache_root=args.cache_root,
        runs_root=args.runs_root,
    )


def add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-uuid", help="Source volume UUID; defaults to project.json.")
    parser.add_argument("--source-mount", type=Path, help="Source mount; defaults to project.json.")
    parser.add_argument("--source-root", type=Path, help="UKB image root; defaults to project.json.")
    parser.add_argument("--label-uuid", help="Label volume UUID; defaults to project.json.")
    parser.add_argument("--label-mount", type=Path, help="Label mount; defaults to project.json.")
    parser.add_argument("--label-root", type=Path, help="Phenotype CSV root; defaults to project.json.")


def resolve_source_arguments(args: argparse.Namespace) -> argparse.Namespace:
    project_root = (getattr(args, "project_root", None) or discover_project_root()).resolve()
    defaults = DeploymentDefaults.load(project_root)
    for name in (
        "source_uuid",
        "source_mount",
        "source_root",
        "label_uuid",
        "label_mount",
        "label_root",
    ):
        if getattr(args, name, None) is None:
            setattr(args, name, getattr(defaults, f"ukb_{name}"))
    return args
