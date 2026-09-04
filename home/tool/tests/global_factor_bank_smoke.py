"""Fit real spatial LOOK banks at multiple shared factors on a tiny cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from look_core.data import UKBBilateralVisitDataset, make_loader
from look_core.graph import build_resnet50_mhd_graph
from look_core.look import greedy_fit_look, validate_global_factor_bank
from look_core.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = ProjectPaths.load(args.project_root)
    dataset_kwargs = {
        "labels_csv": paths.labels_csv,
        "data_root": paths.image_root,
        "image_size": 224,
        "limit": 8,
        "preprocess_cache_root": paths.preprocess_cache_root,
    }
    train = UKBBilateralVisitDataset(split="train", augment=False, **dataset_kwargs)
    validation = UKBBilateralVisitDataset(
        split="validation", augment=False, **dataset_kwargs
    )
    train_loader = make_loader(train, 2, 0, False, 3407)
    validation_loader = make_loader(validation, 2, 0, False, 3407)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph = build_resnet50_mhd_graph(
        "layer3", batch_size=2, pretrained=False, device=device
    )
    nodes = ["fusion_layer3", "fusion_feature"]
    artifacts, _ = greedy_fit_look(
        graph,
        train_loader,
        validation_loader,
        "oct_missing",
        nodes,
        [4, 8, 16],
        [2],
        2,
        device,
        args.output,
        resume=False,
    )
    selection = json.loads(
        (args.output / "factor_selection.json").read_text(encoding="utf-8")
    )
    assert [item["factor"] for item in selection["factor_banks"]] == [4, 8, 16]
    assert all(
        (args.output / "factors" / f"x{factor}" / "bank_complete.json").is_file()
        for factor in (4, 8, 16)
    )
    validate_global_factor_bank(
        artifacts, nodes, int(selection["selected_factor"])
    )
    assert artifacts[0].factor == selection["selected_factor"]
    assert artifacts[1].factor == 1
    print(
        f"GLOBAL FACTOR BANK SMOKE PASS: selected x{selection['selected_factor']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
