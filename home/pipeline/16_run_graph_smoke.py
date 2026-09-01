from __future__ import annotations

import argparse
import os


def main() -> None:
    from look_core.cli import add_runtime_arguments, resolve_runtime_arguments

    parser = argparse.ArgumentParser()
    add_runtime_arguments(parser)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    from look_core.config import ExperimentConfig
    from look_core.data import UKBPairedEyeDataset, make_loader, validate_reference_table
    from look_core.graph import FUSION_POSITIONS, build_resnet50_mhd_graph, graph_summary, reset_and_forward
    from look_core.reproducibility import enforce_single_gpu

    enforce_single_gpu()
    paths = resolve_runtime_arguments(args)
    config = ExperimentConfig(
        data_root=paths.dataset_root,
        output_root=paths.runs_root / "smoke",
        cache_root=paths.cache_root,
        num_workers=0,
        micro_batch_size=2,
        effective_batch_size=2,
    )
    audit = validate_reference_table(config.labels_csv, config.data_root, check_paths=False)
    print("Data audit:", audit)
    dataset = UKBPairedEyeDataset(
        config.labels_csv, config.data_root, "validation", image_size=224, limit=2
    )
    loader = make_loader(dataset, 2, 0, train=False, seed=config.seeds[0])
    batch = next(iter(loader))
    device = torch.device("cuda:0")
    for position in FUSION_POSITIONS:
        graph = build_resnet50_mhd_graph(position, batch_size=2, device=device, pretrained=False)
        with torch.no_grad():
            logits = reset_and_forward(
                graph, batch["oct"].to(device), batch["cfp"].to(device)
            )
        assert logits.shape == (2, 5)
        print(graph_summary(graph))
        del graph
        torch.cuda.empty_cache()
    print("SMOKE PASS: seven topologies, one visible GPU, paired UKB images")


if __name__ == "__main__":
    main()
