#!/usr/bin/env python3
"""Build or validate the lossless deterministic CFP/OCT preprocessing cache."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from look_core.data import PREPROCESS_CACHE_VERSION, ensure_preprocessed_pair_cache
from look_core.paths import ProjectPaths
from look_core.reproducibility import sha256, write_json_atomic


def _warm_one(task: tuple[str, str, str, int, str]) -> str:
    data_root, fundus_path, oct_path, image_size, cache_root = task
    return str(ensure_preprocessed_pair_cache(
        Path(data_root), fundus_path, oct_path, image_size, Path(cache_root)
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.image_size < 1:
        raise ValueError("workers and image-size must be positive")
    paths = ProjectPaths.load(
        project_root=args.project_root,
        data_root=args.data_root,
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
    )
    labels_csv = paths.dataset_root / "reference_labels.csv"
    frame = pd.read_csv(labels_csv, usecols=["split", "fundus_path", "oct_path"])
    pairs = frame.drop_duplicates(["fundus_path", "oct_path"], keep="first")
    cache_root = paths.cache_root / "preprocessed_pairs"
    tasks = (
        (
            str(paths.dataset_root),
            str(row.fundus_path),
            str(row.oct_path),
            args.image_size,
            str(cache_root),
        )
        for row in pairs.itertuples(index=False)
    )
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for _ in tqdm(
            executor.map(_warm_one, tasks, chunksize=8),
            total=len(pairs),
            desc="Preprocessing cache",
        ):
            pass
    summary = {
        "status": "complete",
        "cache_version": PREPROCESS_CACHE_VERSION,
        "image_size": args.image_size,
        "pairs": int(len(pairs)),
        "split_counts": frame.groupby("split").size().astype(int).to_dict(),
        "workers": args.workers,
        "labels_csv": str(labels_csv),
        "labels_sha256": sha256(labels_csv),
        "cache_root": str(cache_root),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(summary, cache_root / PREPROCESS_CACHE_VERSION / "warm_complete.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
