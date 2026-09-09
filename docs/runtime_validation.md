# Runtime validation

`scripts/validate_runtime.py` checks a manifest-accepted train/development cache using real reference weights, four disposable optimizer steps, strict checkpoint roundtrip and one finite inference pass over each split. It reports execution integrity, not scientific performance, and never reads test images or calculates model rankings.

Activate the project environment, then pass `--project`, `--data-root`, `--output` and `--manifest-sha`. Use `--prepare-only` before allocating a GPU to verify dataset construction and batch loading. The runtime requires exactly one visible CUDA device. Output belongs outside the source tree. The paired applications can be scheduled as independent single-GPU Slurm array tasks, so each releases resources when finished. Do not add sleep loops or unplanned training to hold a GPU.

The 2026-09-09 cache copy preserves project-specific preprocessing: LOOK uses bilateral 2D cached images, while Radon Bridge retains bilateral 3D OCT downsampled to 32 slices and its existing CFP224 cache. Their train/development identity and labels are verified separately from image preprocessing. Original full label manifests are retained for provenance; runtime loaders use train/development only.

Batch16 and four updates are bounded engineering checks, not resumed historical training or a convergence experiment. Reference checkpoint scores are not recomputed for publication. Files and labels stay in authorized storage; only aggregate acceptance records may be shared.
