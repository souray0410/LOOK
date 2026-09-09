# Runtime validation

`scripts/validate_runtime.py` checks a manifest-accepted train/development cache using real reference weights, four disposable optimizer steps, strict checkpoint roundtrip and one finite inference pass over each split. It reports execution integrity, not scientific performance, and never reads test images or calculates model rankings.

Activate the project environment, then pass `--project`, `--data-root`, `--output` and `--manifest-sha`. Use `--prepare-only` before allocating a GPU to verify dataset construction and batch loading. The runtime requires exactly one visible CUDA device. Output belongs outside the source tree. The paired applications can be scheduled as independent single-GPU Slurm array tasks, so each releases resources when finished. Do not add sleep loops or unplanned training to hold a GPU.

The 2026-09-09 cache copy preserves project-specific preprocessing: LOOK uses bilateral 2D cached images, while Radon Bridge retains bilateral 3D OCT downsampled to 32 slices and its existing CFP224 cache. Their train/development identity and labels are verified separately from image preprocessing. Original full label manifests are retained for provenance; runtime loaders use train/development only.

Batch16 and four updates are bounded engineering checks, not resumed historical training or a convergence experiment. Reference checkpoint scores are not recomputed for publication. Files and labels stay in authorized storage; only aggregate acceptance records may be shared.

## Two-GPU acceptance

`scripts/validate_distributed_runtime.py` is launched by `torchrun --standalone --nproc-per-node=2`. Both ranks run the same application through `MHD_Trainer` and its V4 DDP adapter. LOOK and R&B run sequentially within a two-GPU allocation. Global batch remains 16 (8 per rank); four optimizer updates are disposable. Gradients are checked for finite values and identical SHA digests across ranks before each optimizer step; updated parameters and original Node IDs are checked too. Dataset shards are disjoint and cover each split exactly once.

Ordinary BN uses each rank's local batch, so these updates are not claimed equivalent to single-GPU batch16 training. Rank0 BN buffers explicitly define the shared checkpoint, which both ranks strictly reload and replay. Both ranks complete finite train/development inference without test access. The entry checks DDP execution, not FSDP, tensor or pipeline parallelism, and does not extend the scientific queue. Each rank writes progress and acceptance; the controller fails on missing progress and measures project process memory separately on each allocated GPU. Resources are released at completion or failure.
