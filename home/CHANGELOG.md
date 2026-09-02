# Changelog

## 2026-09-02 complete-modality baseline search

- Synchronized the upstream V4 Mermaid phase/level visualization update while retaining
  the verified LOOK multi-GPU utility implementation.
- Replaced weighted replacement sampling with deterministic natural sampling without
  replacement and moved effective-number class-balanced cross entropy into the MHD loss
  hyperedge.
- Defined activation-free fusion as concatenation, one linear projection, and
  normalization; classifier dropout remains outside the fusion operation.
- Added a 168-configuration validation-only baseline search over all seven fusion
  positions, class-balance beta, discriminative learning rates, dropout and label
  smoothing. Filling, cGAN, LOOK and sealed-test evaluation are disabled in this stage.
- Added atomic partial leaderboards and grouped diagnostics, plus a detached checker that
  exposes current configuration, complete and per-class validation evidence, confusion
  matrix, best epoch, GPU state and recent logs.
- Calibrated effective-number betas to the observed 113:1 training imbalance. The search
  uses `0.9999`, `0.99995`, and `0.99999`; their normal-class shares of total weighted
  training contribution are approximately 70%, 56%, and 29%, respectively.
- Explicitly releases parent-process CUDA cache between sequential configurations so
  rank-zero validation cannot reduce memory headroom for the next two-rank DDP stage.

## 2026-09-02 data-loader and preprocessing calibration

- Live profiling identified repeated paired PNG decoding, CFP ROI extraction, resizing,
  and augmentation, rather than V4 graph backward or DDP collectives, as the remaining
  throughput bottleneck. Testing 32 workers per rank saturated all 128 CPUs and was
  slower, so the validated 16-worker profile is retained.
- Added a lossless, version-keyed, atomically written cache for deterministic pre-
  augmentation CFP/OCT preprocessing. Dynamic augmentation, batch sizes, optimization,
  and all scientific experiment axes remain unchanged.
- Record an explicit `interrupted` detached-run status when the worker receives
  `SIGINT` or `SIGTERM`, so a stopped session cannot remain labelled as running.

## 2026_09_02_00_00_00

- Replaced the embedded MHD V3 package with the reviewed MHD Framework V4 source.
- Routed classifier backward propagation through V4 Gradient Messages and
  `MHD_Graph.backward(...)` for single-device and DDP training.
- Added backward-topology pruning and numerical gradient/optimizer equivalence tests.
- Added one- and two-GPU DDP selected only through a physical GPU list.
- Added the tested NCCL transport default required by the current `ws` GPU pair.
- Added full-state cross-rank SHA-256 verification before safely skipping the slow,
  redundant DDP initial broadcast on `ws`; gradient synchronization remains enabled.
- Calibrated classifier batch 128/GPU to 16.609 GiB peak and paired-cGAN batch 224/GPU
  to 13.858 GiB peak on two RTX 5000 Ada GPUs.
- Defined micro-batches per device and effective batches globally, with automatic
  single-/multi-GPU gradient accumulation selected only by the GPU list.
- Kept LOOK inference on model levels so loss/metric monitor hyperedges execute only
  when labels are supplied.
- Added per-epoch live validation summaries, curves, generalization gaps, and
  class-specific validation figures to classifier monitoring.
- Added graph-internal label, differentiable loss, and batch-accuracy monitor nodes.
- Added rank-zero training histories and classifier/cGAN curve exports.
- Added validation freeze manifests and read-only sealed test execution.
- Moved source, cache, and runs to `2026_09_02_00_00_00` while reusing the validated
  dataset path from the preceding release.

## 2026_08_30_11_20_47

- Released a portable `home/` source tree and independent `data/` runtime tree.
- Centralized author defaults while supporting explicit Linux path overrides.
- Established canonical `look_core` and `MHD_Project` imports with no aliases.
- Added resumable single-GPU ResNet50, paired-cGAN, LOOK, sweep and matrix analysis.
- Added safe structured batch execution and content-addressed file-selection plans.
- Added self-repairing environment setup and a validated portable Jupyter kernelspec.
- Documented the five-class reference-standard limitation and sealed-test protocol.
- Removed prior migration branches, compatibility wrappers and generated run products.
