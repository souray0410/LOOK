# Changelog

## 2026_09_01_16_06_11

- Added one- and two-GPU DDP selected only through a physical GPU list.
- Added the tested NCCL transport default required by the current `ws` GPU pair.
- Added graph-internal label, differentiable loss, and batch-accuracy monitor nodes.
- Added rank-zero training histories and classifier/cGAN curve exports.
- Added validation freeze manifests and read-only sealed test execution.
- Moved source, cache, and runs to `2026_09_01_16_06_11` while reusing the validated
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
