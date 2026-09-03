# Current Iteration

## 2026_09_03_08_30_00

- Replaced the imbalanced five-class training task with balanced-primary and natural-
  secondary four-class weak-reference cohorts; Cataract is excluded.
- Removed classifier re-training, class-balanced replacement sampling and all related
  configuration/checkpoint fields.
- Added one-stage end-to-end baseline calibration, seven linear fusion topologies,
  three-seed confirmation, OCT-only/CFP-only references and a 0.70 validation review gate.
- Added natural sealed-test evaluation, AUPRC/per-class ranking evidence and automatic
  label/image/model quality audit output.
- Consolidated runtime defaults around one current physical dataset and preprocessing
  cache while retaining explicit path portability.
- Updated notebook, detached operation, documentation and tests for the current protocol.
