# Current Iteration

## 2026_09_03_08_30_00

- Replaced eye-level self-report labels with participant-level bilateral record-derived
  phenotypes combining self-report, ICD timing and procedure evidence.
- Added separate prevalent, incident and temporally uncertain handling; labels are
  explicitly not expert image-grading gold standards.
- Defined one sample as the earliest complete bilateral CFP/OCT visit and added an MHD
  bilateral-mean Edge before participant classification.
- Removed classifier re-training, class-balanced replacement sampling and all related
  configuration/checkpoint fields.
- Added one-stage end-to-end baseline calibration, seven linear fusion topologies,
  three-seed confirmation, OCT-only/CFP-only references and a compound validation gate.
- Added natural sealed-test evaluation, AUPRC/per-class ranking evidence and automatic
  label/image/model quality audit output.
- Consolidated runtime defaults around one current physical dataset and preprocessing
  cache while retaining explicit path portability.
- Updated notebook, detached operation, documentation and tests for the current protocol.
- Processed 77,118 participants with complete bilateral CFP/OCT: 1,810 mutually exclusive
  prevalent cases, 1,946 incident cases and 70,128 strict controls; excluded 3,234 target-
  comorbid or competing-eye-condition participants.
- Final balanced counts are Normal 925, DR 449, Glaucoma 925 and AMD 436. The 70/15/15
  participant split has 1,913/408/414 samples and no missing images or split leakage.
- Added a protocol-level data adequacy audit. The cohort passes method-development and
  internal-testing minimums, while explicitly remaining unsuitable for clinical-deployment
  claims without expert image grading and an independent external test.
- Removed the superseded weak-label cohort, old checkpoints and stale resume state after
  the new cohort passed integrity verification.
- Verified 83 tests, two-GPU criteria aggregation, nine MHD graph topologies, the complete
  filling/LOOK/sealed-test smoke, and exact ImageNet V2 mapping to 12 active branch edges.
- Invalidated the first baseline search after all fusion positions showed rapid training
  memorization and low validation Macro-F1. The audit confirmed aligned labels/predictions,
  correct image pairing, exact ImageNet mapping, and correct MHD metrics/backward behavior.
- Fixed stale epoch-conditioned augmentation by sharing epoch state with persistent
  data-loader workers, retained the final DDP-safe partial training batch, and reduced
  effective batch size from 256 to 128 to double optimizer updates per epoch.
- Replaced bilateral mean-only aggregation with parameter-free mean-plus-max pooling to
  preserve unilateral disease evidence without changing the linear modality-fusion rule.
- Added label-smoothing calibration and per-run evidence-strength diagnostics; removed
  the aggressively overfitting `3e-4/3e-3` learning-rate profile.
