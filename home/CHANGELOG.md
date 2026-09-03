# Current Iteration

## 2026_09_03_19_35_04

- Replaced the weak four-class benchmark with a binary glaucoma task.
- Initially defined a 716-case high-confidence glaucoma profile, then compared it with
  six prespecified alternatives in a fixed validation-only task scout.
- Added deterministic 1:1 controls matched within participant splits by age, sex, and
  assessment centre, plus a natural-prevalence sensitivity cohort.
- Kept bilateral CFP and one central 2D OCT B-scan per eye; no 3D OCT is used.
- Changed bilateral aggregation from mean-plus-max to a parameter-free linear mean.
- Changed baseline ranking and checkpoint selection to complete-validation AUROC, with
  Macro-F1, sensitivity, specificity, calibration, and unimodal comparisons as gates.
- Added raw-zero as a distinct filling baseline beside normalized-mean and paired cGAN.
- Removed residual-strength alpha from LOOK. Ridge GCV lambda remains the sole
  correction regularization parameter.
- Preserved MHD Framework V4 unchanged and retained single- or two-GPU execution through
  one GPU-list setting.
- Preserved superseded Steps 12-20 in a timestamped, non-executable history area and
  continued active development at Steps 21-32.
- Added seven validation-only task candidates spanning evidence quality and sample size,
  with a fixed short feature-fusion scout before any formal baseline commitment.
- Added a validation-only UKB data/task usability summary and explicit parent-process
  CUDA cleanup between scout candidates.
- Selected `glaucoma_all_evidence` for formal baseline qualification: 925 cases and 925
  matched controls; scout AUROC 0.6948 and Macro-F1 0.6402 under the fixed short budget.
- Added a hash-checked Step 32 selection manifest. The decision does not freeze a
  baseline, approve LOOK, or access sealed test data.
