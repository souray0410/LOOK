# Current Iteration

## 2026_09_03_19_35_04

- Replaced the weak four-class benchmark with a binary glaucoma task.
- Defined 716 high-confidence prevalent glaucoma cases using pre-imaging HES ICD,
  glaucoma-specific treatment/procedure evidence, or concordant 6148 and 20002
  self-report.
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
  continued active development at Steps 21-31.
- Added seven validation-only task candidates spanning evidence quality and sample size,
  with a fixed short feature-fusion scout before any formal baseline commitment.
- Added a validation-only UKB data/task usability summary and explicit parent-process
  CUDA cleanup between scout candidates.
