# Publication Protocol

## Status And Claims

- Target venues: IEEE TMI / Medical Image Analysis standard of evidence.
- Current labels are record-derived clinical phenotypes based on available self-report,
  hospital ICD timing and procedure evidence. They are not expert image-grading gold
  standards.
- The unit of analysis is one participant's earliest complete bilateral CFP/OCT visit.
- The balanced primary cohort supports controlled method comparison, not prevalence
  estimation. Natural-distribution evaluation is reported separately.
- LOOK remains an offline linear/affine latent correction fitted on training features;
  it does not fine-tune the frozen classifier backbone.

## Locked Experimental Order

1. Verify phenotype timing, bilateral participant-level splits and cohort manifests.
2. Select a conventional complete-modality baseline using validation only.
3. Require three-seed stability, unimodal references and explicit scientific review.
4. Freeze the baseline before fitting filling or LOOK components.
5. Fit mean filling, independent paired cGAN filling and LOOK from training data only.
6. Select LOOK factor/latent/ridge settings on validation only.
7. Freeze all artifacts before balanced and natural sealed tests.

## Required Reporting

Report Macro-F1 as primary endpoint plus balanced accuracy, weighted F1 as descriptive,
Macro/per-class AUROC and AUPRC, sensitivity, specificity, ECE, confusion matrices,
participant-bootstrap 95% confidence intervals, paired bootstrap differences and
multiple-comparison adjustment. Preserve predictions and participant IDs.

State cohort flow, exclusions, class counts, deterministic Normal sampling, image
preprocessing, ImageNet weight source, fusion operation, optimizer schedule, seeds,
hardware, software versions, checkpoint criterion, early stopping and all searched
settings. Negative and below-gate results remain auditable.

## Quality Gate

Review requires three-seed mean Macro-F1 `>=0.65`, every mean class F1 `>=0.45`, stable
training and multimodal performance no worse than the best unimodal reference. A failure
triggers phenotype-source/timing, confusion, image quality, modality visibility,
preprocessing and weight-mapping audits. Test data may not be consulted to raise scores.

## Data Adequacy Gate

Before baseline search, require at least 2,000 balanced participants, 400 total and 250
training participants per disease class, and 60 participants per class in validation and
internal test. Save class-specific Wilson 95% interval precision. These project thresholds
guard against an obviously underpowered pilot; they are not journal acceptance criteria.

Follow CLAIM 2024 terminology: use `reference standard`, `validation` only for tuning,
`internal test` for the sealed UKB split, and `external test` only for an independent data
source. Report cohort flow, inclusion/exclusion, acquisition context, class prevalence,
demographics, missingness, all partitions, calibration and uncertainty. State explicitly
that expert image grading and external clinical testing are unavailable in this release.
