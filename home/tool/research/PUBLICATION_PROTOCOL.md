# Publication Protocol

## Status And Claims

- Target venues: IEEE TMI / Medical Image Analysis standard of evidence.
- Current labels are laterality-aware doctor-informed self-report and must be called a
  weak reference. They are not expert image-grading gold standard.
- The balanced primary cohort supports controlled method comparison, not prevalence
  estimation. Natural-distribution evaluation is reported separately.
- LOOK remains an offline linear/affine latent correction fitted on training features;
  it does not fine-tune the frozen classifier backbone.

## Locked Experimental Order

1. Verify participant-level splits and four-class cohort manifests.
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

Balanced validation `Macro-F1 >= 0.70` permits review but does not guarantee publication
quality. A lower score triggers label consistency, confusion, image quality, modality
visibility, preprocessing and weight-mapping audits. Test data may not be consulted to
raise the score.
