# Publication Protocol

## Claims

- Target evidence standard: IEEE TMI / Medical Image Analysis method study.
- Labels are record-derived glaucoma phenotypes, not expert image grades.
- The analysis unit is one participant's earliest complete bilateral CFP/OCT visit.
- The matched primary cohort estimates controlled discrimination, not population
  prevalence. Natural-prevalence test results are reported separately.
- LOOK is an offline affine latent correction and never fine-tunes the frozen classifier.

## Locked Order

1. Verify evidence timing, high-confidence rules, matching, image references, and splits.
2. Select a conventional complete-modality baseline using validation only.
3. Require three-seed stability, unimodal references, quality gates, and explicit review.
4. Freeze the baseline before training filling generators or fitting LOOK.
5. Fit raw-zero, normalized-mean, paired cGAN, and LOOK controls from training data only.
6. Select spatial factor, PCA dimension, and ridge lambda on validation only.
7. Freeze all artifacts before primary and natural-prevalence tests.

## Reporting

AUROC is the primary binary endpoint. Also report AUPRC, Macro-F1, balanced accuracy,
weighted F1 as descriptive, sensitivity, specificity, ECE, Brier score, confusion
matrices, participant-bootstrap 95% intervals, paired bootstrap differences, and Holm
multiple-comparison adjustment. Preserve participant-level predictions.

State cohort flow, evidence definitions, exclusions, matching diagnostics, class counts,
preprocessing, ImageNet weight source, fusion operation, optimizer schedule, seeds,
hardware, software versions, checkpoint criterion, early stopping, and every searched
setting. Negative and below-gate results remain auditable.

## Gates And Limitations

Baseline review requires three-seed mean AUROC at least 0.80, mean Macro-F1 at least
0.70, glaucoma sensitivity and specificity at least 0.65, stable training, and no AUROC
deficit against the best unimodal reference. Failure triggers phenotype, timing, image
quality, modality visibility, preprocessing, and weight-mapping audits. Test data cannot
be consulted to raise scores.

The primary cohort requires at least 500 cases, 350 training cases, and 100 cases in both
validation and internal test. These transparent project thresholds are not journal
acceptance rules. Expert grading and an independent external clinical test are
unavailable, so clinical-deployment claims remain out of scope.
