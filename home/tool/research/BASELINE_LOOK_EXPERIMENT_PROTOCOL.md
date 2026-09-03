# Baseline And LOOK Experiment Protocol

## Cohorts

- Primary: balanced four-class record-derived prevalent phenotype cohort.
- Secondary: same four classes at natural observed prevalence.
- Auxiliary only: post-imaging incident phenotype cohort, excluded from selection.
- Classes: Normal, diabetes-related eye disease, Glaucoma, macular degeneration.
- Cataract is outside the current formal task.
- One row is the earliest complete bilateral CFP/OCT visit for one participant.
- Participant-level 70/15/15 splits are deterministic and test remains sealed.

## Baseline

- ImageNet1K V2 ResNet50 for CFP and OCT branches.
- Fusion positions: input, stem, layer1, layer2, layer3, layer4 and feature.
- Fusion: concatenation, linear `1x1`/dense projection, normalization; no fusion
  activation or nonlinear attention.
- One-stage end-to-end AdamW fine-tuning with unweighted cross-entropy and natural
  sampling without replacement.
- Calibration: LR pairs `3e-5/3e-4`, `1e-4/1e-3`, `3e-4/3e-3`; dropout `0.0/0.2`.
- Stage A: seven positions at seed 3407. Stage B: top three at 3407/3408/3409.
- OCT-only and CFP-only use true single active branches under the same profile.
- Left/right eyes share modality weights and are aggregated by a parameter-free MHD
  bilateral-mean Edge before the classifier.

## LOOK Controls

- Complete-modality backbone is frozen before any missing-modality evaluation.
- Filling arms: normalized mean and independently trained paired cGAN pseudo-channel.
- No missing-input fine-tuning and no joint classifier/GAN/LOOK fine-tuning.
- Missing patterns: OCT missing and CFP missing; random missing ratios are secondary.
- Shared spatial compression factor per artifact bank: 4, 8 or 16; vector feature nodes
  use identity compression.
- Node-specific PCA latent dimension and ridge alpha are selected on validation.
- Residual correction is lifted to the original node shape and injected into MHD node
  state. Correction matrices receive spectral/effective-rank/condition/R2 analysis.

## Freeze Boundaries

Baseline approval is explicit. LOOK validation artifacts are then hashed in a study
freeze manifest. Balanced and natural tests reject unlisted or changed configurations.
