# Baseline And LOOK Experiment Protocol

## Cohorts

- Primary: 925 all-evidence prevalent glaucoma cases and 925 matched strict controls.
- Secondary: the same all-evidence definition and all strict controls in the held-out
  natural-prevalence test split.
- Evidence sensitivity: objective, concordant self-report, single-source self-report,
  and the nested 716-case high-confidence subset are reported separately where feasible.
- Auxiliary: post-imaging incident glaucoma, excluded from model selection.
- One sample is one participant's earliest complete bilateral CFP/OCT visit.

## Baseline

- ImageNet1K V2 ResNet50 branches for 2D CFP and central 2D OCT.
- Left/right eyes share modality weights and use a parameter-free linear bilateral mean.
- Fusion positions: input, stem, layer1, layer2, layer3, layer4, and feature.
- Fusion is concatenation, linear projection, and normalization without an added
  activation or attention block.
- One-stage AdamW fine-tuning with unweighted cross-entropy and natural sampling.
- Calibration uses pretrained/new-layer LR pairs `3e-5/3e-4`, `1e-4/1e-3`, and
  `3e-4/3e-3`, each with classifier dropout `0.0` or `0.2`.
- Validation AUROC selects checkpoints and configurations. Macro-F1, balanced accuracy,
  sensitivity, specificity, AUPRC, and ECE remain required evidence.
- OCT-only and CFP-only references must be trained under the selected profile.

## LOOK Controls

- Freeze the complete-modality backbone before missing-modality experiments.
- Filling arms: raw zero, normalized mean, and independently trained paired cGAN.
- No missing-input fine-tuning or joint classifier/GAN/LOOK optimization.
- Missing patterns: OCT missing and CFP missing; ratios 20, 40, 60, 80, and 100 percent.
- A complete artifact bank uses one spatial factor from 4, 8, or 16; vector nodes use
  identity compression.
- Validation selects PCA latent dimension and GCV ridge lambda.
- LOOK applies `z_corrected = z_missing + delta_z`; there is no residual alpha.
- Matrix analysis covers W and I+W, singular spectra, eigenvalues, effective rank,
  condition number, train MSE, and train R2.

## Freeze Boundaries

Step 32 first locks the selected task and label hashes. The baseline must then pass its
validation gate and receive explicit approval. LOOK
validation artifacts are then hashed into a frozen manifest. Primary and natural tests
reject any unlisted or changed configuration.
