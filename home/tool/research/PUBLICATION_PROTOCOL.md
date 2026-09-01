# Publication Protocol: UKB CFP-OCT LOOK

## Claim boundary

This code supports a controlled 2D incomplete-modality feasibility study. The
reference labels are laterality-aware, doctor-informed participant self-reports;
they are not expert image adjudications. OCT input is one central B-scan. CFP
and OCT are paired at eye/visit level but are not spatially registered.

Accordingly, the cGAN output is evaluated as a pseudo-channel for downstream
classification, not as a clinically faithful reconstruction. Publication in
Medical Image Analysis or IEEE TMI cannot be guaranteed by software quality.
Submission-strength claims require stronger reference labels, full-volume OCT,
external or multicenter validation, and comparison with current trained
incomplete-modality methods.

## Locked causal comparison

The classifier is ImageNet-initialized and trained once using complete CFP-OCT
pairs. The selected checkpoint is frozen before any modality is removed. No
missing-input classifier fine-tuning, modality-dropout classifier training, or
joint generator-classifier optimization is part of the proposed experiment.

For every fusion topology and seed, report these five conditions:

1. Complete CFP and OCT.
2. Normalized-mean fill with frozen classifier.
3. Normalized-mean fill plus LOOK with frozen classifier.
4. Independently trained paired-cGAN fill with frozen classifier.
5. Paired-cGAN fill plus LOOK with frozen classifier.

Because filling with the ImageNet normalization mean is numerically zero after
normalization, it is one baseline and must not be double-counted as separate
"mean" and "zero" methods.

## Split and fitting rules

- Official split: deterministic participant-level 70/15/15 train, validation,
  and sealed test; both eyes and repeated visits stay with the participant.
- Classifier: complete-modality training rows only; validation Macro F1 selects
  the checkpoint.
- cGAN: a deterministic participant-level internal split of official training
  rows; two independently optimized directions; validation L1 selects each
  generator. No official validation or test image trains either generator.
- LOOK: official training features fit PCA and ridge correction; official
  validation selects node sequence, downsampling factor, latent rank, and alpha.
- Test: accessed once after architecture, generator, LOOK, endpoints, and
  comparisons are frozen.
- Repetition: preregistered seeds `3407`, `3408`, and `3409`.

## Endpoints and inference

- Primary endpoint: Macro F1 on the natural-distribution participant-held-out
  test set.
- Secondary endpoints: weighted F1, macro one-vs-rest AUROC, Cohen's Kappa,
  balanced accuracy, accuracy, class sensitivity/specificity, ECE, and
  multiclass Brier score.
- Missing settings: fixed OCT missing, fixed CFP missing, and participant-stable
  20/40/60/80% random missingness with balanced missing directions.
- Uncertainty: participant-clustered 95% bootstrap intervals.
- Paired test: LOOK-after-fill versus the same fill-only condition, with Holm
  adjustment across confirmatory fixed-direction comparisons.

## Required ablations

- Seven fusion positions: input, stem, layer1, layer2, layer3, layer4, feature.
- Both missing directions and all preregistered random missing ratios.
- LOOK downsampling factors, PCA latent dimensions, alpha, and correction nodes.
- Three random seeds with identical participant splits.
- Filling-strategy interaction: normalized mean versus paired cGAN, each with
  and without LOOK.
- Resource reporting: parameter count, generator cost, LOOK fitting time,
  inference latency, GPU model/memory, and storage.
- Matrix mechanism: numerical/effective/stable rank, singular spectrum,
  eigen-spectrum, non-normality, cross-latent energy, and the conditioning of
  `I + alpha W`, compared across nodes, seeds, missing directions, and fillers.

Matrix eigenvectors live in the learned PCA latent coordinates. For the
row-vector operation `zW`, left eigenvectors are input directions. Singular
vectors should be the primary interpretation when W is non-normal; eigenvalue
plots must not be presented as direct anatomical or causal evidence.

External trained baselines such as modality dropout, synthesis/diffusion, and
recent incomplete-multimodal methods are recommended for publication context.
They must be trained and tuned under the same participant splits and compute
budget, but they are comparison methods, not variants of the no-retraining LOOK
pipeline.

## Result integrity

One notebook configuration produces one SHA-256-fingerprinted experiment. The
manifest records source hashes, data-table hash, environment, configuration,
and audit. Shared classifier and generator checkpoints have independent
scientific identities. Raw probabilities and participant IDs are retained for
all metrics. Intermediate checkpoints are resume artifacts; only a completed
`experiment_result.json` registered in `experiment_registry.json` is reportable.

For final runs, `SMOKE_LIMIT=None`, `CHECK_ALL_IMAGE_PATHS=True`, and test access
requires `FROZEN_TEST_CONFIRMATION="CONFIGURATION_FROZEN"`. Repeated test runs
must not be used for model selection.

## Submission gate

- [ ] Five locked conditions completed for all declared topologies and seeds.
- [ ] Test split unsealed only after a timestamped analysis freeze.
- [ ] Participant-clustered intervals and corrected paired tests reported.
- [ ] Per-class confusion, calibration, and clinically reviewed failures shown.
- [ ] cGAN image fidelity and downstream utility reported separately.
- [ ] Cross-view non-registration and central-slice OCT limitations explicit.
- [ ] Label provenance never described as expert image-grading gold standard.
- [ ] Expert-adjudicated subset or stronger composite reference standard added.
- [ ] External/multicenter validation completed or study framed as feasibility.
- [ ] Independent reproduction completed from numbered scripts and README only.
