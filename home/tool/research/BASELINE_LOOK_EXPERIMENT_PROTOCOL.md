# Baseline Selection and LOOK Evaluation Protocol

## Purpose

This protocol fixes the order of model selection and incomplete-modality evaluation.
It is designed for a TMI/MIA-oriented study and CLAIM 2024 reporting. Journal acceptance
cannot be guaranteed by an experiment template; external validity, reference-standard
quality, clinical framing, novelty, and the final effect sizes remain decisive.

The code calls the development partition `validation`. In the manuscript, describe it
unambiguously as the **development/model-selection split**. Reserve **internal test** for
the sealed UKB test split and **external test** for a genuinely independent institution
or cohort, following CLAIM 2024 terminology.

## Stage A: Complete-Modality Screening

- Run all 168 preregistered configurations on the development split only.
- Use seed 3407 only for computationally efficient screening.
- Train exclusively from complete CFP-OCT pairs.
- Use ImageNet-pretrained ResNet50 branches, natural sampling without replacement,
  effective-number class-balanced cross entropy, and patient-disjoint partitions.
- Keep fusion activation-free: concatenation, one linear Conv/Linear projection, then
  BatchNorm/LayerNorm. Classifier dropout is outside the fusion operation.
- Rank by macro F1, then balanced accuracy, macro one-vs-rest AUROC, and lower ECE-15.
- Treat this ranking as candidate screening, not a final test result.

## Stage B: Stable Baseline Selection

1. Select the top three Stage A configurations using the fixed ranking rule. Do not
   select candidates by test performance or by a single minority class after inspection.
2. Re-run each candidate with seeds 3407, 3408, and 3409. Reuse the existing seed-3407
   checkpoint when its fingerprint and manifest remain valid.
3. Rank candidates by mean macro F1 across seeds, then mean balanced accuracy, mean
   macro AUROC, lower mean ECE-15, and lower between-seed macro-F1 standard deviation.
4. Freeze the winning **hyperparameter setting** and retain all three seed-specific
   complete-modality checkpoints. These checkpoints constitute the baseline model family.
5. Record the selected configuration, code hash, label-table hash, environment, class
   weights, seeds, checkpoint hashes, and selection table in one frozen manifest.

The sealed test split remains untouched during Stages A and B.

## Stage C: Controlled Incomplete-Modality Study

For every frozen baseline seed, use identical participants and missingness assignments
for all methods. Never fine-tune the classifier after a modality is removed.

Required comparison arms:

1. Complete CFP + OCT: frozen-model upper reference.
2. Missing modality + normalized-mean fill: zero after ImageNet normalization.
3. Missing modality + independently trained paired cGAN fill.
4. Normalized-mean fill + LOOK.
5. Paired-cGAN fill + LOOK.

Normalized zero and channel-wise training-mean filling are the same operation after the
declared normalization and must not be presented as two independent baselines. The cGAN
is trained only from the training partition, uses a participant-disjoint internal GAN
development subset, and is never jointly optimized with the classifier or LOOK.

Controlled axes include OCT missing, CFP missing, cohort-level missing ratios, correction
node, downsample factor, latent dimension, and ridge alpha. All LOOK fitting uses training
features; all LOOK hyperparameter and node selection uses only the development split.
The test split must not influence matrix rank, correction node, downsampling, alpha,
filling choice, or any stopping decision.

## Stage D: Frozen Internal Test

- Access the sealed UKB test split only after the baseline and LOOK configurations are
  frozen and hashed.
- Report each seed and the aggregate mean with dispersion; do not report only the best
  seed.
- Use macro F1 as the primary endpoint. Also report balanced accuracy, per-class F1,
  sensitivity, specificity, macro AUROC, weighted F1, accuracy, Cohen kappa, cross
  entropy, Brier score, ECE-15, and confusion matrices.
- Use participant-clustered paired bootstrap confidence intervals because two eyes or
  repeated observations from one participant are not independent.
- Compare LOOK against its matching fill arm on identical samples. Correct the family of
  prespecified pairwise comparisons with Holm's method and report effect sizes with
  confidence intervals, not p-values alone.
- Preserve prediction tables so every table and figure can be regenerated without
  rerunning inference.

## Stage E: External Evidence

UK Biobank internal testing alone does not establish transportability. A strong TMI/MIA
submission should add an independent or multicenter ophthalmic cohort when access and
label compatibility permit. Report center, scanner/device, protocol, demographic and
disease-spectrum shifts, with no retuning on the external test cohort. If external data
are unavailable, state this explicitly as a limitation and avoid clinical-generalization
claims.

## Reporting and Audit Requirements

- Use `reference standard`, not `ground truth`, and document exactly how each diagnosis
  label was derived. UKB phenotype-derived labels must not be described as adjudicated
  ophthalmologist labels unless that provenance is actually demonstrated.
- Report cohort flow, exclusion reasons, class prevalence, participant counts, image
  counts, acquisition details, missing-data assumptions, and patient-level split logic.
- Report all 168 screening results and all confirmation seeds in supplementary material,
  including unsuccessful configurations. Do not select figures post hoc.
- Publish code, environment lock, configuration files, deterministic run IDs, and enough
  instructions to reproduce results from authorized UKB data. UKB images and records
  cannot be redistributed.
- Distinguish method limitations from engineering limitations: linear latent correction,
  two-modality setting, single central OCT slice, phenotype-label uncertainty, and current
  lack of external testing are separate issues.

## Decision Gate

Do not begin formal LOOK comparison runs until Stage B emits a frozen baseline-selection
manifest. Do not access the sealed test split until Stage C choices are frozen. Any change
to data, labels, backbone, loss, fusion, preprocessing, or LOOK search space invalidates
the downstream freeze and creates a new study fingerprint.

## Reporting Sources

- CLAIM 2024: https://pubs.rsna.org/doi/10.1148/ryai.240300
- IEEE TMI author instructions: https://ieeetmi.org/authors-instructions/
- Medical Image Analysis journal page: https://www.sciencedirect.com/journal/medical-image-analysis
