# Expanded UK Biobank preparation (not a locked training protocol)

The user selected ResNet50 for the next LOOK and R&B preparation, superseding the ResNet34 proposal. Keep infrastructure acceptance on the already accepted reference architectures. Two-rank validation must pass before new architecture resource checks. Both applications pin the new MHD_Framework V4 API.

## Data and disease definitions

Complete original CFP/OCT copying and SHA acceptance, then audit label tables, imaging/record dates, participant and eye/visit pairing, missingness, and image quality. Large files do not establish a usable labelled cohort. Do not enforce 50:50 by discarding most controls. Evaluation prevalence follows the declared cohort; balance or weighting choices belong to training and must be recorded.

The Eye study below reports large-scale manual grading of UKB fundus/OCT retinal features and separately ascertains diagnoses from interviews and linked records. Suspected imaging disease and recorded clinical diagnosis are different targets. Confirm which grading fields are actually available in the local export. Do not treat missing diagnosis as a certified negative or post-imaging diagnosis as prevalent disease without a time-aware definition.

Candidate tasks are glaucoma, AMD and diabetic retinopathy, conditional on available labels and adequate positive counts. For DR, define whether the population is people with diabetes (DR versus no DR) or all eligible participants; these answer different questions. Preserve comorbidity; disease classes are not mutually exclusive.

Start with independent disease-specific binary models using the same architectures and shared participant split assignment. R&B retains CFP/OCT native task heads and independent pretraining per disease, followed by bridge training. Shared-backbone multi-label training is a separate later question, requiring masks for unknown labels and explicit task weighting. Do not select only diseases with favorable pilot performance.

## Model and resource preparation

R&B needs 2D/3D ResNet50 bottleneck support and new native checkpoints; old ResNet18 checkpoints and SVD bases are not interchangeable. ResNet50 stage3 has a different channel dimension, so derive explicit r/h and bridge costs rather than blindly reusing rho. Fit bases only on the new training features after independent native training.

Both projects use two-rank training sequentially on the two-GPU allocation. Determine real device memory from the allocated GPUs, retain approximately 10 GiB free per GPU plus transient safety allowance, and report project process peak separately from other workloads. Similar allocated budgets do not imply identical parameter count or actual memory usage. Choose common effective batch targets from measured representative workloads; microbatch and accumulation may differ, with BN differences recorded. RETFound is a later adapter/weight/3D-input audit, not assumed to be a drop-in 3D network.

## Retuning and evaluation

A changed cohort, backbone and batch require a fresh, bounded development-only tuning protocol. Declare matching search budgets for baselines and proposed methods; tune native pretraining and bridge adaptation separately. Audit class proportions before choosing loss weighting, sampling, primary metric, threshold and validation patience. Record optimization steps and participants processed as well as epochs. Report branch metrics separately from any fused probability output, and include positive-class PR-AUC and sensitivity/specificity for imbalanced tasks.

Assign participants globally across diseases, eyes and visits. Record all prior participant exposure. Previously used development/test participants must not silently become a fresh independent holdout. Freeze a new evaluation policy before reviewing new holdout performance; old results remain historical exploratory evidence.

## Sources and limits

- [UK Biobank retinal imaging grading, Eye](https://www.nature.com/articles/s41433-022-02298-7): manual image grading, separate record-derived disease definitions, substantial divergence between suspected imaging disease and recorded diagnoses. Its cohort counts are literature counts, not this installation's inventory.
- [RETFound, Nature](https://www.nature.com/articles/s41586-023-06555-x): separate CFP/OCT foundation models adapted to multiple downstream tasks. UKB supplies external evaluation of systemic disease prediction; do not misrepresent all ocular benchmarks as UKB tasks or all downstream tasks as one jointly trained classifier.

Pending before scientific launch: export field availability, phenotype definitions and timing, per-task counts, fresh holdout design, architecture acceptance, common optimization/search budgets, finite candidate matrix and resource preflight. Current GPU jobs certify runtime only.
