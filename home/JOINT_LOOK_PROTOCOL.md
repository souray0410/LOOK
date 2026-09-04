# LOOK unified protocol — Step 38

Release `2026_09_04_19_18_07`. This specification supersedes fixed-layer3 Step 37.
The previous release was paused by user instruction; its files and negative results
remain historical exploratory evidence, not inputs to this new study.

## Complete-input backbone and fusion comparison

Keep ResNet50, ImageNet initialization and normalization, the selected record-derived
UKB glaucoma task, participant splits, preprocessing, MHD V4 core and optimizer profile
fixed. Train all seven fusion positions (input, stem, layer1, layer2, layer3, layer4,
feature) at seeds 3407, 3408, 3409. Also train OCT-only and CFP-only at the same seeds.
These are 27 fresh complete-input backbones. Do not reuse earlier checkpoints.

Cross-entropy is the gradient loss. Best epoch and early stopping use full participant
validation Macro-F1 from logits argmax; no threshold tuning. Training may use AMP,
but validation and frozen inference use FP32. Macro-F1 counts/ratios use float64 and
return a Python float to preserve precision through the existing trainer interface.
Checkpoint/frozen-validation Macro-F1 disagreement blocks fusion ranking for diagnosis.

Rank the seven fusion positions by mean validation Macro-F1 across all three seeds,
then lower sample SD, then declared position order. Select exactly the first three.
No winner is chosen by its best seed, AUROC, missing-input performance or test result.
Save all 21 fusion rows, six reference rows, seven aggregate rows and selection rule.
This is a prespecified validation-selection protocol; selected architectures and
LOOK performance still require independent sealed-test confirmation later.

LOOK must strictly reuse each selected position/seed checkpoint with matching identity
and hash. A mismatch fails instead of automatically retraining. ImageNet normalization
is unchanged: normalized_mean inserts zero AFTER normalization; raw_zero is a raw
black image and becomes -mu/std. Neither is a dataset-mean normalization change.
Observed modalities undergo exactly the same normalization in every filling arm.

Independent paired cGAN trains only on an internal split of complete training pairs
(internal validation fraction 0.1; generator selection by reconstruction L1). It does
not use outer validation/test labels or images for generator selection. Its generation
is independent of fusion position, so one generator pair per seed is shared by the
three selected backbones. The GAN is newly trained in this release. Classifier and
LOOK select by Macro-F1; the independent generator keeps its task-agnostic criterion.
No missing-input backbone fine-tuning or residual-strength parameter is introduced.
Both test cohorts remain sealed throughout this queue.

## Ordered joint sites and feature model

For selected fusion position f, visit joint_input through joint_f in original order,
then fusion_f through fusion_feature and fusion_participant_feature. For example,
layer3 uses `joint_input -> joint_stem -> joint_layer1 -> joint_layer2 -> joint_layer3
-> fusion_layer3 -> fusion_layer4 -> fusion_feature -> fusion_participant_feature`.
Input and feature fusion use the corresponding dynamically generated schedule.

Before fusion, concatenate OCT then CFP on channels for the same participant and
eye. For input, `[B,2,3,H,W]` becomes `[2B,3,H,W]` in participant-major, eye-minor
order. Correct the joint state, split by member channel counts, and restore the
saved member shapes. Both retained and missing modalities may change. This is a
dense PCA-coordinate residual map, not a pixel-registration assumption or a new
trainable fusion module. Original fusion and classifier operations are preserved.

All feature collection, candidate evaluation and final inference use the same
LOOK scheduler. Input correction runs before level 0. Complete targets never
receive correction. Missing features inherit every accepted upstream decision.

Spatial features use bilinear downsampling (`align_corners=False`, no antialias),
then per-channel population statistics over samples and spatial positions. Vector
features use per-coordinate statistics. Variance floor is `1e-12`. PCA uses only
unaugmented complete training inputs, including every tail sample. Dmax=512 is
independent of the searched dimension list; vector factor is 1, spatial factors
are 4, 8, 16. A bank under one frozen backbone is shared across filling strategies,
missing directions and ablations; new feature/code identities prevent old-bank reuse.

For flattened compressed state x, feature mean mu, scale sigma, PCA mean m and
row-wise PCs P:

```
z = ((x - mu) / sigma - m) @ P.T
delta_z = z @ W + b
corrected_state = original_state + upsample((delta_z @ P) * sigma)
```

The residual decoder adds neither m nor mu. Ridge fits complete-minus-current
latent residuals from sufficient statistics, including an intercept. GCV uses
`SSE = TSS - sum((2s - s^2) * projected_target_energy)` and degrees of freedom
`1 + sum(s)`, with `s = eigenvalue / (eigenvalue + lambda)`.

## Acceptance, scoring and recovery

Each position evaluates the current bank with that position OFF. The best
candidate is enabled only for strictly higher validation Macro-F1; a tie or decline
leaves it off. Candidate ties choose the smaller dimension. Entire factor banks
are compared by Macro-F1, then fewer enabled sites, then larger factor. All-off is
valid and exactly reproduces filling-baseline logits.

Every AUROC/AUPRC in the new LOOK evaluation and bootstrap uses the original
binary logit difference. Class predictions use logits argmax. Prediction bundles
save logits, logit-difference scores, float64 probabilities, labels, participant
IDs and missing patterns. Report F1, calibration, stable NLL and probability
saturation; probability saturation must not erase ranking evidence.

Each atomic decision records baseline/candidate metrics and prediction hashes,
best dimension, enabled status, PCA source, upstream-decision hash and the best
candidate's matrix diagnostics even when disabled. Saved artifacts include
ordered member names, member shapes and the split rule. Resume walks completed
decisions in order and validates accepted artifact hashes; an intermediate
candidate does not count as a decision. Empty selected manifests are valid.
Completed bank summaries and factor selection expose all decisions and matrices.
Validation bootstrap is exploratory and conditional on validation selection,
not a held-out confirmatory interval. Both test cohorts remain sealed.

## Shared random missingness

`nested_exact_count_fixed_direction_v1`, mask seed 3407, independent of model seed.
On the evaluation cohort's unique participant IDs, hash-rank IDs without labels.
At ratio r, mask the first floor(r*N+0.5) participants. Alternate missing directions
in this fixed order with a seeded offset, so every prefix is balanced within one.
A participant's direction never changes as r grows. The entire modality in both eyes
is missing; the other modality remains. At r=1 all participants lose ONE modality,
not both. At N=296 the five missing counts are 59, 118, 178, 237, 296. Realized fractions
and per-direction counts are saved; nominal percentages are rounded to participants.

Every fusion position, backbone seed, filling strategy and LOOK factor shares these
same masks. Prediction bundles include IDs and patterns; a sidecar records protocol,
seed, nominal/actual ratio, counts, participant-set hash and assignment hash. No labels
enter assignment. Fixed complete OCT missing and CFP missing are separate scenarios.
Matrices are fitted on the training cohort with each corresponding modality masked;
random evaluation reuses these matrices and never refits from evaluation labels.

## Queue, recovery and evidence

After 27 backbone runs and fusion selection, run mean for all three selected positions
and three seeds, then black for the same nine cases, then independent cGAN for the same
nine cases: 27 main cases. Add mean seed 3407 input-only and fusion-only ablations for
each selected position: six ablation cases. Fusion-only sites are derived from each
architecture, not hardcoded to layer3. Every case reports complete, filling and
filling+LOOK for factors 4/8/16, both full-missing directions and all five random ratios.
The ten candidate dimensions are 8,16,32,64,96,128,192,256,384,512, capped by available
rank; Dmax remains independent at 512. No factor curve is assumed monotonic.

Start or resume with unchanged committed source/specification and GPU set:

```bash
bash /home/mengh/LOOK/2026_09_04_19_18_07/tool/operations/start_unified_study.sh --gpus 0,1 --execute
python3 /home/mengh/LOOK/2026_09_04_19_18_07/tool/operations/check_unified_study.py
```

Session: `look-unified-20260904-191807`. Summary: `runs/unified_study/<id>/summary.json`.
Backbone evidence and fusion selection are beside that summary. Runtime histories,
predictions, PCA, matrices, ordered decisions and search traces retain their existing
subdirectories. The launcher refuses a duplicate live session. Resume revalidates
completed checkpoint/results and resumes incomplete training/ordered LOOK decisions.
JSON and prediction writes use atomic replacement with file/directory fsync. Portable
best weights are synced before completion metadata. This does not make a multi-file
training checkpoint immune to power loss; corrupt/inconsistent state must be diagnosed.

Old release `2026_09_04_10_49_20` is paused, with no result deletion. Stop audit:
`/data/mengh/LOOK/2026_09_04_10_49_20/runs/maintenance/unified_protocol_pause_20260904T161807Z.json`.
Historical source is retained in its snapshot branch/tag and in the old deployed tree.
Only the new main release is an active workflow. Step 37 entry/config is archived under
`pipeline/history/2026_09_04_10_49_20` and is not an executable current entrypoint.

## Technical gates and implementation limits

`tool/operations/verify_unified_release.py --project-root <source>` runs the bounded gates.
Run unit tests, nine graph-topology smoke checks, intentional tiny overfit, two-GPU FP32-checkpoint smoke,
joint-site/PCA/restore smoke, three-filling end-to-end smoke and resume checks before
formal training. Save source hashes and technical verification under runs/maintenance.
Disappointing metrics are reportable findings, not reasons to restart or change criteria.
No validation bootstrap accounts for the full architecture/dimension-selection process;
reported validation intervals remain exploratory and conditional on selection.

Frozen-backbone and sufficient-statistic fitting support a future stage-wise execution
backend. Per-level weight streaming remains deferred and has not been implemented or
benchmarked. Do not claim measured memory savings or call this computation-free.
Reference source and hash: references/260810_source.txt and 260810_manifest.json.
