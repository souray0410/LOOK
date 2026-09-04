# LOOK joint sequential optional correction (Step 37, Macro-F1)

Current release: `2026_09_04_10_49_20`. The retired AUROC protocol is preserved
on Git branch `archive/superseded-auroc-2026-09-03` at `383c3c8`. It is not a
valid source of checkpoints or results for this release. Main tracks this protocol.

## Fixed architecture, new checkpoints and inputs

Train complete-input layer3 ResNet50 at seeds 3407, 3408 and 3409 from ImageNet
initial weights. The versioned `configs/macro_f1_study.json` fixes the previous
architecture and optimizer hyperparameters; do not reopen architecture search.
Cross-entropy remains the gradient loss. Full participant-level validation
Macro-F1 determines best checkpoint and early stopping, with argmax predictions
and no threshold tuning. Ties preserve the first best epoch. Metric name, source
identity and labels hash are included in checkpoint metadata and backbone ID.
LOOK must reuse these exact new completed checkpoints; mismatch raises an error.

ImageNet input normalization and the participant splits are unchanged. Missing
`normalized_mean` inputs are zero after normalization; `raw_zero` represents a
raw black image (`-mu/std` after normalization). Observed inputs use the same
normalization in every arm. Independent paired cGAN is a separate filling arm,
trained only on internal training pairs if no compatible generator exists.
No missing-input backbone fine-tuning, residual-strength alpha or test access.

## Ordered joint sites and feature model

`joint_input -> joint_stem -> joint_layer1 -> joint_layer2 -> joint_layer3 ->
fusion_layer3 -> fusion_layer4 -> fusion_feature -> fusion_participant_feature`.

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

## Queue and operations

Nine main cases: mean seed 3407 first, mean 3408/3409, black all three seeds, then
independent cGAN all three seeds. Two additional mean-3407 ablations enable only
joint input, or only the four fusion/post-fusion sites. Each case covers both
complete missing directions and random ratios 0.2, 0.4, 0.6, 0.8, 1.0.

Before these LOOK cases, Step 37 completes all three new backbones. All three
spatial factors (4, 8, 16) are independently evaluated for both complete-missing
directions and every random ratio. Each case saves complete performance, filling
performance, and filling-plus-LOOK performance; factor selection does not hide
factor-specific results. F1, AUROC/AUPRC, sensitivity/specificity, confusion matrix
and calibration are reported. Selection on Macro-F1 does not guarantee that the
other metrics improve, or that gains generalize to independent test participants.

Start/resume on ws with the same source, specification and GPU configuration:

```bash
bash /home/mengh/LOOK/2026_09_04_10_49_20/tool/operations/start_macro_f1.sh --gpus 0,1 --execute
python3 /home/mengh/LOOK/2026_09_04_10_49_20/tool/operations/check_macro_f1.py
```

The launcher exports the new source path and refuses to duplicate a live
`look-macro-f1` session. Training resumes from `last/`; LOOK resumes completed
ordered decisions. Summaries: `runs/macro_f1_study/<id>/summary.json`; history:
`runs/backbones/<id>/history.json`; results and per-factor decisions:
`runs/experiments/<id>/validation_result.json` and `look/`.

## Retirement and technical verification

`retire_auroc_release.py` scopes cleanup to explicitly listed model/output/state
subtrees of runtime `2026_09_03_19_35_04`. It preserves shared data, environment,
independent generators, task-selection provenance and small maintenance audits.
It refuses symlinks, live writers, test outputs and non-target releases. The
minimal audit is `runs/maintenance/retired_auroc_release.json` in the new release;
it records code/config identities and deleted paths/sizes, not large result copies.

`verify_macro_f1_training.py` checks real two-GPU, two-epoch checkpoint selection
on eight train/validation participants per split. `verify_macro_f1_joint.py` uses
that technical checkpoint on nine training and twelve validation participants,
all nine sites, both missing directions and all factors, Dmax 4 and dimensions
1/2. These are technical checks, not scientific evidence. They verify all-off
logit equality, restored selected banks, nondecreasing validation Macro-F1 and
unchanged checkpoint hashes. Tests additionally cover endpoint disagreement,
upstream inheritance, residual decoding, GCV, complete PCA sharing, interruption
recovery, all-factor reporting and cleanup boundaries.

## Implementation advantage and limits

Frozen backbones and local sufficient-statistic fitting permit stage-wise
execution and caching without backpropagation. This is an implementation
advantage, not a claim that CPU offloading is a new algorithm. Current Step 37
keeps the model resident: streaming per-level weight loading/unloading is NOT
implemented or benchmarked. The active layer3 graph has about 130.3 MiB of FP32
parameters, so activations/PCA/workspace/cache must also be measured before
claiming large GPU-memory savings. Any later backend must demonstrate logit and
decision equivalence and report peak allocated/reserved memory, runtime and cache
volume. Describe this method as frozen-backbone/backpropagation-free correction;
PCA, Ridge fitting and validation selection still have data and compute costs.

Historical reference: `references/260810_source.txt`, exact hash in its manifest.
The reference's alpha sweep, missing-input training, old GCV and omitted input
correction during downstream collection are deliberately not carried forward.
