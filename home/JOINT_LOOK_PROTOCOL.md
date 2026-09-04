# LOOK joint sequential optional correction (Step 36)

This is the approved scientific protocol replacing study `e6d740a884be`.
The contribution under investigation is LOOK. Negative results remain evidence;
validation acceptance does not guarantee test, F1 or calibration improvement.

## Fixed backbone and inputs

Reuse the three reviewed layer3 backbones from candidate
`baseline_candidate__025a0ceb64d2.json` (seeds 3407, 3408, 3409). Step 36 verifies
candidate artifacts, exact scientific backbone IDs and checkpoint hashes, and
raises on mismatch; it never retrains or quarantines a protected backbone. The strict backbone reuse
flag is separate from generator training: missing independent cGANs may train on
the internal training split during validation development, never during test.
The seven training-identity source files and MHD core are unchanged.
ImageNet input normalization and the participant splits are unchanged. Missing
`normalized_mean` inputs are zero after normalization; `raw_zero` represents a
raw black image, i.e. `-mu/std` after normalization. Neither changes how observed
images are normalized. Independent paired cGAN is a separate filling arm.
No missing-input fine-tuning, residual-strength alpha, threshold tuning, or test access.

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
candidate is enabled only for strictly higher validation AUROC; a tie or decline
leaves it off. Candidate ties choose the smaller dimension. Entire factor banks
are compared by AUROC, then fewer enabled sites, then larger factor. All-off is
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

On ws, start or resume the exact same command after verifying no live session:

```bash
bash /home/mengh/LOOK/2026_09_03_19_35_04/tool/operations/start_joint_look.sh \
  --candidate /data/mengh/LOOK/2026_09_03_19_35_04/runs/baseline_selection/candidates/baseline_candidate__025a0ceb64d2.json \
  --reviewer-note 'Souray approved joint sequential optional LOOK migration; preserve exact layer3 checkpoints; test sealed' \
  --gpus 0,1 --execute
python3 /home/mengh/LOOK/2026_09_03_19_35_04/tool/operations/check_joint_look.py
```

Keep the reviewer note, GPU configuration, dimensions and source version fixed
when resuming; they determine study identity. A live session is not restarted.
Step 36 writes `runs/joint_look/<id>/summary.json`, plans under `runs/sweeps/`,
per-case predictions and decisions under `runs/experiments/<id>/look/`.

## Retirement and validation evidence

`tool/operations/clean_joint_predecessor.py` is separate from Step 35. It scopes
removal to stopped study e6d740a884be, refuses external references, symlinks,
live writers, non-LOOK/test plans and protected paths, and saves old code/config
identities and aggregate results before deletion. Full runtime audit:
`runs/maintenance/joint_protocol_cleanup_e6d740a884be.json`.
Backbones, baseline evidence, dataset/cache, environment, generators, prior
maintenance audits and Git source history are retained. No large result backup.

The bounded real-checkpoint verifier is `tool/operations/verify_joint_protocol.py`.
It uses nine training participants and twelve validation participants, all nine
sites, both missing directions, Dmax 4 and dimensions 1/2 solely for technical
acceptance. It asserts exact all-off logits, nondecreasing development selection,
restorable banks and unchanged checkpoint hashes. Its scores are not scientific
results. See `docs/joint_migration_evidence.json` for final deployment evidence.

## Implementation advantage and limits

Frozen backbones and local sufficient-statistic fitting permit stage-wise
execution and caching without backpropagation. This is an implementation
advantage, not a claim that CPU offloading is a new algorithm. Current Step 36
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
