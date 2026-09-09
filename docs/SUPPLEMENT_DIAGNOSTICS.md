# Bounded stability, calibration and cost diagnostics

Step 50 is an additive development analysis, declared after observing validation
instability and before opening test. It does not revise LOOK, search new starts,
refit W/b or PCA, or change the 87-configuration test roster.

## Frozen scope

- layer3 fusion, normalized_mean, seeds 3407/3408/3409, OCT/CFP missing.
- Calibration: filling, original LOOK, self-input missing-only, SSF and logit affine:
  30 existing validation prediction sets; no new backbone inference required.
- Stability: original LOOK and self-input missing-only; every accepted prefix,
  including all OFF; full unaugmented train (1264) and validation (296).
- Benchmark: all five methods above, same frozen model, validation batch, device,
  FP32, three warmups and 20 timed repeats. Cost measurement includes filling and
  the actual correction kernel (including CPU affine processing), excludes IO.
- Exactly three new GPU seed jobs. Each performs two-participant train/validation
  acceptance before full replay. Tiny results cannot be counted as full evidence.

## Interpretation

Temperature uses bounded log(T) in [-6,12], with identity included; endpoint hits
are reported. Five deterministic participant-stratified folds are shared across
models and directions. Fit on four folds and predict the fifth; additionally save
the full-validation temperature as a diagnostic only. This does NOT remove the
upstream model selection bias: the backbone/correction used the full validation
set. No calibration parameter is applied to test. Fold-specific temperatures can
change pooled ranking even though each fold preserves ranking and all argmaxes.

Each accepted-prefix trace compares the actual before/after writeback with the
same participant/eye's complete-input features at the exact fitting stop site.
Record norm, relative delta, feature MSE, frozen PCA-coordinate MSE, final feature
norm, logits and classification/calibration metrics. These are conditional path
effects, not independent node contributions. Full-image and latent reconstruction
error are descriptive and are not a new selection objective.

Historical fit/search/PCA costs are retained with their original scopes. Missing
timings remain null. Shared PCA is not charged twice for the two directions.
Controlled warm inference timings are separate from historical fitting times;
shared workstation interference is recorded, so no dedicated-device latency claim.

## Scheduling and resources

Run Step 50 with `--request REQUEST --output OUTPUT`. It freezes a hashed manifest
and completes CPU analysis immediately, then waits until the EXISTING test queue
finishes. Only its status/identity is read; no test prediction/metric files are read.
The original validation and test sources, identities and requests stay unchanged.

The controller acquires the existing project GPU lock, obeys the live arbitrary
device policy (single/multiple cards, drain/interrupt), and runs at most one job per
assigned card. All LOOK processes on each card share the 14 GiB budget; an 11 GiB
allocator limit leaves headroom but is not treated as a process-memory guarantee.
The controller samples aggregate LOOK process VRAM every 0.5 seconds, terminates
an over-budget worker, and retries at 8/4/2/1 participants. A failure at 1 blocks.
External projects are never terminated. Partial outputs are retained; each batch
size has its own execution directory, preventing mixed-batch prefix recovery.

All inference is deterministic; an interrupted prefix restarts from its beginning,
and completed prefixes are verified by identity and hashes. We do not skip samples.
New snapshots, source hashes and the request are checked before dispatch.

## Deliverables

`REPORT.md`, `calibration.csv`, individual calibrated/raw prediction bundles,
`historical_costs.json`, then `stability.csv` and `benchmark.csv` after GPU completion.
Reports explicitly mark incomplete phases and never fill missing values with zero.
Raw participant outputs stay on the workstation; only code/docs/tests go to GitHub.

Reference: Guo et al., ICML 2017, https://proceedings.mlr.press/v70/guo17a.html.

For self-input controls, latent diagnostic error is measured in the same unmasked
complete-training PCA coordinates as the original LOOK, to compare actual states.
It is not the retained-channel-neutralized fitting loss of that control.
