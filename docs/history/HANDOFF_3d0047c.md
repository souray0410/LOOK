# LOOK: New-Session Handoff

Prepared 2026-09-04. This is a context handoff, not a command to restart experiments.
Read this first, then refresh runtime state. The attached snapshot is historical,
not a live dashboard. Later user instructions and current evidence take precedence.

## 1. User And Immediate Objective

- Researcher: Souray Meng, medical image computing PhD at KAUST, targeting TMI/MIA.
- Respond in Chinese, address Souray, and start with the agreed Chinese acknowledgement.
  Explain the conclusion first, then reasoning; avoid reassurance unsupported by evidence.
  Exact greeting, expressed as JSON Unicode escapes: "\u597d\u7684\uff0cSouray\u3002".
- The scientific contribution is LOOK, not endless backbone hyperparameter tuning.
- The user explicitly approved proceeding with an adequate reviewed backbone despite
  failed self-imposed numeric gates. Do NOT stop back at those old gates automatically.
- Preserve a simple method, explicit experiment selection, reproducibility, and negative
  results. Never claim an AUROC/F1 cutoff guarantees journal acceptance.
- User wants early complete comparisons followed by automatic broader searches and
  seed repeats. No manual approval between currently scheduled development cases.
- Do not change data, labels, core method or test policy merely to increase scores.

## 2. Where Things Actually Live

```text
SSH alias              ws
Private GitHub         https://github.com/souray0410/LOOK
Active timestamp       2026_09_03_19_35_04
Deployed source        /home/mengh/LOOK/2026_09_03_19_35_04
Runtime root           /data/mengh/LOOK/2026_09_03_19_35_04
Shared image root      /data/mengh/LOOK/2026_09_03_08_30_00/dataset
Shared image cache     /data/mengh/LOOK/2026_09_03_08_30_00/cache/preprocessed_pairs
Python                 /home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python
Project timeline       /home/mengh/LOOK/PROJECT_TIMELINE.md
Project standard       /home/mengh/LOOK/GENERAL_PROJECT_STANDARD.md
```

The GitHub repository is a release with `home/` (source) and `data/` (empty skeleton).
Only `home/` is deployed into the source root above. That source root is NOT a Git
checkout. Do not run git pull there. `main` is canonical; runtime artifacts stay on ws.
The old Mac source tree was intentionally deleted. Use `/tmp` as command workdir when
the old cwd does not exist. Temporary clones are allowed; verify push/deployment then
delete only those temporary clones. Do not recreate a persistent Mac LOOK project.
Never upload UKB images, patient records, credentials or checkpoints to GitHub.

## 3. Current Dataset And Model

- Task: `glaucoma_all_evidence`, participant-level Normal vs Glaucoma.
- 925 cases + 925 matched controls. Train 1264 (632/632), validation 296 (148/148),
  test 290 (145/145). These aggregate counts were rechecked at handoff.
- Labels are **record-derived clinical phenotypes**, not expert image gold standards.
- Each participant uses the earliest complete bilateral visit: left/right CFP and
  left/right central OCT B-scans. OCT is 2D, not a volume. Tensor per modality is
  `[B, 2, 3, 224, 224]`; OCT grayscale is repeated across channels.
- The source images and preprocessing cache have one shared copy; do not duplicate them.
- Primary labels: runtime root + `/dataset/cohorts/task_scout/glaucoma_all_evidence/primary/reference_labels.csv`.
  Natural labels use the corresponding `/natural/reference_labels.csv`.
- Bilateral ImageNet-V2 ResNet50 branches: eyes share modality-specific weights;
  linear bilateral mean aggregates eye features. Fusion uses linear projection or
  1x1 convolution plus normalization, not attention or added nonlinear fusion blocks.
- MHD V4 represents forward/backward. PyTorch optimizer updates stay outside the graph.
- Seven fusion positions were compared, then Top-3 repeated at seeds 3407/3408/3409.
  That baseline comparison is FINISHED, not a pending prerequisite.

## 4. Reviewed Backbone Decision

Select the ORIGINAL `layer3` configuration, by three-seed mean validation AUROC.

| Candidate | Mean AUROC | Mean Macro-F1 |
| --- | ---: | ---: |
| Original layer3 (selected) | 0.783464 | 0.693760 |
| Later regularized feature | 0.775954 | 0.704713 |

Do not select the second one just because its F1 exceeds 0.70: AUROC was the primary
selection criterion. Neither candidate passed all project-internal automatic gates.
Souray explicitly approved moving into LOOK on 2026-09-04; preserve that distinction.
Candidate evidence file:

```text
/data/mengh/LOOK/2026_09_03_19_35_04/runs/baseline_selection/candidates/baseline_candidate__025a0ceb64d2.json
```

Selected profile: pretrained LR 3e-4, new-layer LR 3e-3, dropout 0, weight decay 1e-4,
label smoothing 0, unweighted CE, AdamW, cosine/warmup, complete-modality fine-tuning.
Use ALL THREE existing checkpoints, not the single best-performing seed:

```text
resnet50_oct_cfp_fusion_layer3__complete_modalities__seed3407__abb01fe16faa
resnet50_oct_cfp_fusion_layer3__complete_modalities__seed3408__757f09b50d84
resnet50_oct_cfp_fusion_layer3__complete_modalities__seed3409__ac12fc3f4a58
```

These are under `runs/backbones/`. Do not retrain or delete them. The current study
keeps `--gpus 0,1`, matching the original training identity; PCA/LOOK inference itself
is single-GPU plus CPU linear algebra. Independent GAN training uses DDP.

## 5. LOOK Method And The Latest Correction

The correct two-stage contract is now implemented:

1. On unaugmented COMPLETE TRAINING features, fit standardization and shared PCA.
2. For each filling/missing scenario, project into that SAME basis and fit Ridge.

```text
complete train features -> adaptive average pooling -> standardization -> PCA bank
missing train features -> same standardization/PCA -> Ridge residual W,b
validation             -> choose latent dimensions and whole-bank spatial factor
inference              -> lift decoded residual -> inject at MHD Node feature state
```

`delta_z = z_missing @ W + b`; `z_corrected = z_missing + delta_z`.
There is NO residual-strength alpha, no DCT, no extra low-rank preprocessing, and no
missing-input backbone fine-tuning. GAN is trained independently, never jointly.
Ridge lambda uses training-set GCV; it is not the removed alpha.

Shared PCA entries live under `runs/pca/<bank_id>/<node>_x<factor>.pt`, with hashes,
progress, manifest and resume state. Keys include checkpoint, data/reference and
feature implementation identities, batching/image settings and Dmax; missing pattern,
filling strategy and candidate dimension list are NOT part of the PCA identity.
Different backbone seeds need different PCA banks. Vector nodes always use factor 1.
New factors supplement a bank; existing valid entries are reused unchanged. Corrupt
entries are quarantined and rebuilt individually. Correction artifacts record
`pca_source_id` and embed a portable slice of the shared PCs; no independent refitting.

`Dmax=512` is a capacity INDEPENDENT of the searched latent dimensions. Actual rank is
bounded by complete training feature count minus one and feature dimension. Current
expanded candidates: `[8,16,32,64,96,128,192,256,384,512]`; factors `[4,8,16]`.
CLI also supports `--latent-min 32 --latent-max 512 --latent-step 32` instead of a list.
Changing a dimension list must not refit PCA when Dmax and PCA inputs are unchanged.

Within a factor, dimensions are selected sequentially per node on validation. This is
greedy node-wise search, NOT exhaustive search over every combination of dimensions.
Then choose one factor for the entire correction bank, not independent per-node factors.

## 6. Active Automatic Queue

Entry: `pipeline/34_run_reviewed_look_study.py`, detached session `look-reviewed`.

1. Prepare needed complete PCA entries for seed 3407, factor 16.
2. Two quick cases: zero/mean, seed 3407, dimensions `[16,64,256]`, both fully missing
   directions. Full training/validation cohorts, not a tiny subset. No random-ratio sweep.
3. Supplement PCA with factors 4/8, then two expanded-search zero/mean cases, seed 3407.
4. Prepare the other seeds' PCA and run four expanded zero/mean cases (3408/3409).
5. Three independently trained cGAN cases, one per seed, with expanded LOOK search.

Total: 2 quick + 9 full outer cases. Full cases include each completely missing
direction and deterministic participant-level RANDOM missing ratios 20/40/60/80/100%.
The random-ratio experiment may mix the two missing directions; do not describe it
as separate direction-specific ratio curves without inspecting evaluation code.
No metric-based stop between scheduled cases. Technical errors stop visibly and can
resume. Preserve adverse results. Case output is available as soon as it completes.

Train fits PCA/Ridge; GAN has a train-internal holdout. Validation selects settings.
The same validation set has already been used for task, backbone and LOOK development;
these are exploratory adaptive results, not independent test estimates. Three seeds
measure initialization sensitivity, not independent validation cohorts. BOTH TEST
COHORTS REMAIN SEALED pending a separate final configuration review. No test tuning.

## 7. Live Monitoring And Output Map

First action in the new session (read-only):

```bash
ssh ws 'python3 /home/mengh/LOOK/2026_09_03_19_35_04/tool/operations/check_reviewed_look.py'
```

Add `--follow` to follow the log. Mac sleep/disconnection does not stop remote tmux.
Low GPU use during CPU PCA/Ridge is normal; GPU1 is normally idle in LOOK inference.

Handoff snapshot identities (refresh rather than assuming still current):

```text
study summary   runs/reviewed_look/e6d740a884be/summary.json
review decision same directory /reviewed_baseline.json
comparisons     same directory /look_comparisons.json
quick sweep     runs/sweeps/validation__bc70f6d5d8dc/
seed3407 PCA    runs/pca/e1860fa373eaf768/
current log     runs/logs/look-reviewed_20260904T051924Z.log
```

Per-case `runs/experiments/<id>/validation_result.json` contains completed metrics;
`predictions/` contains participant-level NPZ (do not upload). `look/<pattern>/factors/`
contains per-node candidates, search history and complete-bank reports. Final selected
banks have `factor_selection.json` and matrix analysis JSON/CSV/figures. Matrix reports
include W and I+W spectra/rank/conditioning, MSE/R2 and PCA source IDs. Shared PCA
resource cost is counted once per source, not once per filling artifact embedding it.
Study summary timestamps update on stages, NOT on every batch; inspect log/bank files.

If a restart is actually needed, use the launch command in README with the same note
and flags; changes to the reviewer note also change the orchestration ID. Do not start
a duplicate session just because a summary timestamp appears old.

## 8. Results At Handoff And Immediate Analysis Priority

At 2026-09-04 05:27:23 UTC, both QUICK cases had completed and the job was preparing
additional shared PCA entries for expanded search. See
`tool/research/HANDOFF_SNAPSHOT_20260904.json` for exact aggregate values.

| Filling / missing | Fill AUROC | LOOK AUROC | Fill F1 | LOOK F1 |
| --- | ---: | ---: | ---: | ---: |
| zero / OCT | 0.7313 | 0.7449 | 0.5845 | 0.6619 |
| zero / CFP | 0.6308 | 0.6570 | 0.5517 | 0.6327 |
| normalized mean / OCT | 0.7022 | 0.5000 | 0.3333 | 0.3333 |
| normalized mean / CFP | 0.5937 | 0.6359 | 0.5808 | 0.6027 |

Same seed's complete-input baseline: AUROC 0.772781, F1 0.712441.
IMPORTANT: mean-filling/OCT-missing LOOK collapsed in the quick bank. This is a real
negative signal, not evidence the job crashed. Do not hide it or assert universal LOOK
improvement. First compare expanded results; then inspect probabilities/logits,
per-node search history, train R2/MSE, W spectra, normalization and PCA variance.
No cause has been established yet. Do not add alpha, edit labels, or open test as a fix.
Even improved discrimination can worsen calibration; inspect ECE as well as AUROC/F1.

## 9. Changes Already Verified And Cleanup Boundaries

- PCA previously repeated per missing pattern. It now has a complete-only shared stage.
- Earlier fixes retained: PCA includes the short final training batch; Ridge GCV uses
  exact residual SSE (2*s - s*s term) and intercept degrees of freedom.
- 122 unit tests passed; actual two-GPU tiny pipeline passed zero/mean/cGAN, shared
  PCA reuse, frozen classifier, LOOK and sealed-test exclusion. Tiny metrics are not
  scientific results. Notebook and Step 28 defaults now expose independent Dmax 512.
- The prior partial LOOK study `d367d81e8f8d` and sweep `735a085af22f`, about 463 MB,
  were deleted by manifest-scoped Step 35 at user request. Its cleanup audit remains
  in `runs/maintenance/`. DO NOT rerun that cleanup against the new live study.
- All baseline checkpoints/evidence, dataset, environment and preprocessing cache were
  retained and hash-checked. No need to reconstruct the dataset or reinstall Python.
- Compute and shared-PCA changes were committed as `6a13406`; empty skeletons as
  `1493a84`. The subsequent handoff commit changes documents only; no restart needed.

MHD files were deliberately unchanged:

```text
MHD_Framework_V4.py 6f499046ff3068b76c01a69e25d11398f2d2c0ff3ebb18f2faf825ff0199e6ba
MHD_Utils_V4.py     695214c85c510631aa7ef51f8f9a8b4e65cd801b70bec530f7874c8e10b005dc
```

## 10. Read More Only As Needed

1. `README.md`: exact launch/resume/path options and full protocol.
2. `PIPELINE_LEDGER.md` and `STRUCTURE.md`: numbered steps, inputs/outputs and layout.
3. Repository-root `PROJECT_TIMELINE.md`: concise timestamp-by-timestamp history.
4. Repository-root `GENERAL_PROJECT_STANDARD.md`: project-wide engineering contract.
5. `tool/look_core/look.py`, `pipeline.py`, `study_grid.py`, then tests: executable truth.
6. Git history / `snapshot/<timestamp>` branches and timestamp tags for old experiments.

Historical tags: 2026_08_30_11_20_47, 2026_09_01_16_06_11, 2026_09_02_00_00_00,
2026_09_03_08_30_00, 2026_09_03_19_35_04. Do not force-update these archival snapshots.
Old work included BraTS 3D-U-Net LOOK, MHD V3 -> V4, five/four-class UKB attempts, cRT,
and seven task-scout definitions. These explain the decisions but are NOT current tasks.
Steps 1-11 and 21-35 are active/maintenance entries; Steps 12-20 are historical under
`pipeline/history/`. Preserve chronological numbering; don't reuse old numbers.

If earlier conversation detail is needed and local Codex memory is available, search
`/Users/mengh/.codex/memories/MEMORY.md` narrowly, then its referenced rollout summaries.
Those memories include separate MHD_Project work and stale paths; they are NOT the
authoritative current LOOK status. Do not assume a new window automatically has the
old hidden context. This document and the repository/runtime files are the handoff.

## 11. Safe Maintenance Workflow

Read status first. For an approved change: temporary clone private LOOK main, inspect
existing code, use apply_patch, run focused/full tests on ws as appropriate, document
the decision and append numbered entry only if a new execution step is needed.
Refresh `home/file_manifest.json` and release manifest with Step 1 helpers; push private
main; sync `home/` to deployed source (exclude environments/caches, no broad --delete).
Sync project-level timeline/standard to `/home/mengh/LOOK/`. Verify deployed hashes,
repository privacy and relevant runtime state, then remove only temporary clones.
Never wipe runs, stop unrelated jobs, change core MHD, or reinstall the environment
merely because a new session has begun. Do not promise unattended debugging by the
assistant: the deterministic server queue runs independently, not a hidden agent loop.
