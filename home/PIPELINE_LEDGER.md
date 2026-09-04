## 2026-09-04 — Step 37: Macro-F1 restart (current)

Release `2026_09_04_10_49_20` supersedes previous model/output protocols. Backbone
best epoch and early stopping now use full-validation Macro-F1; LOOK uses the
same primary endpoint. Fixed layer3 and three seeds are retrained from ImageNet,
without architecture search. New checkpoint identities and PCA banks prevent old
AUROC artifacts from being reused. All factors and random missing ratios are
reported independently. Test cohorts remain sealed. Steps 33–36 and their
launchers are archived on `archive/superseded-auroc-2026-09-03` at `383c3c8`.
The following older entries are historical and do not define current acceptance
or launch rules. See `JOINT_LOOK_PROTOCOL.md` in the home tree for current rules.

# Pipeline Ledger

| Step | Purpose | Durable output | Recovery and integrity |
|---:|---|---|---|
| 1 | Initialize roots and manifests | runtime skeleton | create missing paths only |
| 2 | Sync source | remote source tree | excludes data, environments and runs |
| 3 | Verify/create environment and kernel | environment lock, LOOK kernel | reuses complete environment |
| 4 | Verify source mounts | source audit | requires read-only raw media |
| 5 | Export CFP and central OCT | four image directories | per-file resume; source unchanged |
| 6 | Verify export | image manifest | counts and image readability |
| 7 | Audit bilateral pairs | paired/unpaired reports | participant/visit/eye keys |
| 8 | Remove unpaired derived files | paired export | dry-run, allowlist, explicit execute |
| 9 | Verify paired images | paired verification | exact references and zero extras |
| 10 | Extract selected fields | paired visit map and matched fields | streaming, source hashes |
| 11 | Build record candidates | evidence-level candidate table | source, timing, code and exclusion reason |
| 12-20 | Historical four-class iteration | `pipeline/history/2026_09_03_08_30_00/` | superseded; provenance only, never active |
| 21 | Build glaucoma and task-bank cohorts | profile-specific primary/natural/incident tables | one global split, source fingerprint and deterministic matching |
| 22 | Verify glaucoma and task-bank cohorts | verification and data manifest | paths, classes, shared split, SMD and leakage |
| 23 | Clean partials | clean derived metadata tree | verification gate and explicit execute |
| 24 | Run unit tests | test report | blocks formal execution on failure |
| 25 | Run graph smoke | nine topology checks | binary shape, ImageNet mapping, forward/backward |
| 26 | Run pipeline smoke | three filling arms and LOOK | DDP, resume and sealed-test guard |
| 27 | Interactive experiment | standard runner artifacts | one configuration cell |
| 28 | Baseline/LOOK sweep | checkpoints, rankings and freezes | deterministic IDs and last/best resume |
| 29 | Matrix analysis | aggregate CSV and figures | selected LOOK banks only |
| 30 | Validation-only task scout | task leaderboard and resumable checkpoints | seven prespecified profiles, no test/LOOK access |
| 31 | Summarize task usability | JSON and Markdown audit | validation evidence, phenotype caveats and no automatic winner |
| 32 | Select formal task | hash-checked selected-task manifest | explicit review; validation-only evidence; test remains sealed |
| 33 | Bounded overnight continuation | `runs/overnight/<fingerprint>/summary.json`, grids and checkpoints | waits for successful predecessor; six regularization profiles at most; resume by scientific IDs; machine-gated validation pilot only |
| 34 | Reviewed fixed-backbone LOOK study | `runs/pca/<bank_id>/`, `runs/reviewed_look/<fingerprint>/`, 2 quick + 9 full cases | shared complete-train PCs first per stage; independent Dmax/list/range; exact backbone reuse; validation-only; failed gates retained; incremental PCA-entry and correction-bank resume |
| 35 | Clean superseded reviewed LOOK outputs | `runs/maintenance/cleanup_look__<id>.json` | maintenance only: stopped study manifest allowlist, dry-run then execute; preserve baselines/data/cache; rerun Step 34 afterward |

| 36 | Joint sequential optional LOOK | `runs/joint_look/<id>/`, 9 main cases + 2 ablations | frozen exact layer3 checkpoints; shared full-train PCA; logit ranking; atomic on/off decisions; both tests sealed; see JOINT_LOOK_PROTOCOL.md |

Steps 34 and 35 are retained historical entries; the active replacement is Step 36.

Pipeline numbers are append-only. Superseded executed steps move to the timestamped
history directory; active replacements receive new numbers and are never renumbered.

## Task Scout State Machine

1. Build all candidate labels from the same evidence table and global participant split.
2. Match strict controls 1:1 within each split, age, sex and assessment centre.
3. Run the four main quality-volume profiles first, followed by three diagnostic profiles.
4. Hold model, fusion position, seed, optimizer and short training budget fixed.
5. Rank validation evidence only; never open test and never run LOOK during scouting.
6. Review performance jointly with case count, evidence quality, balance and modality gain.
7. Promote an eligible task explicitly before the full baseline state machine begins.

The current reviewed selection is `glaucoma_all_evidence`: 925 prevalent cases and 925
matched controls, with train/validation/test case counts of 632/148/145. Step 32 records
the decision but does not qualify the baseline or authorize LOOK.

## Baseline State Machine

1. Verify the Step 32 task manifest and label hashes.
2. Calibrate three pretrained/new-layer LR pairs and dropout 0.0/0.2 at feature fusion.
3. Train OCT-only and CFP-only references under the selected profile.
4. Search seven linear fusion positions at seed 3407.
5. Confirm the top three at seeds 3407, 3408, and 3409.
6. Rank by mean AUROC, then Macro-F1, balanced accuracy, ECE, and AUROC stability.
7. Require AUROC 0.80, Macro-F1 0.70, sensitivity/specificity 0.65, and no multimodal
   AUROC deficit.
8. Require explicit approval before LOOK.
9. Freeze all validation-selected artifacts before any test access.

## Recovery Contract

- Existing files are reused only after configuration, code, input, and artifact checks.
- Long training resumes from `last/`; `best/` is preserved separately.
- One sweep case is committed before the next starts.
- Corrupt or mismatched outputs move to quarantine.
- A changed phenotype rule, labels file, model setting, filling strategy, or LOOK setting
  creates a new deterministic run ID.
- Validation and test predictions use separate paths.
- Natural and incident cohorts cannot influence model selection.
- Cleanup never targets `image_root`, `preprocess_cache_root`, or mounted source media.


## Step 38 — 2026_09_04_19_18_07 unified protocol replacement

User requested pause of Step 37 and fresh consistent comparisons. Old processes were
stopped and old results preserved. Train all seven fusion positions and two unimodal
controls at seeds 3407/3408/3409. Checkpoint/architecture/LOOK endpoint is Macro-F1;
validation uses FP32, training AMP. Select top three fusion positions by three-seed mean,
then lower SD and declared order. Run 27 main LOOK cases plus six ablations. Independent
GAN still uses training-internal reconstruction validation. Random masks now use shared
hash ordering, exact rounded participant counts, nested sets and fixed missing direction.
All factors and metrics are reported; both test cohorts remain sealed. Step 37 and the
old method descriptions are historical and do not govern this release.

Current specification: configs/unified_study.json; current entrypoint: pipeline/38_run_unified_study.py.
Technical evidence belongs under the new runtime runs/maintenance; runtime status requires
live verification. No earlier checkpoint, generator, PCA or experimental result is reused.
