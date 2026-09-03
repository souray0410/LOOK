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

## Baseline State Machine

1. Calibrate three pretrained/new-layer LR pairs and dropout 0.0/0.2 at feature fusion.
2. Train OCT-only and CFP-only references under the selected profile.
3. Search seven linear fusion positions at seed 3407.
4. Confirm the top three at seeds 3407, 3408, and 3409.
5. Rank by mean AUROC, then Macro-F1, balanced accuracy, ECE, and AUROC stability.
6. Require AUROC 0.80, Macro-F1 0.70, sensitivity/specificity 0.65, and no multimodal
   AUROC deficit.
7. Require explicit approval before LOOK.
8. Freeze all validation-selected artifacts before any test access.

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
