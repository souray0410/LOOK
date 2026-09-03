# Pipeline Ledger

| Step | Purpose | Durable output | Integrity and recovery |
|---:|---|---|---|
| 1 | Resolve and initialize roots | runtime skeleton | create missing directories, never replace data |
| 2 | Sync source | remote source tree | source only; exclude datasets, environments and runs |
| 3 | Verify/create environment and kernel | package install and lock | reuse a valid environment, fill missing dependencies |
| 4 | Verify source mounts | mount audit | require read-only image and phenotype media |
| 5 | Export CFP and central OCT | four image directories | per-file resume; original media unchanged |
| 6 | Verify export | export verification | counts, readability and manifest references |
| 7 | Audit CFP/OCT pairs | strict paired/unpaired reports | deterministic participant/instance/eye keys |
| 8 | Remove unpaired derived files | paired derived image tree | allowlist, dry-run and explicit `--execute` |
| 9 | Verify paired images | paired verification | exact pair counts and no extra derived images |
| 10 | Extract selected phenotypes | bilateral visit map and selected columns | source read-only check, schema and hashes |
| 11 | Build record phenotypes | evidence-level candidate table | preserve source, code, timing and exclusion reason |
| 12 | Build four-class cohorts | balanced/natural/incident manifests | deterministic participant split and matched Normal sampling |
| 13 | Verify cohorts | verification, analysis readiness and data manifest | bilateral paths, mapping, ratio, uniqueness, zero leakage, protocol sample-size gates and test precision |
| 14 | Clean derived intermediates | compact canonical dataset | only after PASS; allowlist and explicit `--execute` |
| 15 | Run tests | test report | formal execution blocked on failure |
| 16 | Run MHD graph smoke | topology evidence | seven fusion plus two unimodal graphs; forward/backward |
| 17 | Run pipeline smoke | tiny two-filling LOOK runs | DDP, resume, frozen backbone and sealed-test checks |
| 18 | Interactive experiment | standard runner artifacts | one configuration cell; no notebook-only science |
| 19 | Baseline and LOOK sweep | checkpoints, predictions, rankings and freezes | deterministic IDs, best/last resume, frozen test gate |
| 20 | Matrix analysis | aggregate tables and figures | selected LOOK banks only |

## Phenotype Rules

- The analysis unit is one participant at the earliest complete bilateral visit.
- Prevalent disease requires evidence available at or before imaging.
- Post-imaging first evidence is incident-only and cannot enter model selection.
- Undated target evidence is excluded from the prevalent/control task.
- Target comorbidity and competing eye disease are excluded from the primary task.
- Normal requires explicit no-eye-disease self-report and no target evidence in follow-up.
- Labels are named `record_derived_clinical_phenotype`, never gold standard.

## Baseline State Machine

1. Calibrate two conservative pretrained/new-layer LR pairs, dropout 0.0/0.2 and label
   smoothing 0.0/0.1 at feature fusion, with effective batch size 128.
2. Train OCT-only and CFP-only references with the selected profile before interpreting
   multimodal fusion results.
3. Apply the selected profile to all seven linear modality-fusion positions at seed 3407.
4. Re-run the top three positions with seeds 3407, 3408 and 3409.
5. Rank by mean Macro-F1, balanced accuracy, Macro-AUROC, ECE and stability.
6. Require mean Macro-F1 `>=0.65`, each mean class F1 `>=0.45`, and no multimodal deficit.
7. Persist an audit on failure; require explicit reviewer approval on success.
8. Only an approved baseline may enter filling and LOOK validation.
9. Freeze all validation-selected artifacts before balanced/natural test access.

## Recovery Contract

- Reuse only outputs whose configuration, code, input and artifact manifests validate.
- Resume incomplete training from `last/`; preserve `best/` independently.
- Commit one sweep case before starting the next.
- Share epoch state with persistent data-loader workers so deterministic augmentation
  actually changes without worker restart; retain the equal-sized final DDP batch.
- A changed label table or scientific parameter creates a new deterministic run ID.
- Validation and test predictions use disjoint files and directories.
- Natural and incident cohorts cannot influence hyperparameter selection.
- Cleanup never targets mounted source media.
