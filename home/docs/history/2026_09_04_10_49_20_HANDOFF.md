# LOOK handoff — Macro-F1 restart

Current release `2026_09_04_10_49_20`. Respond in Chinese beginning “好的，Souray。”
Local shell commands use `/tmp`; temporary clones only. GitHub `main` is the latest
valid workflow. Archive branch `archive/superseded-auroc-2026-09-03` preserves
commit `383c3c8` and the retired AUROC protocol, for historical tracing only.

Read in order:
1. `project.json` and `configs/macro_f1_study.json`.
2. `JOINT_LOOK_PROTOCOL.md` (method, selection, queue and recovery commands).
3. `PIPELINE_LEDGER.md`, `CHANGELOG.md`, parent `PROJECT_TIMELINE.md`.
4. New runtime `runs/maintenance/`: training/joint verification and retirement audit.
5. Refresh `tool/operations/check_macro_f1.py`; read current study summary and logs.
6. `references/260810_manifest.json` and verbatim reference source when needed.

Source: `/home/mengh/LOOK/2026_09_04_10_49_20` on `ssh ws`.
Runtime: `/data/mengh/LOOK/2026_09_04_10_49_20`.
Environment: `/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv`.
Existing selected cohort metadata remains in the previous release dataset root;
images and preprocessing remain shared with `2026_09_03_08_30_00`. These are
intentional immutable data dependencies, not old model/result reuse.

Souray explicitly replaced the earlier fixed-checkpoint instruction: retrain all
three complete-input backbones with validation Macro-F1 best-epoch selection,
then new full-train PCA and joint sequential optional LOOK, also selected by
Macro-F1. Preserve layer3, ImageNet normalization, hyperparameters and data splits.
No new architecture search. CE remains the training loss; no threshold tuning.
Do not reuse or rename AUROC-best weights. Both test cohorts remain sealed.

Step 37 is the only current formal experiment entrypoint. Old Steps 33–36 and
launchers were removed from main; recover them only from the archived branch.
The current notebook calls Step 37. Earlier ledger entries are historical records.
The queue trains seeds 3407/3408/3409, then mean 3407, remaining mean cases,
three black-image cases, three independent cGAN cases and two mean-3407 ablations.
Report every factor and missing ratio, not just the selected factor. Technical
errors block progress; disappointing metrics do not justify protocol changes.

Per-level subgraph/weight streaming is deferred and is not implemented. Do not
claim measured memory savings or a new offloading algorithm.

All runtime status statements need live verification. Start/resume:
`bash tool/operations/start_macro_f1.sh --gpus 0,1 --execute`.
Check: `python3 tool/operations/check_macro_f1.py`.
