# LOOK handoff — unified fusion and LOOK study

Current release `2026_09_04_19_18_07`. Respond in Chinese beginning “好的，Souray。”
All local shell commands use /tmp; only temporary local checkouts. GitHub main is
current source; ws02 is the workstation (ws remains an SSH alias). Read in order:

1. project.json and configs/unified_study.json.
2. JOINT_LOOK_PROTOCOL.md for the exact method, selection and missingness definitions.
3. PIPELINE_LEDGER.md Step 38, CHANGELOG.md and parent PROJECT_TIMELINE.md.
4. runs/maintenance technical evidence and the previous release pause audit.
5. Refresh tool/operations/check_unified_study.py and read the current summary/log.

Source: /home/mengh/LOOK/2026_09_04_19_18_07
Runtime: /data/mengh/LOOK/2026_09_04_19_18_07
Environment/data/preprocessing are explicit shared dependencies in project.json.
Old result and checkpoint reuse is forbidden. Previous release is paused and retained.

Step 38 is the current formal entrypoint. Twenty-one fusion and six unimodal complete
backbones are trained anew with FP32 validation Macro-F1 checkpointing. Select three
fusion positions by three-seed mean F1; ties use SD and declared order. Then 27 main
LOOK cases (three positions x three seeds x mean/black/independent cGAN) and six mean
seed3407 ablations. Preserve all factor/scenario metrics and negative results. GAN
selection uses training-internal reconstruction validation, never outer validation.
Both test cohorts stay sealed. No threshold tuning or missing-input fine-tuning.

Random masks are shared across every case, exact-count nested and direction-fixed.
LOOK fits only training residuals. Validation is used to select checkpoint/architecture/
LOOK decisions, so validation gains are exploratory and not an independent test claim.
Training AMP remains enabled; evaluation is FP32. MHD core remains unchanged.

Source deployment and tests must be verified before formal launch. Any runtime status
in this document is a setup description and must be refreshed live. Start/resume:
`bash tool/operations/start_unified_study.sh --gpus 0,1 --execute`.
Check: `python3 tool/operations/check_unified_study.py`.
The existing thirty-minute monitor should target this release only after verified launch.

Historical fusion comparisons used a different selection protocol and cannot justify
claiming layer3 is the Macro-F1-best fusion position. Earlier fixed-layer3 LOOK gains
remain exploratory evidence, not invalid solely because architecture selection changed.
Do not delete them or describe all prior results as erroneous. Per-level weight streaming
is deferred. Every new technical failure blocks progress; negative efficacy does not.
