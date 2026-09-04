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
Results from superseded study `7eb81db4949c` are forbidden inputs. Previous release is
paused and retained as historical evidence.

Step 38 is the current formal entrypoint. Seven fusion positions are screened once at
seed 3407 with FP32 validation Macro-F1 checkpointing. Select the first three by F1,
using declared position order only for exact ties, then add seeds 3408 and 3409 for
those positions. OCT-only and CFP-only are auxiliary three-seed controls outside
selection. Then run 27 main
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
There is no automatic status notification; Souray requests live status on demand.

Historical fusion comparisons used a different selection protocol and cannot justify
claiming layer3 is the Macro-F1-best fusion position. Earlier fixed-layer3 LOOK gains
remain exploratory evidence, not invalid solely because architecture selection changed.
Do not describe all earlier historical results as erroneous. Outputs from stopped study
`7eb81db4949c` are protocol-invalid and removed by scoped audit. Per-level weight streaming
is deferred. Every new technical failure blocks progress; negative efficacy does not.


## 2026-09-04: unimodal post-training evaluation repair

The shared LOOK input reset assumed both OCT and CFP input nodes existed. This
crashed after the first OCT-only training run completed, before validation_result
was written. Reset now writes only input nodes present in the graph. Training,
checkpoint selection, graph topology, residual fitting and statistical definitions
were unchanged in that repair. Its checkpoint belonged to the subsequently superseded
all-position three-seed attempt and is no longer a valid input to the current study.

Regression coverage now executes evaluate_missing for all seven fusion positions
and both unimodal controls, checks exact original-forward logits and a partial tail
batch, and checks that an unused modality cannot affect a unimodal model. A real
296-participant validation pass through ExperimentRunner also matches the saved
Macro-F1 exactly. Verification and scoped failure cleanup commands are in
tool/operations/verify_unimodal_evaluation.py and cleanup_failed_unimodal_evaluation.py.
The latter requires --execute, permits only the known failed attempt, preserves
training artifacts and writes a small audit before deletion.

The source-code fingerprint creates a new study ID inside the SAME release and
unchanged specification. The failed attempt is archived as a maintenance audit;
its incomplete evaluation/sweep/state directories are removed. The original
release tag remains immutable; the repair receives a distinct patch tag. Read the
latest live summary rather than relying on the initial study ID.
