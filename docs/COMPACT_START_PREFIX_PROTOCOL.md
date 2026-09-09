# Fixed fusion, three correction starts, one frozen diagnostic

Approved 2026-09-07 after completing the 15+3 controls. This is a development-stage
mechanism supplement, not an independent test or retrospective prespecification.

- Keep layer3 fusion, normalized mean and seeds 3407/3408/3409 fixed.
- Reuse and hash-audit start ordinals 1, 4 and 7: joint_input, joint_layer2,
  fusion_layer4. All nine cases already exist. Earlier sites OFF; eligible sites
  retain the original optional, sequential fitting/search. No new fitting.
- Preserve the full nine-start evidence and every unfavorable outcome. Do not
  choose the best of these three as a new primary method.
- Diagnose only original joint LOOK, layer3/mean/3407/OCT missing. Audit saved
  predictions for the actually accepted matrices, then replay prefixes 0..4 on
  the complete validation set with identical checkpoint, W/b, batch and precision.
- Record all six metrics, stable NLL, logit margins, correction and feature norms.
  Prefix effects are conditional on their upstream path, not isolated node effects.
- No test inference, training, PCA fitting, additional starts or policy grid.
- Pipeline 45 uses the same global LOOK GPU lock and process-memory monitor as
  pipeline 43. At most GPU0/1, 14 GiB of LOOK process memory per GPU, one worker.
  Keep historical inference batch; if replay exceeds budget, stop and report.
- New output identity hashes the frozen evidence and diagnostic code; resume only
  complete, hash-verified prefixes. Historical runs and source remain unchanged.

Run pipeline/45_run_compact_prefix.py with --project-root,
--start-evidence-manifest and --output for CPU evidence. Add --execute and the
existing global --lock-path for bounded GPU replay. Do not use the old full queue.
