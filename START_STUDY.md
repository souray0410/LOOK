# LOOK correction-start supplement

This release adds `suffix_start_validation_selection_v1` to the completed-parent workflow. It does not alter the numerical LOOK algorithm or the running parent release. The parent remains `screen_then_replicate_joint_look_macro_f1_v3` (52 stages).

## Scientific protocol

For each of the three selected fusion positions, three fillings and seeds 3407/3408/3409, enumerate all nine canonical correction starts. Before the start, every site is OFF. At and after the start, refit greedily using that suffix's accepted upstream corrections; the first eligible site is not forced ON. Train fits PCA and W/b; validation selects dimensions, node activation, global spatial factor and start. The original strict-improvement and factor tie rules are unchanged. Choose starts independently for OCT missing and CFP missing by descending validation Macro-F1, then ascending canonical ordinal for exact ties. Re-evaluate these two selected banks together for random missingness, with the same fixed nested masks and no ratio-specific search.

There are 243 suffix cases: 27 original full-path cases and 3 mean/3407 fusion-only cases are referenced after exact configuration and artifact checks, leaving 213 new fits. Combined with the parent's 52 stages, there are 265 unique development stages. Selected-start combined evaluations (27 contexts) are derived evaluations, not additional W/b fitting stages. Each case covers both fixed directions and all declared random ratios. Original input-only ablations remain separate.

The canonical site order includes joint input, joint pre-fusion states, fused states and participant pooling. Fusion position, allowed correction start and first actually enabled correction are separate fields. Original graph metadata is not rewritten; `legacy_topology_audit.json` supplies corrected interpretation.

A larger configuration search guarantees non-decreasing selected validation F1 relative to the included original start. That does not establish generalization. Keep original and selected-start variants in the final comparison; freeze both configurations before independent test evaluation. The primary test is a subset of the natural-distribution test, so those evaluations are not independent cohorts. This entrypoint has no test mode.

## Execution and continuation

From any working directory, with the configured LOOK environment:

```bash
PYTHONPATH="$LOOK_RELEASE/tool" "$LOOK_PYTHON" "$LOOK_RELEASE/pipeline/39_run_start_study.py" \
  --project-root "$LOOK_RELEASE" --parent-source "$LOOK_PARENT_RELEASE" \
  --parent-summary "$LOOK_PARENT_SUMMARY" --gpus 0,1
```

The command above is a deterministic dry run. `--execute --wait-parent` arms a continuation that checks only parent completion every 30 seconds, holds no model/GPU while waiting, and starts after all 52 stages complete. Parent failure exits with an error; it never restarts the parent. To survive disconnects, use `bash tool/operations/start_suffix_study.sh` with the same explicit roots and parent arguments. This is the experiment queue, not a scheduled notification service.

`tool/operations/check_start_study.py --project-root "$LOOK_RELEASE"` reads compact status. The new runtime root is separate from the parent. A parent resource mismatch fails; it does not train a replacement or quarantine a parent artifact. Completed cases and partial decisions resume only within their pinned start/source identity.

## Provenance and output

- `source_inventory.json`: parent summary, results/manifests, label tables, checkpoint, shared PCA and generator paths, sizes and SHA-256 hashes, plus unchanged scientific-source checks.
- `case_records/`: canonical and eligible sites, allowed start, actual first enabled site, ON/OFF/excluded states, dimensions, factor, per-step conditional gains and prediction/result hashes.
- `selected_starts/<context>/frozen_selection.json`: independent direction choices and exact matrix references. Its validation evaluation combines the two banks through normal inference.
- `report/advisor_report.md`: supervisor-facing development report with matched baselines, original/selected-start LOOK, seed variability, existing ablations, unimodal controls, limitations and source-linked evidence. It updates as complete contexts become available.
- `report/`: English PNG/SVG scientific figures; individual seeds and mean ± sample SD only when all three seeds exist; CSV tables; full validation metrics including secondary and calibration metrics. Incomplete combinations are absent, never zero-filled. Negative differences are retained. Spatial shrinkage is always labelled rho = 1/4, 1/8 or 1/16; vector sites use identity factor 1.

The three figure families cover start effects, ON/OFF topology decisions, and sequential conditional gains. Sequential increments are not isolated causal effects of individual nodes. Legacy audits are additive and do not modify accepted results.

## Acceptance

Run the full unit suite and `tool/operations/verify_start_study.py` with explicit source, parent and separate maintenance output paths. The latter uses the real frozen feature/3407 backbone and its existing training PCA with six train and six validation participants, two late starts, both directions and all fillings; its cGAN generator is explicitly an untrained deterministic conversion fixture. It checks all-OFF equality, suffix fits, resumed decisions, upstream input propagation and unchanged parent resources. Its numbers are technical acceptance only, never formal performance results. Existing unchanged training/DDP qualification remains attributable to the parent release.

## Runtime resource guard

The suffix launcher sets the child process open-file soft limit to 65536 (`LOOK_NOFILE_LIMIT` override), including when a tmux server already exists. This changes no scientific configuration. A failed parent is never restarted automatically. On 2026-09-05 at 06:27:56 UTC the parent stopped at 27/52 with DataLoader `received 0 items of ancdata` followed by `Pin memory thread exited unexpectedly`; the observed shell soft limit was 1024. Descriptor exhaustion is a supported working diagnosis, not a proven postmortem count. Recovery requires explicit authorization under the user's no-automatic-restart instruction.
