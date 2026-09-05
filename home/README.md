# Correction-start supplement: 2026_09_05_09_12_55

The latest release adds the suffix-start study after the unchanged parent 52-stage queue. See [START_STUDY.md](START_STUDY.md). Parent paths and status snapshots below are historical; use the read-only monitors for current status.

# LOOK 2026_09_04_19_18_07

Read [HANDOFF.md](HANDOFF.md), [the fixed study specification](configs/unified_study.json)
and [method and operations](JOINT_LOOK_PROTOCOL.md). Step 38 first screens the seven
fusion positions at seed 3407 and selects the top three by validation Macro-F1. Only
those three positions receive seeds 3408 and 3409 before joint sequential LOOK with
mean, black-image and independent cGAN filling. OCT-only and CFP-only are auxiliary
controls and never enter fusion selection.

Main is current source. Previous fixed-layer3 results are retained as exploratory history;
their queue is paused. Data, splits, preprocessing, ImageNet normalization and MHD V4
are shared unchanged. Superseded all-position three-seed results are not inputs.
Tests stay sealed. Stage-wise weight streaming is not implemented.


## Self-input supplement

Release 2026_09_05_23_25_06: append exactly 3 self_input_missing_only cases after the unchanged 15-case method queue. Total 283 stages. Same levels and shared PCA, train-mean clamp before PCA, missing-only writeback, no new start search. See docs/SELF_INPUT_EVIDENCE.md. Test remains sealed.
