# LOOK 2026_09_04_19_18_07

Read [HANDOFF.md](HANDOFF.md), [the fixed study specification](configs/unified_study.json)
and [method and operations](JOINT_LOOK_PROTOCOL.md). Step 38 compares seven fusion
positions and two unimodal controls at three seeds, selecting checkpoints and the top
three fusion positions by validation Macro-F1. It then evaluates joint sequential LOOK
with mean, black-image and independent cGAN filling on each selected position/seed.

Main is current source. Previous fixed-layer3 results are retained as exploratory history;
their queue is paused. Data, splits, preprocessing, ImageNet normalization and MHD V4
are shared unchanged. All models/generators/PCA/results in this release are newly fitted.
Tests stay sealed. Stage-wise weight streaming is not implemented.
