# Self-input information-source control

One additional fixed policy, not a combinatorial search. Fusion: layer3. Filling:
normalized_mean. Seeds: 3407, 3408, 3409. Two fixed missing directions and the existing
five nested random ratios per case. Three new cases (six direction fits), queued
AFTER the immutable 15-case method study. Total computation stages: 283 = 52+213+15+3.
No new level arrangement, node groups, starts, fillings, backbones or hyperparameters.

## Scientific contrast

The primary comparator is existing `missing_only`, which reads both branches and
writes only the missing branch before fusion. `self_input_missing_only` keeps the
same writeback scope, original input start, level sequence, complete-train joint PCA,
dimension/factor search, train Ridge fitting, and strict validation Macro-F1 acceptance.
It clamps retained-branch coordinates to the complete-train PCA mean BEFORE joint
PCA projection, in both complete and missing training features and in inference.
Thus each pre-fusion predictor receives only the missing branch's current sample
information plus fixed training statistics. Clamping AFTER PCA would leak retained
sample information through the joint basis and is not allowed.

The missing residual is added to the original missing state; retained states are
left unchanged. Fused sites use the original full fused features and implementation.
Each policy fits downstream on its own accepted upstream state. OFF and ties behave
as before. No per-ratio or extra start selection is permitted. The shared joint PCA
still contains training-population cross-modal structure: this is not an independent
branch PCA method or evidence for arbitrary heterogeneous network portability.

With deterministic constant filling and no earlier cross-branch writeback, pre-fusion
missing states carry no participant-specific retained information. This is a predicted
limitation of the fixed control, not a runtime failure or permission to add variants.

Primary paired delta is self-input minus missing-only, matched by seed, direction,
scenario and metric. Positive F1 favors self-input. Report all six metrics and all
negative deltas. No three-seed aggregation until all three finish. A secondary contrast
against original joint LOOK is descriptive; it changes both read and write scope.
The contrast is exploratory and was specified after development discussion, before
its results. It cannot establish universal superiority of either policy.

## Source, reuse and execution

The existing parent, suffix and 15-case method releases and queue identities remain
unchanged. The supplement has its own release, identity, output root, state lock,
predecessor freeze and final comparison freeze. It verifies all three missing-only
results, predecessor inventory and pinned source files before running. It reuses
frozen backbones and identical complete-training PCA after hash checks. It refits its
own W/b; resume identities include policy, source, direction, PCA and upstream history.
All development stays test sealed. It requests no new data split or test inference.

```bash
bash tool/operations/start_self_input_evidence.sh --project-root /path/to/release \
  --session look-self-input --predecessor-summary /path/to/method/summary.json --gpus 0,1
```

Refresh the combined advisor report without waiting for GPU completion:

```bash
python pipeline/41_refresh_advisor_report.py --method-summary /path/to/method/summary.json \
  --self-input-summary /path/to/self_input/summary.json --output /path/to/fresh/report
```

The report preserves pending states and exports `paired_self_input_differences.csv`.
The 15 existing method cases and the three new self-input cases retain separate
source summaries. The merged 18-case view is only a derived reporting view.
