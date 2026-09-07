# LOOK test follow-on

The current validation queue is unchanged. Step 49 waits for every declared
validation job and selected-start evaluation, then acquires the SAME exclusive
project GPU lock. This is an event-dependent local process, not a new scheduled
notification or a second concurrent scientific queue.

## Frozen scope

Use the existing test split of 290 participants. Natural test is withdrawn.
No natural-test dataset is constructed. Small original label catalogs remain
only for historical source checks in the immutable validation implementation.

The roster contains 87 inference configurations: 27 original LOOK, 27
validation-selected starts, 6 input/fusion-only ablations, 6 unimodal references,
15 method controls, 3 self-input controls, and 3 previously evaluated terminal-only
controls. Complete multimodal and filling predictions are included with LOOK;
they are not additional trained configurations. The other start candidates are
validation development evidence and are not searched again on test.

Freeze the complete roster, selection rule, checkpoint, generator, embedded
train-PCA/correction artifacts, validation predictions and report rules before
any test evaluation. Select each missing direction using all nine validation
starts; ties retain the earlier start. Never refit W/b, PCA, SSF, generators,
backbone, thresholds or other parameters. No early stopping on test.

## Acceptance and execution

Every job first replays 8 validation participants (deterministic class-balanced
subset defined by the existing loader). Require historical labels/IDs and identical argmax. Record historical FP32
logit drift across devices/batch algorithms, including a strict closeness flag.
Require same-device parity with the original LOOK forward path at atol=1e-6,
rtol=1e-6; method policies call their unchanged original inference kernels.
All original numerical modules are hash-pinned against the validation release. All replays must pass
before a test loader is constructed. Replays also exercise the real GPU memory
budget with the inference adapter. A failure blocks the queue without changing
the scientific configuration or reopening validation search.

Reuse the current device policy, including arbitrary single/multiple GPU choices,
UUID validation and drain/interrupt changes. At most one case per assigned GPU;
all LOOK processes together must stay within 14 GiB per GPU. Other projects do not
count toward this project allowance. The allocator limit is 11 GiB plus measured
NVML process accounting. On OOM, rerun the full deterministic inference case at
8, 4, 2, then 1 participants per batch; never skip examples or raise the budget.

Inference covers complete inputs, both fixed missing directions, and nested
20/40/60/80/100% participant missingness with mask seed 3407. For N=290 these are
58/116/174/232/290 people. Random scenarios reuse the three frozen directional
predictions, with identical participant alignment, and never trigger selection.

## Running

With `PYTHONPATH=<new-source>/tool` and the existing project Python environment:

```sh
python <new-source>/pipeline/49_run_test_queue.py --request <request.json> --output <new-run-root>/test_evaluation
```

`request.json` pins the completed main/method/self-input summaries, reused start
evidence, the live validation plan and identity, shared device policy and project
execution lock. The waiter validates these dependencies without loading test.
A failed predecessor remains blocked; the waiter never restarts validation.

Inspect `queue_status.json`, `events.jsonl`, `frozen_manifest.json`,
`replay_gate.json`, per-case receipts, and `report/REPORT.md`. Completed predictions
and receipts are immutable to a different case, phase or frozen manifest.

## Interpretation

Report all positives and negatives, three seeds individually and mean/sample SD,
paired participant bootstrap intervals for mean seed differences, F1 alongside
AUROC/AUPRC and NLL/Brier/ECE. Seeds share the same test participants and are not
independent cohorts. Single-seed ablations stay explicitly exploratory. This
selected balanced internal cohort does not establish natural-population
calibration or external clinical generalization. Seeing test scores ends this
round of method selection; future changes require a separately defined protocol.
