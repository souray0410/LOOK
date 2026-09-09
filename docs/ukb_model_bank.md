# UKB model bank: architecture verification and reusable native training

Status: design for the current research cycle; no training matrix, new GPU allocation or test access is created by this document. This supersedes the interpretation that the framework background work consists only of synthetic checks or reproduction on original paper datasets.

## Purpose and separation

Train conventional networks and retinal foundation-model adaptations on the available UK Biobank ophthalmology data. Retain complete evidence for MHD correctness, UKB training and downstream model reuse. This is a finite research workstream, not a keepalive workload.

1. **Implementation acceptance:** compare an official/native PyTorch reference with the MHD V4 adapter using the same initialized weights, UKB training-only inputs, targets, batch, loss and numerical settings. Check intermediate features, logits, loss, input/parameter gradients, multiple optimizer updates, BN state, checkpoint reload and interruption/resume. Synthetic edge cases supplement these checks. Numerical tolerance is declared by precision; long stochastic training need not be bitwise identical.
2. **Native UKB training:** after acceptance, train/fine-tune a single-modality task model to the declared convergence rule. This produces reusable native parent checkpoints. Existing pretrained foundation weights are initialization, not UKB-trained results, and fine-tuning is not reproduction of original foundation pretraining.
3. **Downstream research:** LOOK and Radon_Bridge independently import accepted parents and run their own methods/controls. Native training success is neither LOOK nor Radon_Bridge improvement evidence.

MHD must expose meaningful block/stage boundaries, skip branches, token/grid shape and Node IDs. Wrapping a complete opaque network as one edge does not establish the modular integration needed for downstream feature intervention. Preserve native mathematics; any new pooling or classifier is an explicitly documented adapter.

## Finite staged model coverage

| Stage | Candidate | Main role | Input boundary |
|---|---|---|---|
| First | ResNet50 2D | Already requested conventional baseline | CFP and separately OCT 2D |
| First | ResNet50 3D | Already requested volumetric baseline | Ordered OCT volume; bottleneck adapter and initialization audit required |
| Next | ConvNeXt-Tiny | A distinct modern convolutional architecture | 2D first |
| Next | ViT-B/16 | General transformer control | 2D patch-grid features plus explicit class-token handling |
| Next | RETFound-MAE, named CFP/OCT release | Domain-pretrained large vision model | Native 2D inputs; not a ready-made 3D OCT network |
| Later | A selected volumetric transformer | Broaden the actual 3D architecture family | Choose reference, task adapter and initialization before launch |

Candidates are not an exhaustive Cartesian product of every model, disease, seed and input mode. Complete first-stage correctness, timing and label audit, then lock the next finite matrix. RETFound reference repository contains several architectures and checkpoints; pin the exact implementation, license, checkpoint SHA and pretraining cohort description rather than using a floating model name. Public pretrained-data overlap with UKB holdouts must be audited or explicitly reported as unknown.

Initially complete one verified disease definition across the first-stage architectures. Add AMD/DR or other diseases only after reviewing available fields, time alignment and positive/control counts. Do not choose diseases from favorable model results. Independent disease-specific binary parents provide the cleanest initial link to current project questions; shared multi-label or self-supervised UKB pretraining is a separate protocol.

## Shared participants, independent assets

Define one immutable participant-level split/exposure ledger before any new training. Both eyes, all visits, modalities and diseases of a person inherit the same split. Record previous study exposure; previous test participants cannot silently become an untouched new holdout. Self-supervised pretraining, basis fitting and normalization fitting also obey the train-only boundary.

Use a declared paired-cohort view for directly comparable CFP/OCT parents. Training on all available single-modality participants is a valuable additional view, but must be separately identified because the training population changes. Preserve missing labels and timing uncertainty; do not automatically label missing diagnoses negative. Separate raw transfer acceptance from semantic label/volume acceptance.

A reusable parent key includes dataset/cohort and split SHA, phenotype version, modality and eye/participant prediction unit, preprocessing and input shape, architecture/feature-interface version, initialization SHA, framework/code revision, optimization protocol and seed. Equal architecture names are insufficient to establish reuse compatibility.

Proposed storage is a separate restricted `UKB_Model_Bench` research area using the common research skeleton; this document does not create a new public repository. MHD_Framework remains a generic toolbox: portable adapters/examples/correctness tests may belong there, while UKB cohort code, clinical labels and Slurm/account details do not enter its core. Keep bank checkpoints immutable; LOOK and Radon_Bridge take project-local verified regular-file copies with parent provenance. Each project owns subsequent trainable state and results.

## Resource-aware training and priority

Prefer an actively used interactive allocation for development and controlled job steps for work. Every ready task is a real, finite case with a completion criterion. About half actual GPU memory is a resource-sizing starting point, not a reason to allocate tensors or repeat completed work. Measure full-step peaks, compute utilization and throughput; retain explicit transient and free-memory margins.

For formal UKB training, fix effective batch, optimizer and learning-rate rules per protocol. A100/V100 may use different microbatch/accumulation only under a declared policy. Ordinary BatchNorm depends on microbatch and is not made equivalent by accumulation; fix its mode, use a compatible configuration, or report separate settings. Do not change the training recipe merely to reach 50% memory. Reference/MHD acceptance always matches the actual numerical configuration.

User-directed project tasks have priority. Pause background training at a consistent checkpoint boundary, save optimizer/scheduler/scaler/RNG/sampler/BN/early-stop state, then exit all distributed ranks and verify GPU memory release. SIGSTOP retains CUDA memory and is not a valid handoff. Start with epoch-boundary resume if exact mid-epoch sampler/prefetch restoration is not verified. Never mark an interrupted parent ready. Portable inference across GPU classes and numerically identical training continuation are distinct claims.

## Complete retained evidence

Retain protocol, adapter/reference source SHAs, framework dependency, environment, initialization source and license, input/label/split manifests, actual commands, all attempts and logs, per-epoch metrics, best checkpoint, stop/resume state and acceptance receipts. Save train/development predictions and later authorized test predictions in restricted storage with stable participant order. Record parameters, training steps, elapsed/GPU time, throughput and memory, plus feature-interface metadata for downstream integration. Do not delete intermediate evidence under an implicit cleanup rule; any later retention reduction needs an explicit policy.

Registry lifecycle: `planned -> implementation_accepted -> training -> converged -> reusable`, with separate `paused`, `failed`, `needs_attention` and `test_evaluated` records. Reusability requires provenance and strict replay, not only a high development score. Development selects checkpoints; test is unlocked only under the dedicated frozen evaluation protocol. All failed/unfavorable completed cases remain in the registry.

GitHub stores code, protocols, non-identifying aggregate evidence and artifact digests. Restricted participant data and UKB-derived checkpoints stay in authorized storage; any model release requires its own data/license review. Report three questions separately: implementation fidelity, native UKB performance, and downstream method benefit.

## Immediate preparation

- Both label CSV headers are readable; field definitions, dates, image linkage and counts still require audit.
- Complete image copy and ordered-volume integrity checks.
- Implement/accept ResNet50 2D/3D adapters and measure resource candidates without reading holdouts.
- Lock phenotype, split/exposure policy, native training recipe, seed plan and finite task manifest before requesting the first UKB training allocation.
- The generic adaptive/priority runner and complete resume contract are not yet accepted; documentation alone is not an active training queue.

Sources: [Torchvision model references](https://docs.pytorch.org/vision/stable/models.html), [official RETFound implementation and checkpoints](https://github.com/rmaphoh/RETFound).
