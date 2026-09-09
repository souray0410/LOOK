# Native models training and reuse plan

Status: design, not a submitted training queue. This plan extends the user's request for persistent models and traceable hyperparameter studies. Core frameworks, native model training and downstream method tuning are separate layers.

## First cohort and model coverage

Use the previously accepted UKB small train/development cohort (1264/296 participants), its original glaucoma phenotype and its exposure ledger. No access to the previously evaluated holdout is authorized by these preparation runs. Full UKB data is a later dataset version and does not block small-cohort preparation.

The standard ResNet family is 18/34/50/101/152. Three distinct input tracks are relevant: CFP 2D, OCT 2D and ordered OCT 3D. Each track must pass its own cache/participant/eye-pooling/preprocessing acceptance, and each volumetric adaptation is explicitly labeled. These are candidate coverage axes, not proof that all 15 architectures/input views are implemented or feasible. They support the different native inputs of the two research projects; no 2D result is presented as a 3D result.

## Layered selection

1. Accept the MHD V4 implementation against its native reference, including meaningful intermediate features, gradients, updates and resume behavior.
2. Run a finite, prespecified native-model hyperparameter screen using development only. Declare the same search opportunity for comparable ResNet variants; use identical underlying data roles, task definitions and selection metric. Candidate learning rates, effective batch, regularization, augmentation and seed policy must be recorded before launch, with initialization-aware differences justified. Device memory determines feasible microbatches, not the winning training recipe.
3. Freeze each architecture/input track's selected configuration before repeated-seed training. A preparation seed can screen feasibility; a single seed is not a stable performance estimate. Archive the screen separately from the repeated-seed results and record whether any screening seed is reused. Do not silently equate development selection with independent confirmation.
4. Register converged/replayed native checkpoints as immutable parents. LOOK and Radon_Bridge each take verified owned copies. Inside a downstream comparison, no-method, proposed method and alternative methods share the exact native parent and matched data/protocol; no re-selection of a parent based on method advantage.
5. Tune downstream method parameters separately. Bridge width, geometry and attachment nodes belong to Radon_Bridge; correction/missingness choices belong to LOOK. Neither enters the generic pretrained model identity unless it is itself the artifact being trained.

Use architecture-specific settings when required, while documenting tuning budgets and scientific differences. Future RETFound adaptation is not fairly characterized as training from scratch under the same learning rate as ResNet. For BatchNorm networks, gradient accumulation does not erase microbatch differences. Do not claim global optimality: each selected setting is best only within its declared task/cohort/search budget.

## Required retained records

Retain every candidate's full configuration, immutable config ID, run/attempt IDs, source/environment locks, init weights provenance, logs/curves, selected checkpoint, full last/resume checkpoint, stopping reason, metrics/predictions in authorized storage, timing/memory and acceptance receipts. Keep unsuccessful settings visible. Epoch checkpoint retention must be explicit in the locked protocol; this plan never authorizes deleting already retained checkpoints.

Consumers must resolve a unique weights ID and verify the task, class order, preprocessing, feature endpoints, framework/adapter pin and checkpoint SHA before loading. Full-model replay, backbone-only transfer to a new head, and continued training each have different provenance and acceptance. Calling an existing weights ID does not authorize access to its previous holdout.

## Subsequent cycles

Add DenseNet, transformer/foundation-model families and U-Net through separate task-appropriate adapters. U-Net segmentation training requires corresponding segmentation targets; available disease-level labels alone cannot support that claim. Full-cohort UKB training is a new data protocol. V5 support is a version-migration acceptance followed, where scientifically justified, by new training. Existing V4/small-cohort artifacts remain addressable.

Model availability is not downstream efficacy. Report implementation fidelity, native predictive performance and downstream improvement separately. Native models trained on this small phenotype cohort are task-specific parents, not automatically universal foundation models.
