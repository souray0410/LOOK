# Method and implementation review after the unimodal evaluation failure

Scope: unchanged unified_fusion_joint_look_macro_f1_v2 specification; a correction
of the post-training evaluation adapter, not a change in the selection endpoint.
Only look._reset_inputs changes in scientific code. Backbone implementation files,
MHD core, checkpoint hashes and reference notebook source are unchanged.

## Checked data flow

- Complete backbone training uses full modalities; training minimizes CE. Full
  FP32 participant validation and logits argmax determine Macro-F1 checkpointing.
- Unimodal controls contain one input node. Their evaluation adapter must be
  equivalent to graph.reset_and_forward. Both OCT-only and CFP-only are now tested.
- Complete PCA receives unaugmented training data only. Complete targets receive
  no corrections. Shared bases are keyed by the frozen backbone and feature rules;
  no validation/test data enter PCA, and Dmax is independent of candidate d.
- iter_feature_pairs applies all accepted upstream artifacts only to the missing
  branch, including joint input. OCT/CFP members are concatenated and restored in
  fixed order; joint correction can change both branches without moving fusion.
- LatentSufficientStatistics fits y = z_full - z_current from training samples.
  Ridge residual is z_current @ W + b. Decode adds neither PCA mean nor feature mean.
- Every step evaluates the upstream bank with the current site off, then enables
  only a strict validation Macro-F1 improvement; ties choose smaller d. Whole-factor
  ties prefer fewer active sites, then larger factor. All-off is a valid bank.
- Evaluation reuses trained matrices. Random masks are shared across cases, nested
  and fixed in direction. No matrices are refitted from evaluation participants.
- Independent GAN training/selection stays inside the training cohort. It uses
  internal reconstruction L1, while classifier/fusion/LOOK selection uses Macro-F1.
- Test evaluation requires a frozen manifest. Neither sealed test cohort is used
  by this queue. AUROC/AUPRC ranking uses raw logit difference; other metrics and
  negative results are retained.

## Evidence and remaining scientific limits

Before the repair, the new evaluation regression has seven passing fusion cases
and two failing unimodal cases. After repair the full suite has 144 passing tests,
including upstream inheritance, all-off equivalence, residual/GCV calculations,
PCA sharing/training-only guards, atomic decisions/resume and test protection.
The real completed OCT-only checkpoint is additionally evaluated on all 296
validation participants, with exact original-forward logits, identical saved F1,
no training invocation and an unchanged weight hash. Tiny joint and three-fill
end-to-end checks are rerun after the adapter change; their logs are technical
verification, not efficacy evidence.

These checks establish tested implementation properties. They do not prove
universal efficacy, unbiased validation performance after selection, or improvement
of AUROC/calibration/test F1. The zero-map option guarantees only that each greedy
validation decision can retain its current upstream bank under this exact protocol.
Independent test confirmation remains necessary after the protocol is frozen.
