# Bounded method evidence and advisor report

This supplement runs only after all 52 original stages and all 243 suffix cases
(including 30 reused cases) complete. It adds 15 cases: five methods, three seeds,
fixed layer3 fusion and normalized-mean filling. Each case fits two directions;
the five nested random ratios reuse the frozen directional configurations.
Combined development count is 52 + 213 + 15 = 280. Test remains sealed.

## Hypotheses and comparisons fixed before results

1. Original LOOK vs SSF: feature reconstruction compared with supervised
   parameter-efficient adaptation, using the same frozen backbone and head.
2. Original LOOK vs independent-fit LOOK: test the benefit of conditioning W/b
   fitting on accepted upstream corrections. Independent fitting removes upstream
   corrections during training feature extraction only; sequential validation
   evaluation still inherits the accepted corrections.
3. Original LOOK vs missing-only writeback: preserve the joint regression input,
   but replace only the missing branch at pre-fusion joint sites. Post-fusion
   correction is unchanged. This tests retained-branch correction, not whether
   the retained features should be available to the regression.

The terminal-only comparison reuses the ninth suffix. The original and
validation-selected-start configurations are retained. No new architecture,
residual-strength or additional dimension/factor search is introduced.

4. Original LOOK vs bias-only: set W=0, fit the train mean paired latent
   residual b, and retain the same PCA, sequential conditioning and validation
   search. This tests whether sample-dependent mapping is needed beyond recentering.
5. Original LOOK vs logit-affine: fit s_new = a*s+c, a>=1e-8, on unaugmented
   train logits/labels by binary cross-entropy plus 0.5e-6*((a-1)^2+c^2).
   L-BFGS-B max 2000 iterations, fixed tolerances, no optimizer search. Validation
   chooses fitted versus identity by strict Macro-F1; ties stay OFF. Frozen
   positive affine transforms preserve directional AUROC/AUPRC, but mixed-direction
   rankings can change at random ratios. Backbone, head and features stay unchanged.

The two new controls (6 cases, 12 direction fits) are exploratory additions
specified after seeing development results, before any test access. They are
not retrospectively described as prespecified before all validation observations.
The v2 order is logit-affine, bias-only, SSF, independent-fit, missing-only,
each with seeds 3407/3408/3409. Original nine control definitions are preserved.
The previous v1 waiting study has no completed cases; retain its summary and
record the superseding v2 identity in a deployment audit. Running parent and suffix
source releases are unchanged. No partial v1 artifacts are adopted.

## SSF adaptation

Reference: Reza et al., *Robust Multimodal Learning with Missing Modalities via
Parameter-Efficient Adaptation*, https://arxiv.org/abs/2310.03986.
This is a matched LOOK-task implementation, not exact reproduction of the
authors' benchmark. Hooks insert channelwise scale/shift after convolution,
linear and normalization modules in the retained encoder and fusion graph.
The missing encoder is frozen and unadapted; its input uses the same
normalized-mean filling as LOOK. The original classifier is fixed. Scale=1 and
shift=0 reproduce the filling baseline. Batch-normalization buffers stay frozen.

The prespecified search in configs/method_evidence.json uses AdamW, FP32,
learning rates [1e-5,6e-5,3e-4], weight decay 0.01, at most 100 epochs, patience
15, microbatch 8 and effective batch 128. Train labels provide cross-entropy
supervision; full validation Macro-F1 chooses the checkpoint and learning rate.
Identity epoch 0 is allowed; ties retain the earliest best epoch and the declared
learning-rate order. Report supervision and search-cost differences explicitly.

## Reporting and falsification

All negative values and right-to-wrong transitions remain visible. Report paired
final-feature L2 changes, decision margins, and cases where feature error improves
but classification worsens. For final classifier difference vector w,
|m(corrected)-m(full)| <= ||w||_2 ||h(corrected)-h(full)||_2. This bounds preservation
of the full model decision, not correctness or Macro-F1.

Measured inference includes the resident frozen backbone and correction, with
synchronized timings, batch size and GPU peak allocation. A smaller trainable
parameter count does not establish lower inference memory or edge deployment.
Shared PCA cost is reported separately and not multiplied across methods.

Only complete, hash-verified results enter the advisor report. Three-seed mean
and sample SD require all seeds; unfinished cases are pending, never zero.
The existing suffix report provides all starts, ON/OFF/excluded states and
sequential conditional gains. Spatial labels use rho = 1/4, 1/8, 1/16.

Before final test access, freeze configurations, comparisons, uncertainty
procedure and report scope. Natural and primary test participant overlap must
be assessed before describing them as independent replications. The supplement
does not access either test cohort or claim journal acceptance is established.

## Operations

Entry: pipeline/40_run_method_evidence.py. Default is a deterministic dry plan.
The launcher tool/operations/start_method_evidence.sh creates one tmux queue and
waits for both predecessors. A failed predecessor fails closed; no restart or
source replacement is automatic. Separate source, run directory and identities
prevent cross-method W/b reuse. All original numerical kernels stay unchanged.
The report generator updates after each new completed case and after the final
reference analyses; comparison_freeze.json records the completed scope.

Report runtime dependency: reportlab==4.4.3; plotting uses existing matplotlib.

## Advisor deadline and report refresh

The user requested a report for the morning of 2026-09-06 (Asia/Riyadh).
The full 280-stage program cannot be promised complete overnight: the parent
still had 18 stages remaining when this supplement was authorized, followed by
213 new suffix cases. Deliver a timestamped, verified development snapshot even
when the queue remains incomplete; never relax the protocol to meet the deadline.

`pipeline/41_refresh_advisor_report.py --method-summary PATH --output NEW_DIRECTORY`
runs read-only analyses independently of GPU queues, verifies every included
prediction against all six metrics, and writes an English presentation PDF,
Chinese speaker notes, paired effects, seed summaries and provenance. Use a
new timestamped directory each refresh. Render and inspect every PDF page before
copying it to the user. No email to the advisor is authorized.

The report includes F1-versus-AUROC changes, NLL ratios, unfavorable outcomes,
and only complete three-seed aggregates. Main changes may be described as
validation improvements; mechanism hypotheses remain pending until their
controls finish. Actual ON-node amplification still requires direct diagnostic
evidence; large unused candidate weights are not an explanation.
