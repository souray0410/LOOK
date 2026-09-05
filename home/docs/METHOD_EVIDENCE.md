# Bounded method evidence and advisor report

This supplement runs only after all 52 original stages and all 243 suffix cases
(including 30 reused cases) complete. It adds 9 cases: three methods, three seeds,
fixed layer3 fusion and normalized-mean filling. Each case fits two directions;
the five nested random ratios reuse the frozen directional configurations.
Combined development count is 52 + 213 + 9 = 274. Test remains sealed.

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
