# UK Biobank Task Scout Protocol

## Purpose

This scout answers one question before expensive LOOK experiments: which available UKB
record phenotype provides a learnable, adequately sized and scientifically defensible
image benchmark? It is exploratory task selection, not final model selection.

## Controlled Design

- One participant contributes the earliest complete bilateral CFP/OCT visit.
- Every profile inherits the same deterministic participant-level 70/15/15 split.
- Strict controls are matched 1:1 within each split by age, sex and assessment centre.
- Architecture is fixed to ImageNet V2 ResNet50 feature fusion with linear projection,
  bilateral mean and a linear head.
- Optimizer, learning rates, dropout, seed and a 12-epoch/patience-4 budget are fixed.
- Only train and validation are accessed. Test, LOOK, GAN and stage-fusion search are off.

## Candidate Rationale

| Priority | Profile | Scientific role |
|---:|---|---|
| 1 | `glaucoma_high_confidence` | 716 cases; best prespecified balance of evidence quality and sample size |
| 2 | `glaucoma_all_evidence` | 925 cases; tests whether 29% more cases outweigh single-source label noise |
| 3 | `any_target_eye_disease_high_confidence` | 1,052 cases; more data while retaining stronger record evidence |
| 4 | `any_target_eye_disease` | 1,945 cases; maximum eligible data and greatest phenotype heterogeneity |
| 5 | `glaucoma_objective_only` | 236 cases; cleaner objective evidence but underpowered, diagnostic only |
| 6 | `diabetic_eye_disease_all_evidence` | 449 cases; disease-specific and limited, diagnostic only |
| 7 | `macular_degeneration_all_evidence` | 436 cases; disease-specific and limited, diagnostic only |

Incident cases are excluded because combining prevalent diagnosis with future onset would
change the clinical question from classification to risk prediction. Cataract is excluded
because media opacity can create a shortcut rather than a retinal/optic phenotype.

## Review Rule

The leaderboard is ordered among sample-size-eligible candidates by Validation AUROC and
then Macro-F1. It is evidence, not an automatic winner. Final review must jointly consider:

1. case count and validation/test precision;
2. phenotype provenance and likely label noise;
3. sensitivity, specificity, calibration and train-validation gap;
4. whether CFP/OCT appearance plausibly reflects the label;
5. whether multimodal input later outperforms both unimodal controls;
6. suitability for controlled missing-modality recovery with LOOK.

Only after that review is one task promoted to full LR/dropout calibration, unimodal
controls, seven fusion positions, three-seed confirmation and sealed testing.

## LOOK Suitability Gate

The scout only tests whether a task has a reproducible validation signal. The promoted
task must then pass a separate method-suitability review before LOOK begins:

1. the complete-modality baseline is conventionally trained and clearly above chance;
2. multimodal performance is not worse than the strongest unimodal reference;
3. removing CFP or OCT creates a reproducible performance deficit;
4. complete-modality performance has not saturated so strongly that recovery has no
   measurable headroom;
5. three-seed variability and class-wise sensitivity/specificity remain acceptable.

The project never weakens a baseline deliberately. If a strong baseline approaches a
ceiling, that is reported and the missing-modality severity or task suitability is
reconsidered transparently rather than changing the model to manufacture headroom.
