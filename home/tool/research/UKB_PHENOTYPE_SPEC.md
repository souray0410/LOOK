# UKB Record-Derived Glaucoma Phenotype

Profile: `ukb_record_glaucoma_binary_bilateral`

## Analysis Unit

One participant at the earliest assessment with complete left/right colour fundus
photographs and central OCT B-scans. Repeated visits are not independent samples.

## Primary Cases

A prevalent glaucoma case must have evidence available at or before imaging and meet at
least one high-confidence rule:

- Hospital evidence: ICD-10 H40.1, H40.8 or H40.9 in field 41270, or ICD-9 365.1,
  365.8 or 365.9 in field 41271.
- Glaucoma-specific treatment or procedure evidence: field 20004 code 1436, OPCS C60.1
  in 41272, or positive 5326/5327 response.
- Concordant self-report: both field 6148 code 2 and field 20002 code 1277.

Single-source self-report remains in the natural sensitivity cohort but cannot define a
primary case.

## Controls And Time

- Strict controls require an explicit no-eye-disease response and no glaucoma evidence
  during available follow-up.
- Evidence first recorded after imaging is incident-only.
- Undated target evidence is excluded from the prevalent/control task.
- Competing serious ocular disease and target-disease comorbidity remain excluded.
- Participants are split 70/15/15 before matching. Controls are selected without
  replacement within each split by five-year age bin, sex, and assessment centre.

## Interpretation

The available CSVs do not contain expert retinal-grading fields 30905-30943 or
primary-care field 42040. Labels must therefore be reported as a
`record-derived glaucoma phenotype`, not an image-level gold standard. Natural-
prevalence evaluation is a frozen-model sensitivity analysis and cannot select models.
