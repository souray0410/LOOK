# UKB Record-Derived Eye Phenotype Specification

Profile: `ukb_record_prevalent_4class_bilateral`

## Analysis Unit

One participant at the earliest assessment instance with complete left/right CFP and
central OCT. Repeated visits are not independent samples.

## Target Evidence

| Target | Self-report eye field 6148 | Illness field 20002 | ICD-10 | ICD-9 | Procedure evidence |
|---|---:|---:|---|---|---|
| DR | 1 | 1276 | H36.0 or E10-E14 with `.3` | 362.0 or 250.5 | none |
| Glaucoma | 2 | 1277 | H40.1, H40.8 or H40.9 | 365.1, 365.8 or 365.9 | 20004=1436, OPCS C60.1, 5326/5327 positive |
| AMD | 5 | 1528 | H35.3 | 362.5 | none |

Dots and spaces are removed before code matching; valid ICD subcodes retain the listed
prefix. ICD/OPCS dates are compared with field 53 at the selected imaging instance.
Illness and operation diagnosis ages from 20009/20011 are retained in evidence details.

## Temporal Rules

- Evidence dated on/before imaging, or self-reported at the imaging assessment, is prevalent.
- First evidence dated after imaging is incident and stored separately.
- Undated hospital target evidence is excluded from prevalent/control classification.
- More than one lifetime target phenotype is excluded as target comorbidity.

## Controls And Exclusions

Strict Normal requires field 6148 code `-7` at imaging and no target evidence in available
follow-up. Exclusions include cataract, severe eye trauma, retinal detachment/occlusion,
other serious eye disease and relevant ICD prefixes H25, H26, H33, H34 and S05.

## Limitations

The available source exports do not include expert retinal-grading fields 30905-30943 or
primary-care field 42040. The phenotype is therefore a record-derived reference suitable
for controlled method development, not an expert image-level gold standard. These limits
must be reported in manuscripts and cannot be repaired by tuning on sealed test data.
The default extraction uses `ukb670300.csv`: it contains all selected field families and
the relevant imaging instances. The second export contributes only later 5326/5327
instances within this selected field set and is not needed for the earliest-visit task.
