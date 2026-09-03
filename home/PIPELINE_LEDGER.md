# Pipeline Ledger

| Step | Purpose | Primary input | Durable output | Resume/integrity rule |
|---:|---|---|---|---|
| 1 | Initialize declared roots | `project.json` or CLI roots | runtime skeleton/manifests | create missing directories; never replace data |
| 2 | Sync source to server | local `home/` | remote source tree | rsync source only; exclude data/env/results |
| 3 | Verify or create environment/kernel | Python path and requirements | package install, lock, LOOK kernel | reuse valid venv; install missing/inconsistent packages |
| 4 | Verify source mounts | source UUID/mount arguments | mount audit | read-only requirement before data work |
| 5 | Export CFP and central OCT | authorized image media | image tree and export manifest | file-level completion; original media unchanged |
| 6 | Verify export | export manifest/images | verification | count/readability/reference checks |
| 7 | Audit paired observations | exported image tree | pairing report | deterministic pairing keys |
| 8 | Remove unpaired derived files | pairing allowlist | paired derived tree | dry-run plus `--execute`; source media excluded |
| 9 | Verify paired data | paired tree | paired verification | exact pair and leakage checks |
| 10 | Extract phenotypes | authorized CSVs + paired IDs | matched phenotype table | input hashes and schema |
| 11 | Audit weak eye labels | matched phenotypes | five-class weak-reference table | exclude ambiguous/multilabel rows; preserve provenance |
| 12 | Build four-class cohorts | source weak-reference table | balanced/natural CSVs and flow | deterministic SHA-256 Normal sampling; no image copy |
| 13 | Verify cohorts | cohort CSVs + image root | verification JSON | class mapping, counts, references, duplicates, leakage |
| 14 | Clean derived intermediates | PASS verification | compact dataset root | fixed allowlist, dry-run, explicit execute |
| 15 | Unit tests | source + tiny fixtures | test report | no formal run on failure |
| 16 | Graph smoke | balanced validation subset | topology smoke output | seven fusion + two true unimodal graphs, four logits |
| 17 | Pipeline smoke | tiny balanced subset | two filling/LOOK smoke runs | DDP, resume, frozen classifier and sealed-test checks |
| 18 | Interactive experiment | one config cell | same artifacts as library runners | no notebook-only scientific logic |
| 19 | Baseline/formal sweep | validated cohorts and profiles | checkpoints, predictions, rankings, freezes | deterministic IDs, best/last resume, test requires freeze |
| 20 | Matrix aggregation | completed LOOK banks | aggregate CSV/figures | selected banks only |

## Step 19 Baseline State Machine

1. Six feature-fusion LR/dropout calibration configurations.
2. Seven fusion positions at seed 3407 with selected profile.
3. Top-three positions at three seeds.
4. OCT-only and CFP-only references.
5. Three-seed ranking and 0.70 Macro-F1 validation quality gate.
6. If failed, persist label/image/confusion audit and stop.
7. If passed, require a reviewer note before frozen baseline creation.
8. Only then permit filling, paired cGAN and LOOK validation.
9. Freeze all selected validation artifacts before balanced/natural test access.

## Recovery Contract

- Complete outputs are reused only when their identifiers and manifests match.
- Incomplete training resumes from `last/`; best model is preserved separately.
- Each configuration is committed to progress before the next begins.
- New labels, LR, dropout, fusion, seed or relevant code produce a new run ID.
- Validation and test prediction filenames are disjoint.
- Natural-distribution test is secondary and cannot affect selection.
- Cleanup may remove only declared derived/runtime paths and never mounted source media.
