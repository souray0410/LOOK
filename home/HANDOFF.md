# LOOK handoff — joint sequential correction

Updated 2026-09-04 after Souray's approved Step 36 migration. Runtime statements
are snapshots: refresh with the checker before acting. Respond in Chinese and
begin every reply with “好的，Souray。”; remain objective about negative metrics.

## Read in this order

1. `project.json`: canonical runtime roots, task and environment.
2. `JOINT_LOOK_PROTOCOL.md`: current method, queue, exact start/resume commands.
3. `PIPELINE_LEDGER.md`, `CHANGELOG.md`, parent `PROJECT_TIMELINE.md`.
4. `docs/joint_migration_evidence.json`: deployment acceptance and cleanup snapshot.
5. On ws, current `runs/joint_look/*/summary.json`, factor/decision records and
   `runs/maintenance/joint_protocol_cleanup_e6d740a884be.json`.
6. `docs/history/HANDOFF_3d0047c.md` for historical baseline/scouting evidence;
   its Step 34 method and running-state claims are superseded.
7. `references/260810_manifest.json` and source export for method provenance.

Run `python3 tool/operations/check_joint_look.py` on ws after reading. Old
`check_reviewed_look.py` inspects the retired protocol, not the current queue.

## Fixed context

Private canonical source: https://github.com/souray0410/LOOK (`main`). The release
contains `home/` source and `data/` skeleton. Deployed source is NOT a git checkout.

- SSH `ws`; source `/home/mengh/LOOK/2026_09_03_19_35_04`.
- Runtime `/data/mengh/LOOK/2026_09_03_19_35_04`.
- Python `/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv/bin/python`.
- Images and preprocessing are shared with release `2026_09_03_08_30_00`.
- Parent project standard and timeline remain authoritative maintenance records.
- Local terminal workdir `/tmp`; only temporary local checkout, no permanent
  source or downloaded participant data. Verify GitHub/deployment before removal.

Task `glaucoma_all_evidence`: participant-level binary record-derived phenotype,
925 cases and 925 matched controls; train 1264, validation 296, test 290. Each
participant has bilateral CFP and central 2D OCT; OCT grayscale is repeated RGB.
Natural-distribution test is secondary and sealed together with primary test.

Reuse exact layer3 seeds 3407/3408/3409 from baseline candidate
`025a0ceb64d2`; no further backbone search, no missing-input fine-tuning and no
alpha. Complete-input checkpoint selection evidence and failed historical gates
are preserved. The original baseline gate does not block this approved study.

Main filling is zero AFTER ImageNet normalization (`normalized_mean`). Black
raw images are `raw_zero` and are different inputs. Preserve independent cGAN.
Full complete-training PCA is shared across filling and missing directions;
Dmax 512 is independent of dimensions. Nine ordered joint/fusion sites can each
be disabled; all-off is a valid baseline. All ranking uses logit differences.

Study e6d740a884be was stopped for protocol replacement; its exclusive incompatible
outputs are retired using the new scoped cleanup, with negative aggregate results
kept. Never interpret old saturated softmax AUROC=0.5 as proof of absent logit
ranking information. Do not resurrect Step 34 or apply Step 35's old PCA-only reason.

Current implementation advantage: frozen local fits support staged execution.
Per-level weight offloading remains a proposed optimization, not implemented
capability or measured savings. No claim of computation-free or learning-free LOOK.
