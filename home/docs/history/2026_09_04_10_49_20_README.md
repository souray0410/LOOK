# LOOK current workflow

Read [HANDOFF.md](HANDOFF.md), then [JOINT_LOOK_PROTOCOL.md](JOINT_LOOK_PROTOCOL.md).
Current specification: `configs/macro_f1_study.json`. Current entrypoint: Step 37.
The notebook exposes the same configuration and detached launcher.

On ws:
```bash
bash /home/mengh/LOOK/2026_09_04_10_49_20/tool/operations/start_macro_f1.sh --gpus 0,1 --execute
python3 /home/mengh/LOOK/2026_09_04_10_49_20/tool/operations/check_macro_f1.py
```

The first command also resumes interrupted work and refuses a duplicate live
session. Keep source/specification/GPU list unchanged to resume the same identity.
Backbone checkpoint selection and LOOK decisions both use validation Macro-F1;
training loss stays cross-entropy. All factors, both missing directions and random
ratios are evaluated. Both test cohorts remain sealed. Historical AUROC execution
scripts are on `archive/superseded-auroc-2026-09-03`, not active main entrypoints.
