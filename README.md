# LOOK Release Bundle

This repository is a portable two-tree Linux reproduction bundle. It contains source,
configuration, documentation, and an empty runtime skeleton; it does not redistribute
UK Biobank data.

```text
2026_09_02_00_00_00/
|-- home/   deploy as the project source root
`-- data/   deploy as the runtime cache/runs skeleton
```

The author deployment is:

```text
home/          -> /home/mengh/LOOK/2026_09_02_00_00_00
data/cache/    -> /data/mengh/LOOK/2026_09_02_00_00_00/cache
data/runs/     -> /data/mengh/LOOK/2026_09_02_00_00_00/runs
validated data -> /data/mengh/LOOK/2026_08_30_11_20_47/dataset
```

That mapping is centralized in `home/project.json`. A new iteration changes the
release roots there and may keep `dataset_root` unchanged. Pipeline and notebook code
do not contain timestamp-specific runtime paths.

See `home/README.md` for setup and execution commands. The reusable engineering contract
for this and future projects is in `GENERAL_PROJECT_STANDARD.md`.

## Reusable Project Standard

The concise rules below are summarized from `GENERAL_PROJECT_STANDARD.md`; that standalone
document is the normative, project-independent checklist.

Future research projects should follow the same contract:

1. Keep a timestamped `home/` source tree separate from its `data/` runtime tree.
2. Centralize author defaults in one project configuration; resolve paths as explicit
   CLI arguments, then environment variables, then project defaults.
3. Keep internal imports and resources relative to the installed project root.
4. Number only ordered executable pipeline entries; keep reusable libraries unnumbered.
5. Give each scientific configuration a deterministic ID derived from data, code, and
   normalized configuration fingerprints.
6. Resume complete or partial valid artifacts; quarantine invalid artifacts; write new
   files atomically; never silently overwrite a valid formal result.
7. Separate validation/model selection from sealed test evaluation.
8. Record environment, device topology, random seeds, manifests, predictions, metrics,
   and analysis artifacts with every formal run.
9. Treat controlled source data as read-only and exclude data, environments, caches,
   and checkpoints from the source release.
10. Require unit, smoke, integrity, portability, and documentation gates before release.

Defaults make the author's host convenient; explicit parameters make the same release
portable to another Linux account, server, or storage layout.

## Data Access

UK Biobank data are controlled-access resources. Reproducers must obtain authorization
and supply their own image and phenotype sources. No participant data are included.

## Acceptance Gates

- `MHD_Framework_V4.py` is the reviewed core; multi-GPU preparation lives in Utils,
  and classifier backward propagation enters through `MHD_Graph.backward(...)`
  and the LOOK application layer.
- `GPU_DEVICES=[0]`, `[1]`, or `[0,1]` is the only notebook compute selector.
- Classifier and paired-cGAN training support one-process-per-GPU DDP; LOOK fitting and
  final evaluation run on the first selected GPU.
- The graph contains differentiable loss and diagnostic metric nodes with declared
  monitor hyperedges.
- Validation artifacts are hashed before sealed test access.
- Source manifests contain no deprecated package aliases, compatibility links, hidden
  interpreter paths, or runtime data.
