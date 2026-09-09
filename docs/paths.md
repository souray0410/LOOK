# Portable path policy

Internal configuration paths resolve from the repository root, independent of cwd.
Absolute paths are allowed for externally stored data. Server-specific roots belong
in configs/deployment, not reusable scientific source. Runtime receipts may record
resolved absolute paths for provenance. Do not rewrite historical records.

LOOK: the default data root is ../data/LOOK, relative to the checkout. Set
LOOK_DATA_ROOT to override it, or SOURAY_MACHINE=ibex (or ws02) to select an explicit
machine profile. Derived dataset/image/label/cache locations follow that data root.
Other LOOK_* path overrides are also relative to the checkout when not absolute.
The default cohort is pending_protocol; no old split or test permission is adopted.

Radon_Bridge: pass a workspace config explicitly. Both the shared research_deployment_v1
and previous radon_bridge_workspace_v1 schemas are accepted. Relative roots use the
containing project root (or the config's directory when outside a project).
Artifact/basis paths use the editable project root; for a separately installed wheel
set an absolute RB_PROJECT_ROOT. Set RB_ARTIFACT_MAP to an explicit JSON with
schema=artifact_path_map_v1 and mappings=[{source: absolute_old_prefix,
target: new_prefix}]. Relative map filenames and targets are project-root-relative.
Existing files take precedence, the longest mapped source prefix wins, and archived
JSON is never changed. Old server-specific reproduction remains on archived branches.

To share a checkout, change the external roots or supply a personal machine profile.
No data copying, historical cleanup, model changes or training occurs during resolution.

## Any server or local machine

`python scripts/manage.py preflight` defaults to the local profile. `--machine ws02` and `--machine ibex` select optional examples. Use `--profile configs/deployment/my_server.json` for any other machine without changing source code or the shared registry. The profile uses the same schema, project identity and study timestamp, plus code_root/data_root (absolute or repository-relative). No path is created by preflight. LOOK also selects a named application data profile with `SOURAY_MACHINE`; explicit data overrides take precedence.
