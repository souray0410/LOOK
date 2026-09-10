# Architecture

Both research applications use the same eight source roles and unit-test roles.

| Role | Responsibility |
|---|---|
| `data` | 数据读取、标签、队列及预处理 |
| `models` | MHD 原生网络搭建与前后向接口 |
| `methods` | LOOK 矫正或 R&B 通信算子 |
| `training` | 训练、优化、收敛和分布式执行 |
| `evaluation` | 预测、指标、检查点重放及评价 |
| `analysis` | 统计、诊断和报告图表 |
| `runtime` | 配置、路径、状态、调度及管理入口 |
| `studies` | 实验矩阵、协议及研究流程 |

Canonical modules: `data/dataset.py`, `models/graph.py`, `methods/operator.py`, `training/trainer.py`, `evaluation/evaluator.py`, `evaluation/metrics.py`, `runtime/paths.py`, `runtime/cli.py`, `studies/protocol.py`. Scientific submodules may differ by method; identical names must never disguise different mathematical roles.

The native MHD adapter is in `models`; method operators live in `methods`. This move preserves function/class names, state_dict keys and MHD node identities. Configurable entry points and legacy source references are rebased together. No research task starts from a package import.

Internal imports are absolute package paths. `scripts/check_layout.py` verifies actual source roles, unit-test roles, canonical modules, snake_case module names and internal import targets; `scripts/manage.py check` and CI run it. A flat source file or an unresolved old import fails validation.

See [module migration map](module_migration.json) for every previous module path. Python object pickles requiring historical import paths use [archived source](history.md); state_dict compatibility is checked independently.

The installed MHD release selects the API. Import `mhd_framework` / `mhd_framework.utils`; this application pins the V4 release, never floating main. Packaging paths changed; V4 tensor implementations are preserved.

## Reusable native models

Optional architectures are owned by `mhd_framework.models` in the framework repository. The core does not import this package. Architecture documentation and configuration are under that repository's `models/` directory. The selected V4 source commit is pinned in `framework.lock.json`; installing the historical V4 tag alone does not provide these later optional additions.

The project adapter `models/registry.py` resolves an exact training request, verifies and copies an accepted complete model into a project-owned directory, then loads it strictly. A missing match returns an explicit pending request; it does not silently substitute random weights or start a GPU job. The independent training workflow must satisfy that request before method experiments proceed.

Dataset identity and split, preprocessing, architecture, framework source, initialization, seed and training protocol belong to artifact identity. A compatible tensor shape is not sufficient for reuse. Model artifacts include the task head, selected model and a separate stopping/resume state. Loss and optimization belong to the training workflow; their definitions remain recorded with the artifact. Project fine-tuning creates project-owned results and never modifies the shared native model. Restricted participant data and predictions are not distributed with source.

The `models` extra is required for RETFound (`timm==0.9.2`); official pretrained weights require separate authorized access. No model import downloads weights.

The permanent [model/run standard](../workspace/MODEL_RUN_STANDARD.md) separates reusable definitions, configuration identity, timestamped training executions and accepted artifacts. Read-only project materialization can remain content-addressed; new training receives its own execution identity.
