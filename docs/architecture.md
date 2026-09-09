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
