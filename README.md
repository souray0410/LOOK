# LOOK

[维护与交接 / Handoff](docs/handoff/README.md) — 当前版本、部署状态、验收证据与跨项目调用；读取后请刷新实时状态。

LOOK（inear one-one korrektur）：基于固定 MHD V4 的研究项目。`main` 是与 Radon_Bridge 统一的当前开发结构；`2026_09_09_10_30_34` 是本次研究批次标识。修改前的完整代码、配置和报告见[历史复现](docs/history.md)。

```text
src/look/
│   ├── data/            数据读取、标签、队列及预处理
│   ├── models/          MHD 原生网络搭建与前后向接口
│   ├── methods/         LOOK 矫正或 R&B 通信算子
│   ├── training/        训练、优化、收敛和分布式执行
│   ├── evaluation/      预测、指标、检查点重放及评价
│   ├── analysis/        统计、诊断和报告图表
│   ├── runtime/         配置、路径、状态、调度及管理入口
│   ├── studies/         实验矩阵、协议及研究流程
configs/deployment/     机器路径与部署配置
scripts/                两项目相同的安装、结构校验和管理命令
tests/unit/<role>/      与 src 相同的八类职责
tests/integration/      显式执行的合成集成验证
tests/helpers/          测试辅助工具
docs/                   架构、方法、开发、路径与历史索引
experiments/2026_09_09_10_30_34/  本批协议与状态
third_party/MHD_Framework/ 固定 Git 提交的独立依赖
workspace/              共享服务器与仓库管理规范
```

两项目统一目录职责、公共模块名、命令和配置格式；方法特有模块使用各自准确的名称。内部导入使用 `look.<role>.<module>`。完整规范和模块迁移表见[架构](docs/architecture.md)。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/bootstrap.sh
python -m look check
python -m look test
python -m look preflight --machine local
python -m look verify-env
```

上述命令也可统一写为 `python scripts/manage.py <command>`。检查不启动训练、不申请 GPU、不读取研究数据。可复用代码使用项目根目录相对路径；外部影像、CSV、缓存和结果由[路径配置](docs/paths.md)指定。

MHD V4 包版本 `4`，固定提交 `3559caa8d596d4438533a69d39d8a2c32eb21e46`，API 为 V4。它独立安装，不随 MHD_Framework 的后续修改自动升级。本仓库 wheel 只包含 `look`。

新训练需要独立验收数据和协议，不自动继承历史队列或 test 使用权限。受限影像、CSV、参与者预测和检查点保留在授权存储，GitHub 只保存代码与可公开汇总。

The installed MHD release selects the API. Import `mhd_framework` / `mhd_framework.utils`; this application pins the V4 release, never floating main. Packaging paths changed; V4 tensor implementations are preserved.

模型定义、时间戳训练记录、来源映射和保留规则统一遵循[长期模型与训练规范](workspace/MODEL_RUN_STANDARD.md)，适用于后续所有模型、数据集和研究项目。

## UKB mechanism supplement

The frozen-host LOOK supplement, its 378 additional neural training tasks, mechanism and sample-efficiency controls, sealed-test preparation, and external-method audit are defined in [the protocol](docs/look_mechanisms.zh-CN.md) and [the adaptation audit](docs/look_external_audit.zh-CN.md). The original 81-host study remains a separate ledger.
