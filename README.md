# LOOK

本仓库是研发中的研究项目；当前分支和实验批次为 `2026_09_09_10_30_34`。从本次起与 Radon_Bridge 使用同一套研究仓库结构和操作命令。MHD_Project 是独立开源框架依赖，本次使用 V4。

```text
src/look_core/          可复用模型、算法与数据接口
configs/deployment/    机器路径配置
scripts/               统一安装、检查和测试入口
tests/                 单元测试与独立集成验证
docs/                  原理、开发规范、历史索引
experiments/2026_09_09_10_30_34/    当前研究协议及状态
third_party/MHD_Project/    固定提交的独立工具包
workspace/             共用结构规范与校验工具
```

```bash
git submodule update --init --recursive
bash scripts/bootstrap.sh
python scripts/manage.py check
python scripts/manage.py test
python scripts/manage.py preflight --machine ibex
```

依赖安装后，MHD 由自己的包提供；本项目 wheel 只包含 `look_core`，不复制框架源码。不从旧实验自动导入数据划分或任务队列。原始影像和表型CSV留在受限共享存储，新训练需独立的数据/协议验收。

[开发约定](CONTRIBUTING.md) · [架构](docs/architecture.md) · [历史复现](docs/history.md) · [统一结构](workspace/REPOSITORY_STANDARD.md)

旧代码、历史协议、报告及对应源数据表完整保留在[封存分支](https://github.com/souray0410/LOOK/tree/archive/2026_09_09_10_30_34_before_unified_layout)，以及原先各时间戳发布；当前分支不再携带旧报告树。受限预测、检查点和原始数据继续保留在授权存储，未上传GitHub或删除。
