# LOOK

本次准备分支与部署目录：`2026_09_09_10_30_34`。新实验运行于 Ibex，旧 ws02 研究保留用于复现，不自动续跑。

## 统一入口

- 代码根就是 Git 仓库根；不再嵌套 `home/`。
- `tool/look_core/`：LOOK 模型、校正及研究逻辑。
- `third_party/MHD_Project/`：固定提交框架子模块；本次只用 V4，与 Radon_Bridge 一致。
- `framework.lock.json`：API、上游提交、源码 SHA256；`workspace/check_framework.py` 检查漂移。
- `experiments/2026_09_09_10_30_34/`：本次 Ibex 环境与准备说明。
- `workspace/`：LOOK、Radon_Bridge、MHD_Project 共同管理规范。
- `docs/history/pre_ibex_20260909/`：迁移前说明和配置，不能当作新实验验收。

```bash
git submodule update --init --recursive
python workspace/check_framework.py --project-root .
```

完整安装及路径配置见[本次准备说明](experiments/2026_09_09_10_30_34/README.zh-CN.md)。新队列、数据划分与任务尚未锁定，历史 pipeline 和研究参数只是可复用实现，不自动构成新的运行授权。原始 CSV 与影像仅留授权存储，训练需另行验收的派生标签与划分。

[共同规范](workspace/README.zh-CN.md) · [研究通用标准](GENERAL_PROJECT_STANDARD.md) · [历史时间线](PROJECT_TIMELINE.md)
