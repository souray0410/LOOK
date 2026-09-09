# 2026_09_09_10_30_34：统一工作目录准备

见[统一约定](../../workspace/README.zh-CN.md)。本次为统一 Ibex 新实验准备，分支和部署目录同名。旧发布源码及结果保留，通过旧提交和环境复现。

## 后续明确授权：Ibex 新实验准备

用户于2026-09-09授权停止 ws02 LOOK 并准备 Ibex 新实验，明确不续跑旧队列。停止记录位于新时间戳运行目录的 `migration/ws02_stop/`；旧实验结果和未完成 attempt 保留，不能按完成报告。旧队列无设备授权，不重新启动。

本次沿用 V4，`framework.lock.json` 锁定上游提交与源码SHA。MHD V5独立继续开发，两应用不自动切换。`environment_bootstrap.sh` 创建独立 LOOK 环境，安装原精确依赖并做 V4 CPU 验收；不会启动训练或申请GPU。

`ibex_environment.sh` 给出明确 Ibex 根路径，通过既有环境变量接口覆盖历史默认。旧配置归档于 `docs/history/pre_ibex_20260909/project.json`，新的根 `project.json` 使用 Ibex 路径并明确等待新协议。新队列必须先 source 此环境配置，并完成数据与协议验收；`pending_protocol` 标签路径刻意保持未建立，原始表型CSV不自动替代派生标签。

代码目录：`/ibex/project/c2377/souray/home/mengh/LOOK/2026_09_09_10_30_34/`。
运行目录：`/ibex/project/c2377/souray/data/mengh/LOOK/runs/2026_09_09_10_30_34/`。
研究未锁定，不生成可启动训练的 workload，也不申请占用GPU。Slurm适配采用分配的可见设备，不覆盖成ws02卡号；正式单/多卡预检随确定的任务单独完成。

LOOK 已统一为仓库根应用、固定 MHD 子模块及 V4 导入，与 Radon_Bridge 的依赖接入方式相同。旧导入路径完整对象 pickle 由旧发布环境读取；新环境的旧模型兼容验证以 state_dict、初始输出和梯度为准，不承诺任意历史 Python 对象跨包路径反序列化。
