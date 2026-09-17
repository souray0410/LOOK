# 后继融合宿主的接续管理：2026-09-17

16:42 UTC核验：v6 CPU dispatcher已安全接替旧v5，实际周期`workflow_allocations_active`、`error=null`；同一申请账本SHA保持，无allocation取消，无GPU进程信号。

新链为正收益树接续→原best-forward接续→依赖门控的后继任务与原候选队列。middle/features使用已登记同父模型、3416、16倍/q32/tree配置；当前依赖未验收，实际周期正确拒绝提前执行，不能宣称已产生新融合对照结果。12个重复种子仍受原周包门槛约束。

远端事实入口：`OPS/look_active_workflow.json`，`OPS/look_next_fusion_20260917_v1/owner_v6/activation_receipt.json`、`dispatcher_binding.json`、`invariance_receipt.json`及同目录上级`plan_radon_look_priority_v6.json`。OPS沿用既有2026_09_10_11_11_31操作根。

CPU针对测试4项，加之前policy/weekly 11项通过。新管理层源码为独立固定覆盖快照，科学worker继续原08bf6cc，不能把管理层全部称08bf6cc原始源码。新watcher复用原states/锁；Popen前原子记录intent，身份不明拒绝再次启动。

当前验收边界：控制周期已核验；完整watcher周期和科学产物推进仍在复核。manager心跳不替代产物；实际跨allocation恢复、后继完整GPU任务与科研报告仍待验证。保留原科学结果截止时间，不用本次控制面更新时间刷新科研证据。
