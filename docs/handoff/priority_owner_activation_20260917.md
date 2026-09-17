# 2026-09-17：LOOK优先主线自动接续已启用

新版CPU管理入口已在Ibex真实启动，PID678280，实际周期error=null；共享journal/output/account锁不变。10个尚未获批且未启动的LOOK申请已绑定统一priority-owner；未来新申请也使用该入口。它先检查当前有限主线是否需要安全接续，然后执行原GPU owner队列。既有科学worker和allocation owner没有被停止。

当前主线仍为原run `2026_09_17_15_28_21_422026`、原3416/spec/科学源码与环境。正常运行或未知存活状态不会重复领取；确认租期中断、旧step结束且原manager锁释放后才恢复。容量不适合或Slurm查询超时跳过该卡的优先任务；确定性故障单独留档隔离，其他健康候选继续。到期仍按原规则保存，不将48小时视为科研完成。

## 验收和已修复故障

- 3个接续测试、5个owner测试、4个原准入表达式场景以及真实共存资源拒绝检查通过。
- v3实际启动发现管理环境缺少`scheduling.project_priority`；保存失败证据，独立v4修正管理PYTHONPATH，科学worker环境不变。
- v4真实无提交daemon周期核验63个ready/12个拒绝、40个native源绑定；4个关键管理模块同源。新生产周期已刷新error=null。
- 独立核验31个owner pins、172个fallback pins全部SHA一致，10 LOOK和9 R&B待批绑定各自新版，仍共用唯一恢复watcher。
- 原三张运行卡的10个Slurm steps完全保留。待批不是运行；在此次核验时账号3张运行、21张待批。

**尚未发生并验收下一次真实租期切换**。接续路径已自动接入，不再等待手动启用；该边界不同于新搜索结果已完成。首配置尚未scientific accepted，整周匹配包与其他种子门槛不变，test仍封存。

## 授权环境证据

`OPS=/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31`

- `look_efficiency_20260917/priority_owner_v4/{activation,cycle_validation,ready}.json`
- `look_efficiency_20260917/priority_owner_v3/failed_activation.json`
- `lease_recovery_after_reboot_20260917/plan_radon_look_priority_v4.json`
- `look_active_workflow.json`
- `look_efficiency_20260917/continuation_v2/publication/current`

累计observer在两缺失状态和配置报告验收后发布同一配置的四个匹配单元，未完成时显示覆盖状态，不填造数值。管理程序、训练和汇总运行在Ibex，不依赖Mac的SSH连接；这不代表本机Codex在电脑休眠后仍能执行推理，也不承诺登录节点重启后的系统级自启动。
