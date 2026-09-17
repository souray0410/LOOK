# 2026-09-17：完整交付与并行对照

用户批准的当前优先项：3416、16倍空间降采样、每轮最佳下游位置；两种缺失状态贯通后产生完整配置报告。四个代表起点对照为1、2、中间、末级特征，独立拟合并可在资源就绪后并行。原矩阵和独立线性家族补充继续保留，不由本次发布取消。

## 已实现

- `studies/search_delivery_queue.py` 从原五配置序列编译同一run/spec的search feed和优先级政策。它不启动任务或改变claims，不给控制臂增加“主策略完成”的假依赖；首次输出明确为prepared_not_activated。
- `runtime/delivery_policy.py` 校验配置身份、依赖和环路，优先关键依赖、当前交付、匹配控制、准备、通用模型。科学依赖必须通过所属case验收器；不能信任accepted字符串。新版策略不受旧硬编码两条顺序控制上限约束，实际资源准入仍必须通过。
- 原dispatcher消费政策，保留首种子有限周包门槛；后续种子不能只凭某个宿主完成就先行。新版配置验收后可触发累计发布；维护入口也可幂等刷新，不重新训练。
- 同一固定入口累计已验收配置；五配置完整后生成同宿主搜索策略统计/图。新增对照不覆盖旧有效结果，不完整组不排名。

## 核验与限制

本地Python3.11、torch2.8及本仓库锁定MHD来源：296项单元测试通过，结构检查通过。首次共用R&B的旧MHD安装造成4个Swin测试失败，已按LOOK自身框架锁设置独立导入路径并通过全套重验；没有修改模型来绕过错误。

Ibex曾连接重置/超时，域名入口vsc509-29-r又无法看到项目存储；通过原节点vsc509-03-l恢复访问。独立快照18项针对测试通过；原五配置按相同run/spec编译登记并生成累计入口，当前prepared_not_activated。未停止或重启已有worker。

## 上线接续

1. 核对live binding、claims、Slurm步骤、真实进度与源码，不照抄历史job编号。
2. 独立快照完成Ibex目标环境验收，使用现有准入/领取，不另建调度器。
3. 对当前序列运行 `python -m look.studies.search_delivery_queue --sequence <原序列JSON> --output <独立登记目录>`。核对5个run和specSHA完全不变。
4. 原科学worker保持运行。旧串行sequence管理器须明确退出并核对不会再领取后续配置；保留其历史快照。将编译的feed、delivery_policy、delivery_sequence和原weekly_delivery_policy绑定既有dispatcher，核对当前运行任务的共享claim，不能新旧owner同时派发同一序列。
5. 先观察新控制臂真实领取、预检、拟合和暂停/恢复；验收后增量并发。current配置的两个缺失状态、诊断、统计和报告仍是优先交付。
6. 既有维护执行 `python -m look.studies.search_delivery_queue --sequence <原序列JSON> --output <发布目录> --refresh`，仅刷新验收结果；原周包未解决内容仍阻止种子释放，不自动缩减清单。

通用准则已同步PHD及workspace/DELIVERY_STANDARD.md。最佳位置搜索不保证更快，报告保留真实拟合/评价次数与耗时；不能把配置或位置数少直接解释为计算更少。

## 11:18 UTC现场补充

当前主搜索仍为51909172.10，活跃计算进程已核实；缺OCT首轮已写到site_005，尚无完整配置接受。原登录节点管理进程已不在，不能仅凭旧status宣称监督仍活跃。新增attached_search只接续原manager锁、原claim的owner/generation及同一Slurm步骤观察；不启动/取消训练、不变科学源码。3项保护测试在本地及Ibex通过，observer已实际返回observing/step10，负责到期前900秒请求保存、死亡后严格验收和累计发布，不能凭陈旧心跳抢占。

现场证据在OPS/independent_delivery_20260917/{contract_tests.json,live_reconciliation.json,attached_search/}。新独立队列仍未激活；须先核对登录进程消失后待批申请的owner接续，恢复既有派发，再验证独立控制臂的真实资源与下游。当前没有新配置科研结果。
