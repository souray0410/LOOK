# 最新单状态结果：2026-09-17 20:05 Saudi

缺OCT正收益树17候选完成，最终71.33%→75.47%，仍选单末端；见[累计表、路径解释与边界](current_results_20260917.md)。另一缺失状态及完整周包尚未验收。

# 2026-09-17 后继融合接续更新

见[新管理器实际周期与验收边界](next_fusion_activation_20260917.md)。v6已接替CPU管理；middle/features等待原研究依赖，不是新增结果完成。

# LOOK：当前交接入口

**当前主策略已由用户修订为[正收益下游分支树](positive_forward_tree_20260917.md)。** 08bf6cc实现；3416、空间16倍、q32新配置已通过真实宿主资源预检，正式51919716.20已复用根层9个候选并展开3个正分支，Stage1前缀下游统计256→320批。旧best-forward继续作为对照。下列旧快照保留其原日期，不能覆盖本次主线修订。

最新运行更新：[优先主线自动接续已启用](priority_owner_activation_20260917.md)。10个待批及未来申请已绑定新版，真实管理周期通过；新科研结果仍按独立验收判断。

最新实现：[完整交付与依赖并行](independent_delivery_20260917.md)。本地代码通过检查；远端部署尚待资源、领取和下游验收。下列历史运行记录保留其原核验日期。

## 当前计划和数值对比（2026-09-17更新）

先读[完整数值、方法含义、实际队列与限制](current_results_20260917.md)。包括6个已验收末端案例及原始/关闭分数，不能与正在运行的best-forward混用。

## 当前故障恢复：全量dev已通过，正式拟合进行中

详见[9月17日修复与下游证据](allocator_recovery_20260917.md)。新q32曾失败，现原任务恢复到51909172.10；782批验证已过，进入train拟合。不是完整配置接受。旧预检状态以下仅作历史记录。

## 2026-09-17 完整配置优先已部署

见[现场、来源及缓存边界](serial_delivery_20260917.md)。16/32/3416新任务已在
51909172.8执行真实预检，自动衔接正式拟合与报告；尚无该配置正式接受结果。
原16倍仍含10维网格，不能与新单维任务混为一谈。独立仿射家族缺口继续保留。

## 2026-09-16 晚：整周首种子交付门槛

[当前搜索实现和部署证据](https://github.com/souray0410/LOOK/blob/52afe196fe42a2dcf424669d0d3c8ecbcc7c9693/docs/handoff/README.md)：best-forward为主，四个固定起点为对照；先完成3416整周实验、诊断、统计、图表和报告再放行其余种子。Ibex17项测试通过，15条首种子搜索路线已登记，正式搜索仍等待独立预检；其他线性算子独立搜索仍有实施缺口，不宣称全项目完成。实时入口仍为look_active_workflow.json。

# LOOK — 维护与交接

## 2026-09-17：正式最佳位置向后搜索启动

预检跨租期续接通过，58份旧候选SHA不变；完整队列在8GiB主机保护退出后，版本化v2以24GiB额度恢复，51919716.11已越过原失败位置；健康原worker保持，完整规模峰值仍在验证。仍未产生该正式任务的验收结果；独立仿射完整搜索未部署，整周门槛继续关闭。详见[当前实施交接](https://github.com/souray0410/LOOK/blob/f4bb293/docs/handoff/README.md)及[验收记录](https://github.com/souray0410/LOOK/blob/f4bb293/docs/handoff/search_continuation_20260917.json)。


## 2026-09-17：正式最佳位置向后搜索启动

预检跨租期续接通过，58份旧候选SHA不变；原spec锁定的正式任务已在51919716.7执行完整dev评价，健康原worker保持。仍未产生该正式任务的验收结果；独立仿射完整搜索未部署，整周门槛继续关闭。详见[当前实施交接](https://github.com/souray0410/LOOK/blob/5da3a3d/docs/handoff/README.md)及[验收记录](https://github.com/souray0410/LOOK/blob/5da3a3d/docs/handoff/search_continuation_20260917.json)。


## 2026-09-17：搜索预检接续修复已上线

新管理源`ea76569`经Ibex23项测试后部署，预检持久化、exit75暂停和到期保护已接通。当前独立GPU预检仍在运行，正式新搜索尚未验收。activate_v4与新dispatcher已现场核查，旧GPU任务保持。实际跨allocation续接和独立仿射搜索仍待完成；不能称整周研究已全自动验收。详见[当前实施交接](https://github.com/souray0410/LOOK/blob/f4e9fae/docs/handoff/README.md)。


最近核查：2026-09-14 08:46（Asia/Riyadh）。本页是有日期的交接快照，不是实时监控。

## 职责与当前研究

LOOK 是冻结已训练宿主后追加的缺失模态修正研究，使用 MHD_Framework V4。基础模型训练独立于本项目。当前优先完成 UKB 眼科研究，再扩展其他器官方法实验；其他器官的数据准备和合格基础模型可提前进行。

原81个宿主单元及新增378次宿主/学生训练、机制消融、样本效率和外部方法适配审计保持原协议。主输出是完整、缺CFP、缺OCT状态下宿主的最终分类结果，不能套用 Radon_Bridge 的分支平均指标。test仍封存。

## 版本和部署必须分开

| 对象 | 核查基线 |
|---|---|
| 准备分支 | `implementation/ukb_complete_20260913`；文档调整前为 `ba9fc74` |
| 线上调度快照 | `look_roles_e74bb81`；实际控制器与worker版本从绑定文件及配置读取 |
| 框架依赖 | 各代码版本自己的 `framework.lock.json`；准备分支锁定 `c0a27ab`，不改历史worker锁 |
| 最新已核实代码检查 | [ba9fc74 CI成功](https://github.com/souray0410/LOOK/actions/runs/34785048568)；后续提交须按新SHA重新核查 |

读取main上的本页不表示main已包含准备分支全部科学代码，也不授权热更新运行中的源码。

## 运行状态和下一步

08:46读取的控制器状态更新时间为08:23：当前筛选清单116个候选，47个接受；项目队列0个任务、0个接受，等待配对的三种子父模型。这个分母是LOOK当前清单，不能与全模型池或Radon_Bridge清单相加。

1. 验收父模型配方、各模态三种子及开发预测重放。
2. 在Ibex验收宿主训练→冻结→LOOK/对照拟合→匹配dev评价→恢复→报告的完整路径。
3. 依赖满足后滚动派发，不把基础模型完成计作LOOK完成。
4. 新宿主、学生、机制和外部适配按各自门槛接受；全研究锁定后才进入test。

协议和范围以[机制协议](https://github.com/souray0410/LOOK/blob/implementation/ukb_complete_20260913/docs/look_mechanisms.zh-CN.md)、[外部方法审计](https://github.com/souray0410/LOOK/blob/implementation/ukb_complete_20260913/docs/look_external_audit.zh-CN.md)及对应配置/接受记录为准，不从本页重新生成科研参数。

## 如何刷新状态

授权Ibex环境的操作根为 `/ibex/project/c2377/souray/home/mengh/operations/2026_09_10_11_11_31`。
先读取其中 `look_active_workflow.json`，再按其 `controller_config`、`dispatcher_config`、`dispatcher_output` 和配置的 `output` 找状态、队列及日志。核对时间、实际Slurm step、训练状态与接受文件；陈旧心跳不授权重复领取。PID和allocation编号只作当次快照，不能当永久定位。

## 跨项目调用

- [MHD_Models交接](https://github.com/souray0410/MHD_Models/blob/main/docs/handoff/README.md)：请求并验收独立完整父模型，记录架构/输入、标签与数据SHA、预处理、配方、种子、框架、检查点SHA及Node映射。未完成返回等待依赖，不静默换随机权重。
- [Radon_Bridge交接](https://github.com/souray0410/Radon_Bridge/blob/main/docs/handoff/README.md)：共享合格父模型与数据产物，不共享可变训练状态或混合评价定义。
- [MHD接口与版本](https://github.com/souray0410/MHD_Framework/blob/release/v4/docs/installation.md)：按应用锁安装准确提交，不跟随浮动main。项目拥有自己的宿主、修正产物和预测。

## 验收、资源与维护责任

GitHub负责独立安装和代码检查；Ibex是实际GPU完整链路验收环境。ws02仅可选调试，不是前置门槛。健康任务保留，新验证作为经资源准入的独立attempt。GPU完整峰值、CPU/主机内存、恢复、节点/BN/随机状态均需记录；不在登录节点跑GPU验证。所有新增GPU申请至少48小时。细则见[共享规范](../../workspace/GPU_EXECUTION_STANDARD.md)。

部署、协议变更、验收失败/修复、阶段完成和阻塞变化后维护本页及精确证据链接；保留更新时间，区分计划/代码通过/环境通过/部署/结果接受。重大历史变化由Git保留，不覆盖原实验。跨项目读取先刷新，不凭README修改他方任务。GitHub只放汇总和来源引用，不放参与者标识、原始数据、特征或受限预测。

## 2026-09-14 名称与框架对应更新

训练仓库已从 `Model_Training` 更名为 `MHD_Models`，仍为私有仓库并保留完整历史。新引用使用 `souray0410/MHD_Models`；旧源码目录、运行ID、检查点和框架锁不因改名重写。详细规则见 [MHD_Models版本与发布契约](https://github.com/souray0410/MHD_Models/blob/main/docs/framework_compatibility.md)。本条只更新名称与依赖说明，不刷新上文训练进度，也不表示新科学代码已部署。


## 2026-09-14 执行衔接修复（上线前核验）

临时共享提交锁竞争改为明确的`waiting_submission_lock`，只在获取flock的位置捕获；真正的fork/Slurm查询失败仍报错，不将所有BlockingIOError隐藏。原工作队列可追加独立、按SHA核验的native feed；同run同spec的多研究引用去重，保持原领取和科学身份。控制器与GPU worker分开验收和部署，旧健康训练不热改。

## 2026-09-14 11:12 控制器交接与长期自审

控制补丁09be3be；CI34820266134，161项通过。新控制入口已在Ibex通过清单/源码核验及启动验证，旧CPU dispatcher无子进程后通过stop文件正常退出，再移交原journal和共享资源锁；未发送GPU终止信号、未取消任何allocation。当前绑定文件指向新控制快照，现有GPU owner继续使用原配置和科学源码，新allocation才采用新feed。现场23个单卡allocation仍运行，四个项目父模型持续更新；这不等于LOOK或Radon_Bridge方法实验已产生结果。

通用自审原则见[RESEARCH_AUDIT_STANDARD](../../workspace/RESEARCH_AUDIT_STANDARD.md)：主动检查假设、实现、证据与检查本身的盲点，不依赖用户发现问题。监控须同时核对完整研究范围和运行事实，不以部分CI通过或进程存活概括整体就绪。OPS/execution_audit_20260914/{config,latest,repair_ledger}.json记录独立门槛、覆盖与修复状态；脚本只读，不替代完整模型/Slurm/真实GPU验收。

## 2026-09-14：长期自动纠错责任

用户明确自动纠错必须成为通用长期规则，不能只适用于本次故障。AGENTS及共享RESEARCH_AUDIT_STANDARD已规定：已授权流程必须从发现推进到诊断、版本化修复、验收、安全恢复和真实进度复查；不能停在报警、CI通过或调度器重启。未关闭问题保留证据、责任主体、下一步及可执行自动续接。保护健康任务和原科学标准，限制瞬时重试，不盲重启确定性错误。规范适用于未来模型、数据、框架、项目与CI/资源/评价/报告链路，但不是保证所有故障都能自动解决。规则更新本身不表示任何当前训练或故障已完成验收。


## 2026-09-14 15:56：LOOK控制补丁已安全接替

控制提交 `e25910c` 的 [CI 34845530283](https://github.com/souray0410/LOOK/actions/runs/34845530283) 已成功；Ibex七项针对测试、124/124父任务API清单及合成会话保护测试通过。线上使用旧 `look_execution_09be3be` 源副本，仅替换 `runtime/project_dispatch.py` 为该提交版本，并绑定已核验的 `models_023d471` 预检脚本；其他科学模块逐文件保持原SHA。

实际入口改为 `OPS/look_admission_20260914/dispatcher.json`，证据在同目录及 `look_admission_validation_20260914`。新CPU dispatcher当次PID1678754已实际运行、无错误、等待账户容量；64个当时可领取任务API拒绝数0。旧CPU dispatcher2931336由独立会话守护保持，原salloc子进程807373及其健康GPU父训练持续更新到第7轮。不得杀旧会话或重复启动管理器；旧子进程全部自然结束后守护才退休旧CPU。新journal加入原角色政策，复制原三条请求身份，没有取消或重复提交allocation。

此为控制路径安全接替，新的正式GPU worker仍须逐任务完整资源及恢复验收，不是LOOK方法研究完成。Radon独立长GPU恢复探针仍在进行，不能重复启动或提前解除其派发保护。下一维护继续推进首个项目参考匹配组执行依赖与实际验收，不以该控制修复代替科学链路实现。

## 2026-09-14 current-format migration boundary

Shared model standard v4: new releases use one canonical artifact format and reader.
MHD_Models main 0859bfe implements the new package and one-time converters. Running
models and research projects retain original immutable snapshots until accepted
transition. See [migration](../model_migration.md); package-layout deployment is not
permission to change scientific protocols or follow main at runtime.


## Permanent contract review gate (2026-09-14)

The shared [research audit standard](../../workspace/RESEARCH_AUDIT_STANDARD.md)
and AGENTS.md now require one current contract and explicit version migration.
This gate covers the whole workflow, not only the model catalog. New consumers
must reject unconverted legacy inputs; old pinned workers finish unchanged.
Migration, downstream replay and release evidence are required before switching.
This documentation update does not certify remaining production migration or
upgrade MHD V4. See the standard for the mandatory review checklist.


## 2026-09-15 project-first resource policy

The user superseded the generic-model reservation: project work and required
parent models have priority; generic models get only unused capacity. The shared
GPU standard now also requires independent configurations, controls and seeds to
be separately claimable for parallel execution. Real scientific dependencies remain.
MHD_Models prepares a CPU demand publisher for the existing budget contract; it
does not replace this project's running source or automatically split old cases.
Live policy deployment and per-arm parallel execution are separate acceptance gates.


## 2026-09-15 revised allocation targets: 10 / 10 / 4

This supersedes the no-generic-floor policy earlier today: preserve four generic
model slots, cap LOOK and Radon_Bridge targets at ten each, and lend every currently
unused project slot to generic models. Targets govern new admission, not forced
termination of healthy existing work. Explicit publisher configuration and live
consumption must be verified. Independent project-arm parallelism remains a
separate task-DAG integration gate, not automatically solved by quota changes.


## 2026-09-15 resource-lease continuity standard

The shared model/run and GPU execution standards are version 5 and AGENTS.md
requires their continuity gates for training, PCA/SVD, correction and evaluation.
See [the audit](../allocation_continuity.md) for checked production boundaries and
remaining acceptance. Actual new allocation policy stays 48h. Existing healthy
scientific snapshots remain unchanged. MHD_Models centralizes new submissions in
the guarded planned pool; short unguarded legacy writers are retired. Prepared
segment-admission changes do not certify all production recovery paths.

Isolated Ibex management acceptance: 272 tests passed in
`lease_continuity_20260915/validation2/`; the first failed fixture run is preserved.
Shared document hashes match across all three repositories. This source is for
new validated owners; original worker bindings and runtime states were not changed.

## 2026-09-15 afternoon resource handoff

The latest user instruction supersedes the four-card generic minimum: retain two
generic GPUs and distribute other capacity dynamically to ready project work. The
sole deployed publisher is `project_demand_floor2_20260915`; the old publisher is
retired, with GPU workers and allocation owners preserved. Its 15:46 Saudi snapshot
reported LOOK 7, Radon_Bridge 10 and other work 7 (running plus pending), with demand
targets 11/11/2. These counts require a fresh Slurm/role check before action.

The policy implementation passed 13 targeted tests on Ibex, including the two-card
floor, borrowing, reduced limits and uncertain ownership. Healthy project work and
registered project prerequisite models are distinct from generic exploration.
Extra generic workers may yield at a verified checkpoint after a project preflight;
quota publication alone does not transfer an existing allocation. The staged
legacy-to-project handover must verify process identity, source-specific recovery,
old step death, shared claims and actual downstream execution. Never cancel the
allocation owner to rebalance. All new requests remain 48 hours.

The terminal-stage control overlay is deployed from `78b8962`, with 26 targeted
tests and a real GPU resource/reload preflight. `look_active_workflow.json` binds
`look_terminal_20260915/control/dispatcher.json` and `registry_v2.json`. Three
accepted 3416 hosts have independently runnable terminal tasks. This is the approved
single-final correction pulled forward, with equal-q linear comparisons; it is not
full progressive LOOK or spatial-route acceptance. Original PCA workers continue.
At 15:46 the new dispatcher was waiting for account capacity; check the subsequent
`borrow_51898877` handover receipts before claiming formal execution or completion.

### Existing allocation handover accepted at 15:59 Saudi

Allocation `51898877` remains RUNNING. Its generic ConvNeXt-L worker saved the
original run at epoch18 / offset41152 / update16164; the complete checkpoint loaded
with model, optimizer, scheduler, RNG and Node IDs, and the shared claim is paused.
This review did not rerun a next-update equivalence test. The original scientific
source, configuration and run ID remain unchanged and eligible for later recovery.

An independent terminal-stage GPU probe passed while the native worker remained
healthy. Only then did the native worker checkpoint and exit. The old family manager
is held under a supervised, identity-checked one-time handover; its allocation and
original step remain present. The new LOOK owner runs in that same allocation.
The wrapper resumes the old family manager only after its project child exits.
Do not independently resume the old manager while LOOK is live.

LOOK step13 is now executing `terminal/cataract/resnet50/middle/3416`, run
`2026_09_15_15_32_32_778633`. Real train feature-cache shards advanced from153 to408
in independent reads. This is actual method execution, not accepted performance.
The terminal owner uses the shared claims and existing finite feed; no replacement
GPU request was submitted. The demand publisher now counts LOOK8 / Radon_Bridge10
/ generic6, including pending jobs; recheck before future decisions.

Evidence under `OPS/look_terminal_20260915/borrow_51898877`: probe receipt, original
process identity, `checkpoint_accepted.json`, `borrow_v2.py`, borrowed request journal,
project owner logs and handover status. `OPS/heartbeat_20260915_1545/handover_review.json`
records independent progress. V1 waited for a nonexistent final companion status;
V2 checks actual Slurm step death instead, preserving the recorded `stopping` status.
No scientific worker was killed or acceptance criterion relaxed.


## 2026-09-16：独立起点已部署并开始正式拟合

科学源f3c8a475d3348a818ae637e35033808d1b0a535f，[CI35029091261](https://github.com/souray0410/LOOK/actions/runs/35029091261)成功，完整CPU和双rank检查通过。Ibex225项CPU通过；Git archive缺submodule元数据的2项布局失败已补齐锁定Git证据后复测通过。真实GPU两方向拟合、冻结宿主、保存重载及主动中断后恢复通过，8条原已提交节点决策SHA保持不变。

OPS/look_suffix_20260916保存独立源码、registry.json、runtime_environment.json、deployment.json和验证证据；生产数据根为DATA/LOOK/2026_09_10_11_11_31/suffix_starts_20260916_v1。当前3个3416宿主已准备21条新增后缀，首尾复用原完整案例。正式首条deep/start8已在51898670.13执行，须继续核对run/status、Slurm、worker.log与接受文件，不把资源探针当正式结果。现阶段完整后缀接受数仍为0。

唯一CPU dispatcher切换至OPS/look_suffix_20260916/control/dispatcher.json，旧CPU session由guardian保留，所有旧GPU科学worker不改。look_active_workflow.json、需求publisher及三小时监控已更新。原通用ConvNeXt-B按同run保存完整checkpoint后paused回候选池；borrow_51898670监督仅在新项目owner退出后恢复旧family，严禁手工提前SIGCONT。第二条borrow_51896678为有限共存资格验证→断点验收→接替链，通过前保留旧模型，基础模型最低2卡约束不变。第二条不是已完成交接。

每宿主全部9起点完成后自动生成两种缺失方向完整表、10000次配对区间与dev起点选择，再解锁对应3417/3418；无正收益门槛。本周优先完整3416匹配结论，所有批准范围保留。原15主案例、其他机制/空间/线性范围继续按依赖滚动，不把基础模型数替代项目完成数。

关联自审还发现R&B3D Swin预检在50GiB allocator上限下OOM；失败step已死、2个相同资源候选隔离，保留完整失败证据后移除全局新派发hold，其他合格任务恢复准入。这不表示Swin资源问题已解决，不修改batch/精度，也不把资源失败解释为性能差。证据OPS/look_suffix_20260916/rb_resource_incident。跨项目需求publisher已恢复健康、指向新LOOK控制器。

实现源码位于[已部署提交f3c8a47](https://github.com/souray0410/LOOK/tree/f3c8a475d3348a818ae637e35033808d1b0a535f)，本main提交仅同步协议与交接记录，不替换正在运行的源码。参见[完整起点协议](../suffix_starts_20260916.zh-CN.md)。


## 2026-09-16 03:50 Saudi：第二条起点执行与共享心脏数据验收

第二条有限接替已通过独立GPU验证及原模型完整断点验收：51896678.21正式运行同一3416/deep宿主的start7；51898670.13继续start8。两条worker均有新鲜完整dev推理进度，尚无完整后缀或九起点组接受。阶段status超过两小时不等于进程卡死：本次pipeline审计提示后，以实际Slurm step和持续更新日志核查，没有重启健康worker。证据OPS/heartbeat_20260916_0345/live_suffix_liveness.json。

实际pipeline_audit配置此前未收到只写入旧execution_audit配置的起点、周日完整匹配和Swin资源复检门槛。本次将三项追加到真正执行入口，保留原全部门槛及未完成状态；这是审计覆盖修正，不是研究验收通过。

独立数据迁移范围的心脏五组（20205、6025.zip、20207、20208、20209）现共361888文件、8360335258048字节全部源/目标SHA验收。源盘保留，临时rsync授权已精确撤销，其他SSH授权不变。DATA/UKBiobank/multiorgan/manifests/transfer_completed_20260916.json为完成凭证。此状态仅为文件复制完整，标签、访视、配对与任务语义仍待审计；不自动开启其他器官项目实验，也不改变眼科数据划分或test封存。


### 2026-09-16 06:55 Saudi：审计配置状态修复

上次追加门槛把deployed_running/running_not_complete/needs_review写进了冻结审计器仅接受的门槛state，导致audit_input_error。本次保留原运行描述到execution_status，将门槛分别写为not_validated/waiting_dependencies/not_validated；pipeline与旧配置同步，全部门槛保留且没有改为接受。真实入口重跑确认输入错误消失。两条起点阶段status较旧，但Slurm步骤及持续更新日志均核实在推进，不重启健康任务。证据OPS/heartbeat_20260916_0645/{audit_gate_schema_repair,audit_after_repair,review}.json。现有实施缺口和Swin三维资源复检仍未完成，不据此宣布全研究健康或完成。


### 2026-09-16 08:59 Saudi：仿射家族管理层已接通

科学快照`d32380197fd0faf9e1f91e6e3f8ad2cd3323c6a9`，[CI35061029793](https://github.com/souray0410/LOOK/actions/runs/35061029793)成功。Ibex完整CPU运行236项通过，另2项因Git archive不含submodule实体失败；补齐锁定V4检出后，部署/结构与本次相关25项全部通过。另有初步30项算子/旧路径/恢复测试。真实A100预检完成五新臂、两缺失方向、冻结宿主、完整MHD重载和RNG恢复；GPU峰值1,765,801,984 bytes、主机峰值2,678,476,800 bytes。预检使用1024训练参与者与32训练探针，不计正式性能。

`OPS/look_affine_20260916/registry_v2.json`和独立源码`source_d323801`已部署；数据输出`DATA/LOOK/2026_09_10_11_11_31/affine_family_20260916_v2`。初始`v1`仅预登记、从未派发，因最终资源边界补丁另建v2，不改写其原身份。3个白内障ResNet50/3416末级host已登记，正式接受0；逐层等待原完整三臂接受，其他种子等待对应3416家族完整技术接受。

唯一CPU dispatcher现为`OPS/look_affine_20260916/control/dispatcher.json`，PID1686933；registry PID1671215。旧CPU243090在确认无子owner后通过guardian退休；原51898670.13、51896678.21及其他GPU worker持续原快照。管理层handover初次记录缺previous字段导致KeyError，已从guardian验收恢复规范schema，并重新观察到error=null、72项就绪、0接口拒绝、waiting_account_capacity。不是修改训练标准排除错误。共享claims、账户锁、48h和最低2张generic保持不变。

look_active_workflow.json、需求publisher及独立audit门槛已更新。新实验确已进现有调度器，当前24张运行/待批额度占满，因此尚无新family正式worker；不能将预检或72项总就绪解释为72项新实验完成。当前三条新case排在同优先级已就绪补充后，未抢停健康项目；维护时应检查完整3416组交付与队列等待，不能只看进程存活。

下一步：按新feed逐case完成预检与正式拟合/重放，汇总51项配对比较，再自动释放同组重复种子；3种子齐全触发独立CPU报告。统计族内区间不替代全研究全局区间。原起点/空间/扩展研究及Radon未完成门槛继续保留，test仍封存。

## 2026-09-16: Scientific review obligations

Added the identical SCIENTIFIC_REVIEW_STANDARD.md to LOOK, Radon_Bridge and
MHD_Models and linked it from AGENTS and RESEARCH_AUDIT_STANDARD. This documents
mandatory comparison/algorithm/evidence checks; no runtime checker, training
snapshot, scientific result, GPU owner or test access changes in this update.

LOOK open incident: forced candidate comparisons were described too broadly.
See [correction selection audit](../correction_selection_audit_20260916.zh-CN.md).
Branch commit 680ec38 supplies additive gate review code; its 11 targeted tests
passed on Ibex. Production gate-review deployment is not accepted here. Alternative
methods still require independent per-node greedy trajectories and real MHD
acceptance; whole-bank fallback does not close this gate. Original PCA greedy
implementation is distinct and must not be described as absent.

## 2026-09-16：统一阶段引用与周报衔接

workspace/stage_registry.py与Radon_Bridge保持逐字相同：阶段引用精确科学身份，复用/恢复调用所属项目验收器，不复制重训、不自动解封test或抢任务。原LOOK算法、生产worker和既定实验范围不变。本条只同步已在准备分支验证过的共享管理工具到main，不宣称LOOK全部研究已完成。

本周KAUST已建立2026-09-20准备目录；周六证据快照，周日06:45英文PPTX/中文讲稿、09:45刷新、09:55交付10:00汇报。既有ukb自动化继续按独立LOOK协议验收；病例/指标/逐层策略不得套用R&B分支平均。年底前投稿目标保持。

## Current status publication

Machine-readable current evidence is [status.json](status.json). Source review and live runtime verification are distinct; this publication does not change scientific jobs or certify unfinished experiments. Update this record after material evidence review.

Private research overview and weekly archive: [PHD](https://github.com/souray0410/PHD). Adopted hub standards: research-standards-v1, 2026-09-16. Existing pinned study/runtime rules remain authoritative for current executions.

## 连贯性规范采用（2026-09-17）

采用PHD `62fd7e08770f3628a048fd59b82da2b9e545afab`，进入[固定累计结果入口](../reports/current/README.md)和[尝试档案](../archive/README.md)。完整规则见[共同契约](../../workspace/RESULT_CONTINUITY_STANDARD.md)。此次仅更新结果管理入口；未刷新运行事实、未迁移全量旧产物、未删除服务器文件，自动累计发布仍须独立验收。

## 2026-09-17：耗时与文献核查

见[重复扫描、3D配方和文献可比性审计](throughput_literature_audit_20260917.md)。发现LOOK重复特征扫描及3D microbatch1/BN配方待审；本次仅核查，不宣称已优化部署。

## 2026-09-17: shared LOOK extraction acceptance

The [shared extraction audit](look_efficiency_20260917.md) records exact moments,
solutions, logits and selection tests, and an explicit source-only evidence converter.
Small real-host extraction probes improved by about 6.9–8.3 times; this is not an
end-to-end speed claim. Production handover remains separately gated and must be
verified through the live execution receipt and downstream progress.

## 2026-09-17 15:54 Saudi: finite current-search continuation

The [shared-search handoff](look_efficiency_20260917.md) now records a pinned,
single-run continuation entrypoint for existing allocation owners, verified
live/lock refusal, clean lease recovery attempt tests, and a detached cumulative
publication observer. Formal scientific source is unchanged and shared shards
advanced 776→916. Actual next-lease execution and the full two-state delivery
remain pending; this does not certify the weekly matched package.

## 2026-09-17 19:56 UTC：首个best-forward双缺失配置完整交付

已验收[当前结果、完整路径及反例](../reports/current/best_forward/README.md)：白内障/ResNet50/deep/3416/x16/q32，缺OCT与缺CFP macro-F1分别+4.1330/+3.8453pp，缺CFP排序及校准指标下降。远端自动重放、统计、图和publication后，本轮独立核验247+5文件SHA及四份预测指标与顺序；这是保留的best-forward对照，不是当前主树或整周包完成。主树缺CFP仍继续，所有test及重复种子门槛不变。
