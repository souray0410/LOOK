# LOOK — 维护与交接

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
