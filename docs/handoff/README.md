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

## 2026-09-14：9月20日阶段交付优先级

见[本周执行安排](../weekly_delivery_20260920.zh-CN.md)。眼科两项目完整参考匹配组优先于继续增加独立模型配置。现场24张GPU运行，但两项目方法接受数仍为0；LOOK旧父筛选50/116，RB旧6/14、完整父准备25/73，分母分开。未来准入策略已在共享锁内改为LOOK6/RB6/model最多12；现有2/2/20 allocation不强停，预留不等于实际已分配。OPS/weekly_delivery_20260920保存原策略和变更验收。三维父模型及完整case/资源/执行接受仍为阻塞；周中检查不以缩短训练或2D替代3D赶交付。心脏5组数据迁移，其他器官及颈动脉暂缓；test继续封存。本文档未上线新的科学worker。

## 2026-09-14：学期投稿硬截止与分阶段授权

用户明确2026-12-31前正式投稿，应尽早；允许首篇完整研究包与729六网络等长期矩阵分阶段，后续实验保留。见[学期交付安排](../semester_delivery_2026.zh-CN.md)。LOOK首篇优先、RB并行阶段推进；11月30日为LOOK内部争取目标，不是未经测量的完成保证。分阶段test入口尚未实现/验收，当前test保持封存。不能把已查看test在后续调参后再次当作未见证据。

## 2026-09-14：阶段扩展复用管理层

用户明确阶段必须可接续、扩展、复用，不能换阶段就重复训练。新增共用workspace/stage_registry.py和STAGED_RESEARCH_STANDARD.md；科学身份与stage分离、文件SHA、原run复用/恢复、不可变阶段与前驱摘要、跨进程锁、失败不自动重训。它仅做管理规划，必须由项目真实接受器/存活检查完成verify_live，未部署全部生产适配器，也不解封test；具体集成和真实模型恢复继续验收。

## 2026-09-14：长期自动纠错责任

用户明确自动纠错必须成为通用长期规则，不能只适用于本次故障。AGENTS及共享RESEARCH_AUDIT_STANDARD已规定：已授权流程必须从发现推进到诊断、版本化修复、验收、安全恢复和真实进度复查；不能停在报警、CI通过或调度器重启。未关闭问题保留证据、责任主体、下一步及可执行自动续接。保护健康任务和原科学标准，限制瞬时重试，不盲重启确定性错误。规范适用于未来模型、数据、框架、项目与CI/资源/评价/报告链路，但不是保证所有故障都能自动解决。规则更新本身不表示任何当前训练或故障已完成验收。


## 2026-09-14：原生预检接口与短任务失败防护

将Radon已定位的相邻调度风险补入LOOK：新增申请前按原生科学源码与框架SHA检查预检接口；不兼容候选记录拒绝原因；短worker退出后重新读取step记录并有限等待终态；失败保留原run并阻止重复申卡。子step移除继承的owner CPU/内存请求字段，保留实际工作线程与科学配置。

Ibex独立目录 `OPS/look_admission_validation_20260914/acceptance.xml` 7项针对测试通过；首轮上传缺少包依赖导致的collection失败保留为attempt1记录，补齐原包源码后通过。结构检查通过。此记录仅为控制代码准备和CPU测试，不是已接替生产或方法实验完成。新控制器必须使用经过SHA锁定的兼容profile实现并完成实际候选API清单验收；原LOOK调度器有活跃salloc子进程，不能直接杀会话。生产切换须独立验证待机入口和会话保护，现有健康GPU owner继续旧源。


## 2026-09-14 15:56：LOOK控制补丁已安全接替

控制提交 `e25910c` 的 [CI 34845530283](https://github.com/souray0410/LOOK/actions/runs/34845530283) 已成功；Ibex七项针对测试、124/124父任务API清单及合成会话保护测试通过。线上使用旧 `look_execution_09be3be` 源副本，仅替换 `runtime/project_dispatch.py` 为该提交版本，并绑定已核验的 `models_023d471` 预检脚本；其他科学模块逐文件保持原SHA。

实际入口改为 `OPS/look_admission_20260914/dispatcher.json`，证据在同目录及 `look_admission_validation_20260914`。新CPU dispatcher当次PID1678754已实际运行、无错误、等待账户容量；64个当时可领取任务API拒绝数0。旧CPU dispatcher2931336由独立会话守护保持，原salloc子进程807373及其健康GPU父训练持续更新到第7轮。不得杀旧会话或重复启动管理器；旧子进程全部自然结束后守护才退休旧CPU。新journal加入原角色政策，复制原三条请求身份，没有取消或重复提交allocation。

此为控制路径安全接替，新的正式GPU worker仍须逐任务完整资源及恢复验收，不是LOOK方法研究完成。Radon独立长GPU恢复探针仍在进行，不能重复启动或提前解除其派发保护。下一维护继续推进首个项目参考匹配组执行依赖与实际验收，不以该控制修复代替科学链路实现。

## 2026-09-14：Ibex三路线与本周实际结果优先

用户批准[三路线协议](../look_spatial_20260914.zh-CN.md)：插值/PCA、直接PCA、平均池化/PCA。81个宿主单元依赖原已训练宿主；每组3416三路线完整技术/重载/配对报告验收后，自动展开3417/3418，完全不按分数筛掉方法。新spatial feed复用既有dispatcher和原子领取，不新增GPU竞争管理器，test封存。

本周优先形成项目完整参考匹配组，而非等待全部大矩阵。2026-09-14 20:00附近现场白内障ResNet50父模型三种子齐全，9项宿主已注册、方法接受仍0。接替预检漏传CUBLAS确定性环境导致报错，独立重跑通过25次更新及完整恢复，随后按原优先级接口请求当前基础模型保存断点；未取消allocation。证据OPS/look_priority_repair_20260914，后续须核查实际项目训练进度。

三路线新代码在Ibex独立准备目录测试；旧健康科学源码不热改。原分辨率PCA尚无真实大队列资源接受，不能称已获得三路线效果。大矩阵、机制、其他疾病及架构保留后续阶段，逐组自动汇总，不能以新增代码数量代替本周结果交付。

## 2026-09-14晚：空间比较实现与阶段推进

新分支2026_09_14_19_56_49；科学快照7d141fd，CI34873729822成功，Ibex完整191项测试通过，另补三种路线实际MHD节点写回测试后相关13项通过。测试通过不是空间比较GPU整组接受；后者等待已冻结且验收的原宿主，并需384GiB worker内存/512GiB allocation资源类的实际预检。

空间注册器look_spatial_registry_20260914已运行，按81宿主/243路线视图登记；3416同宿主三路线全部技术接受和报告后才自动释出3417/3418，不设性能门槛。控制绑定见OPS/look_active_workflow.json。原宿主训练、来源和allocation均保留，新增控制快照只服务后续worker。旧CPU dispatcher拥有salloc子进程，使用session guardian保持其会话；不得直接杀会话。

首个cataract/resnet50/3416/middle宿主已实际更新；空间拟合尚无接受结果。以[本周阶段安排](../weekly_delivery_20260920.zh-CN.md)中的完整小组验收作为交付，不把基础模型或控制代码完成数充当项目结论。

## 2026-09-14：两项目研究链衔接

共享RESEARCH_AUDIT_STANDARD新增关联研究规则：补充必须映射到既有问题、匹配对照、改变因素、先前证据、可复用产物及下一阶段验收，周报优先完整匹配小组。Radon的[跨队列研究链](https://github.com/souray0410/Radon_Bridge/blob/2026_09_14_19_43_11/docs/research_chain_20260914.zh-CN.md)区分几何、参数化与容量；LOOK继续独立比较三种空间处理/PCA路径，不引入可学习分解或额外反传，不混用最终分类与分支平均指标。共享范式不改变正在执行的协议。


## 2026-09-14 21:25：两项目全链路审查

见[完整范围与执行差距审查](../project_audit_20260914.zh-CN.md)。该文按研究包区分已实现、已接线、真实验收、运行和科学接受；LOOK已出现4个持续更新宿主，但方法完整case接受0；Radon完整父准备和新多来源/分解生产接入仍有缺口。MHD_Models新增只读pipeline_coverage审计覆盖全部活跃项目/补充feed及registry，独立Ibex227测试通过；不把状态文件和合成GPU通过当作真实研究完成。部署证据及未关闭事项由现有UKB维护继续处理；所有健康科学worker保留。

## 2026-09-14 model-format transition

The shared model standard is now v4; see [migration](../model_migration.md).
New releases use one canonical artifact format. Current immutable study snapshots
finish unchanged. MHD_Models owns one-time conversion; no WS02/Ibex or V4/V5
fallback is to be added to the new project runtime. Parent replay and deployment
remain explicit acceptance gates; this documentation does not claim they passed.

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

## 2026-09-15：仿射向量修正与统一秩预算补充

用户确认允许独立线性替代对照，强调各方法使用合适的保留量描述、用共同维数预算匹配，而不是预设最佳实现。新准备入口见[协议与数学定义](../look_linear_vectors.zh-CN.md)。已加入共享PCA岭回归、保持同截距的残差低秩回归、自由截距低秩回归；原空间处理与旧科学代码不变。`linear_registry`及现有dispatcher的可选`linear_feeds`提供依赖/预检/复现入口。新增秩预算工具拒绝将截断谱自归一化成100%，能量阈值仅描述、不事后选型。

本条不是上线或科学结果接受声明：需按精确提交完成CI和Ibex完整资源/数据重放验收；生产绑定未切换。密集求解的高维可行性、PCA特征复用提速、PLS与独立配置选型不是已完成事项。健康旧任务继续旧版本。


本补充实现提交`6b0626e`已推送至`2026_09_14_21_37_57`。Ibex完整Git检出的针对性检查77项通过，含原子恢复、MHD节点/冻结状态、秩与能量语义及调度读取；这里是CPU数值/接口测试，非真实队列GPU验收。对应CI为34965862964（状态需刷新）。准备清单位于OPS/look_linear_20260915/registry_prepared.json，首次登记81宿主、243臂视图，均等待原完整LOOK接受；base feed当前9宿主任务、0完整接受。没有切换生产dispatcher、启用新GPU申请或读取test。下一步由该受验来源完成真实资源预检和安全控制器衔接，不能把登记清单当作已上线自动运行。后续小改将推理bank大小与完整拟合存储分别记账。

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

## 2026-09-15：末级线性对照名称纠正

展示层统一采用PCA约束残差岭回归、低秩残差回归（PCA约束均值）、低秩残差回归（自由均值）。后两组并非直接完整特征与残差目标之比；同配置斜率相同，均值约束不同。见[解释与覆盖边界](../linear_maps_explained.zh-CN.md)。本次不改方法ID、拟合、预测或运行快照。

## 2026-09-15：自由均值候选与持续汇总准备

用户偏向自由均值RRR，但保留三臂匹配研究，不按当前排名删减。新增只读terminal_rollup观察器，独立输出并保留原模型/研究身份；其核验范围为报告/spec摘要和原接受条件，不替代原注册器完整预测/缓存验收。当前live核查4/81末级接受，77等待宿主，两个3418宿主仍在更新；原完整逐层与扩展包不能据此宣称全自动完成。本地缺pytest，结构检查通过；目标环境测试与部署另行记录。

### 持续汇总已在Ibex上线

`ce6c32c`的4项报告完整性/负结果/重复清单/test拒绝/三种子门槛测试在Ibex通过；结构检查通过。真实首轮汇总4个接受案例116行，report/spec核验无异常，完整三种子组为0；不重新读参与者数组或训练缓存。独立源码`OPS/look_terminal_20260915/rollup_ce6c32c`，绑定`rollup_binding.json`，输出`continuous_report`；观察器实际存活并写出healthy状态。原GPU训练与registry快照未修改。三小时UKB监控已加入观察器状态核对和安全单实例恢复；无法解释的SHA矛盾需隔离调查，不自动改证据。终端四案例完成后的原借卡owner已恢复通用模型，不得依据旧15:59条目再次迁移。精确CI状态须刷新，Ibex针对性通过不替代CI。

## 2026-09-15 21:45例行核查与谱诊断准备

LOOK253d2a9与ce6c32c、RB135bb68、MHD_Models3420288的精确CI均成功。live audit仍16项已知缺口、fingerprint a8c2e050...未变。22GPU运行/2待批；末级4/81、observer healthy。两个3418宿主epoch17训练/12验证新鲜；两个3417到期暂停且共享claim paused，未被误记为failed，不盲抢。20209源SHA81001/89303文件且临时hash文件新鲜，4/5数据组验收，不报全完成。巡检记录OPS/heartbeat_20260915_2145/review.json。

新增analysis.affine_spectrum紧凑谱诊断及测试，Ibex独立目录look_terminal_20260915/affine_spectrum_validation_2145的9项测试通过。只完成代码/数值接受，真实高维数据计算尚未派发；不替换terminal_78b8962、ce6c32c报告observer或其他健康科学快照。具体边界见linear_maps_explained.zh-CN.md，后续独立只读资源准入后再接真实谱结果。

## 2026-09-16：单种子贯通的父模型登记修正

核查发现末级/空间/线性对照registry已设置3416技术接受后放行重复种子，但project_orders仍等待两模态三种子全部接受才登记宿主，造成不必要的前置等待。现改为锁定配方后，按同一种子的两路已验收父模型交集登记；3416优先，未锁定配方或单边未接受仍拒绝。不改已有spec、run ID、训练源码、数据角色或三种子最终统计门槛。Ibex隔离3项测试通过，覆盖原9任务恒等、部分配对、后续重复、重复执行和接受失败。该变更不等于完整逐层LOOK已贯通；高维拟合/空间资源验收仍须分别处理。部署证据另记，健康运行重复种子不强停。

单种子登记修复`5628f93`精确CI35024412428成功，Ibex14项关联测试通过。CPU控制器已由1071876安全切换为独立`OPS/pilot_release_20260916/control_src`（仅project_orders.py与原控制器源码不同），原配置/source_pins和GPU workers保持不变；live binding增加controller_source与deployment_receipt。首次对账正在重新核验父模型best.pt，登记仍9项，尚未宣称新增宿主已开跑。复核deployment.json及原projects/queue.json，需证明原9项身份不变、新的同种子配对就绪项登记并被原dispatcher读取。完整逐层3416 PCA现场21/23、22/23、20/21，仍不是完整方法验收。末级种子门槛已上线；其他扩展链路保留各自真实验收限制。

## 2026-09-16：恢复独立校正起点

见[后缀起点协议](../suffix_starts_20260916.zh-CN.md)。新增suffix_protocol/case/registry/report及原dispatcher feed；原81宿主×7中间起点=567拟合任务，首尾复用原LOOK/single_final。逐宿主3416全部起点技术完成后才放重复种子，不按性能筛范围。当前补丁通过Ibex隔离CPU回归，正式GPU资源验收与上线凭远端deployment.json确认；此文档不声称已得到全部实验结果。


## 2026-09-16：独立起点已部署并开始正式拟合

科学源f3c8a475d3348a818ae637e35033808d1b0a535f，[CI35029091261](https://github.com/souray0410/LOOK/actions/runs/35029091261)成功，完整CPU和双rank检查通过。Ibex225项CPU通过；Git archive缺submodule元数据的2项布局失败已补齐锁定Git证据后复测通过。真实GPU两方向拟合、冻结宿主、保存重载及主动中断后恢复通过，8条原已提交节点决策SHA保持不变。

OPS/look_suffix_20260916保存独立源码、registry.json、runtime_environment.json、deployment.json和验证证据；生产数据根为DATA/LOOK/2026_09_10_11_11_31/suffix_starts_20260916_v1。当前3个3416宿主已准备21条新增后缀，首尾复用原完整案例。正式首条deep/start8已在51898670.13执行，须继续核对run/status、Slurm、worker.log与接受文件，不把资源探针当正式结果。现阶段完整后缀接受数仍为0。

唯一CPU dispatcher切换至OPS/look_suffix_20260916/control/dispatcher.json，旧CPU session由guardian保留，所有旧GPU科学worker不改。look_active_workflow.json、需求publisher及三小时监控已更新。原通用ConvNeXt-B按同run保存完整checkpoint后paused回候选池；borrow_51898670监督仅在新项目owner退出后恢复旧family，严禁手工提前SIGCONT。第二条borrow_51896678为有限共存资格验证→断点验收→接替链，通过前保留旧模型，基础模型最低2卡约束不变。第二条不是已完成交接。

每宿主全部9起点完成后自动生成两种缺失方向完整表、10000次配对区间与dev起点选择，再解锁对应3417/3418；无正收益门槛。本周优先完整3416匹配结论，所有批准范围保留。原15主案例、其他机制/空间/线性范围继续按依赖滚动，不把基础模型数替代项目完成数。

关联自审还发现R&B3D Swin预检在50GiB allocator上限下OOM；失败step已死、2个相同资源候选隔离，保留完整失败证据后移除全局新派发hold，其他合格任务恢复准入。这不表示Swin资源问题已解决，不修改batch/精度，也不把资源失败解释为性能差。证据OPS/look_suffix_20260916/rb_resource_incident。跨项目需求publisher已恢复健康、指向新LOOK控制器。


## 2026-09-16：统一仿射修正家族补充

用户批准将LOOK作为一般线性/仿射修正家族研究，新增[关联协议](../look_affine_family_20260916.zh-CN.md)。现有三臂不改；新增PCA自由均值、无秩约束残差岭回归、PLS-SVD方向岭回归、对角岭回归和正交对齐。前四臂形成均值×子空间2×2因素设计，51项预定每宿主scope比较；不声称容量相同或各方法独立最优。全部81宿主的末级/逐层分别登记，共162新case、810新方法视图、神经训练0，3416完整技术接受后放其他种子。

新增独立family schema、拟合/恢复、报告和registry/feed接口；原健康运行快照不变。Ibex隔离30项初步数值/恢复及旧路径回归已通过；实际GPU全路径、CI、部署与正式结果尚须远端证据验收，不以本条视为完成。与空间处理及起点实验通过同宿主引用关联，不盲目扩成全部组合。新增PLS明确是PLS-SVD方向后接岭回归，不是假称迭代PLSRegression。新的家族估计器尚无性能结论。


### 2026-09-16 08:59 Saudi：仿射家族管理层已接通

科学快照`d32380197fd0faf9e1f91e6e3f8ad2cd3323c6a9`，[CI35061029793](https://github.com/souray0410/LOOK/actions/runs/35061029793)成功。Ibex完整CPU运行236项通过，另2项因Git archive不含submodule实体失败；补齐锁定V4检出后，部署/结构与本次相关25项全部通过。另有初步30项算子/旧路径/恢复测试。真实A100预检完成五新臂、两缺失方向、冻结宿主、完整MHD重载和RNG恢复；GPU峰值1,765,801,984 bytes、主机峰值2,678,476,800 bytes。预检使用1024训练参与者与32训练探针，不计正式性能。

`OPS/look_affine_20260916/registry_v2.json`和独立源码`source_d323801`已部署；数据输出`DATA/LOOK/2026_09_10_11_11_31/affine_family_20260916_v2`。初始`v1`仅预登记、从未派发，因最终资源边界补丁另建v2，不改写其原身份。3个白内障ResNet50/3416末级host已登记，正式接受0；逐层等待原完整三臂接受，其他种子等待对应3416家族完整技术接受。

唯一CPU dispatcher现为`OPS/look_affine_20260916/control/dispatcher.json`，PID1686933；registry PID1671215。旧CPU243090在确认无子owner后通过guardian退休；原51898670.13、51896678.21及其他GPU worker持续原快照。管理层handover初次记录缺previous字段导致KeyError，已从guardian验收恢复规范schema，并重新观察到error=null、72项就绪、0接口拒绝、waiting_account_capacity。不是修改训练标准排除错误。共享claims、账户锁、48h和最低2张generic保持不变。

look_active_workflow.json、需求publisher及独立audit门槛已更新。新实验确已进现有调度器，当前24张运行/待批额度占满，因此尚无新family正式worker；不能将预检或72项总就绪解释为72项新实验完成。当前三条新case排在同优先级已就绪补充后，未抢停健康项目；维护时应检查完整3416组交付与队列等待，不能只看进程存活。

下一步：按新feed逐case完成预检与正式拟合/重放，汇总51项配对比较，再自动释放同组重复种子；3种子齐全触发独立CPU报告。统计族内区间不替代全研究全局区间。原起点/空间/扩展研究及Radon未完成门槛继续保留，test仍封存。
