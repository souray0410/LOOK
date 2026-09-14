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
