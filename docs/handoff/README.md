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

- [Model_Training交接](https://github.com/souray0410/Model_Training/blob/main/docs/handoff/README.md)：请求并验收独立完整父模型，记录架构/输入、标签与数据SHA、预处理、配方、种子、框架、检查点SHA及Node映射。未完成返回等待依赖，不静默换随机权重。
- [Radon_Bridge交接](https://github.com/souray0410/Radon_Bridge/blob/main/docs/handoff/README.md)：共享合格父模型与数据产物，不共享可变训练状态或混合评价定义。
- [MHD接口与版本](https://github.com/souray0410/MHD_Framework/blob/release/v4/docs/installation.md)：按应用锁安装准确提交，不跟随浮动main。项目拥有自己的宿主、修正产物和预测。

## 验收、资源与维护责任

GitHub负责独立安装和代码检查；Ibex是实际GPU完整链路验收环境。ws02仅可选调试，不是前置门槛。健康任务保留，新验证作为经资源准入的独立attempt。GPU完整峰值、CPU/主机内存、恢复、节点/BN/随机状态均需记录；不在登录节点跑GPU验证。所有新增GPU申请至少48小时。细则见[共享规范](../../workspace/GPU_EXECUTION_STANDARD.md)。

部署、协议变更、验收失败/修复、阶段完成和阻塞变化后维护本页及精确证据链接；保留更新时间，区分计划/代码通过/环境通过/部署/结果接受。重大历史变化由Git保留，不覆盖原实验。跨项目读取先刷新，不凭README修改他方任务。GitHub只放汇总和来源引用，不放参与者标识、原始数据、特征或受限预测。
