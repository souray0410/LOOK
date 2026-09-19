# LOOK 下一最小缺失方法 A 合同：ShaSpec 2D 分类适配（训练前门槛版）

状态：contract_ready_provenance_and_adapter_gate_pending。本文件只锁定下一有限科研包，不在当前 EmbraceNet 集成包中训练第二个 A。

## 为什么只选这一个

老师的问题仍是：一个已经处理整模态缺失的策略 A，在同一冻结 A 上再加 LOOK，是否还有增量价值。

当前一手证据盘点中，ShaSpec 最适合作为下一机制家族候选：
- 论文明确包含 missing-modality classification，而不只是完整输入融合；
- 核心机制是 shared/specific representation + missing-feature generation，与 EmbraceNet 的逐坐标 categorical embracement 明显不同；
- 因而它能新增“LOOK 在生成/重建式缺失策略之后是否仍有剩余线性可校正结构”这一信息。

这不是因为预期 ShaSpec 分数更高。PCA free-mean 与 residual RRR 在下一包中继续作为并列预注册 LOOK 家族，不预设赢家。

## 训练前硬门槛

固定作者证据：paper + author repository commit 57a04fa88331ea3b2671795a15a1ae9a24fd310d；现有快照中 DualNet_SS.py、train_SS.py、README 已哈希固定。

但当前作者仓库暴露的是 BraTS 3D segmentation 实现；论文中的 classification recipe/code 未在快照中找到，且仓库许可未明确。因此在真实训练前必须同时满足：

1. 实现来源与许可：确认可合法复用的作者代码/许可路径；不得把“公开可见”自动当成可复制许可。
2. 任务适配身份：明确 2D CFP/OCT 二分类 adaptation 哪些部分是作者机制、哪些是本项目适配；若没有作者分类代码，必须标为 project adaptation，不能称 author-classification reproduction。
3. 核心机制保真：保留 shared/specific 分解、missing-feature generation 及对应必要损失；不能为了接 LOOK 删除使 ShaSpec 成为 ShaSpec 的机制。
4. 对称缺失协议：完整 / 缺 OCT / 缺 CFP 的 train policy 在执行前固定，不因 EmbraceNet 结果追分。
5. 合法 LOOK 节点：只能暴露不泄漏缺失真实输入/标签、且不改变 A 训练身份的内部表示。
6. 资源与恢复：正式 GPU 前完成 microbatch/profile、next-update resume、完整 dev replay、artifact storage 和远端离线接续门槛。

任一项未满足则本 A 不进入训练；这不是性能淘汰，而是科研/许可/实现身份未闭环。

## 最小单种子完整包

满足门槛后，有限包固定：

- 数据：沿用已审计 1264 train / 296 development；test 继续封存。
- seed：3416。
- A：只训练 1 个 ShaSpec 2D classification adaptation。
- 选优：只用 development，预先登记 A 自身主要指标/停止规则；不拿 A+LOOK 反向选 A。
- 冻结：A 接受后冻结 checkpoint、missing policy、generator/shared/specific 机制和 classifier。
- 主比较：同一冻结 A 下，两种缺失状态分别比较 A、A+PCA free-mean LOOK、A+residual RRR LOOK。
- 两 LOOK 家族使用同一合法节点集合、同一 complete reference、同一 positive-forward-tree 规则；不按 EmbraceNet 首种子结果预删任何家族。
- 指标：Macro-F1、AUROC、NLL、Brier；保留负结果与分支损伤。
- 统计：同一参与者的 10,000 次 paired bootstrap；普通区间、四主对比同族 simultaneous 区间、Holm。零方差恒等对比继续按“simultaneous 区间未定义”报告，不作等效性推断。
- 停止：一个 A + 两家族×两缺失的四棵完整树 + 统计 + 可读报告即完成本首种子包；不因点估计自动补调参/第二 seed。

## 与 EmbraceNet 包的辨识差异

EmbraceNet 结果说明：在 availability gating 下，输入节点 PCA 的增益来自校正仍可用分支，而不是生成缺失模态。ShaSpec 的下一包只有在 missing-feature generation 机制真实参与 A 时才有新增信息价值：它检验 LOOK 是否在“模型主动生成/重建缺失信息”之后仍能增加价值，而不是重复另一种 gating/fusion 宿主。

## 当前动作

当前集成包只发布此合同与准备远端离线执行要求，不启动 ShaSpec 训练。下一真实训练包由持续授权下的负责人在上述硬门槛全部有证据后再派发。
