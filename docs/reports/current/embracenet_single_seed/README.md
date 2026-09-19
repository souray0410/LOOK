# EmbraceNet + LOOK：首个真实缺失方法 A 单种子候选结果

状态：**self_checked_pending_independent_review**。这是 2026-09-19 完成的 WS02 单种子 development 候选；左侧尚未对本真实训练结果做独立科学验收。**test 保持封存**。

## 这次真正回答的问题

老师的问题是：已有整模态缺失策略 A 已经训练好以后，LOOK 还能不能在**同一个冻结 A**上进一步改善缺失状态？

本包把 A 固定为 EmbraceNet。只训练一个新 A（seed 3416），然后冻结这一份 checkpoint；PCA free-mean 和 residual RRR free-mean 都只能在这同一份 A 上拟合。两种整模态缺失都完整保留：

- 缺 OCT：只剩 CFP；
- 缺 CFP：只剩 OCT。

因此这里不再是 MMTM/fusion-stage 的兼容性证据，而是第一份真正的 **A vs 同一 A + LOOK** 实际结果。

## 宿主与训练

- 数据：1,264 train / 296 development；test sealed。
- 宿主：两路 2D ResNet18 → valid-eye participant mean → EmbraceNet `c=256` → 二分类线性头。
- seed：3416。
- train 输入状态：complete / missing OCT / missing CFP 各 1/3，participant 级抽样，同模态双眼同步缺失。
- complete 选择概率：0.5 / 0.5。
- 优化：FP32，microbatch 16 / effective 128，AdamW；pretrained LR 1e-4，新 docking/head LR 1e-3，wd 1e-4，clip 5，warmup 5，minimum 8，patience 15，cap 100。
- complete checkpoint 选择：解析 `E[logits] = W E[z] + b` 的 development Macro-F1。
- best epoch = **11**，stop epoch = **26**；宿主训练计时约 **194.26 s**。
- complete-dev：Macro-F1 **67.773%**，AUROC **72.270%**，NLL **0.64716**，Brier **0.44201**。

## 两种缺失的实际结果

| 方法 | 缺失状态 | Macro-F1 | AUROC | NLL | Brier |
|---|---|---:|---:|---:|---:|
| EmbraceNet A | 缺 OCT | 66.230% | 73.178% | 0.69527 | 0.46953 |
| A + PCA free-mean LOOK | 缺 OCT | **67.594%** | 73.233% | 0.69463 | 0.46934 |
| A + residual RRR LOOK | 缺 OCT | **67.315%** | 73.265% | **0.63900** | **0.43794** |
| EmbraceNet A | 缺 CFP | 56.341% | 60.806% | 1.00625 | 0.63079 |
| A + PCA free-mean LOOK | 缺 CFP | **57.471%** | 60.783% | **1.00273** | **0.62938** |
| A + residual RRR LOOK | 缺 CFP | 56.341% | 60.806% | 1.00625 | 0.63079 |

### 实际正收益树

| LOOK 家族 | 缺失状态 | 最终严格正收益路径 | site attempts | candidate evals | prefixes |
|---|---|---|---:|---:|---:|
| PCA free-mean | 缺 OCT | `joint_input` | 35 | 35 | 10 |
| PCA free-mean | 缺 CFP | `joint_input` | 20 | 20 | 3 |
| residual RRR | 缺 OCT | `embraced_feature` | 9 | 9 | 2 |
| residual RRR | 缺 CFP | **空路径** | 9 | 9 | 1 |

RRR 在缺 CFP 时没有任何节点达到严格 development Macro-F1 正收益，因此树按预注册规则保持为空；这不是丢结果，也不是失败后改协议。

## 四个预注册主比较

下表的“改善”统一设为正值有利于 LOOK。Macro-F1/AUROC 是 LOOK − A；NLL/Brier 因越低越好，方向已反转。全部使用同一 296 人 development 的 10,000 次 participant-paired bootstrap；同时区间覆盖同一四对比族。

### Macro-F1

| 对比 | 改善 | ordinary 95% | 同族 simultaneous 95% | Holm p |
|---|---:|---:|---:|---:|
| PCA / 缺 OCT | +1.364 pp | [+0.312, +2.811] pp | [-0.284, +3.013] pp | 0.1464 |
| RRR / 缺 OCT | +1.085 pp | [-3.676, +5.978] pp | [-4.834, +7.005] pp | 1.0000 |
| PCA / 缺 CFP | +1.131 pp | [-0.289, +2.767] pp | [-0.753, +3.015] pp | 0.4632 |
| RRR / 缺 CFP | 0.000 pp | [0, 0] | 未定义（零方差恒等对比） | 1.0000 |

PCA 缺 OCT 的 ordinary 95% 不跨 0，但**同族 simultaneous 95% 跨 0，Holm 后也不显著**。因此当前不能把这一个开发集单种子结果写成“已经稳定证明 LOOK 必然提升 EmbraceNet”。

### 其他指标与反例

- AUROC 的四个点估计变化都非常小；PCA 缺 CFP 甚至是 **-0.023 pp**。
- RRR 缺 OCT 的 NLL 点估计改善 **0.05627**、Brier 改善 **0.03159**，但同族 simultaneous 95% 均跨 0。
- PCA 缺 CFP 的 NLL ordinary 95% 为正，但 simultaneous 95% 仍略跨 0；Holm p=0.0632。
- RRR 缺 CFP 因树为空，全部指标与 A 完全相同。

## 这次支持什么、不支持什么

**支持：**在同一冻结 EmbraceNet A 上，LOOK 可以找到 development 上有正 Macro-F1 点估计的修正，且两种 LOOK 家族在缺 OCT 时都出现正点估计；PCA free-mean 在两个缺失状态都选出了正收益树。这说明“已有缺失策略之后仍可能有剩余可线性校正结构”是值得继续验证的。

**反例：**RRR 在缺 CFP 时没有任何严格正收益节点；AUROC 的变化接近 0。三个非退化 Macro-F1 主比较的同族 simultaneous 95% 都包含 0；第四个 RRR/缺CFP 因空路径与 A 预测完全相同，bootstrap 差值和标准差恒为 0，因此标准化 simultaneous 区间未定义。这是恒等结果，不是等效性检验。LOOK 的作用明显依赖缺失方向和拟合家族，不能写成统一稳定优势。

**替代解释：**同一个 296 人 development 同时承担 checkpoint/tree 选择和效果估计，存在选择偏差；当前只有一个训练 seed，participant bootstrap 也不包含训练随机性。因此点估计提升可能部分来自 development 选择，而不是跨 seed 稳定效应。

**不能下的结论：**没有 test 结果；没有多 seed 稳定性；不能据此宣称临床泛化、统计上 family-wise 已确证的优势，或 EmbraceNet+LOOK 普遍优于所有缺失策略。

## 工程与审计状态

- 科学源：`918bac87d68604cb5da5fc1aea612cbdd818ab3e`。
- 限定 fit-profile 管理 overlay：`89be4dfeecf83c8632a83fa976bfed541b0aef92`；其数值关键模块与科学源 SHA 一致，仅补 R3 资源/恢复验收。
- 目标部署使用普通 GitHub immutable commit checkout；**此前被拒的 scp 没有重试或换 SFTP/base64/helper 绕过**。
- WS02 target CPU current-source：33 tests passed。
- GPU profile：uninterrupted vs resumed 下一更新、optimizer/scheduler/global+private RNG/mask/data cursor、完整 dev logits 均 exact；峰值约 2.31 GB。
- fit profile：中断/恢复 moments 与不间断 exact；max-dimension rank32 两 solver 均实际非零写回；`embraced_feature` 两 solver 均在 full dev 产生非零 logits 变化；GPU peak 约 0.84 GB。
- PCA：9 个实际节点，`embraced_feature` complete reference 使用解析作者随机完整二阶矩，保留条件方差。
- 远端有限 pipeline 依据 receipt 自动跳过已完成阶段并连续完成 residual RRR → 统计/中文报告 → `verify_case`，未新建竞争调度器。
- 当前单种子科学结果已由左侧独立验收接受；本页本次仅做发布表述修复和后验机制解释集成。test 仍封存。

## 可独立复核的 aggregate 证据 SHA

| 资产 | SHA256 |
|---|---|
| spec | `7d572fd5a770a29085fabd4fbe33c58533f79e6e35d518293315987c20d1e028` |
| resource profile | `1dce2acec0c0c8cd55f3b9fe5dae4635a59b975b61db2ea7f5607f76ca4e9be1` |
| frozen host receipt | `e881aab6179cb859b971f6380b3f9a91c5b49caa5183e3667b20c5157890ec1f` |
| PCA bank receipt | `630f97dba059e00f61f5d13aeaad45d4f288af3c8fb75e0eed58479baeea5a5d` |
| fit-profile receipt | `d3783f262021b74c3dbddf13907b5b06e89d238698bafb3cc2000868223c48c8` |
| PCA arm receipt | `3a594de68c9aa9b3c3b8cbd0b4d1ab9121e406cc21f7adb2f07588079ae28aff` |
| RRR arm receipt | `10c1efb7482c2821b607b8963e3816614cef05f8d54e44bd2107d2568717ad90` |
| delivery receipt | `9f91112ebf614a30c4637090045b13be6d4a07314003b5ee15f51f3944dea851` |
| aggregate results | `301971921a95f4aadb4bf6b8bffc58717e5511ed8f34eadf4246bd1d6d068847` |
| remote Chinese candidate | `524deeac6bfab1dfc51b69d377985e38ff13eb688060d8cfe4d50be2b7766f94` |
| final pipeline status | `9eb33c28c2e8f7dae335854cb58f0b2a175dcf745f701b74ce477b99086501eb` |

下一门槛是左侧对 WS02 原 aggregate/prediction/tree 产物做独立重算和验收。在此之前，本页不标记 scientifically accepted。

## 2026-09-19 独立验收后的写回机制解释诊断

该诊断在左侧接受原单种子科学结果之后单独登记；不重新训练、不重新拟合、不重新搜索，只复用冻结 A、已选 PCA artifact 和原 296 人 development。joint_input 的人类可读含义是“两个模态各自进入 ResNet18 编码器之前的联合输入节点”；embraced_feature 是“EmbraceNet 完成逐坐标模态选择后的融合特征”。

PCA 在缺 OCT 与缺 CFP 时都选择 joint_input。有限反事实写回得到完全一致的结构：

- 完整 LOOK = 仅写可用分支：两种缺失下 logits 与指标逐项相同；
- 仅写缺失分支 = 原 A：两种缺失下 logits 差均严格为 0；
- 因此观测到的 PCA 收益来自校正当前仍可用模态的输入表征，不是补出一个伪造的缺失模态；缺失分支在作者 availability 语义下仍被门控屏蔽。
- 缺 OCT 时完整/可用分支写回相对 A 的最大绝对 logit 变化为 0.13749；缺 CFP 为 0.13281。诊断 receipt SHA 为 7d903e298b4bdc36c630e470d7524634b70ff875feb2894f02741388a3235921。
- RRR 缺 OCT 的已选节点 embraced_feature 位于 EmbraceNet 选择之后，因此语义是融合特征校正，不是模态生成；RRR 缺 CFP 保持空路径恒等。

这是后验机制解释诊断，目前为右侧自检证据；不改变左侧已接受的四个主比较，也不把它升级为新的确认性效应检验。详见 integration_20260919.md。
