# EmbraceNet 缺失方法适配修复 v2：统一 CPU 合同与统计目标

状态：**工程 CPU 闭环，ready_for_left_review；未训练、未读任何 train/dev/test 参与者数据、未用 GPU、未科学接受。**

本文件取代 v1 中尚未闭合的工程 blocker，并**修订 v1 的 K=32 正式评估提案**。v1 冻结 result/ACK/left_audit 原样保留，既有性能结果与 evidence cutoff 不变。

## 一、老师问题与唯一实验家族

老师问题仍只有一个：**已有缺失方法 A 是否能被同一个冻结 A + LOOK 增强？**

- A：EmbraceNet，作者固定 `c61b63dadc6fc8d719bb77475f4734f62697144e`，MIT。
- 只训练 **1 个新 A / seed 3416**。
- A 训练完成后冻结同一个 checkpoint。
- 主比较族只包含两个单缺失状态：
  - missing_OCT：A vs A+LOOK-PCA；A vs A+LOOK-RRR；
  - missing_CFP：A vs A+LOOK-PCA；A vs A+LOOK-RRR。
- 两个 LOOK 方法均保留原 free-mean 定义与完整 positive-forward tree；不根据结果取消阴性。
- complete-input A 只用于原 checkpoint 选择与诊断，不把其好坏作为是否执行缺失对照的开关。

因此主家族是**一个 A checkpoint、两种缺失、两种 LOOK = 四个配对增量对比**，不是方法排名矩阵。

## 二、train / 普通 A / A+LOOK 的缺失入口已经统一

v1 的 blocking 问题是：普通 `forward_embracenet_host` 曾把 raw tensor 直接送入 R18，而 A+LOOK study 入口才先做 `torch.where`。

v2 现统一为 `src/look/models/embracenet.py::prepare_embracenet_inputs` / `reset_embracenet_inputs`：

1. availability 单位是 participant，模态顺序固定 `[OCT, CFP]`；
2. counts 将 participant availability 扩展到其 1/2 个实际眼；
3. 不可用模态在**进入任何 R18/BN 之前**通过 `torch.where` 变为零；
4. 可用模态若含非有限值直接拒绝；
5. train A 的 `forward_embracenet_host`、普通 A eval 与 `forward_embracenet_with_look` 都使用同一 helper。

CPU mixed-batch train-mode 验证覆盖 complete、missing_OCT、missing_CFP 同批存在：将 raw 不可用值由有限随机数替换为 NaN，logits/loss 与 BN buffers 完全一致；raw 不可用输入梯度为 0。

这里**不声称 BN 不更新**。训练态 BN 仍按既有 R18 规则正常更新，只是同一 missing schedule 下它看见的是相同 zero rows，而不是任意未观测值。

## 三、作者机制与恢复合同

作者机制不变：

- docking：Linear + ReLU；
- availability × selection probability 后归一化；
- 每个 embraced coordinate 独立 `torch.multinomial(..., replacement=True)`；
- 训练继续使用作者随机机制，不替换成期望网络。

v2 对控制状态补强：

- replay indices 必须是整数、正确 shape、合法模态；
- replay 若选择 availability=0 或 normalized probability=0 的模态，显式拒绝；
- forward 失败前即清除 pending replay；
- sampling state restore 也清除旧 pending replay/trace；
- study 入口拒绝重复、非法、乱序或超过 stop boundary 的 artifact；
- replay 在 stop 位于 `embraced_feature` 之前时拒绝，避免永远未消费的 pending replay。

受影响作者等价已在当前 model SHA 重新验证：四类作者合法域输出/输入梯度/docking 参数梯度仍为 0 差；作者全局 multinomial RNG 与项目 private RNG 终态一致；完整 Python/NumPy/Torch + private RNG 恢复后下一 AdamW 更新仍精确一致。

## 四、九个实际 LOOK 节点已经做“非零”动态验证

合法有序节点仍为：

1. `joint_input`
2. `joint_stem`
3. `joint_stage1`
4. `joint_stage2`
5. `joint_stage3`
6. `joint_stage4`
7. `joint_features`
8. `joint_participant_feature`
9. `embraced_feature`

当前 SHA 的本机隔离 CPU 测试不是 identity read/write smoke：

- missing_OCT 与 missing_CFP 两种状态；
- 每个节点构造真实非零 `LOOKArtifact`；
- 节点值确实变化；
- 固定相同抽样 trace 后最终 logits 确实变化；
- 对应 zero-weight/zero-bias no-op artifact 与 baseline 精确一致；
- sampling state restore 后同一非零 artifact 输出与 trace 精确一致；
- 重复、非法、乱序、越 stop artifact 均拒绝。

这证明**实际写回—下游继续—抽样配对—恢复**链路已接通，不再以“写回同一个 tensor”冒充方法接入。

## 五、双模态单缺失为什么不需要 K=32 全网重复

两模态时，单缺失 availability 归一化后必然为：

- missing_OCT：`[0, 1]`；
- missing_CFP：`[1, 0]`。

因此对每个 embraced coordinate (j)，categorical 选择退化成唯一可用模态，固定 eval checkpoint / 固定 LOOK 前缀下：

[
z_j=d_{available,j} quad 	ext{almost surely}, qquad Var(z_j)=0.
]

完整 R18/MHD 合成验证使用 sampling seed 1/9/27/81：

- missing_OCT：普通 A 跨 seed 最大绝对差 = **0**；非零 A+LOOK 最大差 = **0**；
- missing_CFP：普通 A 跨 seed最大绝对差 = **0**；非零 A+LOOK 最大差 = **0**；
- complete 状态相对 seed1 的 logits 最大差为 **1.0715 / 1.8086 / 1.5307**，明确仍随机。

### 单缺失正式成本

主比较的每个 single-missing state **只需一次正常作者式 forward**，而不是 32 次全网 forward。仍调用一次正常 EmbraceNet sampling：

- indices 必然全选唯一可用模态；
- 输出确定；
- private RNG 按正常 forward 消耗一次，resume 行为明确；
- A 与 A+LOOK 同状态记录/复用同一 trace。

开发集评估前后必须 snapshot/restore A 的 private sampling state（或使用等价独立 eval clone），因此 dev eval 不改变后续训练 stochastic stream。

### 可缓存部分

固定 checkpoint + 固定 missing state 下：

- `joint_input` 至 `joint_participant_feature` 均确定，可按既有 LOOK cache 身份缓存；
- 单缺失 `embraced_feature` 也确定；
- 若某 LOOK path 在更早节点写回，只需从最早写回点重算一次确定性下游，不需要因 EmbraceNet 再重复 32 次；
- 同 A checkpoint 的两个 LOOK 方法可共享合法的未修正 upstream cache，具体写回后的后缀仍按现有 DAG 规则重算。

## 六、complete 状态的解析关系与最终统计目标

令 docking 后模态 (m) 在坐标 (j) 的值为 (d_{m,j})，选择概率为 (p_m)。作者实现对各 embraced coordinate 条件独立采样，则：

[
mu_j = E[z_j] = sum_m p_m d_{m,j},
]

[
E[z_j^2] = sum_m p_m d_{m,j}^2,
]

[
E[zz^T] = mumu^T + diag(E[z_j^2]-mu_j^2).
]

当前 eval 分类头 dropout=0 且为 linear：

[
ell = Wz+b,qquad E[ell]=Wmu+b.
]

因此，如果正式目标是 v1 所表达的“**先平均 logits，再算指标**”，其无限 Monte Carlo 极限有**精确解析解**，不需要 K=32：

1. 先把 deterministic upstream / LOOK prefix 真正执行到 docking 输入；
2. 计算 EmbraceNet 的解析 (E[z])；
3. 若 `embraced_feature` 上有当前 affine LOOK artifact，则均值可继续按 affine map 精确传播；
4. 通过 linear head 得到精确 (E[logits])；
5. 用 `softmax(E[logits])` 计算该**integrated-mean-logit predictor** 的 Macro-F1/AUROC/NLL/Brier。

前置 LOOK 即使含确定性非线性（例如 R18/ReLU），只要它发生在抽样之前，先真正执行该前缀后再对 categorical embrace 做上述解析积分即可。

### 解析关系不适用/目标不同的地方

[
softmax(E[ell]) 
eq E[softmax(ell)]
]

一般成立。合成精确枚举中，两者最大绝对差为 **0.31757**。

所以以下是不同统计目标：

- **本 v2 推荐**：integrated mean logits → softmax → 指标；
- 另一个可能目标：expected softmax / expected per-draw NLL/Brier。

后者涉及 softmax/NLL/Brier 非线性，不能用 (E[logits]) 代替；若未来真的要回答该问题，必须另行预注册 MC/积分策略。

同一个合成例子中，K=32 平均 logits 相对解析 (E[logits]) 的最大误差仍为 **0.33937**。这说明 K=32 的作用只能是有限 MC 近似/实现诊断，不能被写成“无误差必要步骤”。

### v2 对 K=32 的唯一定位

**K=32 不进入主评估、不参与 checkpoint 选择、不进入四个主配对统计。**

可保留为 complete stochastic state 的可选工程诊断：

- 检查 MC 均值是否向解析均值靠近；
- 展示随机输出分散程度；
- 不根据真实 dev 分数决定是否增减 K。

普通 participant bootstrap 对 integrated deterministic prediction 只反映 participant 抽样；它仍**不包含训练 seed 随机性**。若报告可选 K32 MC 诊断，其 MC 误差也不能假装被普通 bootstrap 覆盖。

## 七、统一的具体 trial contract

### 继承且不改

| 项目 | 值 |
|---|---|
| 数据 | glaucoma；1264 train / 296 dev；test 封存 |
| seed | 3416 |
| backbone | 两路 2D ResNet18 |
| 初始化 | public ImageNet V1 fresh，SHA `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| precision | FP32 |
| batch | microbatch 16；effective batch 128 |
| optimizer | AdamW |
| LR | pretrained 1e-4；新 docking/head 1e-3 |
| weight decay / clip | 1e-4 / 5 |
| schedule | warmup 5；minimum 8；patience 15；100 epoch cap |
| loss | unweighted CE |
| checkpoint 指标 | complete dev Macro-F1 |
| BN | 继承现有 R18 train/eval 规则，不改策略 |

### 新科学选择——需左侧最终确认后才能训练

1. EmbraceNet `embracement_size=256`；
2. complete selection probability=`[0.5,0.5]`；
3. train participant missing schedule：complete / missing_OCT / missing_CFP 各 **1/3**，两眼一起缺失；
4. 三条入口统一先 zero-mask unavailable raw input，再进 R18；
5. complete dev checkpoint 的 Macro-F1 用**解析 integrated-mean-logit predictor**；
6. 两个 single-missing 主评估各只做一次正常作者式 forward；
7. dev evaluation snapshot/restore private sampling RNG，不改变训练流；
8. K32 仅可选诊断，不是正式估计量。

训练仍随机：complete train 样本继续走作者 multinomial；single-missing train 样本虽 categorical 退化，仍使用同一 EmbraceNet forward。

## 八、CPU 验收与边界

本机隔离环境：

- Python 3.11.16
- torch 2.8.0
- torchvision 0.23.0
- numpy 2.2.6
- pytest 8.4.1
- mhd-framework 4
- MHD source commit `c0a27abb3e0f2153bfd273b1d05d5b7dae9784f0`

当前源码：

- model `5543c74e19acbdeb715ba3d76612282a9e4ef6028e38fab943b53676ef263c8d`
- runtime `2f633c4d012f05915be8d413f29f9fcc5b855ce73b3ebbe7f8d4bf64e40b74e2`
- study `ca16bc208aba4c15d166afeabcdbfef06467e9a463be3e2206e30c3628fe8cc7`
- tests `bf35ef9e55e4808311eceb9fb372da64ae6cc4e7914df51cc79d518a4259dccb`

动态证据：

- current EmbraceNet tests：**16/16 pass**；
- native-host/MMTM/linear-vector regression：**30/30 pass**；
- 当前 model SHA 的作者数值/梯度/RNG/下一 AdamW 恢复：pass；
- 完整 MHD 单缺失/统计目标审计：pass。

这些证明本机 CPU 工程闭环；**不等于 WS02 部署 gate 已通过**。后续若左侧批准真实试运行，WS02 仍须按合法部署流程重新做目标 runtime/import/CPU-or-GPU preflight，不能复用被拒的 scp 路线。

## 九、当前状态

- engineering CPU validation：完成；
- one concrete scientific contract：完成，等待左侧确认；
- real CFP/OCT training：未启动；
- participant train/dev/test data：本包均未读取；
- GPU/profile/remote write：无；
- test：封存；
- main push：无；
- scientific acceptance：**false**。

下一步不是继续改代码，也不是再搜索方法：左侧只需对 v2 的两个新科学核心作一次决定——**train 三状态各1/3** 与 **complete 采用解析 integrated-mean-logit estimand（K32仅诊断）**。若接受，再另发真实 GPU 试运行包。
