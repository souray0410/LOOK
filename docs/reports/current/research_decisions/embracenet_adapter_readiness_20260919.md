# EmbraceNet首个真正缺失方法适配：CPU原型与试运行合同（2026-09-19）

状态：**planned + implemented + author-CPU-validated；未训练、未读test、未GPU执行；传输恢复后新增研究入口安全层尚缺动态CPU重验。**

本页落实老师问题“已有缺失方法 A vs 同一个冻结 A + LOOK”。A 固定为 EmbraceNet；本包只完成作者等价、MHD接口原型、随机参考语义和一个明确训练合同，不把CPU工程证据写成CFP/OCT实验接受。

## 1. 固定来源与已验证事实

作者库固定为 `idearibosome/embracenet@c61b63dadc6fc8d719bb77475f4734f62697144e`，MIT。冻结参照 SHA256：

- `LICENSE`: `63def471f62dc22e08dddc85b61c2c1e46a853542e81081951694d8feabe6ccc`
- `README.md`: `28949a2a0d0ab8211e784eb8b2847b3f18557c219bdf7c79152ab0f90ab3333f`
- `embracenet_pytorch/embracenet.py`: `8b3414181bd77cf37bb2f5b39d43bb2ce6cbd73bcf5afa8dbccb6b4e1d11df05`

作者机制保持为 Linear+ReLU docking、availability×selection probability 后归一化、`torch.multinomial` 逐坐标有放回选择；eval 仍随机。项目实现**没有**换成概率加权均值或“期望网络”。

作者合法输入域的独立CPU对照已经落盘于
`experiments/2026_09_10_11_11_31/look_mechanisms/embracenet_cpu_acceptance_20260919.json`
（SHA256 `30bd876d9d0d07d784c5314486b6210c60d82c92a8696c5260312f1cbf6977ef`）。
对应项目核心 `src/look/models/embracenet.py` SHA256 仍为
`39fd8cfb38e7f8c5f6b1241fa61a63d28bb807a13677bde565b59fedbf615a1f`，传输恢复后未改变，因此以下证据保持有效：

- 全可用等概率、missing OCT、missing CFP、全可用非等概率四案例：输出最大绝对差 0；
- 输入梯度与两路 docking weight/bias 梯度最大绝对差均为 0；
- 作者全局 multinomial RNG 终态与项目 private Generator 终态一致；
- 两种单缺失中，改变不可用输入不改变输出，不可用输入及 docking 分支梯度为 0；
- 完整 Python/NumPy/Torch RNG + EmbraceNet private RNG 恢复后，下一次 AdamW 更新的 logits、loss、参数和 optimizer state 精确一致；
- 作者对 all-missing/零总选择概率最终报运行错误；项目显式 ValueError fail-closed；
- 作者对“不可用分支是 NaN”会产生非有限输出，项目 fusion 层显式屏蔽后有限。

传输前 MHD 原型在隔离 WS02 CPU 环境曾有 5 项通过；本次恢复没有为网页故障重复它。恢复后只新增研究入口的 pre-R18 NaN-safe masking 和共同随机抽样 helper，故旧 5 项不能冒充覆盖这两个新变化；动态重验的当前阻塞见第 7 节。

## 2. 唯一推荐的单A试运行协议

继承已接受 R18 小队列 spec（run `2026_09_18_11_18_28_650020`，spec SHA256
`f09a452f8584861ff5291cb262894e16f4ba4c7c962f43f6ba4cfbccb18929e8`）而不改：

| 项目 | 固定值 |
|---|---|
| 数据 | glaucoma；1264 train / 296 development；test封存 |
| 种子 | 3416 |
| 编码器 | 两路2D ResNet18 |
| 初始化 | public ImageNet V1 fresh，SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| 精度 | FP32 |
| batch | microbatch 16；effective batch 128 |
| 优化器 | AdamW |
| LR | pretrained encoders 1e-4；新 EmbraceNet docking + classifier 1e-3 |
| WD / clip | 1e-4 / 5 |
| 调度 | warmup 5；minimum 8；patience 15；100 epoch保护上限 |
| loss | unweighted cross-entropy |
| 选择 | **development complete-input Macro-F1**；不改成双缺失或三状态平均 |
| BN | 沿用既有 R18 行为，不另改 BN 规则 |

本包新增、必须由左侧作为新科学变化明确确认的唯一推荐值：

1. `embracement_size=256`，直接采用作者默认；
2. 模态顺序固定为 `[OCT, CFP]`；complete 时 selection probability=`[0.5,0.5]`；
3. train participant 级 missing schedule 固定为 complete / missing_OCT / missing_CFP 各 1/3；同一参与者两眼一起缺失，使用独立可恢复 RNG；
4. 缺失输入在任何 R18 encoder 之前用显式 `torch.where` 零填充；不依赖 `0 * NaN`。这与项目既有缺失训练“输入置0”语义一致，但把安全屏蔽提前到 encoder 入口；
5. stochastic evaluation 固定 **K=32** 次 Monte Carlo draw；逐 draw 记录精确 modality indices，32 次 logits 先平均再算 Macro-F1/AUROC/NLL/Brier；
6. dev checkpoint 仍按 complete Macro-F1 选；missing_OCT / missing_CFP 只报告，不参与 checkpoint 选择。不得在看分数后改成“三状态平均选优”。

K=32 是本试运行的预注册有限积分预算，不是作者论文默认，也不宣称足以证明 Monte Carlo 收敛；后续若要改 K，必须在新性能前重新批准。

## 3. 随机完整参考与同checkpoint公平对照

冻结同一个 A checkpoint 还不够，因为 EmbraceNet eval 自身随机。推荐如下：

- 每个 `(config_id, participant_id, draw_index)` 生成稳定 63-bit seed：
  `sha256(config_id | participant_id | draw_index)`；
- **seed故意不包含 missing state**，使 complete 与 missing 使用共同随机数（common random numbers）；
- complete / missing_OCT / missing_CFP 各自按自己的 availability 与 selection probability 运行作者 multinomial，因此可以得到不同、但均合法的 state-specific modality indices；
- 不能把 complete 抽到的“随后会缺失的模态”索引强行回放到 missing state；
- 在**同一 participant、同一 missing state、同一 draw**的 A 与 A+LOOK 之间，必须精确复用已记录的 modality indices；这样二者唯一差异才是 LOOK；
- 每个 draw 保存 seed、availability、归一化概率、indices SHA/原始 trace、模型/LOOK artifact 身份；可精确重放，不只保存“设置过seed”。

完整参考的统计含义：

- `joint_input` 到 `joint_participant_feature` 都在 EmbraceNet 抽样之前，完整/缺失特征本身是确定性的，正常缓存一次；
- `embraced_feature` 是随机节点：train-only LOOK 拟合若选择该节点，complete reference 与对应 missing feature 都使用同 participant/draw 的共同随机数，K=32 形成成对样本；
- 最终 dev 指标同样 K=32，先平均 logits 后计算指标；不改成 expectation network。

## 4. 实际 MHD / LOOK 节点与正收益树范围

当前 R18 EmbraceNet 原型的**实际**有序可读写节点为：

1. `joint_input`
2. `joint_stem`
3. `joint_stage1`
4. `joint_stage2`
5. `joint_stage3`
6. `joint_stage4`
7. `joint_features`
8. `joint_participant_feature`
9. `embraced_feature`

这不是把旧 fusion-stage 的九个名字硬套进来。前八个节点位于两路编码/参与者聚合到 EmbraceNet 之前；第九个是作者随机 embracement 之后的256维表示。

两种缺失状态 `missing_OCT`、`missing_CFP` 都保留；LOOK 两臂仍为：

- PCA free-mean；
- residual RRR free-mean。

未来批准后，完整 positive-forward tree 可以按以上九节点分别跑两方法×两缺失；本CPU包**没有**执行真实树拟合或dev性能搜索。pre-embrace joint artifact 即使写到不可用模态半边，A 的 availability 仍会在 EmbraceNet 处屏蔽该模态，因此这部分不会偷偷恢复为“可用模态”；真正直接作用于随机融合表示的是 `embraced_feature`。

## 5. 老师问题的一页决策表

| 老师问题/决策 | 本包给出的唯一具体答案 | CPU证据 | 未来成本 | 当前局限 |
|---|---|---|---|---|
| A 是什么 | EmbraceNet，作者固定 commit + MIT，保持随机 multinomial | 四合法案例输出/梯度/RNG零差 | 1 个新 A 训练 | 尚无 CFP/OCT 性能 |
| A vs A+LOOK 怎么公平 | 同一个训练后冻结 checkpoint；同 missing state 逐draw复用精确 indices | trace/replay 与下一AdamW恢复已验 | 不重训 A；另做 LOOK fit/replay | K=32 是新预注册选择 |
| 缺失怎么训练 | participant 三状态各1/3；双眼一起缺失；pre-R18显式零填充 | 既有mask语义 + 新入口已实现 | 与单A训练同一run | 新入口恢复后动态重验未完成 |
| checkpoint 怎么选 | 继续 complete dev Macro-F1 | 来自 accepted R18 spec | 无新增选择搜索 | missing表现不参与选优 |
| 随机完整参考怎么定义 | complete/missing 同seed common-random，state-specific合法indices；A/A+LOOK同state精确trace复用 | 作者RNG等价、trace可重放 | K=32 embracement/head replay；pre-embrace可缓存 | 未证明K=32积分误差充分小 |
| LOOK 写哪里 | 实际9节点，从joint_input到embraced_feature；两自由均值×两缺失 | 传输前9节点MHD read/write/backward smoke已过 | 批准后完整正收益树拟合 | 未跑真实train/dev树 |
| 是否现在开训练 | **否** | 本包 scientific_acceptance=false | 左审后才可GPU试运行 | test继续封存 |

推荐主负责人下一步只需确认一件事：**是否接受上述唯一 trial contract（尤其三状态1/3与K=32共同随机评估）作为未来首个 EmbraceNet/3416 GPU 试运行协议。** 不需要再泛搜方法或增加种子/骨干。

## 6. 状态边界

- planned：未来单A/seed3416 CFP/OCT试运行与完整 A vs A+LOOK positive tree；
- implemented：作者等价 fusion、private RNG/state、MHD R18 adapter、9节点接口、研究入口 pre-R18 安全屏蔽、共同随机 trace helper；
- CPU-tested：作者核心数值/梯度/RNG/恢复证据已冻结；传输前 MHD 原型 5 项通过；
- not-trained：没有参与者真实训练、没有新 GPU profile、没有任务派发；
- test：封存、未读；
- old hosts/queues/checkpoints：默认行为未改；
- main：不推送。

## 7. 当前唯一未闭合验证项

恢复后 `src/look/models/embracenet.py` 未变，因此作者逐值/梯度/RNG回执有效。为了修补 raw missing NaN 在 R18 前的安全入口，并落实 common-random trace helper，本次仅改变了 `src/look/studies/embracenet_adapter.py` 与对应测试。尝试把这两个变化同步到已存在的 WS02 隔离 CPU 副本时，WebCodex 的 `scp` 调用被上层安全检查在启动前拒绝；随后没有用 helper/替代通道绕过该拒绝。

因此本包可以提交**实现与合同证据**，但不能把传输前的 5 项 MHD 测试冒称为覆盖恢复后的新 study/test SHA。最终状态应保持 `needs_planner_decision`，直到合法 CPU 环境可对当前 SHA 做一次针对性动态重验；这不要求重做作者数值/RNG对照，也不授权真实训练。
