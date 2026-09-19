# LOOK 缺失方法 A：适配合同就绪度（2026-09-19，待左侧独立审查）

## 1. 边界与主问题

本包只补足“真正的缺失模态方法/策略 A 如何公平接入 LOOK”的证据和适配合同；不启动训练、不访问 test、不改变既有数值/cutoff/GPU owner/停止规则，也不构成新科学接受。

主问题固定为：处理整模态缺失的方法/策略 A，与同一个已冻结 A + LOOK 相比，LOOK 是否提供额外价值。MMTM、fusion-stage、fixed-mean/fixed-prefix 仅保留为宿主兼容、组装或 LOOK 机制证据，不计缺失方法 A；R&B 的通信 A vs A+bridge 继续独立。

## 2. ShaSpec 的准确证据边界

一手来源：
- CVPR 2023 主论文：https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.html
- CVF PDF：https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.pdf
- 官方代码：https://github.com/billhhh/ShaSpec
- 既有证据固定 commit：57a04fa88331ea3b2671795a15a1ae9a24fd310d

### 2.1 分类任务确实存在，但官方仓库不是眼科分类实现

论文 Section 4.1/4.2 同时覆盖 BraTS2018 多 MRI 模态医学图像分割和 Audiovision-MNIST 图像+音频分类，并另报告 OpenI X-ray + clinical text 的二分类补充实验。因此“ShaSpec 主论文有 missing-modality classification 证据”成立。

但既有固定官方仓库公开的是 BraTS 3D 分割路线，README、DualNet_SS.py、train_SS.py 对应 BraTS 实现；本包未发现作者公开的 CFP/OCT 双图像 2D 分类 adapter。因此不能把“论文做过分类”写成“眼科二分类可直接复现”。

### 2.2 机制必须保真

论文 Section 3.1–3.3 的核心合同：
- 每个可用模态产生 shared feature 与 modality-specific feature；
- shared/specific 投影后与 shared feature 残差融合；
- 缺失模态 n 时，Section 3.2 Eq. (4) 用其余可用模态 shared features 的均值生成缺失 embedding；
- distribution alignment objective 约束 shared features，domain classification objective 约束 specific features；
- 缺失模态自身不可计算的辅助项省略，主任务仍通过生成的 missing feature 优化。

对 CFP/OCT 两模态、恰缺一个模态时，Eq. (4) 的均值只有一个可用 shared feature，所以数学接口可直接定义；但 2D 架构、缺失采样、损失权重和停止规则仍是未锁科学项。

### 2.3 作者训练/缺失设定的出处

论文 Section 4.2 区分：
- non-dedicated training：训练时随机丢弃模态，一个模型覆盖不同缺失组合；
- dedicated training：训练使用与评估相同的缺失模态设定，为特定缺失状态训练对应模型。

BraTS 路线为 3D U-Net；官方 README 也描述 random modality dropout，并给出较低初始 dropout/逐步增加或 full-modality warmup 的建议，只能锁定 BraTS 路线。

Audiovision-MNIST 分类路线则明确：跟随 SMIL 的 missing-audio 设定；训练 60 epochs；使用 SMIL image/audio encoders；融合后接两层带 dropout 的 FC；Adam、初始 lr=1e-3、weight decay=1e-2，每 20 epochs 将 lr 降低 10%；分类指标为 accuracy。它证明分类配方存在，但对象是 image+audio，不自动决定 CFP+OCT 协议。

## 3. 现有 5 候选的合同就绪度

|候选|必须保留的机制|当前适配新增量|成本/主要风险|状态|
|---|---|---|---|---|
|ShaSpec|shared/specific、missing feature generation、主任务+DCO/DAO|两套 2D 图像接口、二分类头、CFP/OCT 缺失训练合同、合法 LOOK 节点|中高；分类代码未在固定仓库发现，provenance/license 未锁|证据最贴近，但 not training-ready|
|U-HVED|变分/潜变量补全、available-subset fusion、生成/重建目标|2D 生成模型、分类头、是否需要图像重建|高；任务与生成目标变化大|代表性有价值，非低成本首包|
|RFNet|region-aware available-modality fusion、per-modality auxiliary regularization|需为 image-level classification 重定义 region/辅助监督|高；语义迁移本身是科学取舍|不能删核心机制后仍声称 faithful RFNet|
|mmFormer|availability mask、modality-specific encoders、intra/inter-modal transformer、辅助约束|2D token/encoder、分类头、容量政策|高；新 backbone/容量是强混杂|源码较清楚但首包成本高|
|Missing-Aware Prompts|missing-type prompts、prompt/backbone freeze 关系|image-text→CFP/OCT 双图像语义与预训练身份重定义|参数增量低但科学适配风险高|参数少不等于 adapter 简单|

M2FTrans 继续标 screened-out 而非 invalid；它与 mmFormer/RFNet 的 3D robust-fusion 家族在首包的信息增量有限。

## 4. 漏项审查：只补两个低成本类别

### 4.1 ModDrop：missing-modality training strategy

一手来源：Neverova et al., ModDrop: adaptive multi-modal gesture recognition  
https://arxiv.org/abs/1501.00102

核心是训练时随机丢弃独立模态/通道，使融合模型学习跨模态相关性并对一个或多个输入信号缺失保持鲁棒。它不要求 3D segmentation、生成 decoder 或 image-text pretraining，可直接在两个 2D 图像分支上定义 modality-drop policy，因此补上“低成本缺失模态训练策略”类别。

需左侧锁定：missing CFP/OCT 采样分布、是否保留 full-modality 样本、宿主容量、训练/停止预算。纯工程仅包括 mask/drop plumbing、seed 和日志。

### 4.2 EmbraceNet：classification-native availability-aware fusion

一手来源：Choi & Lee, EmbraceNet: A robust deep learning architecture for multimodal classification  
https://arxiv.org/abs/1904.09078

官方实现：https://github.com/idearibosome/embracenet

核心为 docking layers + stochastic embracement；官方实现提供 availability 输入，缺失模态不参与抽样，概率在可用模态间重新归一化。原始任务就是 multimodal classification，而非 3D segmentation。CFP/OCT 可保留各自 2D encoder，只改变 fusion 接口，因此补上“原生分类的缺失可用性融合”类别。

需左侧锁定：embracement size、selection probability、encoder 预训练/联合训练关系、missing policy、停止预算。官方代码存在不等于训练授权。

这两个补项只填类别空洞，不把候选表变成“必须跑 7 个方法”。

## 5. 最小有限 A vs A+LOOK 合同草案

### 5.1 必须固定

1. 数据角色：沿用已审计 train=1264、dev=296、seed=3416、既有 CFP/OCT 2D preprocessing 和 participant order；test sealed。
2. 同一个 A：先按锁定协议训练一次 A，保存 checkpoint SHA 与 config fingerprint；A 与所有 A+LOOK 必须引用同一冻结 checkpoint。
3. 同一缺失输入：至少分别评估 missing_OCT 与 missing_CFP；同一参与者两臂可用输入完全一致。
4. LOOK 只做增量矫正：不得重训 A、改变 A 的 missing mechanism/core losses、BN/state 或主任务头；只能在事先批准的 legal internal representation 上 fit/read/write。
5. 同一评价：Macro-F1、AUROC、NLL、Brier；保持 participant-level paired comparison 和既有 bootstrap/reporting contract。F1 上升但 NLL/Brier 恶化必须同时报告。
6. 成本分开：A training 与 LOOK fit/replay 单独记账。
7. test 不得用于 A 选择、LOOK fit、节点选择或停止。

### 5.2 最小臂

每个缺失场景至少保留 A、A+LOOK-PCA、A+LOOK-RRR/residual。若未来只批准一个 LOOK 变体，必须在查看 dev 增量结果前写入协议，不能事后按 dev 分数选择。

### 5.3 必须由左侧先锁定的科学取舍

- 选哪个 A 以及它为何代表老师问题中的已有缺失方法/策略；
- dedicated/non-dedicated 或其他 missing schedule、比例、是否含 full-modality samples；
- 2D backbone/capacity 与初始化身份；
- 核心/辅助 loss 与权重；
- optimizer、lr、training/stopping/selection budget；
- legal LOOK node 与 complete/reference feature 语义；
- full-modality performance 的角色；
- implementation provenance/license。

ShaSpec 还必须明确 DCO/DAO 的 2D classification 实现、双图像 shared/specific encoder 设计，以及论文分类配方与 BraTS 仓库实现之间如何取舍。

### 5.4 合同锁定后的纯工程适配

tensor/batch plumbing、availability mask 编码、2D encoder adapter、binary logits 接口、暴露已批准 LOOK node、checkpoint/config/source SHA 记录、participant-order/prediction schema、CPU structure/import/documentation gate、可恢复日志与 attempt identity。

若工程实现迫使改变核心 loss、容量、输入可用性或 selection rule，必须重新归类为科学取舍。

## 6. 审查结论与下一 gate

- ShaSpec 可准确写为“论文有 missing-modality classification 证据”，不能写成“官方仓库已给出 CFP/OCT 2D classification 可直接复现实现”。
- ShaSpec 的两模态 missing-feature generation 数学接口明确；真正未锁的是 2D classification architecture、missing schedule、loss/budget、legal LOOK node 和 provenance/license。
- 现有 5 候选偏向 3D segmentation 或 image-text，容易把 adapter cost/host change 混入真正的 A vs A+LOOK。
- 本包仅新增 ModDrop 与 EmbraceNet 两个低成本类别代表；二者均只是候选，不构成训练批准。
- 下一 gate：左侧选择一个 A，并锁定第 5.3 节科学项。在此之前状态只允许 ready_for_review，不启动 GPU、不访问 test。
