# Improved Modality Dropout target-task组件 A + LOOK：2026-09-20 单种子候选

状态：`right_independent_audit_passed_left_acceptance_pending`。这是一组 **target-task组件适配**，不是论文完整 `Learning Contrastive Multimodal Fusion` 方法复现。

## 方法身份先纠正

固定作者来源：MICCAI 2025 / arXiv:2509.18284；作者仓库 `omron-sinicx/medical-modality-dropout`；commit `8040d96b2dec48cf8fc7d13b45e15af0d07952ed`；MIT。

作者方法分阶段：先做 fused/unimodal 的监督 sigmoid contrastive 预训练（三对：OCT-CFP、OCT-fused、CFP-fused），再做 target task 的 simultaneous modality dropout：完整输入、缺OCT、缺CFP三项分类损失，并用 learnable missing token 替代缺失模态。

本次真实 A **只执行了第二阶段的 target-task 部分**：三态 CE（lambda=1）+ learnable missing tokens；没有执行 contrastive pretraining。Encoder 参数无更新，但训练态下 120 个 encoder state keys 中 60 个 BN running-state buffer 变化、非 BN 变化 0。因此准确名称是 **“IMD target-task component adaptation”**，不能称为作者完整方法，也不能称为严格 frozen-encoder-state 复现。

## 冻结 A

- WS02 run：`improved_dropout_20260920_v1`
- source：`77f94be3ecc3a7632bf34aad5fc288d542036ad6`
- seed3416；1264 train / 296 development；test封存
- best/stop：13/28；280 updates
- complete-dev：Macro-F1 62.777%，AUROC 66.979%，NLL 0.66556，Brier 0.46751
- host receipt SHA `47dcc6a8276332e06003228e894aff61a17d392f6187ff06f1c1d555f409d1da`
- best SHA `edc6f5f26befdda3c742a2683b77aa9f629e7cfcdde60c198b83e7c70c534530`
- dev predictions SHA `212e1e4e32a25bf5873b64c66504b84166d0be9a1262f127ba49f4b9f3a6e0ca`
- 12资产冻结 manifest SHA `1f2fd7b0e81f6ed4ab0140da316832926eb0ad373bae1b179adfbaa9c050a268`

LOOK continuation 结束后 12/12 资产再次逐项 byte-identical；没有重训 A。

## LOOK 安全写入与四树

LOOK 只暴露 token substitution 之后的两个逻辑站点：`imd_fusion_input`（learnable token替换缺失模态并完成LayerNorm后）与 `fusion_feature`（TNF MLP输出、分类器前）。真实缺失模态 participant feature 不作为 LOOK 站点。

选中路径：
- PCA / 缺OCT：`imd_fusion_input`
- PCA / 缺CFP：`fusion_feature`
- residual RRR / 缺OCT：空树（严格恒等）
- residual RRR / 缺CFP：`fusion_feature`

## 结果

未修正 A：缺OCT Macro-F1 63.155%、AUROC 66.189%、NLL 0.65541、Brier 0.46287；缺CFP Macro-F1 56.878%、AUROC 58.962%、NLL 0.69046、Brier 0.49589。

Macro-F1 favorable improvement：
- PCA / 缺OCT：+1.182pp；simultaneous95% [-3.145,+5.510]pp
- RRR / 缺OCT：0；零方差恒等，simultaneous95% 未定义
- PCA / 缺CFP：+1.213pp；simultaneous95% [-3.928,+6.354]pp
- RRR / 缺CFP：+1.568pp；simultaneous95% [-3.989,+7.124]pp

Macro-F1 四项都没有稳定的 simultaneously-positive 结论。

### 关键反例：F1略升，但NLL显著变差

PCA / 缺OCT：Macro-F1点估计 +1.182pp；但 NLL favorable improvement = **-0.04582**，ordinary95% [-0.08402,-0.00894]，simultaneous95% **[-0.09076,-0.00087]**，Holm p=0.0496。favorable正值代表LOOK更好，因此整个区间为负说明概率质量变差。必须与F1点增益同时报告。

## 审计边界

- 296名development同序；四指标；10,000次participant-paired bootstrap；四主对比 simultaneous family + Holm。
- development同时用于A/tree选择与效果估计，区间是条件/探索性；只有一个训练seed。
- 右侧独立审计自行重算四指标与10k bootstrap并逐项对齐报告；audit SHA `9bc62aba2e2829126edf9b8e6e4e692225c768819200c5eace907a822caa3b1c`。
- delivery accepted SHA `edbd3506d3f4381d1ee04068ecc1416b590e29af9ba30c661751d4905c8c9dff`；results SHA `6614a480d4f288f9b926708ce26b7aa94c1c8eb68e08e2164d79d9e0a1960474`；中文报告 SHA `7b50edbbdc615645fb0cc7bbd23a106238c85364560c7e2dac79283674a5c207`。
- 当前仍待左侧独立科学接受，不提前写 accepted science。
