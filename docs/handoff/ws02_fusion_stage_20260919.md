# WS02 R18 fusion-stage v1：有限宿主融合位置机制包

任务：`look-ws02-fusion-stage-20260919-v1`

状态：**开跑前协议与实现合同已锁定；本文件本身不表示GPU结果或科研接受。**

## 问题

保持数据、ResNet18骨干类别、公开ImageNet初始化、训练规则、LOOK方法、搜索规则、factor/rank和缺失模拟不变，只改变**原宿主两路模态何时融合**：

- `middle`：两路各自完成Stage2后融合，之后共享Stage3/4/features。
- `deep`：两路各自完成Stage4后融合，再产生每眼features。
- `features`：两路各自产生每眼features后再做向量融合。

这不是“LOOK从哪个位置开始修正”的消融。每个宿主都使用其实际MHD图的完整合法correction sites重新做完整正收益树。

## 锁定科学范围

- UKB旧小队列青光眼：1264 train / 296 development；test封存。
- ResNet18；seed3416。
- 224×224 CFP/OCT二维输入。
- public ImageNet fresh-host初始化。
- factor=16、rank=32。
- `residual_rrr` 与 `pca_free_mean`。
- `positive_forward_tree` 完整搜索；两种缺失：oct_missing / cfp_missing。
- 原training dict、BN、batch、optimizer、warmup/early-stop、FP32和确定性规则均从accepted deep spec精确复用。
- 不新增种子、rank、方法、外部宿主或test访问。
- 不启Fast/priority；不新建调度器。

## deep严格引用

deep不重训，精确引用accepted run：

- run：`2026_09_18_11_18_28_650020`
- spec SHA256：`f09a452f8584861ff5291cb262894e16f4ba4c7c962f43f6ba4cfbccb18929e8`
- host identity：`583b7902416a7e8bcec39e55655ad09932e77b1fc25119cff6872f59a524c66b`
- delivery SHA256：`105f0d94c9020e9b008f46c4f24543878c63b7461fb1c350bbbd3c2d3706eb52`
- framework：`c0a27abb3e0f2153bfd273b1d05d5b7dae9784f0`
- public ImageNet init SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

旧deep scientific source与当前实现字节不同已显式审计：`observed_host.py`无差异；`native_host.py`仅新增可选MMTM分支，mmtm=None时普通deep路径不变；`cohort_delivery.py`新增MMTM入口、磁盘reserve和pipeline恢复记录，不改变普通宿主训练/停止算法。故deep作为原accepted结果引用，不用当前源码“重放后冒充原运行”。

## 开跑前固定的8项跨宿主比较

每项定义为：

`(新宿主 method - 新宿主 host) - (deep method - deep host)`

固定比较族：

1. middle / residual_rrr / oct_missing
2. middle / residual_rrr / cfp_missing
3. middle / pca_free_mean / oct_missing
4. middle / pca_free_mean / cfp_missing
5. features / residual_rrr / oct_missing
6. features / residual_rrr / cfp_missing
7. features / pca_free_mean / oct_missing
8. features / pca_free_mean / cfp_missing

只有deep/middle/features三宿主都完整接受且296人ID/标签顺序完全一致时，才用一次共同参与者级10,000次bootstrap生成普通95%与8项同时95%区间。中途不按局部点估计排名。跨零不表示等效。

## 开跑前宿主结构签名

CPU-only图结构合同由当前MHD pin构造，不读训练数据、不加载预训练权重：

|position|fusion endpoint|总/可训练参数|LOOK correction sites|
|---|---|---:|---|
|middle|stage2|11,893,634|joint_input → joint_stem → joint_stage1 → joint_stage2 → fusion_stage2 → fusion_stage3 → fusion_stage4 → fusion_features → fusion_participant_feature|
|deep|stage4|22,879,362|joint_input → joint_stem → joint_stage1 → joint_stage2 → joint_stage3 → joint_stage4 → fusion_stage4 → fusion_features → fusion_participant_feature|
|features|features|22,879,362|joint_input → joint_stem → joint_stage1 → joint_stage2 → joint_stage3 → joint_stage4 → joint_features → fusion_features → fusion_participant_feature|

重要替代解释：middle参数量明显更少；deep与features参数量相同但融合算子/拓扑不同。因此跨宿主结果不能被简化为单一“层号”因果效应。

新middle/features的actual host receipt必须与这个预登记结构逐字段一致，否则formal拒绝。

## 执行与故障隔离

使用原 `cohort_sequence` / `cohort_delivery`：
- deep accepted reference只publish，不launch。
- 顺序固定：middle → features。
- GPU0共享既有LOOK设备锁；GPU1留给R&B。
- 每个新宿主完整profile、host、PCA、两方法完整树、预测replay、统计、delivery。
- 一个新宿主失败记needs_review后仍允许另一个继续收证据；不盲重试、不加种子。
- 三宿主未齐时累计fusion-stage publication保持incomplete，不产生8项收益差。

## 解释/发布边界

每宿主独立展示：
- F1、AUROC、NLL、Brier；
- host参数量与真实融合位置；
- 实际correction sites和最终LOOK路径；
- 缓存/feature forward成本；
- 负增益、概率质量反例及搜索成本。

该机制包只回答“宿主融合阶段变化时LOOK收益是否一致”。老师外部方法A/B/C与A/B/C+LOOK的匹配缺口仍然保留，不能用本包冒充已全部回答。

单seed、同dev用于宿主/路径选择；participant bootstrap不包含训练随机性或test泛化。最终科研接受仍需左侧独立核原始产物。
