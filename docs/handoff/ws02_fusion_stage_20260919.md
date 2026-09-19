# WS02 R18 fusion-stage v1：有限宿主融合位置机制包

任务：`look-ws02-fusion-stage-20260919-v1`

状态：**右侧完整执行与独立审计已完成；等待左侧从WS02原始产物独立科研验收。当前不表示左侧已接受。**

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
- 两个新spec固定50 GiB artifact-volume reserve；GPU/RAM/workspace预算沿用accepted deep。
- 每个新宿主先做host full296dev/两更新恢复profile，再正式host与PCA；随后进入隔离的 `fit_profile`，完整跑两方法×两缺失的正收益树与296dev replay，并在同目录第二次调用验证selection/bank/replay/feature-cost SHA稳定。
- `fit_profile`只用于准入，正式两方法从自己的独立目录重新运行，不复用profile结果。
- profile全部通过后才运行两方法正式完整树、预测replay、统计、delivery。
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


## 2026-09-19 00:38 UTC：fusion-stage host结构元数据故障与同run修复

现场状态：原 fusion sequence 的 middle/features 均为 `needs_review`，无活跃 sequence manager / cohort_delivery worker，GPU0/GPU1均空闲。旧失败不代表训练未完成：

- middle run `2026_09_19_02_40_42_281912_middle`：host accepted，best epoch 9 / stop epoch 24，240 updates；best/last/history/dev prediction SHA 与 accepted receipt逐项一致。
- features run `2026_09_19_02_40_42_281912_features`：host accepted，best epoch 30 / stop epoch 45，450 updates；同样全部SHA一致。
- 两run的 PCA、fit_profile、residual_rrr、pca_free_mean 均尚未开始，故不存在半完成拟合缓存。
- 旧 failure 发生在 host 已完成后、进入PCA前：runtime structure字典遗漏 `position` 字段，而预登记 `host_structure()` 含 `position`，完整dict比较必然失败。
- 独立逐字段重算 middle/features 的 fusion endpoint、architecture_id、correction_sites、总/可训练参数量均与spec完全一致，除旧runtime字典缺 `position` 外没有科学结构差异。

修复原则：不修改旧 scientific spec/source_commit `7876dc85388336acb1b9032ff50b9e0b1db66c28`，不重训已完成host。新管理修复使用共享 `runtime_host_structure(graph, position)` 生成预登记与runtime结构，避免双实现再次漂移；同run恢复必须有绑定原spec SHA、原pipeline failure SHA和repair packet SHA的一次性 `repair_resume.json`。默认 `needs_review` 仍禁止自动重试；若修复后再次失败，旧marker因pipeline SHA变化自动失效。

### 完整拟合资源门槛与重复成本修复

原 `fit_profile` 会完整执行两方法×两缺失的四棵正收益树，再由formal阶段重新拟合同样四棵树，工程上重复且不增加科学辨识。修复后：

1. `fit_profile` 仍在**完整1264 train / 296 dev、同accepted host、同PCA、同算法/identity**上跑四棵完整树，不降样本、不缩树、不降低RAM/GPU/50GiB磁盘reserve。
2. 每个 profile correction tree 记录**递归全文件 manifest/SHA**；同一目录第二次调用验证恢复后，manifest与final必须字节/数值稳定。
3. formal arm 不再调用第二遍 `fit_family_trajectory`；它把 profile correction tree 在同一run内原子迁移到正式 `arm/corrections/<pattern>`，逐文件核manifest，然后重新加载bank并做完整296dev预测、序列化replay、未修正基线和正式accepted receipt。
4. 正式 receipt 绑定 profile receipt SHA 与每个 promotion receipt SHA，标记 `no_refit=true`；任一缓存缺失、额外文件、SHA变化、profile身份变化或正式预测变化都会拒绝。
5. 已accepted formal arm在sequence恢复时先完整核预测与迁移receipt，再精确复用，不重复正式重放。

因此资源上界和恢复证据仍来自全量完整四树，正式科学计算只拟合一次；变化仅是缓存角色迁移和恢复编排，不改变算法、树搜索、样本、停止、缺失模拟或指标定义。

老师外部A/B/C匹配缺口继续保留；fusion-stage不能替代这些问题。


## 2026-09-19 01:30UTC repair：fit-profile恢复与缓存成本

两个新host已完成且checkpoint冻结：middle best9/stop24/240 updates，features best30/stop45/450 updates；host runtime结构与预登记结构逐字段一致。原结构故障仅为管理代码构造receipt时漏了 `position`，不是宿主图或训练身份变化。

第二故障发生在fit-profile恢复：原实现把完整trajectory第二次调用后的**整个目录字节manifest**当科学等价条件。目标环境诊断证明这不成立：
- `feature_costs.json` 包含full/missing forward、completed cache hit等运行计数，会因恢复读取而变化；
- 更重要的是，首次长程搜索保存的development evidence在fresh accepted host上可出现边界样本变化。当前fresh-host重放本身是确定的，但features residual/OCT旧selection与fresh replay相差一个参与者，且fresh最佳path改变；middle fresh最佳path也改变。
- 因此不能通过放宽SHA把旧selection直接升格为正式结果。

repair v3采用：
1. `profile/fitting` 原始树、moments、artifacts、selection永久保留不改。
2. 新建 `profile/revalidated`：对已有raw tree用hardlink复制immutable moments/artifacts，清除旧selection/progress authority，fresh-host重算所有已缓存候选的296dev evidence；随后调用**原未修改的positive-forward-tree算法**。已有prefix/candidate直接复用；只有fresh strict-positive拓扑真正需要、而旧树没有的下游分支才补算缺失moments/artifacts。
3. 对尚未开始的另外树，直接在revalidated目录完整拟合一次；不存在“profile完整四树后formal再拟合四树”。
4. 每棵revalidated tree完成后，`fit_family_trajectory`内部序列化bank replay必须exact；随后重新加载fresh accepted host再独立296dev重放，values SHA与metrics必须逐值一致。
5. 原raw tree与revalidated tree分别保存完整manifest；科学receipt另锁selected path、bank fingerprint、moment payload fingerprint、prediction values SHA、participant order。
6. formal阶段**不物理搬目录**，只逻辑引用同run `profile/revalidated/<arm>/<pattern>`，加载bank做296dev replay与host baseline，writer必须写 `look_fusion_profile_migration_v2`、`no_refit=true`、`no_physical_relocation=true`，verifier逐SHA复核revalidation receipt/selection/bank/predictions。
7. repair resume需新的0130 one-shot marker；sequence与每个run必须持有同字节management overlay receipt。overlay receipt pin当前全部`src/look`，并从Git历史逐字节证明native_host/observed_host/family_greedy/positive_forward_tree/family_statistics/operator/evaluator/observed_pair与scientific source `7876dc8`相同。

这套修复不改变数据、算法、搜索准则、候选、停止、test规则或已完成host；只把恢复/缓存管理从“整个目录字节相同”改为“原始证据永久保留＋fresh科学证据严格重验证＋运行计数显式非科学字段”。

## 2026-09-19 实际执行与右侧审计完成

实际执行身份：

- deep严格引用run 2026_09_18_11_18_28_650020，不重训。
- middle run 2026_09_19_02_40_42_281912_middle，host best/stop=9/24，240 updates。
- features run 2026_09_19_02_40_42_281912_features，host best/stop=30/45，450 updates。
- scientific source 7876dc85388336acb1b9032ff50b9e0b1db66c28。
- 最终management/publication overlay a065da430fc280b259db39de8994705aa48b147f，数值模块与scientific source逐字节相同。
- sequence completion SHA e45c3ec119ee5fcdb2e55b44a7cd161de3b2d0ef641cdfe8d48a441ae8e14b2a。
- 三宿主累计publication SHA 7354806d6e912ba2e9b2871c45f8a8d4989ea20b425dc2f470e6ad63c1c459ad。
- 右侧独立final audit SHA 291051a82c4b13f918c0c60afd5f71da3d94710233bcc6cdeab0a3a41192f57d。

### 恢复与缓存验收

两个新host训练完成后没有重训。正式下游前，8棵method×pattern tree均通过：

- 正式worker环境固定CUBLAS_WORKSPACE_CONFIG=:4096:8、deterministic algorithms、matmul/cuDNN TF32关闭、threads=2。
- accepted host完整state_dict、40个BN模块、模式、hooks、requires_grad前后一致。
- source raw manifest前后逐文件一致。
- complete source tree的全部prefix/candidate/selected path/best score/metrics/prediction values/moment payload/artifact SHA与revalidated tree科学投影完全相同。
- 原先缺失的tree只计算缺失分支。
- 296人final bank fresh replay精确。
- JSON元数据使用独立inode；不可变tensor/prediction可hardlink；两阶段绝对路径迁移后所有引用必须落在revalidated tree内。
- formal accepted writer记录profile_migration和no_refit=true，正式阶段仅重放/基线/交付，不第二次拟合同一棵tree。

review evidence index SHA：9435a4d0bdd3c547895edacd22a347e188cd06604d6d6906c5e8680ca55fa67b。该归档明确是左侧审查后可重复生成的review evidence，不伪装成历史自动日志。

### 每宿主主要结果（Macro-F1）

|融合阶段|缺失|host|PCA+树|RRR+树|
|---|---|---:|---:|---:|
|deep|缺OCT|62.34%|69.24%|66.77%|
|deep|缺CFP|56.72%|60.02%|61.37%|
|middle|缺OCT|33.33%|63.67%|65.67%|
|middle|缺CFP|40.59%|56.08%|56.41%|
|features|缺OCT|57.49%|69.16%|68.39%|
|features|缺CFP|52.77%|61.82%|62.50%|

middle未修正宿主明显弱于deep/features，且参数量只有11.89M；所以middle出现更大的LOOK收益不能单独解释成早融合优于深融合。

### 8项预登记跨宿主收益差

定义仍为 (新宿主 method-host) - (deep method-host)，单位pp：

|新宿主|方法|缺失|点差|8项同时95%|
|---|---|---|---:|---|
|middle|RRR|缺OCT|+27.91|[+18.96,+36.86]|
|middle|RRR|缺CFP|+11.17|[-0.34,+22.69]|
|middle|PCA|缺OCT|+23.44|[+13.70,+33.18]|
|middle|PCA|缺CFP|+12.20|[+0.44,+23.95]|
|features|RRR|缺OCT|+6.48|[-3.40,+16.35]|
|features|RRR|缺CFP|+5.08|[-4.06,+14.21]|
|features|PCA|缺OCT|+4.77|[-4.45,+13.99]|
|features|PCA|缺CFP|+5.75|[-3.20,+14.71]|

middle有3项同时区间不跨0、1项略跨0；features四项同时区间均跨0。不能据此排名最佳融合阶段：不同宿主参数量、拓扑、融合算子和训练权重不同，且只有单seed、同一dev参与选模。

### 反例与边界

- middle host缺OCT F1只有33.33%，LOOK后升至63到66%；大增益同时意味着基线修正空间更大。
- probability quality不与F1统一：deep缺OCT PCA F1更高，但NLL从host 1.0803恶化到1.6217；RRR NLL更差到2.5447。
- features与deep参数量相同（22.88M）但融合算子和拓扑不同；其4项跨宿主增益差同时区间均跨0，是重要反例。
- 本机制只回答宿主融合阶段与LOOK收益关系；老师外部A/B/C与A/B/C+LOOK仍是独立未完成缺口。
- test始终封存；participant bootstrap不包含训练随机性或独立test泛化。

公开累计入口见 docs/reports/current/fusion_stage/README.md。当前仅为右侧完整执行/审计，等待左侧最终独立科研验收。
