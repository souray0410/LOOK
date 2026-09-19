# LOOK：缺失模态方法 A 的一手证据盘点（不含训练）

## 纠正后的老师问题

LOOK 的主问题是：一个已经合理训练、明确用于处理整模态缺失的策略 A，在完全相同的 A 上再加 LOOK 是否仍有增量价值？

- MMTM、融合位置与固定均值/前缀结果继续保留，但只回答宿主兼容性/组装或 LOOK 机制，不计作缺失方法 A/B/C 完成。
- Stage A 概率诊断只解释既有预测的 F1/NLL/Brier 反例，不替代缺失方法比较。
- GAN、微调只是类别例子，不是必跑清单。候选按一手论文、作者实现、适配保真和成本筛选。

## A vs A+LOOK 的共同公平合同

- same trained A checkpoint
- same preprocessing/missing-input rules
- same train/dev participants, seed3416 and sealed test
- same complete-input reference from A
- same F1/AUROC/NLL/Brier evaluator
- A+LOOK does not retrain A
- preserve A core missing-modality modules/losses

推理安全：no absent modality or target label at inference
比较边界：within-method A vs same A+LOOK is primary; cross-method absolute ranking is secondary

## 有限候选

|候选|一手缺失机制 / 作者任务|代码身份与许可|当前适配差异|成本 / 处置|
|---|---|---|---|---|
|[ShaSpec](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.html)|shared/specific features + alignment/domain auxiliary tasks + unavailable-feature generation/residual fusion；paper covers missing-modality classification and segmentation; pinned repo exposes BraTS 3D segmentation|[57a04fa8](https://github.com/billhhh/ShaSpec)；no explicit repo license located|classification implementation not found in pinned repo；symmetric CFP/OCT missing training requires protocol；2D adapter/core-loss/LOOK-node contract required；license route unresolved|medium-high；best scientific-role match|
|[U-HVED](https://arxiv.org/abs/1907.11150)|variational shared latent from observed subsets; modality completion + segmentation；3D BraTS segmentation/completion; TensorFlow 1.12/NiftyNet|[1d293a87](https://github.com/ReubenDo/U-HVED)；MIT|3D to 2D classification adaptation；legacy framework；new classifier/reference contract|high；generative/completion representative|
|[RFNet](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.html)|region-aware available-modality fusion + per-modality segmentation regularization；3D BraTS incomplete segmentation|[1030aebf](https://github.com/dyh127/RFNet)；no explicit repo license located|region/segmentation semantics do not map directly to eye classification；license unresolved|high；medical robust-fusion candidate|
|[mmFormer](https://arxiv.org/abs/2206.02425)|modality-specific encoders + intra/inter-modal transformers + explicit mask + auxiliary regularizers；3D BraTS incomplete segmentation|[b18663ad](https://github.com/YaoZhang93/mmFormer)；Apache-2.0|large architecture/capacity confound；new 2D classification transformer required|high；license/source-clear robust-fusion candidate|
|[Missing-Aware Prompts](https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_With_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html)|missing-type-aware prompts in ViLT; explicit missing ratio/type；image-text visual classification|[9aa53815](https://github.com/YiLunLee/missing_aware_prompts)；no explicit repo license located|image-text to CFP/OCT redesign；new backbone/pretraining identity；license unresolved|high；classification/missing-aware but modality mismatch|

### 固定作者代码身份

|候选|固定 commit|关键文件 SHA256|
|---|---|---|
|ShaSpec|57a04fa88331ea3b2671795a15a1ae9a24fd310d|DualNet_SS.py 421afdb5dcfd6fbf020311c9056d07873da8fce73214b1ad064330bfaf72ad17<br>train_SS.py 6d4f07027f7d28f7abdc965d17a2446bbddc52384345bb602f4c245d4e9ce4e6<br>README.md 228d35bce5d821f45b4b1d9357d3487388503f2dc6cb59c9827d5239b82108aa|
|U-HVED|1d293a87d496c52adafd69dbcea37767ac6a7ee8|extensions/u_hved/u_hved_net.py afd9b210354118496c9291b167b562580cd905625a3d2fdfd80b5744d925a4ea<br>inference.py 99acf00657db39ec3ff49922f0c9da7cca37a98f95734d97bfb031ec5494c38c<br>README.md 75d70af10d98d3b2f3c41b60e7f277514099f2b269d139fdd6ce266cffbbb3fb|
|RFNet|1030aebf7036c1518065d56d2b99421715b014e7|models.py 14b8c8a698a6ce8cc03300028395f30f1c18865535134d221a5250078cb85536<br>train.py 8d1f7868059c037980ccd190e6e8b5f27db046570e4636cafd855c3bba2d8732<br>README.md 00cecb1852425c543cb52234738feaff126ddf0e1a2de2db680ff917d8dd5953|
|mmFormer|b18663adbfd0323656083e0470420c8c715c1782|mmformer/mmformer.py 5ec1482c8551a375d2bb13dc2a4987077b31a6182aacf0810ca32c2094b8a30b<br>mmformer/train.py 63991bb630e55d91ddd479664bc8860220f8518cdfa2f416644ae1e555dd742d<br>README.md 53e6630a02bac1c5b3eb329325e5f5cbb41bc3f6a80cc35f9702782603673650|
|Missing-Aware Prompts|9aa538159f4623de1936d25daee880ec47c062be|vilt/modules/vilt_missing_aware_prompt_module.py 384fb8cb2361e0c3a71f4c8a3a5c8a2a4265ac608506710a01fec1899f9d6fac<br>vilt/config.py c4803d8c69d3cb343d577f01e217de083d6fdd08e27d1cd95263d06346a13162<br>README.md 9e0116ae3929b7830b2fd43e270fab0527a2be2b67bed29f0821e92d6d73ae7a|

额外筛选 M2FTrans（MIT，commit 03a0b28ae03ad1a7dea5fb874c7b47c1439caa34）：它确实处理 incomplete multimodal，但与 mmFormer/RFNet 同属高成本3D鲁棒融合家族；首包再加入它不会提供足够新的辨识信息，因此先不扩方法数量。

## 现有资产复用边界

- 可复用：1264 train / 296 dev、seed3416、test封存。
- 可复用：CFP/OCT 2D预处理、participant顺序与标签。
- 可复用：LOOK拟合/搜索/重放、四指标和配对评价。
- Stage A 概率诊断只作为描述性基线。
- 不能冒充缺失方法完成：MMTM兼容包、fusion-stage、fixed-mean/fixed-prefix。
- 现有R18/R50/MMTM权重不能自动作为新A权重。

## 建议的有限首包（只建议，不自动训练）

建议左侧优先评估 ShaSpec 是否能作为首个缺失方法 A。原因不是预期分数，而是其主论文明确包含 missing-modality classification，同时也覆盖医学分割，科学角色最贴合纠正后的老师问题。

仍需左侧裁决：分类实现未在官方repo快照中找到；repo许可未明；对称CFP/OCT缺失训练策略需锁定；2D分类适配、核心loss和合法LOOK节点需锁定。

若批准，最小匹配包：1) 固定paper-faithful 2D A；2) A只训练一次；3) 冻结A；4) 同一A比较A vs A+LOOK两缺失状态；5) 同时报F1/AUROC/NLL/Brier和配对证据；6) A训练成本与LOOK拟合成本分开。

为什么暂不先选其他：U-HVED是高成本3D生成补全；RFNet的region/分割语义过强且许可未明；mmFormer许可清楚但引入大Transformer宿主；Missing-Aware Prompts是image-text ViLT到双医学图像的较大重设计；M2FTrans与3D robust-transformer家族首包信息重复。

## 仍未解决

- 选择哪个A、2D适配、训练/停止预算和合法LOOK节点是新科学协议；本盘点不授权训练。
- 精确GPU/墙钟成本只能在候选协议锁定后做目标环境profile。
- LOOK缺失方法主比较仍未完成；证据盘点完成不能冒充实验完成。
