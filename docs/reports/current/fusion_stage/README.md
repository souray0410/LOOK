# LOOK：R18宿主融合阶段机制比较

同一小队列、同一ResNet18公开ImageNet初始化类别与完全相同训练/停止规则，只改变原宿主融合发生在Stage2后（middle）、Stage4后（deep）或每眼向量（features）。deep严格引用既有接受run，不重训；middle/features是两个新完整宿主。

这不是“LOOK从哪个位置开始修正”的消融：融合位置改变了宿主拓扑、参数量、训练权重与融合算子，LOOK仍在每个真实宿主自己的合法correction_sites上独立搜索。

|宿主融合阶段|执行身份|参数量|实际LOOK correction sites|状态|
|---|---|---:|---|---|
|deep|2026_09_18_11_18_28_650020|22879362|joint_input → joint_stem → joint_stage1 → joint_stage2 → joint_stage3 → joint_stage4 → fusion_stage4 → fusion_features → fusion_participant_feature|accepted_reference|
|middle|2026_09_19_02_40_42_281912_middle|11893634|joint_input → joint_stem → joint_stage1 → joint_stage2 → fusion_stage2 → fusion_stage3 → fusion_stage4 → fusion_features → fusion_participant_feature|accepted|
|features|2026_09_19_02_40_42_281912_features|22879362|joint_input → joint_stem → joint_stage1 → joint_stage2 → joint_stage3 → joint_stage4 → joint_features → fusion_features → fusion_participant_feature|accepted|

## 每个宿主内部结果

|融合阶段|缺失状态|不修正F1|PCA＋树F1|残差＋树F1|残差−PCA(pp)|
|---|---|---:|---:|---:|---:|
|deep|缺OCT（仅CFP）|62.34%|69.24%|66.77%|-2.47|
|deep|缺CFP（仅OCT）|56.72%|60.02%|61.37%|+1.36|
|middle|缺OCT（仅CFP）|33.33%|63.67%|65.67%|+2.00|
|middle|缺CFP（仅OCT）|40.59%|56.08%|56.41%|+0.33|
|features|缺OCT（仅CFP）|57.49%|69.16%|68.39%|-0.76|
|features|缺CFP（仅OCT）|52.77%|61.82%|62.50%|+0.68|

## 跨宿主：相对各自不修正基线的LOOK收益差

差值定义为“新宿主的(method−host)收益 − deep的(method−host)收益”，单位百分点。8项在开跑前固定，并共用一次296人参与者级10,000次bootstrap；同时区间覆盖这8项。跨零不代表等效。

|新融合阶段|方法|缺失状态|收益差(pp)|普通95%|8项同时95%|
|---|---|---|---:|---|---|
|middle|自由低秩残差|缺OCT（仅CFP）|+27.91|[+21.27, +34.43]|[+18.96, +36.86]|
|middle|自由低秩残差|缺CFP（仅OCT）|+11.17|[+2.62, +19.48]|[-0.34, +22.69]|
|middle|PCA方向约束|缺OCT（仅CFP）|+23.44|[+16.10, +30.41]|[+13.70, +33.18]|
|middle|PCA方向约束|缺CFP（仅OCT）|+12.20|[+3.52, +20.75]|[+0.44, +23.95]|
|features|自由低秩残差|缺OCT（仅CFP）|+6.48|[-0.65, +13.76]|[-3.40, +16.35]|
|features|自由低秩残差|缺CFP（仅OCT）|+5.08|[-1.54, +11.72]|[-4.06, +14.21]|
|features|PCA方向约束|缺OCT（仅CFP）|+4.77|[-2.03, +11.54]|[-4.45, +13.99]|
|features|PCA方向约束|缺CFP（仅OCT）|+5.75|[-0.88, +12.31]|[-3.20, +14.71]|

## 解释边界

- 单seed且同一dev参与宿主/路径选择；bootstrap不包含训练随机性或test泛化。
- middle/deep/features的参数量、融合算子和训练权重不同；结果不能解释成单一“融合层编号”因果效应。
- 本机制回答宿主融合阶段与LOOK收益是否一致，不替代老师外部A/B/C方法匹配缺口。
- 概率质量、收益下降、负结果和跨零结果全部保留；不按F1点估计删正式对照。

各宿主完整F1/AUROC/NLL/Brier、路径和缓存成本见configs子页；受限预测、权重和参与者资产不上GitHub。
