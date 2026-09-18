# 同一个位置、还没开始走树：两种拟合有何不同？

复用已验收的第一轮候选，不新增训练或GPU拟合。固定同一宿主、空修正前缀、位置与秩，比较PCA与自由残差各自拟合后的F1。
这排除了此前修正路径不同的影响，但λ仍按各方法自身的既定规则选取，不能叫“固定λ/斜率”的机制消融。负候选也完整展示；这里是强制启用候选，不是最终关闭开关后的性能。

|骨干|缺失|位置|不修正|PCA|残差|残差−PCA(pp)|
|---|---|---|---:|---:|---:|---:|
|resnet50|oct_missing|joint_input|37.51|66.19|66.28|+0.08|
|resnet50|oct_missing|joint_stem|37.51|59.66|62.83|+3.18|
|resnet50|oct_missing|joint_stage1|37.51|66.49|65.53|-0.96|
|resnet50|oct_missing|joint_stage2|37.51|44.66|54.44|+9.78|
|resnet50|oct_missing|joint_stage3|37.51|38.89|46.41|+7.52|
|resnet50|oct_missing|joint_stage4|37.51|44.24|55.48|+11.24|
|resnet50|oct_missing|fusion_stage4|37.51|47.95|46.46|-1.50|
|resnet50|oct_missing|fusion_features|37.51|67.57|69.90|+2.34|
|resnet50|oct_missing|fusion_participant_feature|37.51|66.89|68.58|+1.69|
|resnet50|cfp_missing|joint_input|55.43|57.17|56.98|-0.20|
|resnet50|cfp_missing|joint_stem|55.43|43.02|43.02|+0.00|
|resnet50|cfp_missing|joint_stage1|55.43|55.04|51.67|-3.36|
|resnet50|cfp_missing|joint_stage2|55.43|53.80|56.20|+2.40|
|resnet50|cfp_missing|joint_stage3|55.43|51.77|51.07|-0.70|
|resnet50|cfp_missing|joint_stage4|55.43|54.73|48.65|-6.08|
|resnet50|cfp_missing|fusion_stage4|55.43|58.41|57.42|-0.98|
|resnet50|cfp_missing|fusion_features|55.43|59.29|58.04|-1.25|
|resnet50|cfp_missing|fusion_participant_feature|55.43|57.99|60.91|+2.93|
|resnet18|oct_missing|joint_input|62.34|39.47|42.24|+2.77|
|resnet18|oct_missing|joint_stem|62.34|59.86|57.20|-2.66|
|resnet18|oct_missing|joint_stage1|62.34|63.14|62.63|-0.51|
|resnet18|oct_missing|joint_stage2|62.34|64.82|64.27|-0.54|
|resnet18|oct_missing|joint_stage3|62.34|62.40|64.02|+1.63|
|resnet18|oct_missing|joint_stage4|62.34|61.83|64.02|+2.19|
|resnet18|oct_missing|fusion_stage4|62.34|66.38|64.00|-2.39|
|resnet18|oct_missing|fusion_features|62.34|64.84|66.21|+1.38|
|resnet18|oct_missing|fusion_participant_feature|62.34|67.91|64.86|-3.04|
|resnet18|cfp_missing|joint_input|56.72|43.22|44.11|+0.89|
|resnet18|cfp_missing|joint_stem|56.72|56.76|55.91|-0.85|
|resnet18|cfp_missing|joint_stage1|56.72|56.24|57.83|+1.59|
|resnet18|cfp_missing|joint_stage2|56.72|48.90|50.64|+1.74|
|resnet18|cfp_missing|joint_stage3|56.72|57.78|58.36|+0.57|
|resnet18|cfp_missing|joint_stage4|56.72|59.41|59.46|+0.05|
|resnet18|cfp_missing|fusion_stage4|56.72|57.99|59.46|+1.47|
|resnet18|cfp_missing|fusion_features|56.72|58.91|57.31|-1.61|
|resnet18|cfp_missing|fusion_participant_feature|56.72|58.27|58.63|+0.36|

各单元只描述同一个已看过的dev，不新增确认性显著性结论。原树最终结果仍见[累计入口](../small_cohort/README.md)。

## 这次排除了什么解释

ResNet18缺OCT时，同一空前缀下PCA的最佳单点为67.91%，残差为66.21%；两棵树最终为69.24%与66.77%。差异在树尚未分叉时已经存在，所以不能完全归因于后续路径不同。ResNet50同场景的最佳单点则为PCA67.57%、残差69.90%，完整树为69.58%与69.90%。这支持继续研究“宿主特征与拟合方式的相互作用”，尚不能证明某种拟合普遍更好。

注意两个“最佳单点”可能位于不同位置；严格同位置比较见上面36项原始表，不能把最佳单点差异叫成单因素机制证明。空前缀原始指标/位置完整复用，无新增训练、无test访问。
