# EmbraceNet + LOOK 集成说明：独立接受、发布纠错与冻结写回机制诊断

## 科学状态

左侧已对原单种子包做独立审计并接受：385 个文件哈希、203 个源码 pin、86 个预测文件/296 个有序参与者、四棵树 16 个前缀/73 个候选、四指标 10,000 次配对统计均重算通过。原 result SHA 8265c80f...443d0c 保持冻结。

发布时修正一个统计措辞：RRR / 缺 CFP 的 positive-forward tree 为空，方法预测与 A 完全相同；所有 bootstrap 差值都为 0，标准差为 0，故 simultaneous_95=null。准确表述是：三个非退化 simultaneous 95% 区间均包含 0；第四个是零方差恒等对比，标准化 simultaneous 区间未定义。这不能解释成等效性检验。

## 节点语义

- joint_input：双模态输入联合节点，位于 OCT/CFP 各自进入 ResNet18 编码器之前。
- embraced_feature：EmbraceNet 根据 availability / selection probability 做逐坐标选择之后的融合特征。

## 冻结写回分支诊断

诊断身份：冻结 A + 冻结已选 PCA artifact + 原 296 development；不训练、不拟合、不搜索。WS02 receipt SHA：
7d903e298b4bdc36c630e470d7524634b70ff875feb2894f02741388a3235921。

### 缺 OCT

- A：Macro-F1 66.230%。
- 完整 PCA 写回：67.594%。
- 仅写可用 CFP 分支：与完整 PCA 写回逐项相同。
- 仅写缺失 OCT 分支：与 A 逐项相同，logit delta=0。
- 完整/可用分支相对 A 最大绝对 logit 变化 0.13749。

### 缺 CFP

- A：Macro-F1 56.341%。
- 完整 PCA 写回：诊断重放 57.182%。该后验诊断使用冻结 artifact 的分支限定重放；主比较仍以已独立接受的 57.471% 原预测为准。
- 仅写可用 OCT 分支：与该诊断完整写回逐项相同。
- 仅写缺失 CFP 分支：与 A 逐项相同，logit delta=0。
- 完整/可用分支相对 A 最大绝对 logit 变化 0.13281。

该差异不修改原主比较：诊断的目的只是分解已算修正的分支贡献，而不是重新定义正式预测或重新选 artifact。

## 机制解释边界

PCA 的 joint_input 收益在这两个单缺失场景都完全由仍可用分支的表征校正贡献。缺失分支虽然在 joint tensor 中有可计算修正量，但 availability 安全语义会在编码前把缺失原输入归零并在 EmbraceNet 选择时屏蔽该模态；只写缺失分支不会改变最终 logits。因此不能称 LOOK 在这里“生成/补全了缺失模态”。

RRR / 缺 OCT 选择 embraced_feature，属于 EmbraceNet 选择之后的融合特征校正；RRR / 缺 CFP 是空路径恒等结果。

## 限制

该机制诊断是接受主结果后的后验解释，使用同一 development，不承担新的确认性统计结论。test 仍封存；没有新训练、拟合、搜索或新 seed。
