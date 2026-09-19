# EmbraceNet graph 失败边界修复 v3：一次性 replay、可读公式与完整参考二阶矩合同

状态：**本机 CPU 工程修复完成，ready_for_left_review；scientific_acceptance=false；未训练、未读取真实参与者/train/dev/test、未用 GPU、未做远端操作。**

v1/v2 的冻结 result、ACK、left_audit 与既有证据文件原样保留。本 v3 不覆写 v2 冻结 Markdown；v2 中显示损坏的数学公式由本文件作为**可发布勘误/后继合同**给出。

## 1. 图入口 replay 泄漏的根因与统一边界

左侧真实探针证明：先公开调用 `set_embracenet_replay(graph, indices)`，随后图入口在到达 `EmbraceNetFusion.forward` 之前因“available raw input 为 NaN”等原因失败，模块内部的 replay 清理根本没有执行；下一次合法调用于是错误地消费旧 replay。

v3 将 replay 生命周期提升到**公开图入口边界**：

- `forward_embracenet_host`
- `forward_embracenet_with_look`

二者现在共同使用 `embracenet_graph_replay_scope(graph)`。

合同是：

1. 外部仍可以先用 `set_embracenet_replay` 预装一次 exact replay；API 没有被禁用。
2. 合法 full forward 到达 embracement 时，exact replay 正常被消费，trace 仍记录 `replayed=true`。
3. 任何图入口异常，包括 raw available-input NaN、artifact/stop 预校验失败、图内部失败，都会在入口退出时清除尚未消费的 replay。
4. 正常但提前停止在 `embraced_feature` 之前，同样清除尚未消费的 replay。
5. 下一次未请求 replay 的合法调用必须恢复为普通 stochastic forward，不能继承上次待用 replay。
6. 前置校验失败和 embracement 前正常停止不推进 EmbraceNet sampling RNG。
7. 这不是整图事务回滚：若训练图已经实际执行过 BN 或其他前缀副作用，不要求撤销；本合同只保证 one-shot replay 不泄漏。

结构化图级探针在当前 SHA 上通过：

- plain public entry + available NaN failure → 下一合法调用 `replayed=false`；
- study public entry + available NaN failure → 下一合法调用 `replayed=false`；
- study invalid artifact 预校验失败 → sampling RNG 不变，下一合法调用 `replayed=false`；
- study 正常 stop 于 `joint_stage1`（embrace 前）→ sampling RNG 不变，下一合法调用 `replayed=false`；
- plain/study 有效外部 exact replay → 本次 `replayed=true` 且 indices 精确一致；再下一次合法调用 `replayed=false`。

## 2. 当前 CPU 回归

复用 v2 已建立的本机隔离环境，不重复安装：

- Python 3.11.16
- torch 2.8.0
- torchvision 0.23.0
- pytest 8.4.1
- mhd-framework 4
- MHD source commit `c0a27abb3e0f2153bfd273b1d05d5b7dae9784f0`

当前 v3 源码：

- `src/look/models/embracenet.py`：`46e5c124c41c2b16a070d71e0f4c592ba4ee8caba66d89d9c65c1d25f1ea74f2`
- `src/look/runtime/embracenet_sampling.py`：`2f633c4d012f05915be8d413f29f9fcc5b855ce73b3ebbe7f8d4bf64e40b74e2`
- `src/look/studies/embracenet_adapter.py`：`aacdcdd790422a701218f16ab3e886e1220d17046580e28bf288c9c6dcb92e54`
- `tests/unit/models/test_embracenet.py`：`461db3991a0f9c6a91b6ae89cb8ce6f76b35463e98a43482fe02e19e2ef85785`

动态结果：

- EmbraceNet 当前专项：**24/24 pass**；
- native-host / MMTM / linear-vector 必要回归：**30/30 pass**；
- v3 真实图级 replay lifecycle probe：pass。

v3 没有修改 `EmbraceNetFusion` 作者数学实现本身。当前类体 SHA 与 v2 完全相同：

`aac72c592d691cad976a9e6286dcf3bfe3f031f4f585335eb92eda64c86bd497`

因此 v2 已重新核过的作者输出/梯度/RNG/下一 AdamW 恢复证据可以精确复用，不需要再重复作者检索或等价实验。

## 3. 可读的完整态一阶与二阶矩公式

对一个固定参与者，设 EmbraceNet docking 后第 m 个模态在 embraced 坐标 j 的值为 d_{m,j}，该模态的归一化选择概率为 p_m。

### 一阶矩

```text
μ_j = E[z_j] = Σ_m p_m d_{m,j}
μ   = E[z]
```

### 每个坐标的二阶矩与方差

```text
E[z_j²] = Σ_m p_m d_{m,j}²
v_j     = Var(z_j) = E[z_j²] - μ_j²
```

作者实现对不同 embraced 坐标做条件独立的 categorical draw，因此：

```text
E[z zᵀ] = μ μᵀ + diag(v)
```

这里的 `diag(v)` **不能**在“完整参考分布”的 PCA/协方差合同里丢掉。

## 4. 跨参与者完整参考分布：均值向量不等于完整二阶矩

对 N 个参与者，参与者 i 有条件均值 μ_i 与条件方差向量 v_i。

全体完整参考的总体均值：

```text
μ̄ = (1/N) Σ_i μ_i
```

完整随机参考分布的二阶矩：

```text
M2_full = (1/N) Σ_i [ μ_i μ_iᵀ + diag(v_i) ]
```

因此完整参考协方差应为：

```text
Σ_full = M2_full - μ̄ μ̄ᵀ
```

如果只把每个参与者替换成均值特征 μ_i，得到的是另一个对象：

```text
Σ_mean = (1/N) Σ_i [ μ_i μ_iᵀ ] - μ̄ μ̄ᵀ
```

二者差异恰好是：

```text
Σ_full - Σ_mean = (1/N) Σ_i diag(v_i)
```

这个差是半正定项，代表作者随机 embracement 的**参与者内条件抽样方差**。

### v3 明确推荐

若 `embraced_feature` 的 LOOK PCA 合同仍声称拟合的是**作者随机 complete-reference distribution**，则：

- PCA/标准化的二阶统计必须保留 `diag(v_i)`；
- 可用解析一阶/二阶矩流式累计，不需要用 K=32 近似；
- 不能只用 μ_i 做 PCA 然后仍称“完整随机参考分布”。

如果未来决定把 LOOK 的 complete reference 明确定义成“每个参与者的 integrated mean feature μ_i”：

- 那会把 Σ_full 改成 Σ_mean；
- PCA 子空间可能改变；
- 这是**新的科学目标选择**，必须单独由左侧确认，不能在 runner 中静默发生。

本 v3 推荐：**保留作者随机完整参考分布作为 embraced_feature 的 PCA 参考，用解析二阶矩精确实现。**

## 5. 为什么 complete 分类评估可以只用均值，但 PCA 不能自动只用均值

当前 eval classifier dropout=0 且为线性层：

```text
ℓ = Wz + b
```

所以：

```text
E[ℓ] = W E[z] + b = Wμ + b
```

因此如果 complete 分类的正式 estimand 是 v2 已提出的“integrated mean logits”，直接用 μ 得到 E[ℓ] 是精确的。

但是：

```text
softmax(E[ℓ]) ≠ E[softmax(ℓ)]
```

一般不相等。expected softmax、expected per-draw NLL/Brier 是另一类非线性 estimand，需要另行预注册。

同样，**分类头的一阶均值可精确积分**并不意味着 LOOK PCA 可以抛弃完整参考的二阶矩。二者用途不同：

- complete classifier integrated-mean-logit：只需要 E[z]；
- stochastic complete-reference PCA/covariance：需要 E[z] **和** E[z zᵀ]。

## 6. RRR/平方误差目标的边界

如果某个固定参与者的 complete stochastic target 是随机向量 Y，其条件均值为 μ_Y，对确定预测 f(X)：

```text
E[ ||Y - f(X)||² | X ]
= || μ_Y - f(X) ||² + tr(Cov(Y | X))
```

第二项与 f(X) 无关。因此在**投影/标准化/PCA 基已经固定**的条件下，平方误差回归的最优条件均值目标可以用 μ_Y。

但这不能反推“PCA 也只需 mean feature”。如果 PCA 合同针对完整随机参考分布，它仍必须先用 Σ_full。

## 7. single-missing 评估与 replay/RNG 合同不变

双模态单缺失时，归一化 availability 退化为唯一模态：

- missing_OCT → ([0,1])
- missing_CFP → ([1,0])

因此固定 checkpoint/LOOK prefix 下 embraced output 对 sampling seed 确定；v2 多 seed A 与非零 A+LOOK 最大差都为 0。

正式 single-missing 主比较仍是：

- 每状态一次正常作者式 forward；
- 正常消耗一次 sampling draw；
- A / A+LOOK 同状态可记录并配对 exact trace；
- dev evaluation 前后 snapshot/restore private sampling state，避免影响训练恢复。

v3 只修复 replay 的**图入口失败/提前停止生命周期**，不改变上述统计合同。

## 8. 最终 trial 决策边界

仍然只需左侧对科学选择做决定：

1. train missing schedule：complete / missing_OCT / missing_CFP 是否各 1/3；
2. complete checkpoint/分类评估是否采用解析 integrated-mean-logit estimand；
3. embraced_feature 的 LOOK PCA complete reference 是否按 v3 推荐，保留作者随机完整参考分布的解析二阶矩。

继承不变：

- 1264 train / 296 dev / test sealed；
- seed 3416；
- 两路 2D R18；
- ImageNet V1 fresh init；
- FP32，microbatch16/effective128；
- AdamW，pretrained 1e-4 / new 1e-3，wd1e-4，clip5；
- warmup5，minimum8，patience15，100 epoch cap；
- unweighted CE；
- BN 行为不改；
- 一个 A checkpoint，冻结后两种 LOOK × 两种 single-missing 的四个主配对对比；
- K=32 只允许作为可选 complete-state MC 工程诊断，不参与主估计或 checkpoint 选择。

## 9. 当前边界

- 真实参与者/train/dev/test：未读取；
- real training / LOOK fit：未执行；
- GPU / WS02 / Ibex / remote write：未使用；
- 被拒远端传输：未重试、未绕过；
- main：未推送；
- scientific_acceptance：**false**。

v3 的 CPU 工程目标是把 replay 图失败边界与完整参考二阶矩合同说清并验证，不产生新的 CFP/OCT 性能结果。
