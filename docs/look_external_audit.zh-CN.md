# 当前角色纠正

本页原三项是接口/源码候选审计，不是LOOK缺失方法A/B/C覆盖。2026-09-19后，MMTM降回宿主兼容/组装证据；EyeMoSt+/EDRL源码审查留档但不自动作为下一A。真正缺失方法候选按一手证据重新筛选，见 docs/reports/current/research_decisions/missing_method_evidence_20260919.md。

# 外部候选适配审计

本页是代码级适配审计，不是三方法实验完成。三个候选不依本次性能替换。MMTM兼容性小队列适配协议见[记录](handoff/mmtm_cohort_20260918.md)，实际运行须以其部署回执为准。精确作者commit和文件SHA见experiments/2026_09_10_11_11_31/look_mechanisms/external_sources.json。

| 方法 | 核心 / 输入 | 当前处理 | 尚需锁定 |
|---|---|---|---|
| MMTM，CVPR2020 | 两路空间均值→联合MLP/ReLU→各路sigmoid门控；不要求空间尺寸相同 | 已提供维度无关门控并做公式/梯度验收；当前仅作LOOK宿主兼容/组装证据，不是缺失方法A | Stage3/R18深融合、作者scale1随机初始化、沿用complete训练预算已登记；仍须真实GPU及最终科学验收 |
| EyeMoSt+ | 各路预测分布、Student-t融合与置信度约束；作者列出CFP2D及OCT3D编码器 | 审阅MedIA’24的EyeMost_Plus路径；禁止删掉分布/置信度损失以凑普通分类器 | OCT改2D后的编码器及证据头、保持原损失、无标签推理、完整缺失处理、MHD转换与数值对照 |
| EDRL，MICCAI2025 | Essence-point筛选、共同/独有表示解耦与自蒸馏；原图含3DOCT | 审阅code目录fusion_net及Harvard入口；当前2D队列不是原论文完整输入 | 单切片是否仍有实质token选择、原损失权重和训练/推理分支、随机推理、MHD转换与一致性 |

MMTM作者代码在当前commit中返回 `x*sigmoid(...)`，论文式样通常写作 `2*sigmoid(...)`。本地审计模块要求显式gate_scale=1或2，不偷偷把二者当作相同实现；与作者类对照使用1。作者初始化保留Linear随机初始化；零初始化门控是另外的适配选择，尚未据此派发训练。MHD预定映射为两输入特征节点→联合描述符节点→共享压缩节点→两门控节点→各路调制节点，后接已有输出宿主，不能把两路输出直接当作LOOK最终分类指标。

EyeMoSt+拟保留每来源分布参数节点、分布融合及置信度约束损失节点；EDRL拟保留encoder、EPRL、DiLR、融合分类与各原损失节点。两者作者forward接口接收标签以计算损失，这本身不等于预测泄漏；须专门验证推理logits不依赖传入标签。现阶段未完成这项数值验收，因此不将其标为可运行的正式对照。

不声称三维输入方法与当前二维改造等价，不把同样的优化器视为参数量/搜索预算匹配。若保留核心机制的二维适配不可行，记录不适用原因，继续A—E；有明确适配方案后另建补充协议。

来源：[MMTM作者论文](https://www.microsoft.com/en-us/research/publication/mmtm-multimodal-transfer-module-for-cnn-fusion/)、[MMTM作者代码](https://github.com/haamoon/mmtm)、[EyeMoSt/EyeMoSt+作者代码](https://github.com/Cocofeat/EyeMoSt)、[EDRL论文](https://papers.miccai.org/miccai-2025/paper/0678_paper.pdf)、[EDRL作者代码](https://github.com/xinkunwang111/Robust-Multimodal-Learning-for-Ophthalmic-Disease-Grading-via-Disentangled-Representation)。
