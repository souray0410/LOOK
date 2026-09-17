# LOOK与3D父模型：耗时与文献核查

2026-09-17；仅train/dev与只读运行审计。未改训练配置、运行源码或test权限。Owner：LOOK/Radon_Bridge维护任务。状态：diagnosed，性能修复与生产验收尚未完成。

## LOOK：计算结构已确认，分段耗时仍缺测量

运行：search16_q32_20260917_v2，2026_09_17_09_10_26_594899；白内障/ResNet50/deep/3416。

实际入口为search_case → search_policy.fit_search → operator.fit_look_node。当前仍为原PCA/GCV算子。冻结宿主、batch16、num_workers=0、worker_threads=1、FP32；每个位置遍历train，对每个batch分别计算完整/缺失特征，之后遍历dev评价。共享PCA已有，但跨位置完整参考特征未复用。空间x16在网络特征产生后应用，不减少主干输入和此前卷积。q32拟合最终只解32×32线性系统；不能将全部耗时归因于解析求解。

原调用路径对同一上游修正状态有重复网络扫描，属于优化候选，不是线性方法的必要条件。当前尚无读取/前向/统计/求解/评价的独立计时，不能虚构各部分百分比。

保持科学身份的下一项实现目标：缓存冻结完整参考的投影特征；同一已接受上游状态一次提取多个候选位置；依赖变化只重算下游；固定顺序累积统计；dev候选仅重算必须改变的后缀。增加读取预取/推理batch前必须测资源并重放核验。缓存降维特征可用于拟合，不能直接替代后续非线性网络需要的完整激活。

验收：相同模型/参与者顺序/基/上游状态，比较统计、参数、逐候选logits和选中路径；差异超容限不得混用缓存。先分段profile再决定优化，不承诺固定倍数加速。

## 当前运行的3D DenseNet父候选

run 2026_09_12_11_23_31_801126，macular_degeneration，monai_densenet121_3d，random；train 58,380，dev 12,505。配置为128×224×224体积，SGD lr0.025/momentum0.9/WD1e-4，unweighted CE，FP32，effective_batch16/microbatch1，BN train，固定50轮。此配置声明为架构文献适配而非匹配UKB疾病论文复现；eligible_for_formal_selection=False，须审查正式父模型准入，不能仅凭训练完成自动认定合格。

当前进程运行约29小时；已完成2轮，正在第3轮。运行与LOOK共享A10080GB，整卡瞬时13393MiB、GPU利用率92%。本worker记录峰值约8.14GiB。状态采样1789645339.277→1789645529.057，offset29424→29680，短窗口约1.35参与者/s；线性外推约12小时/train epoch，未含验证，不是稳定独占基准或完成承诺。

第2轮macro-F1=0.49645647，AUROC=0.52967389；混淆矩阵[[12329,0],[176,0]]，仍全部预测阴性。阳性率约1.41%。不能称已收敛/合格，也不能据两轮断言最终不可学习。

microbatch1累积16不等于真实batch16，尤其BN仍训练；实际每microbatch眼数还受valid-eye聚合影响。大batch不是无条件更好或必然线性加速。优先在隔离工程预检中测真实batch、I/O、前后向及恢复峰值；正式更改microbatch/BN/优化规则必须新版本、明确科学身份，不原地改旧run。

## 已核实的公开期刊参照

1. Rasel等，Scientific Reports 2024，Assessing the efficacy of 2D and 3D CNN algorithms in OCT-based glaucoma detection，https://pmc.ncbi.nlm.nih.gov/articles/PMC11116516/ ，DOI 10.1038/s41598-024-62411-6。正文经Europe PMC fullTextXML复核。UKB选择255名POAG与765名健康对照；3D ResNet18/DenseNet121用batch16、Adam lr1e-5、50轮。随机初始化AUC分别0.928/0.938；二维权重扩展初始化0.937/0.945。病例选择、健康排除、预处理、疾病、架构、初始化及评价协议与当前研究不同，不能直接排名。核查正文未找到GPU型号或训练小时数；不能据此宣称我们的墙钟时间正常。
2. Chen等，GeroScience 46,1703–1711 (2024；2023在线)，Deep neural network-estimated age using optical coherence tomography predicts mortality，https://pmc.ncbi.nlm.nih.gov/articles/PMC10828229/ 。年龄回归用ResNet18-3D、128×256×256、SGD；建模8,541名健康参与者。可参考3D输入和实现，不是当前疾病分类或耗时直接基准。

## 未关闭事项与后续顺序

1. LOOK分段profile与保持算子/选择语义的复用；保持已有健康任务。
2. 3D父候选完整运行配方/正式准入审计，不能只看architecture名称或accepted文件。
3. 隔离实测CPU/I/O/设备峰值、吞吐和真实batch；更改训练语义另立配置，不以旧run继续掩盖。
4. 先恢复批准的关键父模型与独立派发，不能把暂停时间计为3D必需计算。
5. 文献协议用于合理性审查；性能比较须在我们同一划分重跑匹配参照，test继续封存。

本条是调查与后续验收记录，不是缓存优化已部署、3D配方已修复或队列已接通的证明。
