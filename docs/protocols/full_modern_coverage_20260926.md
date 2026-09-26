# 完整现代宿主研究覆盖

按用户本次逐项决定：ConvNeXt-B/ConvNeXt-B、Swin-B/Swin-B、DINOv2 ViT-B/DINOv2 ViT-B、CFP ConvNeXt-B/OCT B-scan DINOv2 ViT-B四配对；白内障、青光眼、黄斑变性三任务；EmbraceNet及完整分阶段IMD两方法，共24基础单元。现有IMD组件不能充当完整方法。先完成3416，再做3417/3418，后续种子不按结果正负筛选。完整train/dev，人级划分和test封存保持。

每个接受的冻结A，四种修正族shared_pca_ridge、pca_free_mean、rrr_shared_intercept、residual_rrr均覆盖两种模态缺失和预定有意义站点；分别独立执行单站点枚举、best-forward、positive-forward-tree。每条搜索自行拟合、选优与留存负结果，不把某方法或单站点的胜负用作其他搜索的准入过滤。报告同时覆盖性能与代价，开发集搜索不等于独立泛化。

本次新增 `fit_embracenet_search_suite` 可在同一冻结EmbraceNet宿主上执行上述全覆盖，绑定宿主/数据/源码/种子身份，每轨迹独立保存统计、产物和预测SHA。真实小型MHD图四修正族×两搜索的数值/恢复测试，以及完整覆盖独立性测试共17项通过，PyTorch2.8 CPU。此证据不是24单元GPU执行完成。

原已部署GPU52615061的673844b快照仍为首个ConvNeXt白内障EmbraceNet包（两修正族、正向树），不能热改或冒称全矩阵。新覆盖代码在独立后继身份中使用，先通过真实GPU生命周期与资源预算，再领取合法角色执行。其余宿主、完整IMD、全矩阵执行、独立报告验收仍需实施。RETFound和进一步容量/非对称规模研究放在首阶段之后，不能阻塞当前已批准包。
