# 现代 EmbraceNet 首包的完整 LOOK 覆盖后继

独立后继读取已接受的 seed3416 白内障 ConvNeXt-Base 配对宿主和 train-only PCA。原宿主 spec、训练和正在排队的运行快照不改。入口 `python -m look.studies.modern_coverage --contract ...`；完成拟合并释放GPU后，CPU用同一入口加 `--report-only` 生成报告。

四族：shared_pca_ridge / pca_free_mean / rrr_shared_intercept / residual_rrr。两种单模态缺失。每族每方向独立运行9个单站点、best-forward和正向树，共88条轨迹；单站点结果不筛掉后续搜索。保持原rank32、factor16、train-prefix PCA GCV选择规则，不动test。

已接受PCA按原宿主身份、文件SHA、source_id、样本数及九站点覆盖读取，不因新增搜索代码的文件SHA变化重算相同PCA。每个已完成轨迹保存输入身份、选择、bank、重放和预测摘要；跨租期重入只核对已完成证据，不重复整条开发集搜索。当前未完成轨迹仍按原前缀恢复规则继续。

报告输出：完整88行开发集结果、同A的两方向基线、所有88项比较的10000次配对bootstrap/同时区间、选择路径、拟合调用成本及方法图。区间不消除使用同一开发集进行搜索的偏差，不冒充独立test或三种子接受；3417/3418不因3416阴性取消。选择路径和分支作用的进一步科学解释由主窗口完成，脚本不自动选出下一研究赢家。

状态：独立执行入口及恢复/报告测试已实现；真实现代宿主/PCA尚在原部署链中，本后继的GPU运行及最终独立接受尚未发生。
