# EmbraceNet 单种子真实试运行：已授权、待目标部署执行（2026-09-19）

状态：**authorized_preregistered_not_yet_executed**。CPU 工程基线 `9581f107...` 已独立接受；用户现已明确批准本单种子训练/评估配方，并授权在既有 LOOK/R&B 范围内对必要补充实验与新包训练/评估选择先留档后执行，不再逐项询问。该授权不解除 test 封存、唯一 owner、资源/恢复、无 Fast/priority、外部工具拒绝不可绕过等边界。

## 固定问题与停止范围

只回答：同一个冻结 EmbraceNet A 在整 OCT 或整 CFP 缺失时，加 LOOK 是否改善 development robustness。

- 只训练 1 个新 A，seed=3416。
- 1264 train / 296 development；test 封存。
- 两路 2D ResNet18，ImageNet V1 fresh。
- train complete / missing OCT / missing CFP 各 1/3，participant 级、双眼同步缺失。
- complete checkpoint 选择：解析 `E[logits] = W E[z] + b` 后计算原 complete-dev Macro-F1。
- single-missing：每状态一次作者式 forward。
- LOOK：9 个实际节点，positive-forward tree；PCA free-mean 与 residual RRR free-mean × 两种 single-missing，共 4 个预注册主对比。
- embraced_feature 的 complete-reference PCA 保留作者随机条件方差，解析累计 `E[zzᵀ] = μμᵀ + diag(v)`，不改成 mean-only PCA。
- rank=32；空间节点 factor=16；prefix train-PCA GCV。
- 普通 + 同族同时 95% participant-paired bootstrap，10,000 次；报告 Macro-F1/AUROC/NLL/Brier。
- 本包到 1 个 A + 4 棵完整树 + 注册统计 + 中文累计候选即停止；不因结果阴性自动补 seed/调参/换方法。

## 继承训练合同

FP32；microbatch 16 / effective 128；AdamW；pretrained LR 1e-4 / 新 docking+head LR 1e-3；weight decay 1e-4；clip 5；warmup 5；minimum 8；patience 15；100 epoch cap；unweighted CE；BN/数据增强/participant pooling 沿既有 accepted R18 语义。

## 执行边界

先本地 current-source 动态测试，再合法 source deployment；WS02 GPU0 独占本 A，GPU1 留给 R&B；formal 前必须有真实 GPU/resource/profile/resume gate。此前被上层拒绝的远端 scp 不重试、不换 helper/SFTP/base64 等绕过。未独立审计前不推 main；真实结果只标 `self_checked_pending_independent_review`。

授权 SHA：`bad1710372f092f662c87e4d41e60c591decb0ab8dccadb0a75934e91313bd01`。
任务包 SHA：`fd3300c0cdbfa3d168eca5e9e4563b4c5e3d9fdf4dc7e0111bdea89ccd3339f1`。
PHD research standard SHA：`1d24eddf2f939a82ca3ed821c803739a727b8f403d20457e22291a28530ddb2e`。
