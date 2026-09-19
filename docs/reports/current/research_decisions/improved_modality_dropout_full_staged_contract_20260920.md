# 完整作者分阶段 IMD：下一独立有限合同

状态：`contract_ready_not_launched_in_current_batch`。

当前A没有contrastive pretraining，且encoder BatchNorm running state在target-task train mode中变化。不能在已完成A后补做contrastive然后继续沿用同一个A身份；那会混合两个科学配方。

作者分阶段身份：1）预训练/固定unimodal encoder；2）fused/unimodal projector做三对监督sigmoid contrastive；3）target classifier做simultaneous modality dropout + learnable missing tokens。Pinned作者仓库提供TNF、EmptyToken与contrastive loss核心，但没有可直接把当前CFP/OCT participant aggregation、恢复协议与全部阶段串起来的正式trainer。

因此若做完整作者分阶段project adaptation，必须作为新的A身份预登记：
- 同一1264/296、seed3416、test封存；
- contrastive stage保留三对目标；
- encoder参数与运行状态均固定：encoder `eval()`，训练前后全部state byte-identical；
- 明确projector维度、contrastive checkpoint selection/stopping、participant aggregation、stage间optimizer/RNG恢复；
- target stage保留complete / missing OCT / missing CFP；
- A接受后冻结一个checkpoint，再跑同样两缺失×PCA/RRR四树与四指标10k统计。

这会是“完整作者分阶段机制的CFP/OCT project adaptation”，仍不等于作者原image+tabular实验复现。当前整批不后验改写已完成组件A。
