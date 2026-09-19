# LOOK/R&B 比较角色采用记录（LOOK侧）

依据：PHD standards/research.md 2026-09-19“外部方法对照必须对应所研究的问题”及本周 comparison_scope_correction.zh-CN.md。这里只更新当前问题映射，不改历史数值、模型、科学 cutoff、test 策略或已验收证据。

- LOOK：A 是实际处理整模态缺失的既有策略；主比较是同一个已合理训练的 A 与同一个 A + LOOK。宿主门控/完整模态融合能力不能替代缺失处理能力。
- Radon_Bridge：A 是跨维/跨模态交流方法；主比较是同一个交流 A 与 A + bridge。
- MMTM 在 LOOK：既有结果保留为宿主兼容性/组装机制证据；不计为 LOOK 缺失方法 A/B/C 完成。
- fusion-stage、fixed-mean、fixed-prefix 等继续作为 LOOK 机制证据，不计缺失方法覆盖。
- GAN、缺失鲁棒微调只是类别例子，不是机械必跑表。
- 新 A 的训练、适配、预算和合法 LOOK 节点都需要新的有限协议；本记录本身不授权训练。

一手证据盘点：docs/reports/current/research_decisions/missing_method_evidence_20260919.md
