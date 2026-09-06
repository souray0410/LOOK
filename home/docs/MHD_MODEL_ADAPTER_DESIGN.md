# 后续 MHD 模型适配层：接口设计，尚未迁移运行代码

## 目的与边界

将模型构造和执行契约与 LOOK 的统计拟合、配置选择分开。不是只搬走 graph.py：目前 joint.py、look.py 和诊断模块也依赖 OCT/CFP、fusion_logits、fusion_participant_feature 等约定。

本次仅冻结设计，现有实验保持现有实现。未来新数据/网络保留方法定义，但必须重新检验经验结论；不能认为关键发现必然不变。

## 三层职责

1. **模型提供者**：将模型构造成保留真实stage的 MHD 图，加载自己的checkpoint，提供输入绑定、模型前向levels、训练反向levels、任务logits出口及模型身份。
2. **校正位置适配器**：声明有序的候选位置。每个位置包含执行边界、成员节点列表、成员顺序、读取/打包/拆分/写回以及形状约束；允许同level多个节点，也允许合法的单节点位置。
3. **LOOK 引擎**：仅通过上述接口读取完整/缺失特征，拟合PCA和配对W/b，按顺序继承已接受校正、进行validation选择、保存和加载配置。数据划分、缺失机制和任务指标留在实验协议层。

建议未来目录：`models/providers`、`models/correction_sites`、现有`look_core`。首个提供者包装当前ResNet50双眼OCT/CFP；先做到数值等价，再注册第二个模型，避免未验证的通用重写。

## 最小契约

- `build_model(config, checkpoint, device)` 返回 MHD图及模型描述：输入名映射、合法前后向levels、logits出口、checkpoint与拓扑哈希。
- `bind_inputs(batch_inputs)` 按字典绑定命名输入，禁止由LOOK假定两个模态或固定双眼布局。
- `correction_sites` 返回固定顺序的 `site_id, boundary, members, codec`；位置ID稳定，恢复时核验拓扑、成员顺序和codec身份。
- `codec.read/pack/unpack/write` 必须可逆恢复原成员形状，不改变参与者配对顺序。当前codec继续通道拼接；空间形状不兼容时明确拒绝。异构节点可使用单节点位置或另行验证的展平codec，不能直接concat或自动插入可学习投影。
- `forward_until / forward_remaining` 委托 MHD 的合法level执行。联合位置必须保证全部成员状态已生成、尚未被下游消费，不能简单按各网络层号同名配对。
- 可选的 `diagnostic_feature` 与 `linear_head` 能力只用于相应诊断；模型没有线性头时禁用精确margin线性界，不伪装成通用结论。
- 反向执行服务于原模型训练及SSF等需要梯度的对照；LOOK PCA/Ridge本身不因此变为反向训练方法。SSF插入位置需独立声明能力，不假定每个新模型都适用同一hook列表。

## 后续迁移验收

原ResNet50在完整/两方向缺失、全部OFF、固定ON配置上的logits、写回状态与梯度等价；PCA和W/b来源一致；训练/验证划分不变；保存恢复不跨模型身份。再用不同形状的微型MHD图验证适配接口和拒绝路径，而不是立刻新增大型backbone搜索。

该迁移应在独立版本实施；当前受限实验队列与其结果不能被重写。网络构造、位置codec和引擎接口各自哈希，可精确判断未来哪些资源必须重新拟合。
