# MHD Framework V4（4.0.0）

MHD V4 继续从超图视角统一表示和训练深度学习网络。顶层理论核心仍然只有四个对象：

- `MHD_Node`：节点；
- `MHD_Edge`：超边；
- `MHD_Topo`：拓扑；
- `MHD_Graph`：图。

V4 没有增加“大模型专用核心”。CNN、ResNet、RNN、Transformer、GNN、超图网络和用户自定义网络仍由同一组 Node、Edge、Topo、Graph 组装。`MHD_Node.Message` 与 `MHD_Edge.Operation` 是所属核心对象的嵌套辅助类型，不是新的顶层理论对象。

LOOK 本次干净发行版嵌入的目录为：

```text
MHD_Project/
├── MHD_Framework_V4.py
├── MHD_Utils_V4.py
└── README_V4.md
```

## 1. V3 保持不变的部分

- Node、Edge、Topo、Graph 四个核心名称和超图含义不变。
- Edge 仍承载有序操作序列。
- Role Matrix 仍用 `-1 / 0 / 1` 表示头节点、无关节点、尾节点。
- Sort Matrix 仍决定一条超边内部 Node Message 的稳定顺序。
- 多个 level 仍表示分层传播；同一 Edge/Module 可以跨 level 复用。
- `graph.forward()`、按名称读取 Node、普通 Trainer 的主要使用方式保持一致。

## 2. V3 → V4 完整变化

| 项目 | V3 | 最终 V4 |
|---|---|---|
| Node 状态 | Node 直接持有 `initial_state/current_state` | `feature_message` 与 `gradient_message`；二者各自持有 `initial_state/current_state` |
| Node 构造 | 直接传两个 Tensor | 必须传 `MHD_Node.Message`；Gradient Message 可省略 |
| 状态访问 | `node.current_state` | `node.feature_message.current_state` 或 `node.gradient_message.current_state` |
| 状态聚合 | `transfer_mode`，逐次二元更新 | `feature_aggregation` 与 `gradient_aggregation` 使用同一 n 元聚合约定 |
| Edge 操作 | 裸字符串、Module 或 callable | 每项必须包装为 `MHD_Edge.Operation(...)` |
| 多输入调用 | 捕获运行异常并猜解包/列表/广播 | Sort Matrix 排序后确定性地作为位置参数传入；不吞模型异常 |
| 前向 | Feature 状态逐次更新 | Feature Message 缓冲后确定性 n 元归并 |
| 反向 | 主要直接依赖 loss Tensor | `graph.backward({node_name: seed})` 统一发送 Gradient Message |
| 反向拓扑 | 无独立表达 | 可自动由前向拓扑生成，也可显式给出反向 Role/Sort Matrix |
| 自定义梯度 | 无 Edge 级统一协议 | `backward_operations` 与稳定参数名映射 |
| 参数梯度 | PyTorch autograd | 仍接回 PyTorch autograd；共享 Parameter 按对象身份累加 |
| 多卡 | 基础 DDP 工具 | DDP、FSDP2、TP、PP 四个独立并行族 |
| 混合并行 | 未明确 | 本轮明确拒绝 DP×TP、DP×PP、TP×PP 等组合 |
| Merge | 合并 Feature 状态 | 分别合并两类 Message 的四个 State，并严格检查 Operation/Topo/共享状态 Module |
| Monitor | 当前 Feature 指标 | 旧指标名保持；可选监控四个 Message State |
| Checkpoint | Node 直接状态 | 保存两类 Message 下四个 State；可读取旧 V4 Feature-only 状态 |

这些是有意的破坏性调整：V4 不保留 `node.initial_state`、`node.current_state`、`transfer_mode`、`accumulate_mode`，也不接受未包装的 `edge_operations`。这样避免一套概念同时存在多个别名。

## 3. Message：前向与反向的对称状态

```python
import torch
from V4.MHD_Framework_V4 import MHD_Node

feature_message = MHD_Node.Message(
    initial_state=torch.zeros(2, 8),
    current_state=None,  # 自动克隆 Initial State
)

node = MHD_Node(
    id=0,
    name="tokens",
    feature_message=feature_message,
    # gradient_message 省略时自动生成兼容的零状态
    feature_aggregation="replace",
    gradient_aggregation="sum",
)

node.feature_message.initial_state
node.feature_message.current_state
node.gradient_message.initial_state
node.gradient_message.current_state
```

Feature 为浮点或复数时，默认 Gradient Message 使用同 dtype、shape、device；Feature 为整数或布尔类型时，默认 Gradient Message 使用同 shape/device 的 FP32。显式 Gradient Message 必须是浮点或复数，且其 Initial/Current State 的 shape、dtype、device 必须一致。

两类 Message 使用相同操作：

```python
node.feature_message.reset()
node.feature_message.update_initial(new_feature, update_current=True)
node.gradient_message.update_initial(new_gradient)
node.to_device("cuda")       # 同时移动两类 Message
node.reset()                  # 同时重置两类 Message
```

聚合可选 `replace/sum/avg/max/min/mul`，也可传无可学习参数的 callable：

```python
def aggregate(current, incomings):
    return current + torch.stack(tuple(incomings)).sum(dim=0)
```

Node 聚合只负责 Message 的确定性传输。需要学习参数的注意力聚合、GNN pooling 或 MoE 路由属于模型计算，应写成 `nn.Module` 放进 Edge Operation。这样 Message 负责“怎么汇入节点”，Operation 负责“消息怎么算”，职责清楚但接口仍统一。

## 4. Operation：前向与反向的对称计算

```python
import torch.nn as nn
from V4.MHD_Framework_V4 import MHD_Edge

edge = MHD_Edge(
    id=0,
    name="encoder",
    edge_operations=[
        MHD_Edge.Operation(nn.Linear(8, 16)),
        MHD_Edge.Operation(".relu()"),
        MHD_Edge.Operation(lambda value: value + 1),
    ],
)
```

确定性调用规则只有一套：

- 一个输入 Message：`operation(message)`；
- 多个输入 Message：按 Sort Matrix 顺序调用 `operation(*messages)`；
- 字符串 Operation：逐 Message 应用；
- 一个输出 Tensor 对应一个尾节点，Tensor 序列按顺序对应多个尾节点。

V4 不再像 V3 那样捕获 `TypeError/ValueError/RuntimeError` 后猜另一种调用方式，因为这会吞掉模型内部的真实错误。需要逐 Message 应用普通函数时显式返回列表即可：

```python
MHD_Edge.Operation(
    lambda *messages: [message.relu() for message in messages]
)
```

早期开发稿中的 `MHD_ModuleAdapter(keyword_inputs=..., output_keys=...)` 没有保留。把整张 Graph 暴露为标准 `nn.Module` 的桥接逻辑仍是 DDP/FSDP2/TP/compile 所必需的，但它已经收成 Utils 私有实现 `_MHD_ForwardAdapter`，不是用户需要理解或配置的建模概念。对于关键字专用输入或字典输出，直接用普通 `nn.Module` 表达绑定即可：

```python
class ModelOperation(nn.Module):
    def __init__(self, native_module):
        super().__init__()
        self.native_module = native_module

    def forward(self, value, mask):
        result = self.native_module(value, attention_mask=mask)
        return result["last_hidden_state"], result["auxiliary_output"]

edge = MHD_Edge(
    0,
    "model",
    [MHD_Edge.Operation(ModelOperation(native_module))],
)
```

这里必须使用 `nn.Module` 而不是捕获有参数 Module 的 lambda，这样内部 Module/Parameter 才会被 Graph 注册，optimizer、DDP、FSDP2 和 TP 才能正确发现。V4 测试覆盖了关键字绑定、dict 输出拆分、嵌套参数注册、Node Gradient 和 Parameter Gradient。

自定义反向仍使用 Operation 序列。每一步按同样的规则接收一个或多个 Gradient Message，并额外获得只读前向上下文：

```python
def custom_backward(
    gradient,
    *,
    forward_inputs,
    forward_outputs,
    parameters,
):
    weight = parameters["op0.weight"]
    input_gradient = gradient @ weight
    weight_gradient = gradient.transpose(-2, -1) @ forward_inputs[0]
    return [input_gradient], {"op0.weight": weight_gradient}

edge = MHD_Edge(
    id=0,
    name="projection",
    edge_operations=[MHD_Edge.Operation(nn.Linear(8, 8, bias=False))],
    backward_operations=[MHD_Edge.Operation(custom_backward)],
)
```

参数名稳定为 `op{operation_index}.{parameter_name}`。多步贡献求和；未知、缺失、形状错误或非法梯度会明确报错；显式 `None` 表示该参数无梯度。Backward Operation 自身不得持有可学习参数。stop、scale、reverse、straight-through estimator 等规则都可由同一协议表达。

## 5. Topo 与统一前后向

```python
from V4.MHD_Framework_V4 import MHD_Topo

topo = MHD_Topo(
    role_matrices=[forward_role],
    sort_matrices=[forward_sort],
)
```

未传反向矩阵时，V4 自动生成：

```text
backward_role = -forward_role
backward_sort = forward_sort
```

也可同时显式传 `backward_role_matrices` 与 `backward_sort_matrices`。二者必须成对出现，并与前向具有相同层数和矩阵形状。反向依赖与 level 顺序反转，但一条 Edge 内的 sort 不反转，所以自定义 backward 接收 Gradient Message 的顺序仍稳定。

用户无论使用自动拓扑、显式反向拓扑、标准聚合还是自定义聚合，都只调用：

```python
graph.forward()
graph.backward({"loss": None})
graph.backward({
    "loss_a": None,               # ones_like seed
    "loss_b": explicit_seed,
})
```

外部 seed 是额外 Gradient Message，不覆盖 Gradient Initial State。单根和多根统一使用 Mapping。

V4 没有公开的 `backend=`、`fast_path=` 或执行器选择。内部只在 Node Gradient、Parameter Gradient、共享参数及 optimizer step 都经数值测试证明等价时，透明使用 PyTorch 原生 autograd；非零 Gradient Initial State、自定义聚合、显式反向拓扑等情况自动使用通用超图路由。两者共享完全相同的公开接口和结果语义。

## 6. V3 与 V4 并排迁移

```python
# V3
node = MHD_Node(id, name, initial_state, current_state)
value = node.current_state
```

```python
# V4
node = MHD_Node(
    id=id,
    name=name,
    feature_message=MHD_Node.Message(initial_state, current_state),
)
value = node.feature_message.current_state
```

```python
# V3 Edge
edge = MHD_Edge(0, "encoder", [nn.Linear(8, 8), ".relu()"])

# V4 Edge
edge = MHD_Edge(
    0,
    "encoder",
    [
        MHD_Edge.Operation(nn.Linear(8, 8)),
        MHD_Edge.Operation(".relu()"),
    ],
)
```

## 7. 传统网络与 Transformer

普通网络不需要并行配置，也不需要适配器：

```python
nodes = {
    MHD_Node(0, "input", MHD_Node.Message(torch.zeros(4, 8))),
    MHD_Node(1, "hidden", MHD_Node.Message(torch.zeros(4, 16))),
    MHD_Node(2, "output", MHD_Node.Message(torch.zeros(4, 2))),
}
edges = {
    MHD_Edge(0, "encoder", [MHD_Edge.Operation(nn.Linear(8, 16))]),
    MHD_Edge(1, "head", [MHD_Edge.Operation(nn.Linear(16, 2))]),
}
role = torch.tensor([[-1, 1, 0], [0, -1, 1]])
sort = torch.tensor([[0, 1, 0], [0, 0, 1]])
graph = MHD_Graph(nodes, edges, {MHD_Topo([role], [sort])}, device="cpu")

graph.get_node_by_name("input").feature_message.current_state = torch.randn(4, 8)
graph.forward()
```

Transformer 仍只是一个 Operation：

```python
block = nn.TransformerEncoderLayer(
    d_model=128,
    nhead=8,
    dim_feedforward=512,
    batch_first=True,
)
edge = MHD_Edge(
    0,
    "transformer",
    [MHD_Edge.Operation(block)],
)
```

Attention、残差、归一化等细节继续由原生 Module 定义，MHD 不重复实现 PyTorch 已有网络层。

## 8. GNN 与超图网络的表达

- 一条 Edge 可消费多个头节点并产生多个尾节点，因此能直接表达超边计算。
- 多条 Edge 向同一 Node 发送 Feature Message，再由 Node 的 n 元 aggregation 合并，可表达邻居消息聚合。
- 同一 Module 可跨 Edge 或 level 共享；多个 level 可静态展开循环消息传递。
- 动态改变图结构时，构造新的 Role/Sort Matrix；V4 不把模型特有的动态路由硬编码进核心。

测试中已经把“带自环的三节点有向图、共享 Linear、两轮消息传递”与邻接矩阵 PyTorch 实现逐项比较，前向、输入梯度和共享权重梯度一致；也验证了一条三头两尾的真实超边。

## 9. Trainer、AMP 与梯度累积

原 Trainer 的主要调用方式保持：

```python
from V4.MHD_Utils_V4 import MHD_Monitor, MHD_Trainer, create_mhd_optimizer

optimizer = create_mhd_optimizer(graph, default_optimizer_type="adamw")
trainer = MHD_Trainer(
    graph,
    optimizer,
    MHD_Monitor(["loss"]),
    backward_node="loss",
    precision="fp32",
    grad_accum_steps=1,
)
```

Trainer 内部把旧的 `backward_node` 转为 `graph.backward({backward_node: seed})`。FP16 GradScaler 的参数梯度保持缩放语义，而公开的 Node Gradient Message 会转换为未缩放值。BF16、梯度累积和梯度裁剪沿用同一 Trainer。

`MHD_Monitor(["loss"])` 的旧指标名保持，例如 `loss_mean`。需要时可显式选择：

```python
MHD_Monitor(
    ["loss"],
    node_states=(
        "feature_message.initial_state",
        "feature_message.current_state",
        "gradient_message.initial_state",
        "gradient_message.current_state",
    ),
)
```

## 10. 独立多卡支持

并行属于 `MHD_Utils_V4.py` 的可选训练能力，不改变 Framework 核心。默认不开启；一次运行只选择一个并行族。

```python
from V4.MHD_Utils_V4 import MHD_ParallelConfig

ddp = MHD_ParallelConfig(data_parallel="ddp")
fsdp2 = MHD_ParallelConfig(data_parallel="fsdp2")

tp = MHD_ParallelConfig(
    tensor_parallel_size=2,
    tensor_parallel_plan={
        "transformer:0.encoder.linear1": "colwise",
        "transformer:0.encoder.linear2": "rowwise",
    },
)

pp = MHD_ParallelConfig(
    pipeline_size=2,
    pipeline_stages={"encoder": 0, "head": 1},
    pipeline_microbatches=2,
    pipeline_schedule="1f1b",  # 或 gpipe
)
```

各模式仍由同一个准备入口接收同一张 MHD Graph：

```python
from V4.MHD_Utils_V4 import prepare_mhd_model

model = prepare_mhd_model(
    graph,
    input_nodes=["input", "target"],
    output_nodes=["loss"],
    parallel=ddp,  # 也可换为 fsdp2、tp 或 pp
)
```

DDP/FSDP2/TP/PP 都使用一进程一卡的原生启动方式，例如双卡：

```bash
torchrun --standalone --nproc-per-node=2 train.py
```

- DDP：代码不锁死卡数，接受任意合法 `WORLD_SIZE`。
- FSDP2：代码不锁死卡数，使用原生 `fully_shard`。
- TP：`tensor_parallel_size` 可为任意合法大小；简单 `colwise/rowwise` 方案在内部完成 DTensor 布局衔接。
- PP：`pipeline_size` 可为任意合法 stage 数；根据 Edge、Topo 与 `pipeline_stages` 自动生成 stage Module。
- TP 模型若同时含普通 Tensor 与 DTensor 参数，应使用 `create_mhd_optimizer`；它会自动关闭 PyTorch 2.8 不兼容的 foreach 批量更新，无需用户配置布局。
- 本轮不支持 DP×TP、DP×PP、TP×PP 或 DP×TP×PP。请求这些组合会立即报错，不会静默降级。

`pipeline_stages` 是 Edge 到 stage 的必要归属信息，不是另一套建模 API。旧的手工 `pipeline_stage_modules` 入口没有保留。

## 11. Checkpoint 与 V3 迁移

V4 checkpoint 保存：

```text
feature_message.initial_state
feature_message.current_state
gradient_message.initial_state
gradient_message.current_state
```

读取旧 V4 checkpoint 时，旧 `nodes` 或 `node_initial_states/node_current_states` 迁移为 Feature Message；缺失 Gradient Message 时使用默认零状态。

上游 V4 可选提供独立的 V3 checkpoint 迁移工具；LOOK 本次发行版从零开始训练，
不包含该兼容入口，也不复用 V3 checkpoint。

## 12. 实际验证结果与边界

最终验证环境是 `ws` 上已有隔离虚拟环境：Python 3.12.3、PyTorch 2.8.0+cu128、两张 NVIDIA RTX 5000 Ada。测试使用小 batch/microbatch，并保留机器上原有任务。

| 验证项 | 实际结果 |
|---|---|
| 49 项 CPU/模型测试：Message/Operation/Topo/Graph、聚合、反向、Merge、Monitor、checkpoint、V3 迁移 | 通过 |
| 原生 autograd 与通用 Message 路由：单根、多根、各 Node Gradient、各 Parameter Gradient、共享参数、optimizer step | 数值一致 |
| ResNet 与 Transformer 对直接 PyTorch | 前向、loss、参数梯度一致 |
| GNN 两轮传播与三头两尾超边 | 前向和梯度一致 |
| DDP 两卡，标准/自定义 backward | 通过 |
| FSDP2 两卡，标准/自定义 backward | 通过 |
| TP 两卡，标准/自定义 backward；真实 Transformer | 通过 |
| PP 两卡，GPipe/1F1B，标准/自定义 backward | 通过 |
| BF16 + DDP，标准/自定义 backward | 通过 |
| `torch.compile` 图捕获兼容性（记录 backend） | 通过 |

默认 Inductor 的 ws 实测没有完成：该虚拟环境缺少系统 `Python.h`，Triton C helper 编译失败。这是当前主机开发头文件缺失，但在补齐环境前不能声称 `compile=True` 的默认 Inductor 路径已经通过。PyTorch 2.13 是 V4 的正式目标版本，本轮可用环境只有 PyTorch 2.8，因此 2.13 仍需在对应环境复验。

代码层面 DDP/FSDP2/TP/PP 均按任意合法并行度实现，并用四卡配置做了不启动进程的 topology/config 验证；硬件实测只有 `ws` 两卡。三卡以上的性能、通信稳定性和具体模型切分不能由两卡结果替代。

在共享 GPU 负载下，TransformerEncoderLayer（batch 4、序列 128、hidden 512、BF16）采用同步主机墙钟、交替顺序、每轮前向 200 次和训练 100 次，共独立运行三轮。三轮中位结果为：

- 原生前向 0.1109 ms，MHD 0.1199 ms；额外约 9.08 µs，8.19%。
- 原生前向+反向 0.5788 ms，MHD 0.6338 ms；相对额外开销三轮中位数 9.77%。

结果满足本轮门槛：前向相对回归不超过 15%且绝对额外耗时不超过 25µs，标准训练相对回归不超过 10%。这是该主机当时负载下的回归测试，不是跨设备吞吐承诺。自定义 backward 的目标是正确性与原生分布式 hook 衔接，本轮未把其额外开销混入标准训练基准。

本轮不包含独立 PyTorch/FX/export 图生成。

## 13. 兼容性结论

完全保持的核心：

- `MHD_Node`、`MHD_Edge`、`MHD_Topo`、`MHD_Graph` 名称；
- Role Matrix、Sort Matrix、level 与超图表达方式；
- `graph.forward()` 和 Trainer 的主要训练流程；
- 原生 `nn.Module` 作为模型计算主体。

需要迁移的调用：

- Node 状态改为嵌套 Message；旧直接字段不存在；
- Edge 中每个操作必须用嵌套 Operation 包装；
- 早期 `MHD_ModuleAdapter` 不再提供；关键字绑定和结构化输出由普通 `nn.Module` Operation 表达，整图的标准 Module 桥接只作为 Utils 私有实现保留；
- `transfer_mode` 改为 `feature_aggregation`，反向对应 `gradient_aggregation`；
- 多输入 callable 使用确定性位置参数，不再猜调用方式；
- checkpoint 使用四个 Message State，旧文件通过兼容脚本迁移。

V4 的统一含义是：一个超图接口、一套 Message/Operation 规则、一个 `graph.backward` 入口；内部可以透明选择已经证明等价的执行实现，但不把实现选择变成用户配置。
