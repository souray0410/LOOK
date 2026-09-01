# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Framework (MHD) - Version 4.0
Author: Souray Meng (孟号丁)
Core Framework: Hypergraph-based computational graph with multi-level topology
License: MIT
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Set, Any, Sequence, Mapping, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
import ast
import logging
from types import MappingProxyType

logger = logging.getLogger(__name__)


def _evaluate_tensor_ast(node: ast.AST, x: torch.Tensor) -> Any:
    """Evaluate the deliberately small expression grammar used by string operations."""
    if isinstance(node, ast.Name) and node.id == "x":
        return x
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_tensor_ast(item, x) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate_tensor_ast(item, x) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _evaluate_tensor_ast(node.operand, x)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Attribute):
        owner = _evaluate_tensor_ast(node.value, x)
        if node.attr.startswith("_"):
            raise ValueError("字符串操作不能访问私有属性")
        return getattr(owner, node.attr)
    if isinstance(node, ast.Call):
        function = _evaluate_tensor_ast(node.func, x)
        args = [_evaluate_tensor_ast(arg, x) for arg in node.args]
        kwargs = {kw.arg: _evaluate_tensor_ast(kw.value, x) for kw in node.keywords}
        if None in kwargs:
            raise ValueError("字符串操作不支持 **kwargs")
        return function(*args, **kwargs)
    raise ValueError(f"字符串操作包含不受支持的语法: {type(node).__name__}")


def parse_string_operation(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """
    解析并执行字符串形式的张量操作（新规范：直接拼接 x + op_str）

    用户需提供以 '.' 开头的操作字符串，如 ".relu()" 或 ".mean(dim=1)"，
    函数将直接执行 x.op_str，无需额外解析。

    Args:
        op_str: 操作字符串，必须以 '.' 开头，如 ".relu()" 或 ".mean(dim=1)"
        x: 输入张量

    Returns:
        操作后的张量
    """
    if not isinstance(op_str, str) or not op_str.startswith("."):
        raise ValueError("字符串操作必须以 '.' 开头")
    expression = "x" + op_str
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"字符串操作语法错误: {op_str}") from exc
    result = _evaluate_tensor_ast(tree.body, x)
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"字符串操作必须返回 Tensor，实际为 {type(result)!r}")
    return result


@dataclass(frozen=True)
class _MHD_ExecutionStep:
    edge_id: int
    head_ids: Tuple[int, ...]
    tail_ids: Tuple[int, ...]
    edge: Any = field(compare=False, repr=False)
    tail_nodes: Tuple[Any, ...] = field(compare=False, repr=False)


# ===================== 核心超图框架类 =====================

@dataclass(eq=False)
class MHD_Node:
    """Hypergraph node carrying symmetric Feature and Gradient messages."""

    @dataclass
    class Message:
        """Two-state message owned by an :class:`MHD_Node`."""

        initial_state: torch.Tensor
        current_state: Optional[torch.Tensor] = None

        def __post_init__(self) -> None:
            if not isinstance(self.initial_state, torch.Tensor):
                raise TypeError("Message initial_state 必须是 Tensor")
            if self.current_state is None:
                self.current_state = self.initial_state.clone(
                    memory_format=torch.contiguous_format
                )
            if not isinstance(self.current_state, torch.Tensor):
                raise TypeError("Message current_state 必须是 Tensor")
            self.validate()

        def validate(self) -> None:
            if self.initial_state.device != self.current_state.device:
                raise ValueError(
                    "Message Initial State 与 Current State 设备不一致: "
                    f"{self.initial_state.device} vs {self.current_state.device}"
                )
            if self.initial_state.shape != self.current_state.shape:
                raise ValueError(
                    "Message Initial State 与 Current State 形状不一致: "
                    f"{self.initial_state.shape} vs {self.current_state.shape}"
                )
            if self.initial_state.dtype != self.current_state.dtype:
                raise ValueError(
                    "Message Initial State 与 Current State dtype 不一致: "
                    f"{self.initial_state.dtype} vs {self.current_state.dtype}"
                )

        def reset(self) -> 'MHD_Node.Message':
            self.current_state = self.initial_state.clone(
                memory_format=torch.contiguous_format
            )
            return self

        def update_initial(
            self,
            new_tensor: torch.Tensor,
            update_current: bool = True,
        ) -> 'MHD_Node.Message':
            if not isinstance(new_tensor, torch.Tensor):
                raise TypeError("new_tensor 必须是 Tensor")
            if not update_current:
                if (
                    new_tensor.shape != self.current_state.shape
                    or new_tensor.device != self.current_state.device
                    or new_tensor.dtype != self.current_state.dtype
                ):
                    raise ValueError("仅更新 Initial State 时必须与 Current State 完全兼容")
            self.initial_state = new_tensor
            if update_current:
                self.current_state = new_tensor.clone(
                    memory_format=torch.contiguous_format
                )
            self.validate()
            return self

        def to_device(self, device: torch.device) -> 'MHD_Node.Message':
            device = torch.device(device)
            if self.initial_state.device != device:
                self.initial_state = self.initial_state.to(device, non_blocking=True)
            if self.current_state.device != device:
                self.current_state = self.current_state.to(device, non_blocking=True)
            return self

    id: int
    name: str
    feature_message: Message
    gradient_message: Optional[Message] = None
    feature_aggregation: Union[str, Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor]] = "replace"
    gradient_aggregation: Union[str, Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor]] = "sum"
    _default_gradient_initial_identity: Optional[int] = field(
        init=False, default=None, repr=False
    )
    _default_gradient_initial_version: Optional[int] = field(
        init=False, default=None, repr=False
    )

    _BUILTIN_AGGREGATIONS = frozenset({"replace", "sum", "avg", "max", "min", "mul"})

    def __post_init__(self) -> None:
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("节点 id 必须是非负整数")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("节点 name 必须是非空字符串")
        if not isinstance(self.feature_message, MHD_Node.Message):
            raise TypeError("feature_message 必须是 MHD_Node.Message")
        self._validate_aggregation(self.feature_aggregation, "feature_aggregation")
        self._validate_aggregation(self.gradient_aggregation, "gradient_aggregation")
        gradient_message_was_omitted = self.gradient_message is None
        if gradient_message_was_omitted:
            feature = self.feature_message.initial_state
            gradient_dtype = (
                feature.dtype
                if feature.is_floating_point() or feature.is_complex()
                else torch.float32
            )
            zeros = torch.zeros(
                feature.shape,
                dtype=gradient_dtype,
                device=feature.device,
            )
            self.gradient_message = MHD_Node.Message(zeros)
        if not isinstance(self.gradient_message, MHD_Node.Message):
            raise TypeError("gradient_message 必须是 MHD_Node.Message 或 None")
        gradient = self.gradient_message.initial_state
        if not (gradient.is_floating_point() or gradient.is_complex()):
            raise TypeError("Gradient Message 必须使用浮点或复数 dtype")
        if gradient.shape != self.feature_message.initial_state.shape:
            raise ValueError("Feature Message 与 Gradient Message 的形状必须一致")
        if gradient.device != self.feature_message.initial_state.device:
            raise ValueError("Feature Message 与 Gradient Message 的设备必须一致")
        if gradient_message_was_omitted:
            self._remember_default_zero_gradient()

    def _remember_default_zero_gradient(self) -> None:
        initial = self.gradient_message.initial_state
        self._default_gradient_initial_identity = id(initial)
        self._default_gradient_initial_version = initial._version

    def _gradient_initial_is_zero(self) -> bool:
        """Check the Gradient Initial State without syncing the common case."""
        initial = self.gradient_message.initial_state
        if (
            id(initial) == self._default_gradient_initial_identity
            and initial._version == self._default_gradient_initial_version
        ):
            return True
        return torch.count_nonzero(initial).item() == 0

    @staticmethod
    def _validate_aggregation(aggregation: Any, name: str) -> None:
        if isinstance(aggregation, str):
            if aggregation not in MHD_Node._BUILTIN_AGGREGATIONS:
                raise ValueError(f"不支持的 {name}: {aggregation}")
            return
        if not callable(aggregation):
            raise TypeError(f"{name} 必须是内置字符串或 callable")
        if isinstance(aggregation, nn.Module) and any(
            True for _ in aggregation.parameters()
        ):
            raise ValueError(f"{name} callable 不得持有可学习参数")

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MHD_Node) and self.id == other.id

    def reset(self) -> 'MHD_Node':
        self.feature_message.reset()
        self.gradient_message.reset()
        return self

    def to_device(self, device: torch.device) -> 'MHD_Node':
        default_zero_unchanged = (
            id(self.gradient_message.initial_state)
            == self._default_gradient_initial_identity
            and self.gradient_message.initial_state._version
            == self._default_gradient_initial_version
        )
        self.feature_message.to_device(device)
        self.gradient_message.to_device(device)
        if default_zero_unchanged:
            self._remember_default_zero_gradient()
        return self

    @staticmethod
    def _aggregate_messages(
        current: torch.Tensor,
        incomings: Sequence[torch.Tensor],
        aggregation: Union[str, Callable[[torch.Tensor, Sequence[torch.Tensor]], torch.Tensor]],
    ) -> torch.Tensor:
        incoming_list = list(incomings)
        if not incoming_list:
            return current
        for incoming in incoming_list:
            if current.device != incoming.device:
                raise ValueError(
                    f"Message 聚合设备不一致: {current.device} vs {incoming.device}"
                )
        if callable(aggregation) and not isinstance(aggregation, str):
            result = aggregation(current, tuple(incoming_list))
            if not isinstance(result, torch.Tensor):
                raise TypeError("自定义 Message aggregation 必须返回 Tensor")
            return result
        if aggregation == "replace":
            return incoming_list[-1]
        result = current
        if aggregation in {"sum", "avg"}:
            for incoming in incoming_list:
                result = result + incoming
            return result / (len(incoming_list) + 1) if aggregation == "avg" else result
        if aggregation == "max":
            for incoming in incoming_list:
                result = torch.maximum(result, incoming)
            return result
        if aggregation == "min":
            for incoming in incoming_list:
                result = torch.minimum(result, incoming)
            return result
        if aggregation == "mul":
            for incoming in incoming_list:
                result = result * incoming
            return result
        raise ValueError(f"不支持的 Message aggregation: {aggregation}")

    def aggregate_feature_messages(
        self,
        current: torch.Tensor,
        incomings: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        values = [incomings] if isinstance(incomings, torch.Tensor) else list(incomings)
        return self._aggregate_messages(current, values, self.feature_aggregation)

    def aggregate_gradient_messages(
        self,
        current: torch.Tensor,
        incomings: Union[torch.Tensor, Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        values = [incomings] if isinstance(incomings, torch.Tensor) else list(incomings)
        return self._aggregate_messages(current, values, self.gradient_aggregation)


@dataclass
class MHD_Edge:
    """Hyperedge carrying ordered forward and optional backward Operations."""

    @dataclass(eq=False)
    class Operation:
        """One transform owned by an :class:`MHD_Edge`."""

        function: Union[str, nn.Module, partial, Callable]

        def __post_init__(self) -> None:
            if isinstance(self.function, str):
                if not self.function.startswith("."):
                    raise ValueError("字符串 Operation 必须以 '.' 开头")
            elif not callable(self.function):
                raise TypeError("Operation 必须包装字符串、nn.Module、partial 或 callable")

        def to_device(self, device: torch.device) -> 'MHD_Edge.Operation':
            if isinstance(self.function, nn.Module):
                self.function = self.function.to(device, non_blocking=True)
            return self

    id: int
    name: str
    edge_operations: List[Operation]
    backward_operations: Optional[List[Operation]] = None

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError("边 id 必须是非负整数")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("边 name 必须是非空字符串")
        if not isinstance(self.edge_operations, list):
            raise TypeError("edge_operations 必须是 list")
        if not self.edge_operations:
            raise ValueError("edge_operations 不能为空")
        if not all(isinstance(operation, MHD_Edge.Operation) for operation in self.edge_operations):
            raise TypeError("edge_operations 中每一项都必须是 MHD_Edge.Operation")
        if self.backward_operations is not None:
            if not isinstance(self.backward_operations, list):
                raise TypeError("backward_operations 必须是 list 或 None")
            for operation in self.backward_operations:
                if not isinstance(operation, MHD_Edge.Operation):
                    raise TypeError("backward_operations 中每一项都必须是 MHD_Edge.Operation")
                function = operation.function
                if isinstance(function, nn.Module) and any(
                    True for _ in function.parameters()
                ):
                    raise ValueError("backward_operations 不得持有可学习参数")

    def __hash__(self):
        """基于ID的哈希函数"""
        return hash(self.id)

    def __eq__(self, other):
        """基于ID的相等判断"""
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id

    def to_device(self, device: torch.device) -> 'MHD_Edge':
        """
        将边中的可学习模块迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的边自身
        """
        for operation in self.edge_operations:
            operation.to_device(device)
        for operation in self.backward_operations or []:
            operation.to_device(device)
        return self

    def named_edge_parameters(self) -> List[Tuple[str, nn.Parameter]]:
        """Return unique learnable parameters using stable edge-local names."""
        result: List[Tuple[str, nn.Parameter]] = []
        seen: Set[int] = set()
        for operation_index, operation in enumerate(self.edge_operations):
            function = operation.function
            if not isinstance(function, nn.Module):
                continue
            for parameter_name, parameter in function.named_parameters():
                if not parameter.requires_grad or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                result.append((f"op{operation_index}.{parameter_name}", parameter))
        return result

    @staticmethod
    def _apply_callable_to_list(op, tensor_list: List[torch.Tensor]):
        """Call one Operation with Sort-Matrix-ordered positional Messages.

        V3 guessed between positional, list-valued and elementwise calls by
        swallowing exceptions.  V4 deliberately has one deterministic rule:
        one Message is one argument and multiple Messages are positional
        arguments.  An Operation that wants elementwise application can return
        ``[fn(message) for message in messages]`` explicitly; real computation
        errors are therefore never mistaken for a different calling convention.
        """
        if len(tensor_list) == 1:
            return op(tensor_list[0])
        return op(*tensor_list)

    def execute_edge_operations(self, input_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        执行 edge_operations 操作序列

        多个 Message 按 Sort Matrix 顺序作为位置参数传入；字符串操作逐
        Message 应用。最终输出统一转换为 Message 列表。

        Args:
            input_list: 输入张量列表（头节点状态，按 sort_matrix 排序）

        Returns:
            输出张量列表
        """
        data = input_list
        for operation in self.edge_operations:
            op = operation.function
            if isinstance(data, tuple):
                data = list(data)
            if isinstance(data, list):
                if isinstance(op, str):
                    data = [parse_string_operation(op, t) for t in data]
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = self._apply_callable_to_list(op, data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
            else:
                if isinstance(op, str):
                    data = parse_string_operation(op, data)
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = op(data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
        if isinstance(data, torch.Tensor):
            data = [data]
        elif isinstance(data, tuple):
            data = list(data)
        elif not isinstance(data, list):
            raise TypeError(f"edge_operations 最终输出类型错误: {type(data)}")
        if not all(isinstance(tensor, torch.Tensor) for tensor in data):
            bad_types = [type(value).__name__ for value in data if not isinstance(value, torch.Tensor)]
            raise TypeError(f"edge_operations 输出必须全部为 Tensor，发现: {bad_types}")
        return data

    def execute_backward_operations(
        self,
        gradients: Sequence[Optional[torch.Tensor]],
        *,
        forward_inputs: Sequence[torch.Tensor],
        forward_outputs: Sequence[torch.Tensor],
    ) -> Tuple[List[Optional[torch.Tensor]], Dict[str, Optional[torch.Tensor]]]:
        """Execute one edge's explicit backward operation sequence."""
        if self.backward_operations is None:
            raise RuntimeError(f"边 '{self.name}' 未定义 backward_operations")
        parameter_items = self.named_edge_parameters()
        distributed_reference = next(
            (
                parameter
                for _, parameter in parameter_items
                if hasattr(parameter, "placements") and hasattr(parameter, "device_mesh")
            ),
            None,
        )

        def to_custom_context(value: Optional[torch.Tensor]):
            if value is None or distributed_reference is None:
                return value
            if hasattr(value, "placements"):
                return value
            from torch.distributed.tensor import DTensor, Replicate

            placements = tuple(
                Replicate() for _ in range(distributed_reference.device_mesh.ndim)
            )
            return DTensor.from_local(
                value,
                device_mesh=distributed_reference.device_mesh,
                placements=placements,
                run_check=False,
            )

        def match_layout(
            value: Optional[torch.Tensor],
            reference: torch.Tensor,
        ) -> Optional[torch.Tensor]:
            if value is None:
                return None
            value_distributed = hasattr(value, "placements")
            reference_distributed = hasattr(reference, "placements")
            if reference_distributed:
                if value_distributed:
                    return value.redistribute(placements=reference.placements)
                from torch.distributed.tensor import DTensor

                return DTensor.from_local(
                    value,
                    device_mesh=reference.device_mesh,
                    placements=reference.placements,
                    run_check=False,
                )
            if value_distributed:
                from torch.distributed.tensor import Replicate

                replicated = tuple(
                    Replicate() for _ in range(value.device_mesh.ndim)
                )
                return value.redistribute(placements=replicated).to_local()
            return value

        data: Any = [to_custom_context(gradient) for gradient in gradients]
        parameter_view = MappingProxyType(dict(parameter_items))
        context_inputs = tuple(to_custom_context(value) for value in forward_inputs)
        context_outputs = tuple(to_custom_context(value) for value in forward_outputs)
        accumulated: Dict[str, Optional[torch.Tensor]] = {}
        for wrapped_operation in self.backward_operations:
            operation = wrapped_operation.function
            if isinstance(operation, str):
                source = data if isinstance(data, (list, tuple)) else [data]
                data = [
                    None if value is None else parse_string_operation(operation, value)
                    for value in source
                ]
                continue
            if not callable(operation):
                raise TypeError(f"不支持的 backward operation: {type(operation)!r}")
            gradient_inputs = tuple(data) if isinstance(data, (list, tuple)) else (data,)
            context = {
                "forward_inputs": context_inputs,
                "forward_outputs": context_outputs,
                "parameters": parameter_view,
            }
            result = (
                operation(gradient_inputs[0], **context)
                if len(gradient_inputs) == 1
                else operation(*gradient_inputs, **context)
            )
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "backward operation 必须返回 (next_gradients, parameter_gradient_mapping)"
                )
            data, contributions = result
            if not isinstance(contributions, Mapping):
                raise TypeError("parameter_gradient_mapping 必须是 Mapping")
            unknown = set(contributions) - set(parameter_view)
            if unknown:
                raise KeyError(f"边 '{self.name}' 返回未知参数梯度: {sorted(unknown)}")
            for parameter_name, contribution in contributions.items():
                if contribution is not None and not isinstance(contribution, torch.Tensor):
                    raise TypeError(f"参数梯度 '{parameter_name}' 必须是 Tensor 或 None")
                parameter = parameter_view[parameter_name]
                if contribution is not None and (
                    contribution.shape != parameter.shape
                ):
                    raise ValueError(
                        f"参数梯度 '{parameter_name}' 的 shape 与参数不一致"
                    )
                previous = accumulated.get(parameter_name)
                if contribution is None:
                    accumulated.setdefault(parameter_name, None)
                elif previous is None:
                    accumulated[parameter_name] = contribution
                else:
                    accumulated[parameter_name] = previous + contribution
        required = set(parameter_view)
        missing = required - set(accumulated)
        if missing:
            raise KeyError(f"边 '{self.name}' 缺少参数梯度: {sorted(missing)}")
        if isinstance(data, torch.Tensor) or data is None:
            result_gradients = [data]
        elif isinstance(data, tuple):
            result_gradients = list(data)
        elif isinstance(data, list):
            result_gradients = data
        else:
            raise TypeError("backward_operations 最终梯度必须是 Tensor、序列或 None")
        if len(result_gradients) != len(forward_inputs):
            raise ValueError(
                f"边 '{self.name}' 的输入梯度数 {len(result_gradients)} "
                f"与前向输入数 {len(forward_inputs)} 不一致"
            )
        for index, value in enumerate(result_gradients):
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"第 {index} 个输入梯度必须是 Tensor 或 None")
        normalized_inputs = [
            match_layout(value, reference)
            for value, reference in zip(result_gradients, forward_inputs)
        ]
        normalized_parameters = {
            name: match_layout(accumulated[name], parameter)
            for name, parameter in parameter_items
        }
        return normalized_inputs, normalized_parameters


class _MHD_CustomEdgeFunction(torch.autograd.Function):
    """Connect MHD custom backward operations to native PyTorch backward."""

    @staticmethod
    def forward(ctx, edge: MHD_Edge, num_inputs: int, *arguments: torch.Tensor):
        inputs = tuple(arguments[:num_inputs])
        parameters = tuple(arguments[num_inputs:])
        outputs = tuple(edge.execute_edge_operations(list(inputs)))
        ctx.edge = edge
        ctx.num_inputs = num_inputs
        ctx.num_parameters = len(parameters)
        sample = next((value for value in arguments if isinstance(value, torch.Tensor)), None)
        ctx.device_type = sample.device.type if sample is not None else "cpu"
        ctx.autocast_enabled = torch.is_autocast_enabled(ctx.device_type)
        ctx.autocast_dtype = torch.get_autocast_dtype(ctx.device_type)
        ctx.save_for_backward(*inputs, *outputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @staticmethod
    def backward(ctx, *output_gradients: Optional[torch.Tensor]):
        saved = ctx.saved_tensors
        inputs = saved[:ctx.num_inputs]
        outputs = saved[ctx.num_inputs:]
        with torch.autocast(
            device_type=ctx.device_type,
            dtype=ctx.autocast_dtype,
            enabled=ctx.autocast_enabled,
        ):
            input_gradients, parameter_mapping = ctx.edge.execute_backward_operations(
                output_gradients,
                forward_inputs=inputs,
                forward_outputs=outputs,
            )
        parameter_gradients = [
            parameter_mapping[name]
            for name, _ in ctx.edge.named_edge_parameters()
        ]
        return (None, None, *input_gradients, *parameter_gradients)


@dataclass(frozen=True)
class _MHD_ForwardTrace:
    level: int
    edge_id: int
    head_ids: Tuple[int, ...]
    tail_ids: Tuple[int, ...]
    inputs: Tuple[torch.Tensor, ...]
    outputs: Tuple[torch.Tensor, ...]


@dataclass
class MHD_Topo:
    """
    超图拓扑类 - 多层级矩阵列表

    特性：
    1. role_matrices 和 sort_matrices 是等长的列表，每个元素对应一个执行层级
    2. 每个层级内角色矩阵元素为 -1/0/1，同一条边可在不同层级出现
    3. 所有矩阵形状必须一致（边数 × 节点数）

    Attributes:
        role_matrices: 角色矩阵列表，形状均为 (边数, 节点数)
        sort_matrices: 排序矩阵列表，形状均为 (边数, 节点数)
    """
    role_matrices: List[torch.Tensor] = field(default_factory=list)
    sort_matrices: List[torch.Tensor] = field(default_factory=list)
    backward_role_matrices: Optional[List[torch.Tensor]] = None
    backward_sort_matrices: Optional[List[torch.Tensor]] = None
    backward_is_auto: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        """Validate the symmetric forward/backward topology."""
        self._validate_matrix_pair(self.role_matrices, self.sort_matrices, "前向")
        one_backward_missing = (self.backward_role_matrices is None) != (
            self.backward_sort_matrices is None
        )
        if one_backward_missing:
            raise ValueError("backward_role_matrices 与 backward_sort_matrices 必须同时提供")
        self.backward_is_auto = self.backward_role_matrices is None
        if self.backward_is_auto:
            self.backward_role_matrices = [(-matrix).clone() for matrix in self.role_matrices]
            self.backward_sort_matrices = [matrix.clone() for matrix in self.sort_matrices]
        self._validate_matrix_pair(
            self.backward_role_matrices,
            self.backward_sort_matrices,
            "反向",
        )
        if len(self.backward_role_matrices) != len(self.role_matrices):
            raise ValueError("前向与反向拓扑层数必须一致")
        for level, (forward_role, backward_role) in enumerate(
            zip(self.role_matrices, self.backward_role_matrices)
        ):
            if forward_role.shape != backward_role.shape:
                raise ValueError(f"第 {level} 层前向与反向拓扑形状必须一致")

    @staticmethod
    def _validate_matrix_pair(
        roles: Optional[List[torch.Tensor]],
        sorts: Optional[List[torch.Tensor]],
        phase: str,
    ) -> None:
        if not roles or not sorts:
            raise ValueError(f"{phase} role_matrices 和 sort_matrices 不能为空")
        if len(roles) != len(sorts):
            raise ValueError(f"{phase} role_matrices 与 sort_matrices 长度必须相同")
        ref_shape = roles[0].shape
        ref_device = roles[0].device
        for i, (r, s) in enumerate(zip(roles, sorts)):
            if r.ndim != 2 or s.ndim != 2:
                raise ValueError(f"{phase}第{i}层拓扑矩阵必须是二维矩阵")
            if r.shape != ref_shape:
                raise ValueError(f"{phase}第{i}层 role 矩阵形状不一致")
            if s.shape != ref_shape:
                raise ValueError(f"{phase}第{i}层 sort 矩阵形状不一致")
            if r.device != ref_device:
                raise ValueError(f"{phase}第{i}层 role 矩阵设备不一致")
            if s.device != ref_device:
                raise ValueError(f"{phase}第{i}层 sort 矩阵设备不一致")
            if r.dtype == torch.bool or r.is_floating_point() or r.is_complex():
                raise TypeError(f"{phase}第{i}层 role 矩阵必须使用整数 dtype")
            valid_roles = torch.logical_or(torch.logical_or(r == -1, r == 0), r == 1)
            if not bool(valid_roles.all().item()):
                raise ValueError(f"{phase}第{i}层 role 矩阵只允许 -1、0、1")

    def to_device(self, device: torch.device) -> 'MHD_Topo':
        """
        将所有层级的矩阵迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的拓扑自身
        """
        for i in range(len(self.role_matrices)):
            if self.role_matrices[i].device != device:
                self.role_matrices[i] = self.role_matrices[i].to(device, non_blocking=True)
            if self.sort_matrices[i].device != device:
                self.sort_matrices[i] = self.sort_matrices[i].to(device, non_blocking=True)
            if self.backward_role_matrices[i].device != device:
                self.backward_role_matrices[i] = self.backward_role_matrices[i].to(
                    device, non_blocking=True
                )
            if self.backward_sort_matrices[i].device != device:
                self.backward_sort_matrices[i] = self.backward_sort_matrices[i].to(
                    device, non_blocking=True
                )
        return self

    def get_topo(self, level: int, edge_id: int, node_id: int, matrix_type: str = "role") -> int:
        """
        获取指定层级、边和节点的拓扑值

        Args:
            level: 层级索引
            edge_id: 边索引
            node_id: 节点索引
            matrix_type: 矩阵类型，'role'或'sort'

        Returns:
            拓扑值，如果索引越界返回0
        """
        matrices_by_type = {
            "role": self.role_matrices,
            "sort": self.sort_matrices,
            "backward_role": self.backward_role_matrices,
            "backward_sort": self.backward_sort_matrices,
        }
        if matrix_type not in matrices_by_type:
            raise ValueError("matrix_type 必须是 role/sort/backward_role/backward_sort")
        matrices = matrices_by_type[matrix_type]
        if 0 <= level < len(matrices):
            mat = matrices[level]
            if 0 <= edge_id < mat.shape[0] and 0 <= node_id < mat.shape[1]:
                return int(mat[edge_id, node_id].item())
        return 0

    def to_list(self, matrix_type: str = "role") -> List[List[List[int]]]:
        """
        转换为嵌套列表形式（层级 × 边 × 节点）

        Args:
            matrix_type: 矩阵类型，'role'或'sort'

        Returns:
            三维列表
        """
        matrices_by_type = {
            "role": self.role_matrices,
            "sort": self.sort_matrices,
            "backward_role": self.backward_role_matrices,
            "backward_sort": self.backward_sort_matrices,
        }
        if matrix_type not in matrices_by_type:
            raise ValueError("matrix_type 必须是 role/sort/backward_role/backward_sort")
        matrices = matrices_by_type[matrix_type]
        return [m.tolist() for m in matrices]

    def __hash__(self):
        """基于所有矩阵内容的哈希函数"""
        flat_role = tuple(tuple(r.flatten().tolist()) for r in self.role_matrices)
        flat_sort = tuple(tuple(s.flatten().tolist()) for s in self.sort_matrices)
        flat_backward_role = tuple(
            tuple(r.flatten().tolist()) for r in self.backward_role_matrices
        )
        flat_backward_sort = tuple(
            tuple(s.flatten().tolist()) for s in self.backward_sort_matrices
        )
        return hash((flat_role, flat_sort, flat_backward_role, flat_backward_sort))

    def __eq__(self, other):
        """基于所有矩阵内容的相等判断"""
        if not isinstance(other, MHD_Topo):
            return False
        if len(self.role_matrices) != len(other.role_matrices):
            return False
        for r1, r2 in zip(self.role_matrices, other.role_matrices):
            if not torch.equal(r1, r2):
                return False
        for s1, s2 in zip(self.sort_matrices, other.sort_matrices):
            if not torch.equal(s1, s2):
                return False
        for r1, r2 in zip(self.backward_role_matrices, other.backward_role_matrices):
            if not torch.equal(r1, r2):
                return False
        for s1, s2 in zip(self.backward_sort_matrices, other.backward_sort_matrices):
            if not torch.equal(s1, s2):
                return False
        return True

    def validate_topo(self, num_edges: int, num_nodes: int) -> None:
        """
        验证所有层级的拓扑矩阵维度

        Args:
            num_edges: 预期的边数
            num_nodes: 预期的节点数

        Raises:
            ValueError: 当维度不匹配时
        """
        for i, (r, s) in enumerate(zip(self.role_matrices, self.sort_matrices)):
            if r.shape[0] != num_edges or r.shape[1] != num_nodes:
                raise ValueError(
                    f"第{i}层拓扑维度不匹配: 边{num_edges}×节点{num_nodes}，实际{r.shape}"
                )
        for i, (r, s) in enumerate(
            zip(self.backward_role_matrices, self.backward_sort_matrices)
        ):
            if r.shape != (num_edges, num_nodes) or s.shape != (num_edges, num_nodes):
                raise ValueError(f"第{i}层反向拓扑维度不匹配")

    @property
    def num_levels(self) -> int:
        """返回拓扑的层级数"""
        return len(self.role_matrices)


class MHD_Graph(nn.Module):
    """
    多超图动态框架核心类 - Version 4.0（双向 Message + 多层级拓扑）

    特性：
    1. Node 同时承载 Feature Message 与 Gradient Message
    2. 两类 Message 均包含 Initial State 与 Current State
    3. 前向与反向都由 Role/Sort Matrix 和确定性 n 元聚合驱动
    4. Operation 支持原生 nn.Module、函数、partial 和受限字符串操作
    5. 同一条 Edge 可跨层复用，表达循环网络的静态展开
    6. 自动反向拓扑与显式反向拓扑共享 graph.backward 接口
    7. 图合并与可视化保留稳定的 Node/Edge/Topo/Graph 概念

    Author: Souray Meng (孟号丁)
    """

    def __init__(self, nodes: Set[MHD_Node], edges: Set[MHD_Edge], topos: Set[MHD_Topo],
                 device: torch.device = None):
        """
        初始化MHD图

        Args:
            nodes: 节点集合，每个节点包含 Feature/Gradient Message
            edges: 超边集合，每条边包含 Operation 序列
            topos: 拓扑集合（应只包含一个 MHD_Topo 对象）
            device: 计算设备，默认为CUDA(可用)或CPU
        """
        super().__init__()

        # 统一设备配置
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 核心超图对象
        self.nodes = nodes
        self.edges = edges

        # 拓扑验证：确保只有一个拓扑对象
        if len(topos) != 1:
            raise ValueError(f"topos 必须且只能包含一个 MHD_Topo，实际为 {len(topos)}")
        self.topo = next(iter(topos))

        # 统一设备
        self._unify_device()

        # 建立索引系统
        self._node_by_id: Dict[int, MHD_Node] = {}
        self._node_by_name: Dict[str, MHD_Node] = {}
        self._edge_by_id: Dict[int, MHD_Edge] = {}
        self._edge_by_name: Dict[str, MHD_Edge] = {}
        self._build_indices()

        # 参数注册容器。名称只由稳定的数值 ID 组成，用户的边名不会污染 FQN。
        self.edge_module_map = nn.ModuleDict()

        # 获取层级数
        self.num_levels = self.topo.num_levels if self.topo else 0

        # 验证拓扑维度
        if self.topo:
            self.topo.validate_topo(len(self.edges), len(self.nodes))

        # 拓扑排序和参数注册
        self._register_all_params()
        self.compact_topological_sort()
        self._forward_trace: List[_MHD_ForwardTrace] = []

        logger.info(
            "MHD图初始化完成 | 设备=%s 节点=%d 边=%d 层级=%d",
            self.device,
            len(nodes),
            len(edges),
            self.num_levels,
        )

    def _unify_device(self) -> None:
        """
        统一所有组件的计算设备
        """
        if self.topo:
            self.topo = self.topo.to_device(self.device)
        for node in self.nodes:
            node.to_device(self.device)
        for edge in self.edges:
            edge.to_device(self.device)

    def _build_indices(self) -> None:
        """
        构建节点和边的索引字典
        """
        self._node_by_id.clear()
        self._node_by_name.clear()
        self._edge_by_id.clear()
        self._edge_by_name.clear()

        for node in self.nodes:
            if node.id in self._node_by_id:
                raise ValueError(f"节点ID重复: {node.id}")
            if node.name in self._node_by_name:
                raise ValueError(f"节点名称重复: {node.name}")
            self._node_by_id[node.id] = node
            self._node_by_name[node.name] = node

        for edge in self.edges:
            if edge.id in self._edge_by_id:
                raise ValueError(f"边ID重复: {edge.id}")
            if edge.name in self._edge_by_name:
                raise ValueError(f"边名称重复: {edge.name}")
            self._edge_by_id[edge.id] = edge
            self._edge_by_name[edge.name] = edge

        expected_node_ids = list(range(len(self.nodes)))
        expected_edge_ids = list(range(len(self.edges)))
        if sorted(self._node_by_id) != expected_node_ids:
            raise ValueError(f"节点 ID 必须连续且从 0 开始，期望 {expected_node_ids}")
        if sorted(self._edge_by_id) != expected_edge_ids:
            raise ValueError(f"边 ID 必须连续且从 0 开始，期望 {expected_edge_ids}")
        self._nodes_in_id_order = tuple(self._node_by_id[index] for index in expected_node_ids)
        self._edges_in_id_order = tuple(self._edge_by_id[index] for index in expected_edge_ids)

    def update_indices(self, new_nodes: Set[MHD_Node] = None, new_edges: Set[MHD_Edge] = None) -> None:
        """
        更新索引（当节点或边集合发生变化时调用）
        """
        if new_nodes:
            for node in new_nodes:
                self._node_by_id[node.id] = node
                self._node_by_name[node.name] = node
        if new_edges:
            for edge in new_edges:
                self._edge_by_id[edge.id] = edge
                self._edge_by_name[edge.name] = edge

    def to(self, *args, **kwargs) -> 'MHD_Graph':
        """Move registered modules, node states and the two topology matrices together."""
        super().to(*args, **kwargs)
        try:
            target_device = torch._C._nn._parse_to(*args, **kwargs)[0]
        except (AttributeError, TypeError):
            target_device = kwargs.get("device", args[0] if args else None)
        if target_device is not None:
            self.device = torch.device(target_device)
            self._unify_device()
        return self

    def get_node_by_id(self, node_id: int) -> Optional[MHD_Node]:
        """通过ID获取节点"""
        return self._node_by_id.get(node_id)

    def get_node_by_name(self, node_name: str) -> Optional[MHD_Node]:
        """通过名称获取节点"""
        return self._node_by_name.get(node_name)

    def get_edge_by_id(self, edge_id: int) -> Optional[MHD_Edge]:
        """通过ID获取边"""
        return self._edge_by_id.get(edge_id)

    def get_edge_by_name(self, edge_name: str) -> Optional[MHD_Edge]:
        """通过名称获取边"""
        return self._edge_by_name.get(edge_name)

    def compact_topological_sort(self) -> 'MHD_Graph':
        """Compile deterministic forward and backward execution plans."""
        if self.topo is None or self.num_levels == 0:
            self._edge_sequence_per_level = []
            self._execution_plan_per_level = []
            self._backward_edge_sequence_per_level = []
            self._backward_execution_plan_per_level = []
            return self
        (
            self._edge_sequence_per_level,
            self._execution_plan_per_level,
        ) = self._compile_topology_phase(
            self.topo.role_matrices,
            self.topo.sort_matrices,
            "前向",
        )
        (
            self._backward_edge_sequence_per_level,
            self._backward_execution_plan_per_level,
        ) = self._compile_topology_phase(
            self.topo.backward_role_matrices,
            self.topo.backward_sort_matrices,
            "反向",
        )
        return self

    def _compile_topology_phase(
        self,
        role_matrices: Sequence[torch.Tensor],
        sort_matrices: Sequence[torch.Tensor],
        phase: str,
    ) -> Tuple[List[List[int]], List[Tuple[_MHD_ExecutionStep, ...]]]:
        num_edges, num_nodes = len(self.edges), len(self.nodes)
        sequences: List[List[int]] = []
        plans: List[Tuple[_MHD_ExecutionStep, ...]] = []
        role_levels = [matrix.detach().cpu().tolist() for matrix in role_matrices]
        sort_levels = [matrix.detach().cpu().tolist() for matrix in sort_matrices]
        for level, role in enumerate(role_levels):
            role = role_levels[level]
            active_edge_ids = [eid for eid in range(num_edges) if any(value != 0 for value in role[eid])]
            if not active_edge_ids:
                sequences.append([])
                plans.append(tuple())
                continue
            node_to_out_edges = defaultdict(set)
            for eid in active_edge_ids:
                for nid in range(num_nodes):
                    if role[eid][nid] > 0:
                        node_to_out_edges[nid].add(eid)
            edge_deps = defaultdict(set)
            for eid in active_edge_ids:
                for nid in range(num_nodes):
                    if role[eid][nid] < 0:
                        edge_deps[eid].update(node_to_out_edges.get(nid, set()))
                edge_deps[eid].discard(eid)
            edge_in_degree = {eid: len(edge_deps.get(eid, set())) for eid in active_edge_ids}
            reverse_deps = defaultdict(set)
            for eid, deps in edge_deps.items():
                for dep in deps:
                    reverse_deps[dep].add(eid)
            remaining = set(active_edge_ids)
            level_sequence: List[int] = []
            while remaining:
                current = sorted([e for e in remaining if edge_in_degree[e] == 0])
                if not current:
                    edge_names = [self.get_edge_by_id(eid).name for eid in remaining]
                    raise ValueError(f"{phase}第{level}层边拓扑存在环: {edge_names}")
                level_sequence.extend(current)
                for eid in current:
                    remaining.remove(eid)
                    for next_eid in reverse_deps.get(eid, set()):
                        edge_in_degree[next_eid] -= 1
            sequences.append(level_sequence)
            sort_values = sort_levels[level]
            steps: List[_MHD_ExecutionStep] = []
            for edge_id in level_sequence:
                head_ids = [nid for nid in range(num_nodes) if role[edge_id][nid] < 0]
                tail_ids = [nid for nid in range(num_nodes) if role[edge_id][nid] > 0]
                if not head_ids or not tail_ids:
                    continue
                head_ids.sort(key=lambda nid: sort_values[edge_id][nid])
                tail_ids.sort(key=lambda nid: sort_values[edge_id][nid])
                steps.append(
                    _MHD_ExecutionStep(
                        edge_id,
                        tuple(head_ids),
                        tuple(tail_ids),
                        self._edges_in_id_order[edge_id],
                        tuple(self._nodes_in_id_order[nid] for nid in tail_ids),
                    )
                )
            plans.append(tuple(steps))
        return sequences, plans

    def _register_all_params(self) -> 'MHD_Graph':
        """
        注册边中的可学习模块到 ModuleDict
        """
        registered_by_identity: Dict[int, str] = {}
        for edge in sorted(self.edges, key=lambda x: x.id):
            for idx, operation in enumerate(edge.edge_operations):
                op = operation.function
                if isinstance(op, nn.Module):
                    identity = id(op)
                    if identity not in registered_by_identity:
                        module_name = f"e{edge.id}_o{idx}"
                        self.edge_module_map[module_name] = op.to(self.device)
                        registered_by_identity[identity] = module_name
                    operation.function = self.edge_module_map[registered_by_identity[identity]]
        return self

    def sort_nodes_by_topo(self, level: int = 0, edge_id: int = 0) -> List[Tuple[int, int]]:
        """
        按指定层级和边的 sort_matrix 排序节点

        Args:
            level: 层级索引，默认为0
            edge_id: 边索引

        Returns:
            排序后的 (节点索引, 排序值) 列表
        """
        if self.topo is None or level >= self.num_levels:
            return []
        if edge_id >= self.topo.sort_matrices[level].shape[0]:
            return []
        indexed = list(enumerate(self.topo.sort_matrices[level][edge_id].tolist()))
        return sorted(indexed, key=lambda p: p[1])

    def forward(self, levels: Optional[List[int]] = None) -> 'MHD_Graph':
        """Execute deterministic n-ary Feature Message routing."""
        if not self.nodes or not self.edges or self.topo is None:
            return self
        if levels is None:
            levels = list(range(self.num_levels))
        self._forward_trace = []
        record_backward_trace = torch.is_grad_enabled()
        for level in levels:
            if level < 0 or level >= self.num_levels:
                raise IndexError(f"层级索引 {level} 超出范围 [0, {self.num_levels-1}]")
            node_current = [
                node.feature_message.current_state for node in self._nodes_in_id_order
            ]
            pending: Dict[int, List[torch.Tensor]] = defaultdict(list)

            def flush(node_id: int) -> None:
                incomings = pending.pop(node_id, None)
                if incomings:
                    node_current[node_id] = self._nodes_in_id_order[
                        node_id
                    ].aggregate_feature_messages(node_current[node_id], incomings)

            for step in self._execution_plan_per_level[level]:
                for node_id in step.head_ids:
                    flush(node_id)
                edge = step.edge
                head_tensors = [node_current[nid] for nid in step.head_ids]
                if edge.backward_operations is None:
                    output_list = edge.execute_edge_operations(head_tensors)
                else:
                    parameters = [parameter for _, parameter in edge.named_edge_parameters()]
                    result = _MHD_CustomEdgeFunction.apply(
                        edge,
                        len(head_tensors),
                        *head_tensors,
                        *parameters,
                    )
                    output_list = list(result) if isinstance(result, tuple) else [result]
                if len(output_list) != len(step.tail_ids):
                    raise ValueError(
                        f"边 '{edge.name}' 输出数量 ({len(output_list)}) 与尾节点数 ({len(step.tail_ids)}) 不匹配"
                    )
                if record_backward_trace:
                    self._forward_trace.append(
                        _MHD_ForwardTrace(
                            level,
                            step.edge_id,
                            step.head_ids,
                            step.tail_ids,
                            tuple(head_tensors),
                            tuple(output_list),
                        )
                    )
                for node_id, output in zip(step.tail_ids, output_list):
                    pending[node_id].append(output)
            for node_id in sorted(pending):
                flush(node_id)
            for node, value in zip(self._nodes_in_id_order, node_current):
                node.feature_message.current_state = value
                if value.requires_grad:
                    value.retain_grad()
        return self

    def backward(
        self,
        seeds: Mapping[str, Optional[torch.Tensor]],
        *,
        retain_graph: bool = False,
    ) -> 'MHD_Graph':
        """Backpropagate one or more named Gradient Message seeds."""
        if not isinstance(seeds, Mapping) or not seeds:
            raise ValueError("seeds 必须是非空的节点名称 Mapping")
        unknown = set(seeds) - set(self._node_by_name)
        if unknown:
            raise KeyError(f"未知反向根节点: {sorted(unknown)}")
        # This is an internal implementation choice, not a public execution
        # mode.  Both branches implement the same graph.backward(...) contract.
        # The native route is selected only for the subset whose equivalence is
        # covered by node-, parameter- and optimizer-level numerical tests.
        use_equivalent_pytorch_backward = (
            self.topo.backward_is_auto
            and all(
                node.gradient_aggregation == "sum"
                for node in self._nodes_in_id_order
            )
            and self._gradient_initial_states_are_zero()
        )
        if use_equivalent_pytorch_backward:
            self._backward_with_pytorch(seeds, retain_graph=retain_graph)
        else:
            for node in self._nodes_in_id_order:
                node.gradient_message.reset()
            self._execute_backward_messages(seeds, retain_graph=retain_graph)
        return self

    def _gradient_initial_states_are_zero(self) -> bool:
        """Return whether native autograd has the same Gradient Message seeds.

        A non-zero Gradient Initial State is a real message source, not merely
        metadata.  Such states must therefore use the general hypergraph route.
        ``count_nonzero`` keeps the test exact across floating, complex and
        distributed-compatible gradient dtypes; this check is private and does
        not introduce a user-facing execution option.
        """
        return all(node._gradient_initial_is_zero() for node in self._nodes_in_id_order)

    def _normalize_seeds(
        self,
        seeds: Mapping[str, Optional[torch.Tensor]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        roots: List[torch.Tensor] = []
        gradients: List[torch.Tensor] = []
        for name, seed in seeds.items():
            value = self._node_by_name[name].feature_message.current_state
            if not (value.is_floating_point() or value.is_complex()):
                raise TypeError(f"反向根节点 '{name}' 必须是浮点或复数 Tensor")
            if not value.requires_grad:
                raise RuntimeError(f"反向根节点 '{name}' 不在可微计算图中")
            if seed is None:
                seed = torch.ones_like(value)
            if not isinstance(seed, torch.Tensor):
                raise TypeError(f"节点 '{name}' 的 seed 必须是 Tensor 或 None")
            if seed.shape != value.shape or seed.device != value.device:
                raise ValueError(f"节点 '{name}' 的 seed shape/device 与 Feature Message 不一致")
            roots.append(value)
            gradients.append(seed.to(dtype=value.dtype))
        return roots, gradients

    def _backward_with_pytorch(
        self,
        seeds: Mapping[str, Optional[torch.Tensor]],
        *,
        retain_graph: bool,
    ) -> None:
        roots, gradients = self._normalize_seeds(seeds)
        previous_gradients = {
            id(node.feature_message.current_state): (
                None
                if node.feature_message.current_state.grad is None
                else node.feature_message.current_state.grad.detach().clone()
            )
            for node in self._nodes_in_id_order
        }
        if len(roots) == 1:
            torch.autograd.backward(
                roots[0], gradients[0], retain_graph=retain_graph
            )
        else:
            torch.autograd.backward(roots, gradients, retain_graph=retain_graph)
        for node in self._nodes_in_id_order:
            feature = node.feature_message.current_state
            gradient = feature.grad
            if gradient is None:
                node.gradient_message.reset()
                continue
            previous = previous_gradients[id(feature)]
            if previous is not None:
                gradient = gradient - previous
            # Native execution is selected only when every Initial State is
            # zero and aggregation is sum, so this assignment is exactly the
            # same value without a redundant clone and addition kernel.
            node.gradient_message.current_state = gradient.detach()
        if not retain_graph:
            self._forward_trace = []

    def _execute_backward_messages(
        self,
        seeds: Mapping[str, Optional[torch.Tensor]],
        *,
        retain_graph: bool,
    ) -> None:
        if not self._forward_trace:
            raise RuntimeError("graph.backward() 前必须先执行 graph.forward()")
        roots, seed_tensors = self._normalize_seeds(seeds)
        del roots  # Values are routed by node name below rather than native traversal.
        current = [
            node.gradient_message.current_state for node in self._nodes_in_id_order
        ]
        pending: Dict[int, List[torch.Tensor]] = defaultdict(list)
        for name, seed in zip(seeds, seed_tensors):
            pending[self._node_by_name[name].id].append(seed)

        def flush(node_id: int) -> None:
            incomings = pending.pop(node_id, None)
            if incomings:
                current[node_id] = self._nodes_in_id_order[
                    node_id
                ].aggregate_gradient_messages(current[node_id], incomings)

        traces = {(trace.level, trace.edge_id): trace for trace in self._forward_trace}
        parameter_contributions: Dict[int, Tuple[nn.Parameter, torch.Tensor]] = {}
        for level in reversed(range(self.num_levels)):
            for step in self._backward_execution_plan_per_level[level]:
                for node_id in step.head_ids:
                    flush(node_id)
                trace = traces.get((level, step.edge_id))
                if trace is None:
                    raise RuntimeError(
                        f"缺少第 {level} 层边 '{step.edge.name}' 的前向执行轨迹"
                    )
                output_gradients: List[Optional[torch.Tensor]] = []
                for output, tail_id in zip(trace.outputs, trace.tail_ids):
                    final_state = self._nodes_in_id_order[
                        tail_id
                    ].feature_message.current_state
                    if final_state is output:
                        output_gradients.append(current[tail_id])
                    else:
                        output_gradients.append(
                            torch.autograd.grad(
                                final_state,
                                output,
                                grad_outputs=current[tail_id],
                                retain_graph=True,
                                allow_unused=True,
                            )[0]
                        )
                parameter_items = step.edge.named_edge_parameters()
                local_targets = [*trace.inputs, *(p for _, p in parameter_items)]
                differentiable_outputs = []
                differentiable_gradients = []
                for output, gradient in zip(trace.outputs, output_gradients):
                    if output.requires_grad and gradient is not None:
                        differentiable_outputs.append(output)
                        differentiable_gradients.append(gradient)
                if differentiable_outputs:
                    target_indices = [
                        index
                        for index, target in enumerate(local_targets)
                        if target.requires_grad
                    ]
                    differentiable_targets = [local_targets[index] for index in target_indices]
                    computed_gradients = (
                        torch.autograd.grad(
                            differentiable_outputs,
                            differentiable_targets,
                            grad_outputs=differentiable_gradients,
                            retain_graph=True,
                            allow_unused=True,
                        )
                        if differentiable_targets
                        else tuple()
                    )
                    local_gradient_list: List[Optional[torch.Tensor]] = [
                        None for _ in local_targets
                    ]
                    for target_index, gradient in zip(
                        target_indices, computed_gradients
                    ):
                        local_gradient_list[target_index] = gradient
                    local_gradients = tuple(local_gradient_list)
                else:
                    local_gradients = tuple(None for _ in local_targets)
                input_gradients = local_gradients[:len(trace.inputs)]
                if len(step.tail_ids) != len(input_gradients):
                    raise ValueError(
                        f"显式反向边 '{step.edge.name}' 的目标数必须等于前向输入数"
                    )
                for node_id, gradient in zip(step.tail_ids, input_gradients):
                    if gradient is not None:
                        pending[node_id].append(gradient)
                for (_, parameter), gradient in zip(
                    parameter_items,
                    local_gradients[len(trace.inputs):],
                ):
                    if gradient is None:
                        continue
                    identity = id(parameter)
                    if identity in parameter_contributions:
                        previous_parameter, previous = parameter_contributions[identity]
                        parameter_contributions[identity] = (
                            previous_parameter,
                            previous + gradient,
                        )
                    else:
                        parameter_contributions[identity] = (parameter, gradient)
            for node_id in sorted(pending):
                flush(node_id)
        for node, gradient in zip(self._nodes_in_id_order, current):
            node.gradient_message.current_state = gradient.detach()
        backward_targets: List[torch.Tensor] = []
        backward_gradients: List[torch.Tensor] = []
        if parameter_contributions:
            parameters, gradients = zip(*parameter_contributions.values())
            backward_targets.extend(parameters)
            backward_gradients.extend(gradients)
        for node, gradient in zip(self._nodes_in_id_order, current):
            feature = node.feature_message.current_state
            if feature.is_leaf and feature.requires_grad:
                backward_targets.append(feature)
                backward_gradients.append(gradient)
        if backward_targets:
            torch.autograd.backward(
                backward_targets,
                backward_gradients,
                retain_graph=retain_graph,
            )
        if not retain_graph:
            self._forward_trace = []

    def generate_mermaid(self, levels: Union[int, slice, List[int], None] = None) -> str:
        """
        生成 Mermaid 可视化描述，可指定绘制层级范围

        Args:
            levels: 层级选择。默认为 None 表示全部；
                    可为 int（单层）、slice 或 list。
                    当为 list 时，按列表顺序（保持去重）绘制连接。

        Returns:
            Mermaid 图描述字符串
        """
        if self.topo is None or self.num_levels == 0:
            levels_iter = []
        elif levels is None:
            levels_iter = list(range(self.num_levels))
        elif isinstance(levels, int):
            levels_iter = [levels]
        elif isinstance(levels, slice):
            levels_iter = list(range(self.num_levels))[levels]
        elif isinstance(levels, list):
            # 保持传入顺序，同时去重（dict.fromkeys 保留顺序）
            levels_iter = list(dict.fromkeys(levels))
        else:
            raise TypeError("levels 参数类型应为 int / slice / list / None")

        mermaid = [
            "graph TD",
            "",
            " classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            " classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        # 添加节点
        for node in sorted(self.nodes, key=lambda x: x.id):
            mermaid.append(f" {node.name}:::MHD_Node_Style")

        # 添加边（合并选定层级中的连接，按 levels_iter 的顺序叠加）
        for edge in sorted(self.edges, key=lambda x: x.id):
            edge_id = edge.id
            head_names = set()
            tail_names = set()
            for lvl in levels_iter:
                if lvl >= self.num_levels:
                    continue
                role = self.topo.role_matrices[lvl]
                if edge_id >= role.shape[0]:
                    continue
                row = role[edge_id]
                for nid in range(row.shape[0]):
                    if row[nid] < 0:
                        n = self.get_node_by_id(nid)
                        if n:
                            head_names.add(n.name)
                    elif row[nid] > 0:
                        n = self.get_node_by_id(nid)
                        if n:
                            tail_names.add(n.name)
            if not head_names and not tail_names:
                continue
            mermaid.append(f" {edge.name}:::MHD_Edge_Style")
            for hn in sorted(head_names):
                mermaid.append(f" {hn} --> {edge.name}")
            for tn in sorted(tail_names):
                mermaid.append(f" {edge.name} --> {tn}")
            mermaid.append("")

        mermaid_code = "\n".join(mermaid)
        print("=== 超图可视化 ===")
        print(mermaid_code)
        return mermaid_code

    # ---------- 图合并辅助 ----------
    @staticmethod
    def _merge_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        合并多个子图的节点状态，始终采用均值融合（无序且稳定）。
        自动处理 dtype 转换。
        """
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1:
            raise ValueError(f"合并时张量 dtype 不一致: {dtypes}")
        target_dtype = dtypes.pop()
        if any(t.shape != tensors[0].shape or t.device != tensors[0].device for t in tensors):
            raise ValueError("合并时张量 shape/device 必须一致")
        if target_dtype.is_complex or target_dtype.is_floating_point:
            return torch.stack(tensors, dim=0).mean(dim=0)
        return torch.stack([tensor.float() for tensor in tensors], dim=0).mean(
            dim=0
        ).to(target_dtype)

    @classmethod
    def merge_graph(cls, graphs: Set['MHD_Graph'], device: torch.device = None) -> 'MHD_Graph':
        """Deterministically merge compatible graphs by node and edge name."""
        if not graphs:
            raise ValueError("图集合不能为空")
        target_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ordered_graphs = sorted(
            graphs,
            key=lambda graph: (
                tuple(sorted(node.name for node in graph.nodes)),
                tuple(sorted(edge.name for edge in graph.edges)),
            ),
        )
        node_groups: Dict[str, List[MHD_Node]] = defaultdict(list)
        edge_groups: Dict[str, List[MHD_Edge]] = defaultdict(list)
        for graph in ordered_graphs:
            for node in graph.nodes:
                node_groups[node.name].append(node)
            for edge in graph.edges:
                edge_groups[edge.name].append(edge)

        def compatible_value(left: Any, right: Any) -> bool:
            if isinstance(left, str) or isinstance(right, str):
                return left == right
            return left is right

        merged_nodes: Set[MHD_Node] = set()
        name_to_global_nid: Dict[str, int] = {}
        for global_nid, name in enumerate(sorted(node_groups)):
            grouped = node_groups[name]
            for attribute in ("feature_aggregation", "gradient_aggregation"):
                reference = getattr(grouped[0], attribute)
                if not all(
                    compatible_value(reference, getattr(node, attribute))
                    for node in grouped[1:]
                ):
                    raise ValueError(f"节点 '{name}' 的 {attribute} 不兼容")
            feature = MHD_Node.Message(
                cls._merge_tensors(
                    [node.feature_message.initial_state for node in grouped]
                ),
                cls._merge_tensors(
                    [node.feature_message.current_state for node in grouped]
                ),
            )
            gradient = MHD_Node.Message(
                cls._merge_tensors(
                    [node.gradient_message.initial_state for node in grouped]
                ),
                cls._merge_tensors(
                    [node.gradient_message.current_state for node in grouped]
                ),
            )
            merged_nodes.add(
                MHD_Node(
                    global_nid,
                    name,
                    feature,
                    gradient,
                    grouped[0].feature_aggregation,
                    grouped[0].gradient_aggregation,
                )
            )
            name_to_global_nid[name] = global_nid

        graph_id_to_global_node = {
            id(graph): {
                node.id: name_to_global_nid[node.name] for node in graph.nodes
            }
            for graph in ordered_graphs
        }

        def validate_operations(
            edge_name: str,
            operation_groups: Sequence[Optional[List[Any]]],
            label: str,
        ) -> None:
            reference = operation_groups[0]
            for operations in operation_groups[1:]:
                if reference is None or operations is None:
                    if reference is not operations:
                        raise ValueError(f"同名边 '{edge_name}' 的 {label} 不兼容")
                    continue
                if len(reference) != len(operations):
                    raise ValueError(f"同名边 '{edge_name}' 的 {label} 长度不兼容")
                for left, right in zip(reference, operations):
                    left_function = left.function
                    right_function = right.function
                    if isinstance(left_function, str) and isinstance(right_function, str):
                        valid = left_function == right_function
                    elif isinstance(left_function, nn.Module) and isinstance(right_function, nn.Module):
                        stateful = bool(left_function.state_dict()) or bool(right_function.state_dict())
                        valid = left_function is right_function if stateful else type(left_function) is type(right_function)
                    else:
                        valid = left_function is right_function
                    if not valid:
                        raise ValueError(
                            f"同名边 '{edge_name}' 的 {label} 操作不兼容"
                        )

        merged_edges: Set[MHD_Edge] = set()
        name_to_global_eid: Dict[str, int] = {}
        for global_eid, name in enumerate(sorted(edge_groups)):
            grouped = edge_groups[name]
            validate_operations(
                name,
                [edge.edge_operations for edge in grouped],
                "edge_operations",
            )
            validate_operations(
                name,
                [edge.backward_operations for edge in grouped],
                "backward_operations",
            )
            merged_edges.add(
                MHD_Edge(
                    global_eid,
                    name,
                    grouped[0].edge_operations.copy(),
                    None
                    if grouped[0].backward_operations is None
                    else grouped[0].backward_operations.copy(),
                )
            )
            name_to_global_eid[name] = global_eid

        graph_id_to_global_edge = {
            id(graph): {
                edge.id: name_to_global_eid[edge.name] for edge in graph.edges
            }
            for graph in ordered_graphs
        }
        max_levels = max(graph.num_levels for graph in ordered_graphs)

        def merge_topology_phase(
            role_attribute: str,
            sort_attribute: str,
        ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
            merged_roles: List[torch.Tensor] = []
            merged_sorts: List[torch.Tensor] = []
            for level in range(max_levels):
                role = torch.zeros(
                    (len(merged_edges), len(merged_nodes)),
                    dtype=torch.int64,
                    device=target_device,
                )
                sort = torch.zeros_like(role)
                occupied: Set[Tuple[int, int]] = set()
                for graph in ordered_graphs:
                    if level >= graph.num_levels:
                        continue
                    source_roles = getattr(graph.topo, role_attribute)
                    source_sorts = getattr(graph.topo, sort_attribute)
                    source_role = source_roles[level]
                    source_sort = source_sorts[level]
                    edge_map = graph_id_to_global_edge[id(graph)]
                    node_map = graph_id_to_global_node[id(graph)]
                    for local_edge in range(source_role.shape[0]):
                        for local_node in range(source_role.shape[1]):
                            role_value = int(source_role[local_edge, local_node].item())
                            sort_value = int(source_sort[local_edge, local_node].item())
                            if role_value == 0 and sort_value == 0:
                                continue
                            cell = (edge_map[local_edge], node_map[local_node])
                            if cell in occupied and (
                                int(role[cell].item()) != role_value
                                or int(sort[cell].item()) != sort_value
                            ):
                                raise ValueError(
                                    f"合并拓扑在 level={level}, edge={cell[0]}, "
                                    f"node={cell[1]} 发生冲突"
                                )
                            role[cell] = role_value
                            sort[cell] = sort_value
                            occupied.add(cell)
                merged_roles.append(role)
                merged_sorts.append(sort)
            return merged_roles, merged_sorts

        forward_roles, forward_sorts = merge_topology_phase(
            "role_matrices", "sort_matrices"
        )
        all_auto = all(graph.topo.backward_is_auto for graph in ordered_graphs)
        if all_auto:
            merged_topo = MHD_Topo(forward_roles, forward_sorts)
        else:
            backward_roles, backward_sorts = merge_topology_phase(
                "backward_role_matrices", "backward_sort_matrices"
            )
            merged_topo = MHD_Topo(
                forward_roles,
                forward_sorts,
                backward_roles,
                backward_sorts,
            )
        return cls(
            nodes=merged_nodes,
            edges=merged_edges,
            topos={merged_topo},
            device=target_device,
        )
