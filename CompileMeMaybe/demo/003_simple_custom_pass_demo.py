"""
This script demonstrates how to find nodes/remove nodes in the graph.

Also read `remove_assert_ops` in `torch/_inductor/fx_passes/post_grad.py`
"""

from typing import Optional, Any

import torch
from torch._inductor.custom_graph_pass import CustomGraphPass, get_hash_for_files


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


class MyCustomPass(CustomGraphPass):
    def __call__(self, graph: torch.fx.Graph) -> None:
        for node in graph.nodes:
            print(f"Node: {node}, src: {node.args[0] if len(node.args) > 0 else None}")
        print("-" * 100)
        print(f"Graph: {graph}")
        # Passing `op` as string, `target` is the actual type, not string
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.mul.Tensor
        ):
            print(f"Found mul node: {node}")
        print("-" * 100)
        for node in graph.find_nodes(
            op="placeholder",
        ):
            print(f"Found placeholder node: {node}")
        print("-" * 100)
        for node in graph.find_nodes(
            op="call_function", target=torch.ops.aten.relu.default
        ):
            print(f"Found ReLU node: {node}")
            src = node.args[0]
            # Replace all uses of the node with the source node
            node.replace_all_uses_with(src)
            graph.erase_node(node)

        graph.lint()
        print("-" * 100)
        print(f"after erasing ReLU node, Graph: {graph}")

    def uuid(self) -> Optional[Any]:
        return get_hash_for_files((__file__,))


if __name__ == "__main__":
    torch._inductor.config.post_grad_custom_post_pass = MyCustomPass()

    module = MyModule()
    module.compile()

    with torch.no_grad():
        x = torch.randn(10)
        print(module(x))

"""
expected output:

Node: arg0_1, src: None
Node: arg1_1, src: None
Node: arg2_1, src: None
Node: view, src: arg2_1
Node: permute_1, src: view
Node: permute, src: arg0_1
Node: mul, src: permute_1
Node: sum_1, src: mul
Node: mul_1, src: sum_1
Node: mul_2, src: arg1_1
Node: add, src: mul_1
Node: view_1, src: add
Node: relu, src: view_1
Node: output, src: (relu,)
----------------------------------------------------------------------------------------------------
Graph: graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
    %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [1, 10]), kwargs = {})
    %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [1, 0]), kwargs = {})
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0]), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %permute), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [0], True), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [10]), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
    return (relu,)
Found mul node: mul
Found mul node: mul_1
Found mul node: mul_2
----------------------------------------------------------------------------------------------------
Found placeholder node: arg0_1
Found placeholder node: arg1_1
Found placeholder node: arg2_1
----------------------------------------------------------------------------------------------------
Found ReLU node: relu
----------------------------------------------------------------------------------------------------
after erasing ReLU node, Graph: graph():
    %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
    %arg1_1 : [num_users=1] = placeholder[target=arg1_1]
    %arg2_1 : [num_users=1] = placeholder[target=arg2_1]
    %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%arg2_1, [1, 10]), kwargs = {})
    %permute_1 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [1, 0]), kwargs = {})
    %permute : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%arg0_1, [1, 0]), kwargs = {})
    %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %permute), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [0], True), kwargs = {})
    %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_1, 1), kwargs = {})
    %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg1_1, 1), kwargs = {})
    %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
    %view_1 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%add, [10]), kwargs = {})
    return (view_1,)
tensor([ 0.4351,  1.6840,  0.1542,  0.8497, -0.4261, -0.0349,  0.3844, -2.0585,
         0.3058,  1.2893])
"""
