"""
This script demonstrates how to replace a node type with another node type in the graph.
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
        for node in list(
            graph.find_nodes(op="call_function", target=torch.ops.aten.relu.default)
        ):
            print(f"Found ReLU node: {node}")
            src = node.args[0]
            # Create a replacement node (example: replace ReLU with Sigmoid)
            # Set an insertion point right after (or before) the current node
            with graph.inserting_after(node):
                new_node = graph.call_function(
                    torch.ops.aten.sigmoid.default, args=(src,), kwargs={}
                )
            # Redirect all uses to the new node, then erase the old node
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

        graph.lint()
        print(f"after replacing ReLU with Sigmoid, Graph: {graph}")

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

Found ReLU node: relu
after replacing ReLU with Sigmoid, Graph: graph():
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
    %sigmoid_default : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_1,), kwargs = {})
    return (sigmoid_default,)
tensor([0.6349, 0.4902, 0.4508, 0.3701, 0.6583, 0.6748, 0.5612, 0.4386, 0.5007,
        0.5910])
"""
