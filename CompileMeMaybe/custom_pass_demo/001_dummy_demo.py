"""
This script demonstrates how to write a simple custom graph pass and how to register it.
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
        # Pass gets the `torch.fx.Graph` not the `torch.fx.GraphModule`
        print(graph)

    def uuid(self) -> Optional[Any]:
        return get_hash_for_files((__file__,))


if __name__ == "__main__":
    """
    post_grad_custom_pre_pass and post_grad_custom_post_pass happen at different stages
    post_grad_custom_pre_pass is called right after basic setup and before optimizations
    post_grad_custom_post_pass is called right after optimizations and before final cleanup.


    How to choose?
    I Still don't know.
    It seems like post_grad_custom_pre_pass is used in vllm.

    """
    torch._inductor.config.post_grad_custom_post_pass = MyCustomPass()

    module = MyModule()
    module.compile()

    with torch.no_grad():
        x = torch.randn(10)
        print(module(x))

"""
expected output:

graph():
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
tensor([0.3088, 0.0000, 0.0033, 0.0000, 0.0000, 0.0000, 0.8597, 0.0000, 0.6738,
        0.4057])
"""
