import torch
import torch.nn as nn

# def cus_maximum(x, y):
    

class Model(nn.Module):
    def forward(self, x, y):
        assert x.dtype == y.dtype
        return torch.maximum(x, y)

torch.onnx.export(Model(), (torch.randn(1, 2), torch.randn(1, 2)), "model.onnx",
    dynamic_axes={"0": {0: "batch", 1: "width"}, "1": {0: "batch", 1: "width"}, "2": {0: "batch", 1: "width"}})
# args = (torch.randn(1, 2), torch.randn(1, 2))
# 
# torch_script_graph, unconvertible_ops = torch.onnx.utils.unconvertible_ops(
    # Model(), args, opset_version=11
# )
# 
# print(set(unconvertible_ops))