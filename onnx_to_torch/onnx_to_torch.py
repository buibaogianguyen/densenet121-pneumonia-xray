import onnx
from onnx2pytorch import ConvertModel
import torch

onnx_model = onnx.load("onnx_to_torch/densenet.onnx")
pytorch_model = ConvertModel(onnx_model)

torch.save(pytorch_model, "densenet_torch.pt")