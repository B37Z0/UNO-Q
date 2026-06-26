import torch
from torchvision.models import mobilenet_v2

model = mobilenet_v2(weights="IMAGENET1K_V1")
model.eval()

# QNN EP requires a fixed shape
# For benchmarking, use MobileNetV2 default size
dummy_input = torch.randn(1, 3, 224, 224) 

torch.onnx.export(
    model,
    dummy_input,
    "mobilenet_v2_float.onnx",
    input_names=["input"],
    output_names=["output"],
)
print("Exported.")