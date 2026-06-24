import time
import numpy as np
from PIL import Image
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep


# Register QNN as a plugin EP, then select the GPU backend specifically
ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]

options = ort.SessionOptions()
# Force hard failure if operations not supported by GPU
# Want to check if this is genuinely a GPU-only number
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
options.add_provider_for_devices(
    devices,
    {"backend_path": "/home/arduino/edgeai/.venv/lib/python3.13/site-packages/onnxruntime_qnn/libQnnGpu.so"}
)

session = ort.InferenceSession("/home/arduino/edgeai/benchmarking/mobilenet_v2_float.onnx", sess_options=options)


# Preprocess image
# Float ONNX model (not TFLite INT8), so preprocessing differs:
# channel-first (NCHW), float32, ImageNet mean/std normalization.
img = Image.open("test_image.jpg").convert("RGB").resize((224, 224))
input_data = np.array(img, dtype=np.float32) / 255.0

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
input_data = (input_data - mean) / std

# HWC -> CHW
input_data = input_data.transpose(2, 0, 1)
# Add batch dim -> (1,3,224,224)
input_data = np.expand_dims(input_data, axis=0)   

# Run inference
times = []
for _ in range(10):
    start = time.perf_counter()
    # Forward pass
    outputs = session.run(None, {"input": input_data})
    end = time.perf_counter()
    times.append((end - start) * 1000)

# labels.txt comes from TF 1001-class convention (0 = background)
# Using it would print a wrong label, so just ignore it; this is just a benchmark
top_index = int(np.argmax(outputs[0][0]))

# Still calculate confidence for sanity check
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
probs = softmax(outputs[0][0])
confidence = probs[top_index]

print(f"Prediction:    {top_index}")
print(f"Confidence:    {confidence:.1%}")
print(f"Avg latency:   {sum(times)/len(times):.2f} ms")
print(f"Min latency:   {min(times):.2f} ms")
print(f"Max latency:   {max(times):.2f} ms")
