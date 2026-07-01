import time
import numpy as np
from PIL import Image
import onnxruntime as ort

# CPU backend
session = ort.InferenceSession(
    "/home/arduino/edgeai/benchmarking/mobilenet_v2_float.onnx",
    providers=["CPUExecutionProvider"],
)

# Preprocess image
img = Image.open("/home/arduino/edgeai/benchmarking/test_image.jpg").convert("RGB").resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
img_array = (img_array - mean) / std
img_array = img_array.transpose(2, 0, 1)
input_data = np.expand_dims(img_array, axis=0)

# Run inference
times = []

session.run(None, {"input": input_data}) # warmup

for i in range(10):
    start = time.perf_counter()
    outputs = session.run(None, {"input": input_data})
    end = time.perf_counter()
    times.append((end - start) * 1000)

for i, t in enumerate(times):
    print(f"Run {i+1}: {t:.2f} ms")

top_index = int(np.argmax(outputs[0][0]))
print(f"Top class index: {top_index}")
print(f"Avg latency:     {sum(times)/len(times):.2f} ms")
print(f"Min latency:     {min(times):.2f} ms")
print(f"Max latency:     {max(times):.2f} ms")

"""
Run 1: 66.41 ms
Run 2: 69.29 ms
Run 3: 70.18 ms
Run 4: 68.12 ms
Run 5: 67.96 ms
Run 6: 68.27 ms
Run 7: 68.97 ms
Run 8: 68.77 ms
Run 9: 69.59 ms
Run 10: 69.33 ms
Top class index: 281
Avg latency:     68.69 ms
Min latency:     66.41 ms
Max latency:     70.18 ms
"""