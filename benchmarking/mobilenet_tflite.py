import time
import numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter

# Load model
interpreter = Interpreter(model_path="mobilenet_v2_1.0_224_quant.tflite")
interpreter.allocate_tensors() # reserve memory buffers for every layer of the model

# Get I/O details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess test image
img = Image.open("test_image.jpg").convert("RGB").resize((224, 224))
# Convert to int8 array (224,224,3) -> add batch dim (1,224,224,3)
input_data = np.expand_dims(np.array(img, dtype=np.uint8), axis=0)

# Load labels
with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# Run inference
times = []
for _ in range(10):
    start = time.perf_counter()
    # Write image data to model input buffer
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # Forward pass
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

# Get results
# Read output buffer, shape (1,1001) for class scores
output_data = interpreter.get_tensor(output_details[0]['index'])
top_index = np.argmax(output_data[0])
confidence = output_data[0][top_index] / 255.0 # 0-255 scores by default

print(f"Prediction:    {labels[top_index]}")
print(f"Confidence:    {confidence:.1%}")
print(f"Avg latency:   {sum(times)/len(times):.2f} ms")
print(f"Min latency:   {min(times):.2f} ms")
print(f"Max latency:   {max(times):.2f} ms")