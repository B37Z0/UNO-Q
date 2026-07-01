import time
import cv2
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/home/arduino/edgeai/person_detection_int8.tflite", num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
outputs = {o['name'].split(':')[-1]: o for o in output_details}
boxes_out, scores_out = outputs['3'], outputs['1']

_, height, width, _ = input_details[0]['shape']
scale, zero_point = input_details[0]['quantization']

frame = cv2.imread("/home/arduino/edgeai/person_test.jpg")
img = cv2.resize(frame, (width, height))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
data = (img_rgb.astype(np.float32) / 255.0 / scale + zero_point).astype(np.int8)
input_data = np.expand_dims(data, axis=0)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()  # warmup

times = []
for _ in range(10):
    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Avg latency: {sum(times)/len(times):.2f}ms")

boxes = interpreter.get_tensor(boxes_out['index'])[0]
scores = interpreter.get_tensor(scores_out['index'])[0]
h, w = frame.shape[:2]

for box, score in zip(boxes, scores):
    if score < 0.5:
        continue
    ymin, xmin, ymax, xmax = box
    print(f"score={score:.2f} box=({int(xmin*w)},{int(ymin*h)})-({int(xmax*w)},{int(ymax*h)})")

'''
Avg latency: 61.16ms
score=0.77 box=(51,53)-(195,276)
score=0.59 box=(92,0)-(162,101)
'''