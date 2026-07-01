import cv2
import numpy as np
import tensorflow as tf
import time

# Basic real-time person detection on a webcam.
# No GUI, just print the number of detections and bounding boxes to console.
# Also measure inference time for each frame. 

# capture frame -> resize/prepare frame -> run model -> 
# filter detections by confidence -> print results to console

MODEL_PATH = "/home/arduino/edgeai/person_detection_int8.tflite"
THRESHOLD = 0.5

# Load the TFLite model into TFLite's runtime and allocate tensors for the I/O buffers.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# Map outputs by name suffix to access them more cleanly ("3:0" -> "3", "1:0" -> "1").
outputs = {o['name'].split(':')[-1]: o for o in output_details}
bboxes, confidences = outputs['3'], outputs['1']

# Extract input shape and quantization parameters for preprocessing.
_, height, width, _ = input_details[0]['shape']
scale, zero_point = input_details[0]['quantization']

cap = cv2.VideoCapture(1) # external usb webcam for arduino

while True:
    ret, frame = cap.read()
    if not ret: # if frame not captured successfully 
        break

    img = cv2.resize(frame, (width, height)) # fixed input size
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # correct color channel order for model
    data = (img_rgb.astype(np.float32) / 255.0 / scale + zero_point).astype(np.int8) # quantize input 
    input_data = np.expand_dims(data, axis=0) # add batch dimension for model

    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], input_data) # pass into model input tensor
    interpreter.invoke() # forward pass
    ms = (time.perf_counter() - start) * 1000 # ms

    boxes = interpreter.get_tensor(bboxes['index'])[0]
    scores = interpreter.get_tensor(confidences['index'])[0]
    h, w = frame.shape[:2]

    detections = [(box, score) for box, score in zip(boxes, scores) if score >= THRESHOLD]

    # Print inference time, number of detections, and bounding box coordinates for each detection.
    print(f"{ms:.1f}ms | {len(detections)} person(s) detected")
    for box, score in detections:
        ymin, xmin, ymax, xmax = box # normalized coords in [0,1] range -> convert to pixel coords
        print(f"  score={score:.2f} box=({int(xmin*w)},{int(ymin*h)})-({int(xmax*w)},{int(ymax*h)})")

cap.release()