import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "person_detection_float32.tflite"
SCORE_THRESHOLD = 0.5

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Map outputs by their actual name suffix rather than assuming list order
outputs = {o['name'].split(':')[-1]: o for o in output_details}
boxes_out, scores_out, classes_out = outputs['3'], outputs['1'], outputs['2']

_, height, width, _ = input_details[0]['shape']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (width, height))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(boxes_out['index'])[0]
    scores = interpreter.get_tensor(scores_out['index'])[0]

    h, w = frame.shape[:2]
    for box, score in zip(boxes, scores):
        if score < SCORE_THRESHOLD:
            continue
        ymin, xmin, ymax, xmax = box
        x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
        x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()