import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# Real-time person detection on a webcam with HTTP streaming.
# Detection runs in a background thread and annotated frames are streamed to browser.
# I am unfamiliar with HTTP; any related code is vibe-coded. 

MODEL_PATH = "/home/arduino/edgeai/person_detection_int8.tflite"
THRESHOLD = 0.5
PORT = 8080

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

outputs = {o['name'].split(':')[-1]: o for o in output_details}
bboxes, confidences = outputs['3'], outputs['1']
_, height, width, _ = input_details[0]['shape']
scale, zero_point = input_details[0]['quantization']

cap = cv2.VideoCapture(1)

# Share most recent annotated frame between inference thread and HTTP server.
# Lock prevents one thread of reading/overwriting the frame while the other is using it.
latest_frame = None
frame_lock = threading.Lock()

def inference_loop():
    global latest_frame
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.resize(frame, (width, height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = (img_rgb.astype(np.float32) / 255.0 / scale + zero_point).astype(np.int8)
        input_data = np.expand_dims(data, axis=0)

        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        ms = (time.perf_counter() - start) * 1000

        boxes = interpreter.get_tensor(bboxes['index'])[0]
        scores = interpreter.get_tensor(confidences['index'])[0]
        h, w = frame.shape[:2]

        detections = [(box, score) for box, score in zip(boxes, scores) if score >= THRESHOLD]

        for box, score in detections:
            ymin, xmin, ymax, xmax = box
            x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
            x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
            # Draw blue bboxes and label with confidence scores.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"person {score:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Overlay the inference time and detection count directly on the frame.
        # Coords (10, 30) top-left corner / font size 0.7 / color (0, 255, 0) / thickness 2
        cv2.putText(frame, f"{ms:.1f}ms | {len(detections)} person(s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert annotated frame to JPEG so it can be streamed over HTTP.
        _, jpeg = cv2.imencode('.jpg', frame)
        with frame_lock:
            latest_frame = jpeg.tobytes()

class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass # suppress per-request access logs (noisy)

    def do_GET(self):
        # Respond with a multipart stream so the browser can display a continuously updating image.
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with frame_lock:
                    frame = latest_frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                # Send one JPEG frame at a time as part of the multipart response.
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                self.wfile.write(frame)
                self.wfile.write(b'\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected

threading.Thread(target=inference_loop, daemon=True).start()

print(f"Streaming at http://192.168.2.101:{PORT}")
HTTPServer(('0.0.0.0', PORT), StreamHandler).serve_forever()