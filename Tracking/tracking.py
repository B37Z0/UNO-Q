"""
Person-tracking pan/tilt loop.

python3 tracking.py
    --debug           # annotated MJPEG on :8080
    --no-servo        # detection + tracking, no motion

Upon Ctrl-C, recenters camera and exits cleanly.

Pipeline
--------
    Camera thread  ──>  latest frame (shared via lock)
                            │
    main loop (wall-clock paced at DETECT_HZ)
        detect ──> track ──> servo target ──> RPC ──> MCU slew
                            │
    DebugStream thread (optional) ──> annotated MJPEG on :8080

Threads
-------
    1. Camera._loop       - daemon, continuously reads/drains frames
                            so main loop stays up to date
    2. DebugStream server - daemon, serves MJPEG to browser clients
    3. main thread        - runs the detect -> track -> servo loop

All cross-thread data is guarded by threading.Lock. Daemon threads
die automatically when the main thread exits.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np
import tensorflow as tf

from tracker import Tracker


# ----- Configuration -----

MODEL_PATH = "/home/arduino/edgeai/person_detection/person_detection_int8.tflite"
THRESHOLD = 0.5

CAMERA_PATH = "/dev/video2"
CAPTURE_W = 640
CAPTURE_H = 360

# Wall time cadence - tune for performance / tracking speed
DETECT_HZ = 25.0

# Minimum degree delta for new servo command
SERVO_MIN_DELTA_DEG = 1.0

DEBUG_PORT = 8080


# ----- Camera -----

class Camera:
    """
    Continuously drains the camera in a background thread so the main loop
    stays up-to-date (no queueing old frames)
    """
    def __init__(self, path=CAMERA_PATH, width=CAPTURE_W, height=CAPTURE_H):
        self.cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {path}")

        # Order matters: FOURCC -> dimensions -> read()
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Camera open but no frame returned")

        self.h, self.w = frame.shape[:2]
        if (self.w, self.h) != (width, height):
            print(f"  WARNING: requested {width}x{height}, got {self.w}x{self.h}")
            print("  The FOV presets in tracker.py are only valid for 16:9 aspect ratios.")

        self._frame = frame
        self._lock = threading.Lock()
        self._running = True

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """Runs in the background thread. Continuously reads and
        overwrites _frame with the newest frame."""
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return self._frame.copy()

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# ----- Detector -----

class Detector:
    def __init__(self, model_path=MODEL_PATH, threshold=THRESHOLD, threads=4):
        self.threshold = threshold

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=threads
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        outputs = {o['name'].split(':')[-1]: o for o in output_details}
        self.bboxes = outputs['3']
        self.confidences = outputs['1']

        _, self.height, self.width, _ = self.input_details[0]['shape']
        self.scale, self.zero_point = self.input_details[0]['quantization']

    def detect(self, frame):
        """
        Returns (boxes, scores, inference_ms)
            - boxes: filtered + normalized [ymin, xmin, ymax, xmax]
            - scores: confidence scores for each box
            - inference_ms: time taken for inference in milliseconds
        """
        img = cv2.resize(frame, (self.width, self.height))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = (img_rgb.astype(np.float32) / 255.0 / self.scale
                + self.zero_point).astype(np.int8)
        input_data = np.expand_dims(data, axis=0)

        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        ms = (time.perf_counter() - start) * 1000

        raw_boxes = self.interpreter.get_tensor(self.bboxes['index'])[0]
        raw_scores = self.interpreter.get_tensor(self.confidences['index'])[0]

        boxes, scores = [], []
        for box, score in zip(raw_boxes, raw_scores):
            if score >= self.threshold:
                boxes.append(box)
                scores.append(float(score))

        return boxes, scores, ms


# ----- Debug stream -----

class DebugStream:
    """Annotated MJPEG over HTTP for debugging"""
    def __init__(self, port=DEBUG_PORT):
        self.port = port
        self.latest = None
        self.lock = threading.Lock()

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass  # suppress per-request log spam

            def do_GET(self):
                # MJPEG-over-HTTP: browser opens one long-lived GET, and 
                # successive JPEG frames are pushed with a multipart boundary.
                # Browser renders each frame as it arrives -> live video feed.
                self.send_response(200)
                self.send_header(
                    'Content-Type',
                    'multipart/x-mixed-replace; boundary=frame'
                )
                self.end_headers()
                try:
                    while True:
                        with outer.lock:
                            jpeg = outer.latest
                        if jpeg is None:
                            time.sleep(0.01) # no frame, back off
                            continue
                        # Write one MJPEG frame boundary + payload
                        self.wfile.write(b'--frame\r\n')
                        self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                        self.wfile.write(jpeg)
                        self.wfile.write(b'\r\n')
                        time.sleep(0.03) # ~30 fps cap to browser
                except (BrokenPipeError, ConnectionResetError):
                    pass # client disconnected, let handler exit

        self.server = HTTPServer(('0.0.0.0', port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, frame):
        """Called from main thread each loop iteration. Encodes annotated frame
        to JPEG and stashes it for HTTP handler(s) to pick up"""
        ok, jpeg = cv2.imencode('.jpg', frame)
        if ok:
            with self.lock:
                self.latest = jpeg.tobytes()


def annotate(frame, boxes, scores, tracker, pan, tilt, det_ms, loop_ms):
    """Draw detections, tracked target, and telemetry onto frame (copy)"""
    h, w = frame.shape[:2]

    # Detections - blue
    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        x1, y1 = max(0, int(xmin * w)), max(0, int(ymin * h))
        x2, y2 = min(w, int(xmax * w)), min(h, int(ymax * h))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{score:.2f}", (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 12, cy), (cx + 12, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 12), (cx, cy + 12), (200, 200, 200), 1)

    # Deadband box
    db = tracker.deadband_frac
    cv2.rectangle(frame,
                  (int(w * (0.5 - db)), int(h * (0.5 - db))),
                  (int(w * (0.5 + db)), int(h * (0.5 + db))),
                  (120, 120, 120), 1)

    # Selected target - green
    if tracker.target_center is not None:
        tx = int(tracker.target_center[0] * w)
        ty = int(tracker.target_center[1] * h)
        cv2.circle(frame, (tx, ty), 7, (0, 255, 0), 2)
        cv2.line(frame, (cx, cy), (tx, ty), (0, 255, 0), 1)

    cv2.putText(frame, f"pan={pan} tilt={tilt}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(frame,
                f"det {det_ms:.0f}ms  loop {loop_ms:.0f}ms  "
                f"{len(boxes)}p  lost {tracker.lost_frames}",
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return frame


# ----- Main -----

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true",
                   help="serve annotated MJPEG on --port")
    p.add_argument("--no-servo", action="store_true",
                   help="run detection and tracking without moving anything")
    p.add_argument("--port", type=int, default=DEBUG_PORT)
    p.add_argument("--hz", type=float, default=DETECT_HZ,
                   help="detector cadence")
    p.add_argument("--kp", type=float, default=0.4)
    p.add_argument("--deadband", type=float, default=0.04)
    args = p.parse_args()

    # Start camera background thread
    print("Opening camera...")
    camera = Camera()
    print(f"  {camera.w}x{camera.h}")

    # Load TFLite model and pre-allocate tensors
    print("Loading model...")
    detector = Detector()
    print(f"  input {detector.width}x{detector.height}")

    tracker = Tracker(kp=args.kp, deadband_frac=args.deadband)

    servos = None
    if not args.no_servo:
        from servo_controller import ServoController
        servos = ServoController()
        servos.center()
        print("Servos centred")
    else:
        print("Servos DISABLED (--no-servo)")

    # HTTP server thread for debug if requested
    stream = None
    if args.debug:
        stream = DebugStream(args.port)
        print(f"Debug stream on http://<board-ip>:{args.port}")

    # Target wall time per iteration
    interval = 1.0 / args.hz

    last_sent_pan, last_sent_tilt = tracker.targets()

    # Rolling stats
    n = 0                             # iterations
    det_total = 0.0                   # detection time
    loop_total = 0.0                  # loop time
    last_report = time.perf_counter() # time of last report

    print(f"\nTracking at {args.hz:.1f} Hz. Ctrl-C to stop.\n")

    # ----- Tracking loop -----
    try:
        while True:
            loop_start = time.perf_counter()

            frame = camera.read()
            boxes, scores, det_ms = detector.detect(frame)
            pan, tilt = tracker.update(boxes)
            
            # MCU slews toward last target regardless of new commands.
            # Only send and update if the target has moved enough.
            if servos is not None:
                if (abs(pan - last_sent_pan) >= SERVO_MIN_DELTA_DEG
                    or abs(tilt - last_sent_tilt) >= SERVO_MIN_DELTA_DEG):
                    servos.set_cam(pan, tilt)
                    last_sent_pan, last_sent_tilt = pan, tilt

            loop_ms = (time.perf_counter() - loop_start) * 1000

            if stream is not None:
                stream.publish(
                    annotate(frame, boxes, scores, tracker,
                             pan, tilt, det_ms, loop_ms)
                )

            # ----- Stats -----
            n += 1
            det_total += det_ms
            loop_total += loop_ms

            now = time.perf_counter()
            if now - last_report >= 10.0:
                print(f"  det {det_total / n:5.1f}ms  "
                      f"loop {loop_total / n:5.1f}ms  "
                      f"actual {n / (now - last_report):4.1f} Hz  "
                      f"pan={pan:3d} tilt={tilt:3d}  "
                      f"{len(boxes)} person(s)")
                n = 0
                det_total = 0.0
                loop_total = 0.0
                last_report = now

            # Run off remaining time in loop for consistent rate
            remaining = interval - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if servos is not None:
            servos.center()
            time.sleep(0.5) # leeway for slew time
            servos.close()
        camera.close()
        print("Complete.")


if __name__ == "__main__":
    main()
