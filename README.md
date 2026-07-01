## Real-Time Human Detection — Arduino UNO Q

Run on a **Qualcomm Dragonwing QRB2210** CPU, using a MobileNetV2 SSD FPN-Lite 320×320 model quantized to INT8. Two scripts: console output and a browser-accessible live video stream with annotated bounding boxes.

- **Board:** Arduino UNO Q (QRB2210, Debian Linux) - SSH over WiFi
- **Camera:** USB webcam via USB hub

**Model:** MobileNetV2 SSD FPN-Lite
- Adapted from [Edge Impulse public project #121370](https://studio.edgeimpulse.com/public/121370/latest)
- 320×320 RGB input / INT8 quantized / up to 10 detections

---

**`console_output.py`** — captures webcam frames, runs inference, prints detections and inference time to terminal w/o display.

**`gui_output.py`** — Stream annotated frames as MJPEG over HTTP. `http://<board-ip>:8080` in browser for live video with drawn bounding boxes and confidence scores.

| Mode | Avg inference | Notes |
|---|---|---|
| Console output | ~60ms (16.7 FPS) | Inference only |
| GUI stream | ~80ms (12.5 FPS) | Inference + MJPEG encoding + HTTP stream |

---

**Dependencies**
```
tensorflow
opencv-python
numpy
```

**Notes**
- QRB2210 -> 4 threads used via `tf.lite.Interpreter(num_threads=4)`.
- Confidence threshold set to 0.5.
- GUI stream uses Python's built-in `http.server` with a background inference thread.
