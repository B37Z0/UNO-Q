# CV Exploration

Benchmarks comparing computer vision models for person detection on the
Arduino UNO Q (Qualcomm QRB2210). The goal was to find a model that runs
at real-time rates on a CPU-only ARM SoC with no GPU or NPU acceleration.

**Outcome:** YOLO is too slow — even quantized and at reduced resolution it
barely reaches 4 FPS. The Edge Impulse MobileNetV2 SSD FPN-Lite model
(INT8, 320×320) hits ~16 FPS with four threads, making it the clear
winner and the model used in the final project.

---

## Structure

```
CV exploration/
├── YOLO/
│   ├── benchmark_base.py             YOLOv26n baseline (PyTorch, 640px)
│   ├── benchmark_quantization.py     INT8 TFLite at 640px and 320px
│   └── benchmark_multithread.py      INT8 TFLite at 320px, 1/2/4 threads
│
├── Edge Impulse/
│   ├── benchmark_multithread.py      float32 vs INT8, 1 vs 4 threads
│   ├── benchmark_real_time.py        INT8, 4 threads, real image + detection output
│   ├── person_detection_float32.tflite
│   ├── person_detection_int8.tflite
│   └── person_test.jpg
│
└── requirements.txt
```

---

## Results

All benchmarks are 10-iteration averages, measured on the QRB2210.

### YOLO

| Configuration | Avg Latency | FPS |
|---|---|---|
| YOLOv26n PyTorch (640px) | 1561.9 ms | 0.6 |
| INT8 TFLite (640px) | 1021.5 ms | 1.0 |
| INT8 TFLite (320px) | 254.3 ms | 3.9 |
| INT8 TFLite 320px, 1 thread | 222.5 ms | 4.5 |
| INT8 TFLite 320px, 2 threads | 128.0 ms | 7.8 |
| INT8 TFLite 320px, 4 threads | 75.9 ms | 13.2 |

Quantization and resolution reduction bring YOLO from ~0.6 FPS to ~13 FPS
at best, but this is raw inference only — adding capture, tracking, and
streaming overhead would push it well below a usable rate.

### Edge Impulse (MobileNetV2 SSD FPN-Lite)

| Configuration | Avg Latency | FPS |
|---|---|---|
| float32, 1 thread | 304.0 ms | 3.3 |
| float32, 4 threads | 126.9 ms | 7.9 |
| INT8, 1 thread | 190.2 ms | 5.3 |
| INT8, 4 threads | 60.4 ms | 16.6 |
| INT8, 4 threads (real image) | 61.2 ms | 16.3 |

The INT8 model with four threads runs at ~16 FPS for raw inference. In the
final system (capture → detect → track → servo → encode), the full loop
runs at 12–13 Hz — comfortably real-time.

---

## Dependencies

```shell
pip install -r requirements.txt
```

See `requirements.txt` — requires `ultralytics`, `tensorflow`,
`opencv-python`, and `numpy`. Install PyTorch CPU-only first to avoid
pulling the CUDA build:

```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Models

- **Edge Impulse person detection** — MobileNetV2 SSD FPN-Lite, single
  class (person), up to 10 detections, 320×320 RGB input. Adapted from
  [Edge Impulse public project #121370](https://studio.edgeimpulse.com/public/121370/latest).
  Both float32 and INT8 TFLite variants are included in the repo.

- **YOLOv26n** — exported to INT8 TFLite at 640 and 320 input sizes via:
  ```shell
  yolo export model=yolo26n.pt format=tflite quantize=int8 data=coco8.yaml end2end=False
  yolo export model=yolo26n.pt format=tflite quantize=int8 data=coco8.yaml end2end=False imgsz=320
  ```
  The exported `.tflite` files are not checked in (too large / not the
  chosen model).
