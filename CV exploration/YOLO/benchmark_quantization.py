import time
from ultralytics import YOLO

models = [
    ("/home/arduino/edgeai/yolo_models/yolo26n_int8_640.tflite", 640),
    ("/home/arduino/edgeai/yolo_models/yolo26n_int8_320.tflite", 320),
]
# yolo export model=yolo26n.pt format=tflite quantize=int8 data=coco8.yaml end2end=False                               
# yolo export model=yolo26n.pt format=tflite quantize=int8 data=coco8.yaml end2end=False imgsz=320                                 

for model_path, size in models:
    model = YOLO(model_path, task="detect")
    model.predict("bus.jpg", imgsz=size, verbose=False) # warmup

    times = []
    for _ in range(10):
        start = time.perf_counter()
        model.predict("bus.jpg", imgsz=size, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg=sum(times) / len(times)
    print(f"imgsz={size}: avg {avg:.1f}ms ({1000/avg:.1f} FPS)")


'''
imgsz=640: avg 1021.5ms (1.0 FPS)
imgsz=320: avg 254.3ms (3.9 FPS)
'''
