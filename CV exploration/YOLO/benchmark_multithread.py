import time
import numpy as np
import tensorflow as tf

MODEL_PATH = "/home/arduino/edgeai/yolo_models/yolo26n_int8_320.tflite"

for n_threads in [1, 2, 4]:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH, num_threads=n_threads)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()

    # Pure timing test, just use dummy input
    shape = input_details[0]['shape']
    dtype = input_details[0]['dtype']
    dummy_input = np.zeros(shape, dtype=dtype)

    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke() # warmup

    times = []
    for _ in range(10):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = sum(times) / len(times)
    print(f"num_threads={n_threads}: avg {avg:.1f}ms ({1000/avg:.1f} FPS, raw inference only)")


'''
num_threads=1: avg 222.5ms (4.5 FPS, raw inference only)
num_threads=2: avg 128.0ms (7.8 FPS, raw inference only)
num_threads=4: avg 75.9ms (13.2 FPS, raw inference only)
'''
# Even with multithreading, real-time performance is unfeasible.
# Pivoting away from YOLO is recommended.