import time
import numpy as np
import tensorflow as tf

MODELS = {
    "float32": "/home/arduino/edgeai/person_detection_float32.tflite",
    "int8": "/home/arduino/edgeai/person_detection_int8.tflite",
}

for label, path in MODELS.items():
    for n_threads in [1, 4]:
        interpreter = tf.lite.Interpreter(model_path=path, num_threads=n_threads)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        shape = input_details[0]['shape']
        dtype = input_details[0]['dtype']
        dummy_input = np.zeros(shape, dtype=dtype)

        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()  # warmup

        times = []
        for _ in range(10):
            start = time.perf_counter()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg = sum(times) / len(times)
        print(f"{label}, threads={n_threads}, input_shape={shape}: avg {avg:.2f}ms ({1000/avg:.1f} FPS)")


'''
float32, threads=1, input_shape=[  1 320 320   3]: avg 303.99ms (3.3 FPS)
float32, threads=4, input_shape=[  1 320 320   3]: avg 126.91ms (7.9 FPS)
int8, threads=1, input_shape=[  1 320 320   3]: avg 190.16ms (5.3 FPS)
int8, threads=4, input_shape=[  1 320 320   3]: avg 60.35ms (16.6 FPS)
'''