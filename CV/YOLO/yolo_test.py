import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Warmup
model("https://ultralytics.com/images/bus.jpg", verbose=False)

times = []
for _ in range(10):
    start = time.perf_counter()
    results = model("https://ultralytics.com/images/bus.jpg", verbose=False)
    end = time.perf_counter()
    times.append((end - start) * 1000)

print(f"Avg latency: {sum(times)/len(times):.2f} ms")
print(f"Min latency: {min(times):.2f} ms")
print(f"Max latency: {max(times):.2f} ms")

'''
Avg latency: 1561.89 ms
Min latency: 1545.62 ms
Max latency: 1581.08 ms
'''