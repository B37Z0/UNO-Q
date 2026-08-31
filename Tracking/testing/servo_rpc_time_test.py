import time
from servo_controller import ServoController

servos = ServoController()
servos.center()
time.sleep(1)

N = 200
t0 = time.perf_counter()
for i in range(N):
    servos.set_cam(94 + (i % 2), 94)
elapsed = time.perf_counter() - t0

print(f"{N} calls in {elapsed*1000:.1f} ms")
print(f"mean round trip: {elapsed/N*1000:.2f} ms/call")
print(f"max update rate: {N/elapsed:.1f} Hz")

servos.center()
servos.close()