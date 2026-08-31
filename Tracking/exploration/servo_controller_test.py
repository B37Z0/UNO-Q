import time
from servo_controller import ServoController

servos = ServoController()

try:
    print("Center")
    servos.center()
    time.sleep(2)

    print("Pan right")
    servos.set_cam(60, 94)
    time.sleep(2)

    print("Pan left")
    servos.set_cam(130, 94)
    time.sleep(2)

    print("Tilt down")
    servos.set_cam(94, 75)
    time.sleep(2)

    print("Tilt up")
    servos.set_cam(94, 110)
    time.sleep(2)

    print("Center")
    servos.center()
finally:
    servos.close()