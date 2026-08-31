// void setup() {
// }

// void loop() {
// }

#include <Servo.h>
#include "Arduino_RouterBridge.h"

// Slew-limited pan/tilt controller.
// Linux sents TARGET angle when it has a new one (detector rate expected ~10 Hz).
// Sketch runs loop at fixed 50 Hz that incrementally adjusts the angles so motion
// is smooth regardless of the irregular Linux side.

Servo pan;
Servo tilt;

// ----- Calibration -----

const int PAN_CENTER  = 94;
const int PAN_MIN     = 4;
const int PAN_MAX     = 184;

const int TILT_CENTER = 94;
const int TILT_MIN    = 64;
const int TILT_MAX    = 124;

// ----- Motion profile -----

// 1000ms / 20ms = 50 Hz servo update period
const unsigned long UPDATE_INTERVAL_MS = 20;

// 1.5 deg/tick @ 50 Hz = 75 deg/s
const float MAX_STEP_DEG = 1.5f;

// ----- State -----

volatile int pan_target  = PAN_CENTER;
volatile int tilt_target = TILT_CENTER;

// Float so MAX_STEP_DEG < 1.0 isn't rounded to zero
float pan_current  = PAN_CENTER;
float tilt_current = TILT_CENTER;

unsigned long last_update = 0;


// ----- Helper -----

// Step `current` toward `target` by at most `max_step` degrees.
float step(float current, float target, float max_step) {
  float error = target - current;

  if (error > max_step)  return current + max_step;
  if (error < -max_step) return current - max_step;

  return target;
}

bool set_cam(int pan_angle, int tilt_angle) {
  pan_target  = constrain(pan_angle,  PAN_MIN,  PAN_MAX);
  tilt_target = constrain(tilt_angle, TILT_MIN, TILT_MAX);
  return true;
}

bool set_pan(int angle) {
  pan_target = constrain(angle, PAN_MIN, PAN_MAX);
  return true;
}

bool set_tilt(int angle) {
  tilt_target = constrain(angle, TILT_MIN, TILT_MAX);
  return true;
}

bool center() {
  pan_target  = PAN_CENTER;
  tilt_target = TILT_CENTER;
  return true;
}


// ----- Main -----

void setup() {
  pan.attach(9);
  tilt.attach(10);

  pan.write(PAN_CENTER);
  tilt.write(TILT_CENTER);

  Bridge.begin();

  Bridge.provide_safe("set_cam",  set_cam);
  Bridge.provide_safe("set_pan",  set_pan);
  Bridge.provide_safe("set_tilt", set_tilt);
  Bridge.provide_safe("center",   center);
}

void loop() {
  unsigned long now = millis();

  if (now - last_update >= UPDATE_INTERVAL_MS) {
    last_update = now;

    pan_current  = step(pan_current,  (float)pan_target,  MAX_STEP_DEG);
    tilt_current = step(tilt_current, (float)tilt_target, MAX_STEP_DEG);

    pan.write((int)(pan_current + 0.5f)); 
    tilt.write((int)(tilt_current + 0.5f));
  }

  delay(1); // yield since provide_safe defers RPC handlers to loop()
}