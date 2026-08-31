#include <Servo.h>
#include "Arduino_RouterBridge.h"

Servo pan;
Servo tilt;

// ----- Calibration -----

const int PAN_CENTER  = 94;
const int PAN_MIN     = 4;
const int PAN_MAX     = 184;

const int TILT_CENTER = 94;
const int TILT_MIN    = 64;
const int TILT_MAX    = 124;

// ----- RPC functions -----

bool set_pan(int angle) {
  angle = constrain(angle, PAN_MIN, PAN_MAX);
  pan.write(angle);
  return true;
}

bool set_tilt(int angle) {
  angle = constrain(angle, TILT_MIN, TILT_MAX);
  tilt.write(angle);
  return true;
}

bool set_cam(int pan_angle, int tilt_angle) {
  pan_angle = constrain(pan_angle, PAN_MIN, PAN_MAX);
  tilt_angle = constrain(tilt_angle, TILT_MIN, TILT_MAX);

  pan.write(pan_angle);
  tilt.write(tilt_angle);

  return true;
}

// ----- Main -----

void setup() {
  pan.attach(9);
  tilt.attach(10);

  pan.write(PAN_CENTER);
  tilt.write(TILT_CENTER);

  Bridge.begin();
  
  Bridge.provide_safe("set_pan", set_pan);
  Bridge.provide_safe("set_tilt", set_tilt);
  Bridge.provide_safe("set_cam", set_cam);
}

void loop() { 
}