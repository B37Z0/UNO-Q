"""
Pan/tilt tracking controller. 

Accepts normalized bbox coordinates in [ymin, xmin, ymax, xmax] 
(matches TFLITE SSD output from detector) and outputs servo target 
angles in [pan, tilt] format.

Note on angle clamping:
The MCU also clamps servo angles to [4, 184] and [64, 124], but the
angles must be bounded here since it's the controller's belief
about where the camera is pointing. If a person is detected outside 
the limits, the controller 'would' drive the servos beyond the 
limits while the camera would actually stay at the limit.

Coordinate conventions
----------------------
Image space:  
    - x increases RIGHT
    - y increases DOWN

Servo space:  
    - pan  + => camera looks LEFT
    - tilt + => camera looks UP

Because of that, both axes SUBTRACT their correction:
    - person right of centre -> e_x > 0 -> pan right  -> pan  decreases
    - person below centre    -> e_y > 0 -> tilt down  -> tilt decreases
"""

# ----- Calibration -----

PAN_CENTER = 94
PAN_MIN = 4
PAN_MAX = 184

TILT_CENTER = 94
TILT_MIN = 64
TILT_MAX = 124

# Standard FOV @ 16:9 (verified)
FOV_H_DEG = 82.2
FOV_V_DEG = 52.5


class Tracker:
    def __init__(
        self,
        fov_h_deg=FOV_H_DEG,
        fov_v_deg=FOV_V_DEG,
        kp=0.35,
        deadband_frac=0.04,
        pan_limits=(PAN_MIN, PAN_MAX),
        tilt_limits=(TILT_MIN, TILT_MAX),
        pan_center=PAN_CENTER,
        tilt_center=TILT_CENTER,
        lost_frames_before_reacquire=15,
    ):
        """
        kp (proportional gain)
            - Fraction of the angular error applied each update
            - 1.0 = attempt to centre the person in a single move (overshoots)
            - 0.3-0.4 = converge over a few detections without oscillating

        deadband_frac
            - Errors smaller than this fraction of the frame are ignored
            - Stop servo buzzing on detector jitter when person is roughly centered

        lost_frames_before_reacquire
            - Number of consecutive detection-free updates before the current
              target is considered lost and a new one can be selected
        """
        self.fov_h_deg = fov_h_deg
        self.fov_v_deg = fov_v_deg
        self.kp = kp
        self.deadband_frac = deadband_frac

        self.pan_min, self.pan_max = pan_limits
        self.tilt_min, self.tilt_max = tilt_limits
        self.pan_center = pan_center
        self.tilt_center = tilt_center

        self.pan = float(pan_center)
        self.tilt = float(tilt_center)

        self.lost_frames_before_reacquire = lost_frames_before_reacquire
        self.lost_frames = 0

        # [0, 1] normalized coordinates of the current target / None if no target
        self.target_center = None

    @staticmethod
    def _box_center(box):
        """Normalized [ymin, xmin, ymax, xmax] -> (xc, yc)"""
        ymin, xmin, ymax, xmax = box
        return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)

    # ----- Target selection -----

    def _select_target(self, boxes):
        """
        0 people  -> hold + decay the lost counter
        1 person  -> follow
        2+ people -> stay with whoever is closest to the previous target
        reacquire -> pick whoever is closest to frame centre
        """
        if not boxes:
            self.lost_frames += 1
            if self.lost_frames >= self.lost_frames_before_reacquire:
                self.target_center = None
            return None

        self.lost_frames = 0
        centers = [self._box_center(b) for b in boxes]

        if self.target_center is None:
            ref = (0.5, 0.5) # reacquire closest to frame centre
        else:
            ref = self.target_center # maintain closest to last target position

        # Identify closest detection to ref (Pythagorean theorem!)
        best = min(
            centers,
            key=lambda c: (c[0] - ref[0]) ** 2 + (c[1] - ref[1]) ** 2,
        )
        self.target_center = best
        return best

    # ----- Control -----

    def update(self, boxes):
        """
        boxes
            - List of filtered + normalized [ymin, xmin, ymax, xmax]
            - Empty list if nobody visible

        Returns (pan, tilt) as integer servo angles. Values returned
        unchanged if no targets are detected
        """
        center = self._select_target(boxes)
        if center is None:
            return self.targets()

        xc, yc = center

        # Error as a signed fraction of the frame, relative to frame centre
        ex = xc - 0.5
        ey = yc - 0.5

        # Apply deadband - hopefully reduces servo buzzing
        if abs(ex) < self.deadband_frac:
            ex = 0.0
        if abs(ey) < self.deadband_frac:
            ey = 0.0

        # Fractional error -> angular error, via FOV (directly proportional)
        err_pan_deg = ex * self.fov_h_deg
        err_tilt_deg = ey * self.fov_v_deg

        # Proportional correction. Both axes subtract - see module docstring...
        self.pan -= self.kp * err_pan_deg
        self.tilt -= self.kp * err_tilt_deg

        self.pan = min(max(self.pan, self.pan_min), self.pan_max)
        self.tilt = min(max(self.tilt, self.tilt_min), self.tilt_max)

        return self.targets()

    def targets(self):
        """Returns current angles rounded for transmission"""
        return int(round(self.pan)), int(round(self.tilt))

    def reset(self):
        self.pan = float(self.pan_center)
        self.tilt = float(self.tilt_center)
        self.target_center = None
        self.lost_frames = 0