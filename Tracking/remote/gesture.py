"""
Static hand-pose gesture recognition - PC side.

Uses the MediaPipe Gesture Recognizer task out-of-the-box with no
fine-tuning. This module serves to turn the noisy stream from the 
UNO Q into clean discrete events.

    python3 gesture.py                                    # local webcam
    python3 gesture.py --source 2                         # specified local device
    python3 gesture.py --source http://10.88.111.30:8080/ # board stream
    python3 gesture.py --no-preview
"""

import time
import argparse

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from media_control import MediaController


# ----- Tuning -----

MODEL_PATH = "gesture_recognizer.task"

# Wall time is used for gesture timeouts, since images are being streamed
# from the UNO Q at variable (lower) framerates.

# Minimum pose hold time for a gesture
HOLD_TIME_S = 0.30

# Repeat interval for held gestures (volume)
REPEAT_INTERVAL_S = 0.40

# Minimum time between any two events
GLOBAL_COOLDOWN_S = 0.15

# Grade period for momentary classification dropouts since 
# it keeps flickering between poses
GRACE_S = 0.15

# Classifier confidence threshold
MIN_SCORE = 0.60


# ----- Gesture vocab -----

# MediaPipe canned gesture names -> commands
COMMANDS = {
    "Open_Palm": "PLAY_PAUSE",
    "Pointing_Up": "NEXT",
    "Victory": "PREVIOUS",
    "Thumb_Up": "VOLUME_UP",
    "Thumb_Down": "VOLUME_DOWN",
}

# Commands repeated while gesture is held
REPEATING = {"VOLUME_UP", "VOLUME_DOWN"}

# This gesture was intended as an explicit reset. However, the current
# implementation worked fine when testing so I decided to leave it as is...
# Itermediary gesture to explicitly reset system - DEAD CODE
NEUTRAL = "Closed_Fist"

# ----- Event generation -----
 
class GestureFilter:
    """
    Filter frame-by-frame classification stream into discrete events.

    Three procedures:
        - HOLD
          Poses must persist for hold_time before being recognized
          to prevent accidental activations from transitional poses.
 
        - RESET
          Poses should not repeatedly fire by default - a neutral pose
          or absence resets the system for the next gesture.
 
        - REPEAT
          Exception to RESET for incremental controls. Events are 
          registered every repeat_interval while a pose is held.
    """
    def __init__(self, hold_time=HOLD_TIME_S,
                 repeat_interval=REPEAT_INTERVAL_S,
                 cooldown=GLOBAL_COOLDOWN_S,
                 grace=GRACE_S):
        self.hold_time = hold_time
        self.repeat_interval = repeat_interval
        self.cooldown = cooldown
        
        self.grace = grace
        self.pose_lost_at = None

        self.current_pose = None
        self.pose_since = 0.0
 
        self.reset = True
        self.last_fire_time = 0.0
        self.last_event_time = 0.0
 
    def update(self, pose, now=None):
        """
        Accepts canned pose label and returns a command if a gesture is 
        registered. Accepts / returns None otherwise.
        """
        if now is None:
            now = time.perf_counter()
 
        # Apply grace period for momentary None dropouts
        if pose is None and self.current_pose is not None:
            if self.pose_lost_at is None:
                self.pose_lost_at = now
            if now - self.pose_lost_at < self.grace:
                pose = self.current_pose  # suppress the dropout
            else:
                self.pose_lost_at = None  # grace period expired
        else:
            self.pose_lost_at = None # clear grace period timer

        # Reset hold timer upon pose change excepting flickers
        # ANYTHING that is not a valid command triggers a reset:
        # - absence, neutral fist, unmapped gestures (ILoveYou)
        if pose != self.current_pose:
            self.current_pose = pose
            self.pose_since = now
            if pose is None or COMMANDS.get(pose) is None:
                self.reset = True
            return None
 
        # Ignore neutral pose or absence
        if pose is None or pose == NEUTRAL:
            return None
 
        command = COMMANDS.get(pose)
        if command is None:
            return None # ignore excluded poses (ILoveYou)
 
        # Ignore pose if not held long enough
        if now - self.pose_since < self.hold_time:
            return None
 
        # Ignore pose if global cooldown not met
        if now - self.last_event_time < self.cooldown:
            return None
 
        # Repeated commands: register first time, then periodically
        if command in REPEATING:
            if self.reset:
                self.reset = False
                self.last_fire_time = now
                self.last_event_time = now
                return command
            if now - self.last_fire_time >= self.repeat_interval:
                self.last_fire_time = now
                self.last_event_time = now
                return command
            return None
 
        # Discrete commands: register once if system is reset
        if self.reset:
            self.reset = False
            self.last_fire_time = now
            self.last_event_time = now
            return command
 
        return None
 
    def hold_progress(self, now=None):
        """0.0-1.0 fraction of hold time elapsed for debugging."""
        if self.current_pose is None or COMMANDS.get(self.current_pose) is None:
            return 0.0
        if not self.reset and COMMANDS[self.current_pose] not in REPEATING:
            return 1.0
        if now is None:
            now = time.perf_counter()
        return min(1.0, (now - self.pose_since) / self.hold_time)
 
# ----- Preview -----
 
# Landmark connections (21) for drawing skeleton
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # index
    (5, 9), (9, 10), (10, 11), (11, 12),    # middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # ring
    (13, 17), (17, 18), (18, 19), (19, 20), # pinky
    (0, 17),                                # palm base
]
 
def draw_preview(frame, landmarks, pose, score, command, filter, fps):
    """Draw hand skeleton, pose label, and HUD on preview."""
    h, w = frame.shape[:2]
 
    # Skeleton
    if landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
        for a, b in CONNECTIONS: # bones (white)
            cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
        for pt in pts: # joints (blue)
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)
 
    # System reset indicator (green/orange)
    colour = (0, 255, 0) if filter.reset else (0, 165, 255)
 
    # Pose label + confidence score (top left)
    label = pose or "---"
    if pose and score:
        label = f"{label} - {score:.2f}"
    cv2.putText(frame, label, (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
 
    # Hold progress bar (a real one!)
    progress = filter.hold_progress()
    if progress > 0:
        cv2.rectangle(frame, (10, 46), (210, 58), (80, 80, 80), 1)
        cv2.rectangle(frame, (10, 46), (10 + int(200 * progress), 58),
                      colour, -1)
 
    # Display command name if gesture registered
    if command:
        cv2.putText(frame, command, (10, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
 
    # FPS counter (top right) + reset status (bottom left)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 100, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, "reset" if filter.reset else "pending reset...",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
 
    return frame
 
 
# ----- Main -----
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0",
                   help="device index, /dev/videoN, or MJPEG URL")
    p.add_argument("--no-preview", action="store_true")
    p.add_argument("--hold", type=float, default=HOLD_TIME_S)
    p.add_argument("--repeat", type=float, default=REPEAT_INTERVAL_S)
    p.add_argument("--min-score", type=float, default=MIN_SCORE)
    p.add_argument("--no-mirror", action="store_true",
                   help="disable horizontal flip")
    args = p.parse_args()
 
    # Accept device index ("0") / path ("/dev/video2") / URL
    source = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        return
 
    # VIDEO mode - feed frame-by-frame with timestamps
    options = vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)

    gesture_filter = GestureFilter(hold_time=args.hold, repeat_interval=args.repeat)
    controller = MediaController()

    print("\nGestures:")
    print("  fist           neutral / reset")
    print("  open palm      play/pause")
    print("  point up       next")
    print("  victory        previous")
    print("  thumb up       volume up   (repeats while held)")
    print("  thumb down     volume down (repeats while held)")
    print("\nq to quit.\n")
 
    t0 = time.perf_counter() # ms timestamp ref for VIDEO
    last_t = time.perf_counter() # fps ref
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame capture failed")
                break
 
            # Mirror preview
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
 
            # Why does OpenCV use BGR... (MediaPipe expects RGB)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - t0) * 1000)

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            pose = score = landmarks = None
 
            if result.gestures:
                top = result.gestures[0][0] # gestures = list of (hand) lists
                if top.score >= args.min_score and top.category_name != "None":
                    pose = top.category_name
                    score = top.score
 
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
 
            command = gesture_filter.update(pose)
            if command:
                print(f"{command}")
                controller.execute(command)
 
            # EMA FPS (Claude suggested it)
            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
 
            if not args.no_preview:
                draw_preview(frame, landmarks, pose, score,
                             command, gesture_filter, fps)
                cv2.imshow("gesture", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        recognizer.close()

if __name__ == "__main__":
    main()