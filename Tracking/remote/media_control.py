"""
Media control - PC side, Ubuntu + PipeWire.

Playback via MPRIS (playerctl), volume via PipeWire (wpctl).
Module to map gesture commands to OS media actions.

    python3 media_control.py PLAY_PAUSE VOLUME_UP VOLUME_UP
"""

import subprocess
import sys

VOLUME_STEP = 3
TIMEOUT_S = 2.0 # timeout in case playerctl hangs
SINK = "@DEFAULT_AUDIO_SINK@" # whatever the current output device is


class MediaController:
    """
    Volume control commands. Failures are non-fatal, just reported
    and swallowed to not distrub the GestureFilter loop.
    """
    def __init__(self, volume_step=VOLUME_STEP):
        # -l 1.0 caps volume at 100%
        vol = lambda d: ["wpctl", "set-volume", "-l", "1.0", SINK, d]
        self.commands = {
            "PLAY_PAUSE":  ["playerctl", "play-pause"],
            "NEXT":        ["playerctl", "next"],
            "PREVIOUS":    ["playerctl", "previous"],
            "VOLUME_UP":   vol(f"{volume_step}%+"),
            "VOLUME_DOWN": vol(f"{volume_step}%-"),
        }
 
    def execute(self, command):
        argv = self.commands.get(command)
        if argv is None:
            print(f"Unknown command: {command}")
            return False
        try:
            # Capture output here even if not used to keep it out 
            # of the gesture loop's console output
            result = subprocess.run(argv, capture_output=True,
                                    text=True, timeout=TIMEOUT_S)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            # FileNotFoundError: no playerctl or wpctl
            # TimeoutExpired: command hung (see TIMEOUT_S)
            print(f"{argv[0]}: {type(e).__name__}")
            return False
        if result.returncode != 0:
            # Likely no player running
            print(f"{argv[0]}: {result.stderr.strip() or 'failed'}")
            return False
        return True

if __name__ == "__main__":
    # Just to play around
    controller = MediaController()
    for command in sys.argv[1:]:
        print(command)
        controller.execute(command)