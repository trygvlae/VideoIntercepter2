import os
import shutil
import subprocess
import time
from typing import List, Optional


class VideoTransmitter:
    def __init__(self, video_path: str, preferred_players: Optional[List[str]] = None) -> None:
        self.video_path = os.path.abspath(video_path)
        self.available_player = None  # type: Optional[str]

        candidates = preferred_players or [
            "cvlc",  # VLC console
            "vlc",   # VLC GUI (falls back to headless if X not available)
            "ffplay",
            "omxplayer",
        ]
        for name in candidates:
            if shutil.which(name):
                self.available_player = name
                break

        if not os.path.isfile(self.video_path):
            print(f"[WARN] Video file not found: {self.video_path}")
        if not self.available_player:
            print("[WARN] No media player found (cvlc/vlc/ffplay/omxplayer). Will only sleep for duration.")

    def play_for(self, duration_seconds: float) -> None:
        """Play the configured video for approximately duration_seconds. Blocks until complete.

        If no player is available or file is missing, this method just sleeps for the duration.
        """
        if not self.available_player or not os.path.isfile(self.video_path) or duration_seconds <= 0:
            time.sleep(max(0.0, float(duration_seconds)))
            return

        player = self.available_player
        dur = max(0.1, float(duration_seconds))

        try:
            if player in ("cvlc", "vlc"):
                # Use VLC run-time limit and exit after
                cmd = [
                    player,
                    "--no-video-title-show",
                    "--quiet",
                    f"--run-time={dur}",
                    "--play-and-exit",
                    self.video_path,
                ]
                subprocess.run(cmd, check=False)
                return

            if player == "ffplay":
                cmd = [
                    "ffplay",
                    "-loglevel", "error",
                    "-autoexit",
                    "-t", str(dur),
                    self.video_path,
                ]
                subprocess.run(cmd, check=False)
                return

            if player == "omxplayer":
                # omxplayer doesn't have a simple duration limit; kill after timeout
                proc = subprocess.Popen(["omxplayer", "--no-osd", self.video_path])
                try:
                    proc.wait(timeout=dur)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                return

        except Exception as exc:
            print(f"[WARN] Video playback failed with {player}: {exc}. Falling back to sleep.")
            time.sleep(dur)


