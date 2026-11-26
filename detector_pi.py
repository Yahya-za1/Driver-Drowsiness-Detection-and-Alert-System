"""
detector_pi.py — Raspberry Pi version of the team's drowsiness_detector.py.

Same detection logic as the repo (MediaPipe Face Landmarker -> EAR ->
closure duration + PERCLOS), with three changes for the Pi:
  1. Camera opened with V4L2 (Linux) instead of AVFoundation (macOS)
  2. Alerts drive the red LED + buzzer through alert.py (levels 0-3)
  3. Logs FPS, processing time and alert latency to detector_log.csv

Put this file in the repo folder (next to the models/ folder) together
with alert.py, then run:   python3 detector_pi.py
Keys:  q = quit
"""

import csv
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from alert import Alert

# ----------------------------------
# Configuration
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20

# Alert-level thresholds
L1_CLOSED_S = 0.75      # level 1: eyes closed this long ...
L1_PERCLOS = 15.0       # ... or PERCLOS above this
L2_CLOSED_S = 1.5       # level 2: team's original alert rule
L2_PERCLOS = 30.0
L3_CLOSED_S = 3.0       # level 3: critical

PERCLOS_WINDOW = 20.0
MIN_OBSERVED_SECONDS = 10.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

LABELS = {0: "NORMAL", 1: "WARNING", 2: "DROWSY", 3: "CRITICAL"}
COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 120, 255), 3: (0, 0, 255)}


# ----------------------------------
# Eye Aspect Ratio
# ----------------------------------
def calculate_ear(landmarks, indices, width, height):
    p = np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in indices])
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return None if c == 0 else (a + b) / (2.0 * c)


# ----------------------------------
# PERCLOS tracker (unchanged from repo)
# ----------------------------------
class PerclosTracker:
    def __init__(self, window_seconds):
        self.window = window_seconds
        self.intervals = deque()
        self.previous_time = None
        self.previous_closed = None

    def update(self, now, closed):
        if closed is None:
            self.previous_time = None
            self.previous_closed = None
        else:
            if self.previous_time is not None:
                dt = now - self.previous_time
                if 0 < dt <= 0.5:
                    self.intervals.append((self.previous_time, now, self.previous_closed))
            self.previous_time = now
            self.previous_closed = closed

        cutoff = now - self.window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()

        observed = closed_time = 0.0
        for start, end, was_closed in self.intervals:
            d = max(0.0, end - max(start, cutoff))
            observed += d
            if was_closed:
                closed_time += d

        if observed == 0:
            return 0.0, 0.0
        return 100.0 * closed_time / observed, observed


# ----------------------------------
# Map measurements to alert level 0-3
# ----------------------------------
def compute_level(closed, closure_s, perclos, observed):
    if closed is None:                       # no face -> no alert
        return 0
    perclos_ok = observed >= MIN_OBSERVED_SECONDS
    if closed and closure_s >= L3_CLOSED_S:
        return 3
    if (closed and closure_s >= L2_CLOSED_S) or (perclos_ok and perclos >= L2_PERCLOS):
        return 2
    if (closed and closure_s >= L1_CLOSED_S) or (perclos_ok and perclos >= L1_PERCLOS):
        return 1
    return 0


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")
    return cap


def main():
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )

    camera = open_camera()
    alerter = Alert()
    tracker = PerclosTracker(PERCLOS_WINDOW)

    log = open("detector_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["t_s", "fps", "proc_ms", "ear", "closed", "closure_s",
                     "perclos", "level", "alert_latency_ms"])

    closed_since = None
    start = time.monotonic()
    frames, fps, fps_t = 0, 0.0, time.monotonic()
    last_log = 0.0
    logged_lat = 0

    print("Drowsiness detection started. Press Q in the window to quit.")
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Frame grab failed")
                    break

                now = time.monotonic()
                t_proc = time.perf_counter()
                h, w = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(rgb))
                result = detector.detect_for_video(image, int((now - start) * 1000))

                closed, ear = None, None
                if result.face_landmarks:
                    lm = result.face_landmarks[0]
                    left = calculate_ear(lm, LEFT_EYE, w, h)
                    right = calculate_ear(lm, RIGHT_EYE, w, h)
                    if left is not None and right is not None:
                        ear = (left + right) / 2.0
                        closed = ear < EAR_THRESHOLD
                        for idx in LEFT_EYE + RIGHT_EYE:
                            p = lm[idx]
                            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (255, 0, 0), -1)

                if closed:
                    closed_since = closed_since or now
                else:
                    closed_since = None
                closure_s = now - closed_since if closed_since else 0.0

                perclos, observed = tracker.update(now, closed)
                level = compute_level(closed, closure_s, perclos, observed)
                alerter.set_level(level)

                proc_ms = (time.perf_counter() - t_proc) * 1000

                # FPS over 1-second windows
                frames += 1
                if now - fps_t >= 1.0:
                    fps = frames / (now - fps_t)
                    frames, fps_t = 0, now

                # Log ~5 times per second, plus latency of every level change
                lat = ""
                if logged_lat < len(alerter.latencies):
                    lat = f"{alerter.latencies[-1][2]:.3f}"
                    logged_lat = len(alerter.latencies)
                if lat or now - last_log >= 0.2:
                    writer.writerow([f"{now - start:.3f}", f"{fps:.1f}", f"{proc_ms:.1f}",
                                     "" if ear is None else f"{ear:.3f}",
                                     "" if closed is None else int(closed),
                                     f"{closure_s:.2f}", f"{perclos:.1f}", level, lat])
                    last_log = now

                # Overlay
                status = "NO FACE" if closed is None else f"Level {level}: {LABELS[level]}"
                color = (0, 255, 255) if closed is None else COLORS[level]
                lines = [
                    (status, 0.8),
                    (f"EAR: {ear:.3f}" if ear is not None else "EAR: N/A", 0.6),
                    (f"Closure: {closure_s:.1f}s   PERCLOS: {perclos:.1f}%", 0.6),
                    (f"FPS: {fps:.1f}   Proc: {proc_ms:.0f} ms", 0.6),
                ]
                for i, (text, scale) in enumerate(lines):
                    cv2.putText(frame, text, (15, 35 + i * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

                cv2.imshow("Driver Drowsiness Detection (Pi)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        alerter.close()
        camera.release()
        cv2.destroyAllWindows()
        log.close()
        print("Saved detector_log.csv")


if __name__ == "__main__":
    main()
