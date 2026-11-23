
import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import subprocess
from pathlib import Path
from collections import deque

# ----------------------------------
# Configuration
# ----------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20

CLOSED_DURATION = 1.5

PERCLOS_WINDOW = 20.0
PERCLOS_THRESHOLD = 30.0
MIN_OBSERVED_SECONDS = 10.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# ----------------------------------
# Calculate Eye Aspect Ratio
# ----------------------------------
def calculate_ear(landmarks, indices, width, height):
    points = np.array([
        [landmarks[i].x * width,
         landmarks[i].y * height]
        for i in indices
    ])

    a = np.linalg.norm(points[1] - points[5])
    b = np.linalg.norm(points[2] - points[4])
    c = np.linalg.norm(points[0] - points[3])

    if c == 0:
        return None

    return (a + b) / (2.0 * c)


# ----------------------------------
# Audible alert
# ----------------------------------
def play_alert():
    if sys.platform == "darwin":
        sound = Path(
            "/System/Library/Sounds/Glass.aiff"
        )

        if sound.exists():
            subprocess.Popen(
                ["afplay", str(sound)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    else:
        # Terminal bell; may depend on terminal settings
        print("\a", end="", flush=True)


# ----------------------------------
# PERCLOS tracker
# ----------------------------------
class PerclosTracker:
    def __init__(self, window_seconds):
        self.window = window_seconds
        self.intervals = deque()
        self.previous_time = None
        self.previous_closed = None

    def update(self, now, closed):
        # If face tracking was lost, do not count the gap
        if closed is None:
            self.previous_time = None
            self.previous_closed = None
        else:
            if self.previous_time is not None:
                dt = now - self.previous_time

                # Ignore unusually long frame gaps
                if 0 < dt <= 0.5:
                    self.intervals.append((
                        self.previous_time,
                        now,
                        self.previous_closed
                    ))

            self.previous_time = now
            self.previous_closed = closed

        cutoff = now - self.window

        # Remove intervals outside the window
        while self.intervals and self.intervals[0][1] <= cutoff:
            self.intervals.popleft()

        observed = 0.0
        closed_time = 0.0

        for start, end, was_closed in self.intervals:
            duration = max(
                0.0,
                end - max(start, cutoff)
            )

            observed += duration

            if was_closed:
                closed_time += duration

        if observed == 0:
            return 0.0, 0.0

        perclos = 100.0 * closed_time / observed

        return perclos, observed


# ----------------------------------
# MediaPipe setup
# ----------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

# ----------------------------------
# Start camera
# ----------------------------------
camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not camera.isOpened():
    raise RuntimeError("Unable to open camera")

tracker = PerclosTracker(PERCLOS_WINDOW)

closed_since = None
alert_active = False

start_time = time.monotonic()

print("Drowsiness detection started.")
print("Press Q in the camera window to quit.")

try:
    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                break

            now = time.monotonic()
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            timestamp_ms = int(
                (now - start_time) * 1000
            )

            result = detector.detect_for_video(
                image,
                timestamp_ms
            )

            status = "NO FACE"
            color = (0, 255, 255)
            closed = None
            average_ear = None

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                left = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )

                right = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )

                if left is not None and right is not None:
                    average_ear = (left + right) / 2.0
                    closed = average_ear < EAR_THRESHOLD

                    # Draw eye landmarks
                    for idx in LEFT_EYE + RIGHT_EYE:
                        p = landmarks[idx]
                        cv2.circle(
                            frame,
                            (int(p.x * width), int(p.y * height)),
                            2,
                            (255, 0, 0),
                            -1
                        )

                    if closed:
                        status = "EYES CLOSED"
                        color = (0, 0, 255)

                        if closed_since is None:
                            closed_since = now
                    else:
                        status = "EYES OPEN"
                        color = (0, 255, 0)
                        closed_since = None

            if closed is None:
                closed_since = None

            # Calculate continuous closure duration
            closure_duration = (
                now - closed_since
                if closed_since is not None
                else 0.0
            )

            # Update PERCLOS
            perclos, observed = tracker.update(
                now, closed
            )

            # Alert conditions
            duration_alert = (
                closed is True
                and closure_duration >= CLOSED_DURATION
            )

            perclos_alert = (
                observed >= MIN_OBSERVED_SECONDS
                and perclos >= PERCLOS_THRESHOLD
            )

            should_alert = (
                closed is not None
                and (duration_alert or perclos_alert)
            )

            # Play sound on new alert only
            if should_alert and not alert_active:
                print("DROWSINESS ALERT!")
                play_alert()

            alert_active = should_alert

            if should_alert:
                status = "DROWSINESS ALERT!"
                color = (0, 0, 255)

            # ----------------------------------
            # Display metrics
            # ----------------------------------
            cv2.putText(
                frame, status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )

            ear_text = (
                f"EAR: {average_ear:.3f}"
                if average_ear is not None
                else "EAR: N/A"
            )

            cv2.putText(
                frame, ear_text,
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            cv2.putText(
                frame,
                f"Closure: {closure_duration:.1f}s",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            cv2.putText(
                frame,
                f"PERCLOS: {perclos:.1f}%",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            cv2.putText(
                frame,
                f"Observed: {observed:.1f}s",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )

            cv2.imshow(
                "Driver Drowsiness Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
