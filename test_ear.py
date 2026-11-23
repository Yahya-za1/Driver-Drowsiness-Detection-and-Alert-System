
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "face_landmarker.task"
)

CAMERA_INDEX = 0
EAR_THRESHOLD = 0.20
CLOSED_DURATION = 2.0

# Eye landmark positions
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def calculate_ear(landmarks, indices, width, height):
    points = np.array([
        [
            landmarks[i].x * width,
            landmarks[i].y * height
        ]
        for i in indices
    ])

    # Vertical distances
    a = np.linalg.norm(points[1] - points[5])
    b = np.linalg.norm(points[2] - points[4])

    # Horizontal distance
    c = np.linalg.norm(points[0] - points[3])

    if c == 0:
        return 0.0

    return (a + b) / (2.0 * c)


# -----------------------------
# Initialize MediaPipe
# -----------------------------
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

# -----------------------------
# Start camera
# -----------------------------
camera = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_AVFOUNDATION
)

if not camera.isOpened():
    raise RuntimeError("Camera could not be opened")

closed_since = None
start_time = time.monotonic()

print("Camera started. Press Q to quit.")

try:
    with FaceLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                break

            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            result = detector.detect_for_video(
                mp_image,
                timestamp_ms
            )

            now = time.monotonic()
            status = "NO FACE"
            color = (0, 255, 255)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]

                left_ear = calculate_ear(
                    landmarks, LEFT_EYE, width, height
                )

                right_ear = calculate_ear(
                    landmarks, RIGHT_EYE, width, height
                )

                average_ear = (left_ear + right_ear) / 2

                # Display eye positions
                for idx in LEFT_EYE + RIGHT_EYE:
                    p = landmarks[idx]
                    x = int(p.x * width)
                    y = int(p.y * height)
                    cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                if average_ear < EAR_THRESHOLD:
                    status = "EYES CLOSED"
                    color = (0, 0, 255)

                    if closed_since is None:
                        closed_since = now

                    duration = now - closed_since

                    if duration >= CLOSED_DURATION:
                        status = "DROWSINESS ALERT!"
                else:
                    status = "EYES OPEN"
                    color = (0, 255, 0)
                    closed_since = None

                cv2.putText(
                    frame,
                    f"EAR: {average_ear:.3f}  "
                    f"L: {left_ear:.3f} R: {right_ear:.3f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )
            else:
                closed_since = None

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.imshow("Eye Closure Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
