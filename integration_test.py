"""
integration_test.py — Phase 4 integration test.

Runs the full pipeline the final system will use:
    webcam frame -> "detector" -> level -> alert module -> LED/buzzer

The model isn't plugged in yet, so the keyboard stands in for it:
    0 / 1 / 2 / 3  set alert level
    q              quit

Every level change is logged to phase4_log.csv (time, level, fps, latency)
so the numbers can go straight into the report.
"""

import csv
import time

import cv2

from alert import Alert


def estimate_level(frame):
    """Placeholder for the team's model.
    Return 0-3, or None to keep the current level.
    Later: run the model on `frame` and map its output to a level."""
    return None


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")
    return cap


LABELS = {0: "NORMAL", 1: "WARNING", 2: "DROWSY", 3: "CRITICAL"}
COLORS = {0: (0, 200, 0), 1: (0, 200, 255), 2: (0, 120, 255), 3: (0, 0, 255)}


def main():
    cap = open_camera()
    alerter = Alert()

    log = open("phase4_log.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["t_s", "from_level", "to_level", "source", "fps", "latency_ms"])

    t0 = time.perf_counter()
    frames, fps, fps_t = 0, 0.0, time.perf_counter()
    logged = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed")
                break

            # FPS over 1-second windows
            frames += 1
            now = time.perf_counter()
            if now - fps_t >= 1.0:
                fps = frames / (now - fps_t)
                frames, fps_t = 0, now

            # 1) model stand-in
            source = None
            new_level = estimate_level(frame)
            if new_level is not None:
                source = "model"

            # 2) keyboard override
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("0"), ord("1"), ord("2"), ord("3")):
                new_level, source = key - ord("0"), "key"

            if new_level is not None and new_level != alerter.level:
                alerter.set_level(new_level)

            # write any new latency measurements to the log
            while logged < len(alerter.latencies):
                frm, to, ms = alerter.latencies[logged]
                writer.writerow([f"{now - t0:.3f}", frm, to, source, f"{fps:.1f}", f"{ms:.3f}"])
                print(f"Level {frm} -> {to}  latency {ms:.3f} ms  fps {fps:.1f}")
                logged += 1

            # overlay
            lvl = alerter.level
            cv2.putText(frame, f"Level {lvl}: {LABELS[lvl]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[lvl], 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "keys 0-3 = level, q = quit", (10, 465),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Drowsiness Detection - Phase 4", frame)
    finally:
        alerter.close()
        cap.release()
        cv2.destroyAllWindows()
        log.close()
        print("Saved phase4_log.csv")


if __name__ == "__main__":
    main()
