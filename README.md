# Driver Drowsiness Detection and Alert System

A camera-based driver drowsiness detector running on a Raspberry Pi 4. It tracks the driver's eyes in real time and raises an escalating LED and buzzer alert when signs of drowsiness appear.

---

## How it works

```
USB webcam ──► MediaPipe Face Landmarker ──► Eye Aspect Ratio (EAR)
                                                   │
                                                   ▼
                                   Eye-closure duration + PERCLOS
                                                   │
                                                   ▼
                                  Alert level 0–3 ──► Red LED + Buzzer
```

1. **Capture:** frames from a USB webcam (640×480, MJPG) via OpenCV.
2. **Landmarks:** MediaPipe Face Landmarker finds six points around each eye.
3. **Eye state:** the Eye Aspect Ratio (EAR) is computed for both eyes; the eyes count as closed when the average EAR is below `0.20`.
4. **Drowsiness:** the system tracks how long the eyes stay closed and the **PERCLOS** (percentage of time the eyes were closed over a 20 s window).
5. **Alert:** the measurements are mapped to an alert level that drives the LED and buzzer.

### Alert levels

| Level | State | Trigger | LED | Buzzer |
|---|---|---|---|---|
| 0 | Normal | eyes open, or no face detected | off | off |
| 1 | Warning | eyes closed ≥ 0.75 s, or PERCLOS ≥ 15% | slow blink (1 Hz) | silent |
| 2 | Drowsy | eyes closed ≥ 1.5 s, or PERCLOS ≥ 30% | blink (2 Hz) | beep with each blink |
| 3 | Critical | eyes closed ≥ 3 s | rapid blink (5 Hz) | rapid beeping |

---

## Hardware

| Component | Pi pin (physical) | GPIO (BCM) | Circuit |
|---|---|---|---|
| Red LED | 13 | 27 | GPIO → 220 Ω → LED → GND |
| Active buzzer | 11 | 17 | GPIO → 1 kΩ → BC547 base; buzzer between 5 V and collector; emitter → GND |
| 5 V | 2 | — | breadboard (+) rail |
| GND | 6 | — | breadboard (−) rail |

The buzzer is switched through an NPN transistor because it draws ~25–30 mA, more than a GPIO pin can safely supply.

---

## Repository structure

| File | Purpose |
|---|---|
| `detector_pi.py` | **Main program for the Raspberry Pi** — detection + LED/buzzer alerts + logging |
| `alert.py` | Alert module: drives the LED and buzzer for levels 0–3 in a background thread |
| `integration_test.py` | Hardware test: webcam preview, keys `0`–`3` set the alert level |
| `latency_test.py` | Measures alert response time (level request → GPIO output) |
| `drowsiness_detector.py` | Original detector for macOS (used during development) |
| `test_ear.py` | Early EAR-only test script |
| `train_eye_state_cnn.ipynb` | Training notebook for the alternative CNN eye-state model |
| `test_cnn_webcam.py`, `compare_cnn_models.py` | CNN webcam test and Keras vs. TFLite comparison |
| `models/` | `face_landmarker.task` (MediaPipe), trained CNN (`.keras`, `.tflite`) |
| `requirements.txt` | Python dependencies |

---

## Running on the Raspberry Pi

> **Note:** The Pi 4's CPU has no ARMv8 AES extension, so MediaPipe 1.x crashes on it with an *illegal instruction* error. Use **MediaPipe 0.10.18**, which needs **Python 3.11 or 3.12**. Raspberry Pi OS (Trixie) ships Python 3.13, so create a separate 3.11 environment with `uv`.

```bash
# 1. Install uv and create a Python 3.11 environment
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv --python 3.11 ~/dds311
source ~/dds311/bin/activate

# 2. Get the code and install dependencies
git clone https://github.com/Yahya-za1/Driver-Drowsiness-Detection-and-Alert-System.git
cd Driver-Drowsiness-Detection-and-Alert-System
uv pip install -r requirements.txt

# 3. Test the LED and buzzer (steps through levels 0–3)
python alert.py

# 4. Run the detector (press Q in the window to quit)
python detector_pi.py
```

> **Note:** `requirements.txt` also includes the training libraries (e.g. TensorFlow), which the Pi doesn't need. On a laptop with Python 3.13, install the latest `mediapipe` instead of 0.10.18.

`detector_pi.py` writes `detector_log.csv` with FPS, processing time, EAR, PERCLOS, alert level and alert latency.

---

## Eye-state model: why EAR instead of the CNN

A MobileNetV2 CNN was trained on the CEW dataset (`train_eye_state_cnn.ipynb`) and reached **86.6% test accuracy**. In live webcam tests, however, it did not generalise well to real camera images. The prototype therefore uses MediaPipe landmarks with EAR, which needs no training and works reliably in real time. The trained model is kept in `models/` for reference.

---

## Team

- Yahya Al-Khateeb
- Osama Abdulghani
- Oluwakorede Oseni
- Oloto Arni