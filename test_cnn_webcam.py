
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "eye_state_model.tflite"
)

# Load trained model
interpreter = tf.lite.Interpreter(
    model_path=str(MODEL_PATH)
)
interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

# Load face detector
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError("Camera not available")

print("Camera started. Press Q to exit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    gray = cv2.cvtColor(
        frame, cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]

        # Convert BGR to RGB
        face = cv2.cvtColor(
            face, cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        face = cv2.resize(face, (160, 160))
        # Save the actual RGB input to the model
        from PIL import Image

        Image.fromarray(face).save("debug_face.jpg")
        # Press S to save the current face crop
        # Add this immediately after resizing the face

        debug_face = face.copy()


        image = np.expand_dims(
            face.astype(np.float32),
            axis=0
        )

        # Handle float and quantized model inputs
        if input_info["dtype"] == np.float32:
            model_input = image
        else:
            scale, zp = input_info["quantization"]
            dtype = input_info["dtype"]

            model_input = np.clip(
                np.round(image / scale + zp),
                np.iinfo(dtype).min,
                np.iinfo(dtype).max
            ).astype(dtype)

        interpreter.set_tensor(
            input_info["index"],
            model_input
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_info["index"]
        )[0][0]

        if output_info["dtype"] != np.float32:
            scale, zp = output_info["quantization"]
            output = (output - zp) * scale

        open_probability = float(output)
        print(
    f"Open probability: {open_probability:.4f}",
    end="\r",
    flush=True
)

        eye_state = (
            "OPEN"
            if open_probability >= 0.5
            else "CLOSED"
        )

        color = (
            (0, 255, 0)
            if eye_state == "OPEN"
            else (0, 0, 255)
        )

        cv2.rectangle(
            frame, (x, y),
            (x+w, y+h),
            color, 2
        )

        cv2.putText(
            frame,
            f"{eye_state}: {open_probability:.2f}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

    cv2.imshow("Drowsiness Model Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

