
import numpy as np
import tensorflow as tf
from PIL import Image

# Load original Keras model
keras_model = tf.keras.models.load_model(
    "models/best_cew_mobilenet.keras"
)

# Load TensorFlow Lite model
interpreter = tf.lite.Interpreter(
    model_path="models/eye_state_model.tflite"
)

interpreter.allocate_tensors()

input_info = interpreter.get_input_details()[0]
output_info = interpreter.get_output_details()[0]

# Load image saved from camera
image = Image.open("debug_face.jpg").convert("RGB")
image = image.resize((160, 160))

image_array = np.asarray(
    image, dtype=np.float32
)[None, ...]

# Keras prediction
keras_prob = float(
    keras_model.predict(
        image_array, verbose=0
    )[0][0]
)

# Prepare TFLite input
if input_info["dtype"] == np.float32:
    tflite_input = image_array
else:
    scale, zp = input_info["quantization"]
    dtype = input_info["dtype"]

    tflite_input = np.clip(
        np.round(image_array / scale + zp),
        np.iinfo(dtype).min,
        np.iinfo(dtype).max
    ).astype(dtype)

# TFLite prediction
interpreter.set_tensor(
    input_info["index"], tflite_input
)

interpreter.invoke()

result = interpreter.get_tensor(
    output_info["index"]
)[0][0]

if output_info["dtype"] != np.float32:
    scale, zp = output_info["quantization"]
    result = (result - zp) * scale

tflite_prob = float(result)

print(f"Keras open probability: {keras_prob:.4f}")
print(f"TFLite open probability: {tflite_prob:.4f}")
print(f"Difference: {abs(keras_prob - tflite_prob):.4f}")
