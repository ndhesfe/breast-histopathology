
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = [
    "Jinak",
    "Ganas"
]

st.title("Breast Cancer Histopathology Classification")

uploaded_file = st.file_uploader(
    "Upload gambar histopathology",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # Sesuaikan dengan ukuran training
    img = image.resize((128, 128))

    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Input ke model TFLite
    interpreter.set_tensor(
        input_details[0]['index'],
        img
    )

    interpreter.invoke()

    prediction = interpreter.get_tensor(
        output_details[0]['index']
    )

    pred_idx = np.argmax(prediction)
    pred_class = labels[pred_idx]
    confidence = float(np.max(prediction) * 100)

    st.success(
        f"Prediksi: {pred_class} | Confidence: {confidence:.2f}%"
    )

    st.write("Probabilitas:")
    st.write(f"Jinak: {prediction[0][0]*100:.2f}%")
    st.write(f"Ganas: {prediction[0][1]*100:.2f}%")
