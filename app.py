import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("fakevsreal_weights.keras")

# App UI
st.title("Fake vs Real Image Detector 🕵️‍♀️")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Preprocess the image to match the model function
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match model input
    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 4:  # RGBA -> RGB
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]

    if prediction < 0.5:
        st.success(f"🟢 This image is **Real** ({(1-prediction)*100:.2f}% confidence)")
    else:
        st.error(f"🔴 This image is **Fake** ({prediction*100:.2f}% confidence)")
