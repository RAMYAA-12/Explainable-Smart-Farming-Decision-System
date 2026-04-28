import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import shap
import cv2
import os
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Farming AI", layout="wide")

# ---------------- LOAD MODELS ----------------
crop_model = joblib.load("models/crop_model.pkl")
irrigation_model = joblib.load("models/irrigation_model.pkl")
disease_model = tf.keras.models.load_model("models/disease_model.h5")

# 🔥 IMPORTANT: Build model once to avoid Grad-CAM error
dummy = np.zeros((1, 128, 128, 3))
disease_model.predict(dummy)

# 🔥 Dynamic class names
CLASS_NAMES = sorted(os.listdir("data/PlantVillage"))

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌾 Smart Farming AI")
option = st.sidebar.selectbox(
    "Choose Module",
    ["Crop Recommendation", "Disease Detection", "Irrigation"]
)

# ========================= CROP =========================
if option == "Crop Recommendation":
    st.title("🌱 Crop Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        N = st.slider("Nitrogen", 0, 140)
        P = st.slider("Phosphorus", 0, 140)
        K = st.slider("Potassium", 0, 200)
        temp = st.slider("Temperature", 0, 50)

    with col2:
        humidity = st.slider("Humidity", 0, 100)
        ph = st.slider("pH", 0.0, 14.0)
        rainfall = st.slider("Rainfall", 0, 300)

    input_data = np.array([[N, P, K, temp, humidity, ph, rainfall]])

    if st.button("Predict Crop"):
        result = crop_model.predict(input_data)
        st.success(f"🌾 Recommended Crop: {result[0]}")

        # SHAP
        explainer = shap.TreeExplainer(crop_model)
        shap_values = explainer.shap_values(input_data)

        st.subheader("📊 Feature Importance (SHAP)")
        shap.force_plot(
            explainer.expected_value[0],
            shap_values[0][0],
            input_data,
            matplotlib=True,
            show=False
        )
        st.pyplot(bbox_inches='tight')


# ========================= DISEASE =========================
elif option == "Disease Detection":
    st.title("🌿 Disease Detection")

    # ---------- PREPROCESS ----------
    def preprocess(img):
        img = img.resize((128, 128))
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)

    # ---------- GRAD-CAM ----------
    def grad_cam(model, img_array):
        # Ensure model is built
        _ = model(img_array)

        # Find last Conv2D layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            raise ValueError("No Conv layer found in model")

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer).output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap.numpy()

    # ---------- UI ----------
    file = st.file_uploader("Upload Leaf Image")

    if file:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess(image)

        preds = disease_model.predict(img_array)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds)) * 100

        st.write("Predicted index:", idx)
        st.write("Available classes:", len(CLASS_NAMES))

        # Safe prediction
        if idx < len(CLASS_NAMES):
            st.success(f"Prediction: {CLASS_NAMES[idx]}")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Prediction index mismatch")

        # Grad-CAM
        heatmap = grad_cam(disease_model, img_array)
        heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
        heatmap = np.uint8(255 * heatmap)

        st.subheader("🔥 Affected Area (Grad-CAM)")
        st.image(heatmap)


# ========================= IRRIGATION =========================
elif option == "Irrigation":
    st.title("💧 Irrigation Recommendation")

    temp = st.slider("Temperature", 0, 50)
    humidity = st.slider("Humidity", 0, 100)
    moisture = st.slider("Soil Moisture", 0, 100)

    input_data = np.array([[temp, humidity, moisture]])

    if st.button("Predict Water Requirement"):
        result = irrigation_model.predict(input_data)
        st.success(f"💧 Water Needed: {round(result[0], 2)} units")