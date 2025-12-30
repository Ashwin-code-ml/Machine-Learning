import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Jellyfish Classifier",
    page_icon="🪼",
    layout="centered"
)

# =========================
# CUSTOM CSS (ROYAL UI)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
.main {
    background: linear-gradient(135deg, #1f1c2c, #928DAB);
    padding: 30px;
    border-radius: 20px;
}
h1 {
    color: gold;
    text-align: center;
    font-weight: 800;
}
h3 {
    color: #e0e0e0;
}
.stButton>button {
    background: linear-gradient(90deg, #FFD700, #FFB347);
    color: black;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    padding: 10px 24px;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #FFB347, #FFD700);
}
.card {
    background: rgba(255,255,255,0.12);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL & ENCODER
# =========================
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("jellyfish_mobilenetv2.keras")
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()
class_names = label_encoder.classes_

# =========================
# TITLE
# =========================
st.markdown("<h1>🪼 Jellyfish Species Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 align='center'>AI-Powered Image Classification</h3>", unsafe_allow_html=True)
st.write("")

# =========================
# IMAGE UPLOAD CARD
# =========================
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload a Jellyfish Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("🔮 Predict"):
            preds = model.predict(img_array)
            pred_index = np.argmax(preds)
            confidence = preds[0][pred_index] * 100

            st.success(f"🧠 Prediction: **{class_names[pred_index]}**")
            st.progress(int(confidence))
            st.info(f"Confidence: **{confidence:.2f}%**")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<hr>
<p style="text-align:center; color:#ddd;">
Built with ❤️ using <b>TensorFlow</b> & <b>Streamlit</b><br>
AI Jellyfish Classifier
</p>
""", unsafe_allow_html=True)
