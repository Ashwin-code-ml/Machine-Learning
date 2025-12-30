import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Fashion AI",
    page_icon="👗",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #1b2735, #090a0f);
    color: white;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}

.glass {
    background: rgba(255,255,255,0.12);
    backdrop-filter: blur(14px);
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    margin-bottom: 25px;
}

.title {
    font-size: 42px;
    font-weight: 800;
}

.sub {
    opacity: 0.85;
    font-size: 18px;
}

.good { color: #2ecc71; font-weight: 700; }
.mid { color: #f1c40f; font-weight: 700; }
.low { color: #e74c3c; font-weight: 700; }

div.stButton > button {
    background: linear-gradient(90deg, #8e2de2, #4a00e0);
    color: white;
    border-radius: 14px;
    height: 48px;
    border: none;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
@st.cache_resource
def load_fashion_model():
    return load_model("fashion_mnist_model.keras")

model = load_fashion_model()

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ================= HEADER =================
st.markdown("""
<div class="glass">
<div class="title">👗 Fashion AI Classifier</div>
<div class="sub">A deep learning model that understands fashion images</div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.write("Upload an image and explore AI predictions interactively.")

uploaded_file = st.file_uploader(
    "📤 Upload Clothing Image",
    type=["png", "jpg", "jpeg"]
)

def preprocess_image(img):
    img = img.convert("L").resize((28, 28))
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 784), arr

# ================= MAIN FLOW =================
if uploaded_file:

    tabs = st.tabs(["📸 Image", "🧠 Prediction", "📊 Probabilities"])

    image = Image.open(uploaded_file)

    with tabs[0]:
        st.markdown('<div class="glass"><h3>Uploaded Image</h3></div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    with tabs[1]:
        st.markdown('<div class="glass"><h3>Model Thinking…</h3></div>', unsafe_allow_html=True)

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        img_flat, img_2d = preprocess_image(image)
        preds = model.predict(img_flat)

        pred_idx = int(np.argmax(preds))
        pred_name = CLASS_NAMES[pred_idx]
        confidence = float(np.max(preds)) * 100

        if confidence > 85:
            conf_class = "good"
        elif confidence > 60:
            conf_class = "mid"
        else:
            conf_class = "low"

        st.markdown(f"""
        <div class="glass">
            <h2>🎯 Prediction Result</h2>
            <p><b>Class:</b> {pred_name}</p>
            <p><b>Confidence:</b> <span class="{conf_class}">{confidence:.2f}%</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.image(img_2d, width=160, caption="28×28 Grayscale")

    with tabs[2]:
        st.markdown('<div class="glass"><h3>Class Probability Distribution</h3></div>', unsafe_allow_html=True)
        prob_dict = {CLASS_NAMES[i]: float(preds[0][i]) for i in range(10)}
        st.bar_chart(prob_dict)

else:
    st.markdown("""
    <div class="glass">
    <h3>👆 Get Started</h3>
    <p>Upload a fashion image to see the AI in action.</p>
    </div>
    """, unsafe_allow_html=True)
