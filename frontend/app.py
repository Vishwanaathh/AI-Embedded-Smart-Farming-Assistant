import streamlit as st
import requests
import cv2
import numpy as np
import time
import winsound
from ultralytics import YOLO

# ================= CONFIG =================
BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="SMAIRT Farming",
    page_icon="🌱",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #2e7d32;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #555;
    text-align: center;
}
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.result {
    font-size: 20px;
    font-weight: 700;
    color: #1b5e20;
}
.alert {
    color: red;
    font-weight: 800;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown('<div class="main-title">🌾 Farming AIssistance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-powered Smart Agriculture Platform</div>', unsafe_allow_html=True)
st.write("")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs([
    "🌱 Fertilizer",
    "🌾 Crop",
    "💧 Irrigation",
    "🎥 Smart Vision"
])

# =====================================================
# 🌱 FERTILIZER RECOMMENDATION
# =====================================================
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌱 Fertilizer Recommendation")

    col1, col2 = st.columns(2)

    with col1:
        fert_temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="fert_temp")
        fert_hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="fert_hum")
        fert_moist = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0, key="fert_moist")

    with col2:
        fert_crop = st.text_input("Crop Type", "Rice", key="fert_crop")
        fert_soil = st.text_input("Soil Type", "Loamy", key="fert_soil")

    if st.button("🔍 Recommend Fertilizer", key="fert_btn"):
        payload = {
            "temp": fert_temp,
            "hum": fert_hum,
            "moist": fert_moist,
            "crop": fert_crop,
            "soil": fert_soil
        }
        res = requests.post(f"{BACKEND_URL}/fertilizerrecommend", json=payload)

        if res.status_code == 200:
            st.markdown(
                f"<p class='result'>Recommended Fertilizer: {res.json()['fertilizer']}</p>",
                unsafe_allow_html=True
            )
        else:
            st.error("Backend error")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🌾 CROP RECOMMENDATION
# =====================================================
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌾 Crop Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        crop_n = st.number_input("Nitrogen (N)", 0.0, 200.0, 50.0, key="crop_n")
        crop_p = st.number_input("Phosphorus (P)", 0.0, 200.0, 40.0, key="crop_p")
        crop_k = st.number_input("Potassium (K)", 0.0, 200.0, 40.0, key="crop_k")

    with col2:
        crop_temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="crop_temp")
        crop_hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="crop_hum")

    with col3:
        crop_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="crop_ph")
        crop_rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 120.0, key="crop_rain")

    if st.button("🌱 Recommend Crop", key="crop_btn"):
        payload = {
            "n": crop_n,
            "p": crop_p,
            "k": crop_k,
            "temp": crop_temp,
            "hum": crop_hum,
            "ph": crop_ph,
            "rainfall": crop_rain
        }
        res = requests.post(f"{BACKEND_URL}/croprecommend", json=payload)

        if res.status_code == 200:
            st.markdown(
                f"<p class='result'>Best Crop to Grow: {res.json()['crop to grow']}</p>",
                unsafe_allow_html=True
            )
        else:
            st.error("Backend error")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 💧 IRRIGATION & PH CORRECTION
# =====================================================
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💧 Irrigation & pH Correction")

    col1, col2 = st.columns(2)

    with col1:
        irr_temp = st.number_input("Temperature (°C)", 0.0, 60.0, 25.0, key="irr_temp")
        irr_hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, key="irr_hum")
        irr_moist = st.number_input("Soil Moisture (%)", 0.0, 100.0, 40.0, key="irr_moist")

    with col2:
        irr_ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="irr_ph")
        irr_rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0, key="irr_rain")

    if st.button("💡 Get Recommendation", key="irr_btn"):
        payload = {
            "temp": irr_temp,
            "hum": irr_hum,
            "ph": irr_ph,
            "rainfall": irr_rain,
            "moist": irr_moist
        }
        res = requests.post(f"{BACKEND_URL}/irrigationandphcorrection", json=payload)

        if res.status_code == 200:
            data = res.json()
            st.markdown(
                f"""
                <p class='result'>Irrigation: {data['irrcorrection']}</p>
                <p class='result'>pH Correction: {data['phcorrection']}</p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.error("Backend error")

    st.markdown('</div>', unsafe_allow_html=True)

# =====================================================
# 🎥 SMART FARMING VISION SYSTEM
# =====================================================
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🎥 Smart Farming Vision System")
    st.write("Motion-triggered YOLO detection with sound alert")

    start = st.button("▶️ Start Camera", key="start_cam")
    stop = st.button("⛔ Stop Camera", key="stop_cam")

    FRAME_WINDOW = st.image([])

    if "run_cam" not in st.session_state:
        st.session_state.run_cam = False

    if start:
        st.session_state.run_cam = True

    if stop:
        st.session_state.run_cam = False

    if st.session_state.run_cam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            st.error("Camera not found")
        else:
            model = YOLO("../models/best.pt")

            ret, prev_frame = cap.read()
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

            last_sound_time = 0
            sound_cooldown = 5

            while st.session_state.run_cam:
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (21, 21), 0)

                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

                if np.sum(thresh) > 50000:
                    results = model.predict(frame, verbose=False)

                    label = "Unknown"
                    conf = 0.0

                    if results and len(results[0].boxes.cls) > 0:
                        boxes = results[0].boxes
                        idx = boxes.conf.argmax().item()
                        label = model.names[int(boxes.cls[idx])]
                        conf = float(boxes.conf[idx])

                        if time.time() - last_sound_time > sound_cooldown:
                            winsound.PlaySound("../media/alert_tone.wav", winsound.SND_ASYNC)
                            last_sound_time = time.time()

                    cv2.putText(
                        frame,
                        f"DISTURBANCE: {label.upper()} ({conf:.2f})",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                    st.markdown(
                        f"<p class='alert'>⚠️ Disturbance Detected: {label}</p>",
                        unsafe_allow_html=True
                    )

                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                prev_gray = gray

            cap.release()

    st.markdown('</div>', unsafe_allow_html=True)
