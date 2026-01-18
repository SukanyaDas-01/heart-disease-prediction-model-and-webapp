import streamlit as st
import joblib
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
from io import BytesIO

from src.preprocessing import encode_inputs
from src.predict import predict_risk

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# -------------------- Config --------------------
MODEL_PATH = "models/pipeline.pkl"

st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="❤",
    layout="wide"
)

# -------------------- Load Model --------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# -------------------- Lottie Loader --------------------
@st.cache_resource
def load_lottie(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# -------------------- PDF Generator --------------------
def create_pdf(prediction, proba, patient_data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    if proba < 0.3:
        recommendation = "Maintain a healthy lifestyle."
    elif proba < 0.7:
        recommendation = "Schedule regular medical check-ups."
    else:
        recommendation = "Consult a cardiologist immediately."

    content = [
        Paragraph("Heart Disease Risk Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Risk Probability: {proba*100:.1f}%", styles["Normal"]),
        Paragraph(f"Prediction: {'High Risk' if prediction else 'Low Risk'}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Recommendation:", styles["Heading2"]),
        Paragraph(recommendation, styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Patient Summary:", styles["Heading2"]),
    ]

    for k, v in patient_data.items():
        content.append(Paragraph(f"{k}: {v}", styles["Normal"]))

    doc.build(content)
    buffer.seek(0)
    return buffer

# -------------------- Sidebar --------------------
st.sidebar.title("📌 About")
st.sidebar.info(
    "ML-based **Heart Disease Risk Predictor**\n\n"
    "⚠ Educational purpose only."
)

# -------------------- Header --------------------
lottie = load_lottie("https://assets9.lottiefiles.com/packages/lf20_qp1q7mct.json")
if lottie:
    st_lottie(lottie, height=140)

st.markdown("<h1 style='text-align:center;color:#FF4B4B;'>❤ Heart Disease Risk Prediction</h1>", unsafe_allow_html=True)

# -------------------- Input Maps (Strings for preprocessing) --------------------
cp_map = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"]
restecg_map = ["Normal", "ST-T wave abnormality", "Left Ventricular Hypertrophy"]
slope_map = ["Upsloping", "Flat", "Downsloping"]
thal_map = ["Fixed Defect", "Normal", "Reversible Defect"]

# -------------------- High-Risk Test --------------------
if "high_risk" not in st.session_state:
    st.session_state.high_risk = False

def load_high_risk_profile():
    st.session_state.high_risk = True

st.button("🔥 High-Risk Test", on_click=load_high_risk_profile)

# -------------------- Inputs --------------------
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 20, 90, 65 if st.session_state.high_risk else 50)
    sex = st.radio("Sex", ["Male", "Female"], index=0 if st.session_state.high_risk else 1)
    cp_val = st.selectbox("Chest Pain", cp_map,
                          index=cp_map.index("Asymptomatic") if st.session_state.high_risk else 0)

with col2:
    trestbps = st.slider("Resting BP", 80, 200, 170 if st.session_state.high_risk else 120)
    chol = st.slider("Cholesterol", 100, 500, 350 if st.session_state.high_risk else 200)
    fbs = st.radio("Fasting Sugar > 120", ["No", "Yes"], index=1 if st.session_state.high_risk else 0)

with col3:
    thalach = st.slider("Max HR", 60, 220, 100 if st.session_state.high_risk else 150)
    exang = st.radio("Exercise Angina", ["No", "Yes"], index=1 if st.session_state.high_risk else 0)
    restecg_val = st.selectbox("Rest ECG", restecg_map,
                               index=restecg_map.index("ST-T wave abnormality") if st.session_state.high_risk else 0)

oldpeak = st.slider("ST Depression", 0.0, 6.0, 4.0 if st.session_state.high_risk else 1.0)
slope_val = st.selectbox("Slope", slope_map, index=slope_map.index("Flat") if st.session_state.high_risk else 0)
ca = st.slider("Major Vessels", 0, 3, 2 if st.session_state.high_risk else 0)
thal_val = st.selectbox("Thalassemia", thal_map, index=thal_map.index("Fixed Defect") if st.session_state.high_risk else 0)

# -------------------- Prediction --------------------
if st.button("🔍 Analyze Risk"):

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp_val,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": "Yes" if fbs == "Yes" else "No",
        "restecg": restecg_val,
        "thalach": thalach,
        "exang": "Yes" if exang == "Yes" else "No",
        "oldpeak": oldpeak,
        "slope": slope_val,
        "ca": ca,
        "thal": thal_val
    }

    # Encode inputs properly
    encoded = encode_inputs(input_dict)
    prediction, proba = predict_risk(pipeline, encoded)

    # Show metrics
    st.metric("Risk Probability", f"{proba*100:.1f}%")
    st.metric("Prediction", "High Risk" if prediction else "Low Risk")

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba*100,
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # PDF report
    pdf = create_pdf(prediction, proba, input_dict)
    st.download_button("📥 Download Report", pdf, "heart_risk_report.pdf")
