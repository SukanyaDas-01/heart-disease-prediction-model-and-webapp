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

# Add the project root to sys.path
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

    if proba < 30:
        recommendation = "Maintain a healthy lifestyle."
    elif proba < 70:
        recommendation = "Schedule regular medical check-ups."
    else:
        recommendation = "Consult a cardiologist immediately."

    content = [
        Paragraph("Heart Disease Risk Report", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Risk Probability: {proba:.1f}%", styles["Normal"]),
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

# -------------------- Encoding Maps --------------------
cp = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
restecg = {"Normal": 0, "ST-T Abnormality": 1, "LVH": 2}
slope = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
thal = {"Fixed Defect": 1, "Reversible Defect": 2, "Normal": 3}

# -------------------- Inputs --------------------
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 20, 90, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp_val = st.selectbox("Chest Pain", cp.keys())

with col2:
    trestbps = st.slider("Resting BP", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 500, 200)
    fbs = st.radio("Fasting Sugar > 120", ["No", "Yes"])

with col3:
    thalach = st.slider("Max HR", 60, 220, 150)
    exang = st.radio("Exercise Angina", ["No", "Yes"])
    restecg_val = st.selectbox("Rest ECG", restecg.keys())

oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
slope_val = st.selectbox("Slope", slope.keys())
ca = st.slider("Major Vessels", 0, 3, 0)
thal_val = st.selectbox("Thalassemia", thal.keys())

# -------------------- Predict --------------------
if st.button("🔍 Analyze Risk", use_container_width=True):

    input_dict = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": cp[cp_val],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs == "Yes" else 0,
        "restecg": restecg[restecg_val],
        "thalach": thalach,
        "exang": 1 if exang == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": slope[slope_val],
        "ca": ca,
        "thal": thal[thal_val],
    }

    encoded = encode_inputs(input_dict)
    prediction, proba = predict_risk(pipeline, encoded)

    st.metric("Risk Probability", f"{proba:.1f}%")
    st.metric("Prediction", "High Risk" if prediction else "Low Risk")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba,
        gauge={'axis': {'range': [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)

    pdf = create_pdf(prediction, proba, input_dict)
    st.download_button("📥 Download Report", pdf, "heart_risk_report.pdf")
