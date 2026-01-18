# src/preprocessing.py

import numpy as np

# -----------------------------
# Encoding dictionaries
# -----------------------------
CP_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

RESTECG_MAP = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

SLOPE_MAP = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

THAL_MAP = {
    "Fixed Defect": 1,
    "Normal": 2,
    "Reversible Defect": 3
}

# -----------------------------
# Model Feature Order (CRITICAL)
# -----------------------------
FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# -----------------------------
# Input Encoding Function
# -----------------------------
def encode_inputs(inputs: dict) -> np.ndarray:
    """
    Converts raw user inputs into model-ready numpy array.
    Ensures medical validity and feature order consistency.
    """

    # Handle thal missing / invalid values defensively
    thal_value = THAL_MAP.get(inputs.get("thal"))
    if thal_value is None:
        raise ValueError("Invalid Thal value provided.")

    encoded = [
        int(inputs["age"]),
        1 if inputs["sex"] == "Male" else 0,
        CP_MAP.get(inputs["cp"]),
        int(inputs["trestbps"]),
        int(inputs["chol"]),
        1 if inputs["fbs"] == "Yes" else 0,
        RESTECG_MAP.get(inputs["restecg"]),
        int(inputs["thalach"]),
        1 if inputs["exang"] == "Yes" else 0,
        float(inputs["oldpeak"]),
        SLOPE_MAP.get(inputs["slope"]),
        int(inputs["ca"]),
        thal_value
    ]

    # Safety check
    if any(v is None for v in encoded):
        raise ValueError("One or more input features could not be encoded.")

    return np.array([encoded], dtype=float)
