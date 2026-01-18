# src/predict.py

from typing import Tuple
import numpy as np

def predict_risk(
    pipeline,
    encoded_input: np.ndarray,
    threshold: float = 0.5
) -> Tuple[int, float]:
    """
    Predicts heart disease risk.

    Parameters:
    - pipeline: Trained sklearn pipeline
    - encoded_input: Preprocessed input array (shape: [1, n_features])
    - threshold: Probability threshold for positive class

    Returns:
    - prediction (0 or 1)
    - probability of heart disease
    """

    if encoded_input.ndim != 2:
        raise ValueError("Encoded input must be a 2D array.")

    # Predict probability of positive class
    probability = pipeline.predict_proba(encoded_input)[0][1]

    # Apply threshold explicitly
    prediction = int(probability >= threshold)

    return prediction, round(float(probability), 4)
