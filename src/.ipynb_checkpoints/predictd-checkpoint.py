# src/predict.py

def predict_risk(pipeline, encoded_input):
    """
    Returns prediction class and probability
    """
    probability = pipeline.predict_proba(encoded_input)[0][1]
    prediction = pipeline.predict(encoded_input)[0]
    return prediction, probability
