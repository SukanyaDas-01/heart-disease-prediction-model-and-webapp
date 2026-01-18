# **🫀 Heart Disease Risk Prediction System**

### End-to-End Machine Learning Pipeline with Deployment

A production-ready machine learning project that predicts heart disease risk using clinical parameters, built with proper preprocessing, model encapsulation, evaluation, and Streamlit deployment.

## 📌 Project Overview

This project implements an end-to-end ML workflow for heart disease risk prediction, covering:

- Exploratory Data Analysis (EDA)

- Model experimentation & comparison

- Reproducible training using pipelines

- Probability-based prediction

- Deployment using Streamlit

- PDF report generation for predictions

The goal is not just accuracy, but engineering correctness and deployability.


## 🧠 Machine Learning Approach

- **Problem Type**: Binary Classification

- **Target**: Presence of heart disease (0 / 1)

- **Dataset**: Cleveland Heart Disease Dataset (UCI ML Repository)

- **Features**: 13 clinical attributes


## 🏗️ ML Pipeline Architecture

**🔁 End-to-End Flow**
┌────────────┐
│   Dataset  │
│ heart.csv  │
└─────┬──────┘
      ↓
┌───────────────┐
│ Preprocessing │
│ (Encoding +   │
│ Feature Order)│
└─────┬─────────┘
      ↓
┌───────────────┐
│ StandardScaler│
└─────┬─────────┘
      ↓
┌───────────────┐
│ ML Model      │
│ (RF / LR /    │
│ XGBoost etc.) │
└─────┬─────────┘
      ↓
┌───────────────┐
│ Evaluation    │
│ (Recall, F1,  │
│ ROC-AUC)      │
└─────┬─────────┘
      ↓
┌───────────────┐
│ Saved Pipeline│
│ pipeline.pkl  │
└─────┬─────────┘
      ↓
┌───────────────┐
│ Streamlit App │
│ (Prediction + │
│ Visualization)│
└───────────────┘

✅ Scaling + model are encapsulated inside a single `Pipeline`, ensuring safe and consistent inference.


## 📁 Project Structure

heart-disease-prediction/
│
├── app/
│   └── app.py                  # Streamlit UI
│
├── src/
│   ├── preprocessing.py        # Input encoding & feature order
│   ├── train.py                # Reproducible training pipeline
│   ├── predict.py              # Inference logic
│   └── evaluate.py             # Metrics & evaluation
│
├── data/
│   └── heart.csv               # Dataset
│
├── models/
│   └── pipeline.pkl            # Trained ML pipeline
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb  # Analysis & experiments
│
├── reports/
│   └── model_comparison.xlsx   # Model performance results
│
├── requirements.txt
├── README.md
└── .gitignore


## 📊 Model Experimentation

Multiple models were evaluated using a consistent pipeline:

- Logistic Regression
- Naive Bayes
- SVM
- KNN
- Decision Tree
- Random Forest
- XGBoost
- Neural Network
- Voting Classifier (ensemble)


## Evaluation Metrics

- Recall
- F1 Score
- ROC-AUC

📁 Results saved to:
`reports/model_comparison.xlsx`


## 🚀 Training the Model (Reproducible)

`python src/train.py`

This will:
- Load data
- Apply preprocessing + scaling
- Train the model
- Evaluate performance
- Save the entire pipeline to `models/pipeline.pkl`


## 🌐 Running the Web App

`streamlit run app/app.py`

Features:
- Probability-based risk prediction
- Gauge & comparison charts
- PDF report generation
- Medical disclaimer
- Clean UI with Streamlit

## 📄 PDF Report Output

Each prediction includes:
- Risk probability (%)
- Risk category (Low / Medium / High)
- Patient input summary
- Actionable recommendation


## ⚠️ Medical Disclaimer

This project is strictly for educational and research purposes.
- Not a medical diagnosis tool
- Always consult healthcare professionals
- Predictions are based on historical datasets


## 👩‍💻 Author

**Sukanya Das**
🎓 B.Tech CSE (2022–2026)
📧 Email: sukusukanyadas2001@gmail.com
💼 LinkedIn: linkedin.com/in/sukanya-das-a05935244
🐙 GitHub: github.com/SukanyaDas-01