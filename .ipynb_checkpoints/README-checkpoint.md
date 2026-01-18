# 🫀 Heart Disease Risk Predictor

> AI-powered cardiovascular risk assessment tool with interactive web interface

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)](https://scikit-learn.org/)


A machine learning web application that predicts heart disease risk using clinical parameters. Built with Streamlit and featuring real-time predictions, interactive visualizations, PDF report generation, and custom model training capabilities.

![Heart Disease Predictor Demo](screenshots/demo.gif)

## ✨ Features

### 🔮 **Smart Prediction Engine**
- **Real-time risk assessment** with 85%+ accuracy
- **Multiple ML algorithms** (Random Forest, Logistic Regression, SVM)
- **Proper data scaling** for accurate predictions
- **Risk categorization** (Low/Medium/High)

### 🎨 **Interactive User Interface**
- **Beautiful Streamlit UI** with animations and dark theme
- **Lottie animations** for enhanced user experience
- **Responsive design** that works on all devices
- **Intuitive input forms** with helpful tooltips

### 📊 **Advanced Visualizations**
- **Gauge charts** for risk probability display
- **Comparison charts** (Patient vs Normal values)
- **Feature importance plots** for model interpretability
- **Progress indicators** and color-coded risk levels

### 📋 **Professional Reports**
- **PDF report generation** with patient summary
- **Download functionality** for medical records
- **Comprehensive risk analysis** with recommendations
- **Medical disclaimers** and professional formatting

### 🧪 **Testing & Validation**
- **Quick test profiles** (Low/Medium/High risk scenarios)
- **Debug mode** to inspect model inputs
- **Model performance metrics** and confusion matrices
- **Cross-validation support**

### 📈 **Custom Model Training**
- **CSV file upload** for custom datasets
- **Real-time model training** with progress tracking
- **Multiple algorithm selection**
- **Model comparison** and performance analysis
- **Save/download trained models**

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open in browser**
```
Local URL: http://localhost:8501
Network URL: http://192.168.0.102:8501
```

## 📁 Project Structure

```
heart-disease-predictor/
│
├── app.py                      # Main Streamlit application
├── best_model.pkl              # Pre-trained ML model
├── heart_disease.ipynb         # Model training notebook
├── heart.csv                   # Cleveland Heart Disease Dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── screenshots/                # App screenshots and demo
│   ├── demo.gif
│   ├── prediction-results.png
│   └── custom-training.png
│
└── models/                     # Additional trained models
    ├── custom_model.pkl
    └── scaler.pkl
```

## 🧠 Machine Learning Model

### Dataset
- **Source**: Cleveland Heart Disease Dataset (UCI ML Repository)
- **Samples**: 303 patients
- **Features**: 13 clinical parameters
- **Target**: Binary classification (0: No disease, 1: Disease)

### Features Used
| Feature | Description | Range |
|---------|-------------|--------|
| Age | Patient age in years | 29-77 |
| Sex | Gender (0: Female, 1: Male) | 0-1 |
| CP | Chest pain type | 0-3 |
| Trestbps | Resting blood pressure (mm Hg) | 94-200 |
| Chol | Serum cholesterol (mg/dl) | 126-564 |
| FBS | Fasting blood sugar > 120 mg/dl | 0-1 |
| RestECG | Resting ECG results | 0-2 |
| Thalach | Maximum heart rate achieved | 71-202 |
| Exang | Exercise induced angina | 0-1 |
| Oldpeak | ST depression induced by exercise | 0-6.2 |
| Slope | Slope of peak exercise ST segment | 0-2 |
| CA | Number of major vessels (0-4) | 0-4 |
| Thal | Thalassemia test result | 1-3 |

### Model Performance
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 85.2%
- **Precision**: 84.7%
- **Recall**: 86.1%
- **F1-Score**: 85.4%

## 🎯 Usage Examples

### Basic Prediction
1. Enter patient details in the input form
2. Click "🔍 Analyze Heart Disease Risk"
3. View risk probability and recommendations
4. Download PDF report if needed

### Quick Testing
Use the sidebar test buttons for instant validation:
- **🟢 Low Risk**: Young, healthy patient (~15% risk)
- **🟡 Medium Risk**: Middle-aged with moderate factors (~50% risk)
- **🔴 High Risk**: Older patient with multiple risk factors (~80% risk)

### Custom Model Training
1. Navigate to "📊 Train New Model" tab
2. Upload your CSV file with heart disease data
3. Configure training parameters
4. Train and evaluate the model
5. Use the new model for predictions

### Sample High-Risk Profile
```
Age: 65, Male
Chest Pain: Asymptomatic
BP: 170 mmHg, Cholesterol: 350 mg/dl
Fasting Sugar: Yes, Max HR: 100
Exercise Angina: Yes, ST Depression: 4.0
Expected Risk: ~75-85%
```

## 🛠️ Development

### Adding New Features
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Model Improvements
- Add new algorithms (XGBoost, Neural Networks)
- Implement hyperparameter tuning
- Add cross-validation
- Include feature selection

### UI Enhancements
- Add more visualization options
- Implement patient history tracking
- Add multi-language support
- Include mobile responsiveness improvements

## 📊 Screenshots

### Main Interface
![Main Interface](screenshots/main-interface.png)

### Prediction Results
![Prediction Results](screenshots/prediction-results.png)

### Custom Training
![Custom Training](screenshots/custom-training.png)

### PDF Report
![PDF Report](screenshots/pdf-report.png)

## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is designed for **educational and research purposes only**. It should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. 

- Always consult qualified healthcare professionals for medical decisions
- This tool provides risk estimates based on limited clinical parameters
- Individual patient cases may require additional diagnostic procedures
- The predictions are based on historical data and may not reflect individual variations

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- 🐛 **Bug fixes** and performance improvements
- ✨ **New features** and enhancements
- 📚 **Documentation** improvements
- 🧪 **Testing** and validation
- 🎨 **UI/UX** improvements
- 🧠 **Model** enhancements

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sukanya Das**
- 📧 Email: [sukusukanyadas2001@gmail.com](mailto:sukusukanyadas2001@gmail.com)
- 💼 LinkedIn: [sukanya-das-a05935244](https://www.linkedin.com/in/sukanya-das-a05935244/)
- 🐙 GitHub: SukanyaDas-01(https://github.com/SukanyaDas-01)

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Cleveland Heart Disease Dataset
- **Streamlit** team for the amazing web app framework
- **Scikit-learn** contributors for machine learning tools
- **Plotly** for interactive visualizations
- **Open source community** for continuous inspiration

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/SukanyaDas-01/heart-disease-predictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/SukanyaDas-01/heart-disease-predictor?style=social)
![GitHub issues](https://img.shields.io/github/issues/SukanyaDas-01/heart-disease-predictor)
![GitHub pull requests](https://img.shields.io/github/issues-pr/SukanyaDas-01/heart-disease-predictor)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ by [Sukanya Das](https://github.com/YOUR_USERNAME)

</div>
