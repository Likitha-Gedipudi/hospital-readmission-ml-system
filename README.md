# 🏥 Hospital Readmission Risk Prediction System

A production-grade end-to-end machine learning system for predicting 30-day hospital readmission risk using the Diabetes 130-US Hospitals dataset.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Impact](#business-impact)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [API Documentation](#api-documentation)
- [Dashboard](#dashboard)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🎯 Project Overview

This system predicts hospital readmission risk within 30 days using machine learning, helping healthcare providers identify high-risk patients and intervene proactively to prevent readmissions.

### Problem Statement

Hospital readmissions are costly and often preventable. This system:
- Predicts readmission risk at discharge (3 categories: NO, <30 days, >30 days)
- Identifies key risk factors for each patient
- Provides actionable recommendations for care teams
- Estimates potential cost savings from readmission prevention

### Dataset

- **Source**: Diabetes 130-US Hospitals (1999-2008)
- **Size**: 100,104 patient encounters (after cleaning)
- **Features**: 63 engineered features including demographics, clinical history, medications, and lab results
- **Target Classes**: 
  - NO readmission: 53.2%
  - Readmission <30 days: 11.3%
  - Readmission >30 days: 35.5%

---

## 💰 Business Impact

### Key Metrics

- **Predicted Readmissions**: 6,688 patients at risk
- **Preventable (25% intervention rate)**: 1,672 readmissions
- **Est. Cost Savings**: **$20 Million annually**
- **Model Accuracy**: 53.2%
- **F1-Score (Macro)**: 0.443
- **ROC-AUC**: 0.652

### Clinical Value

1. **Early Risk Identification**: Flag high-risk patients at discharge
2. **Resource Optimization**: Target interventions to high-risk patients
3. **Care Coordination**: Facilitate follow-up scheduling and medication reconciliation
4. **Quality Metrics**: Reduce 30-day readmission rates (CMS penalty avoidance)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
│  Raw Data → Preprocessing → Feature Engineering → Splits    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Model Training                           │
│  Logistic │ Random Forest │ XGBoost │ LightGBM              │
│            (Best: Random Forest - F1: 0.443)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Model Deployment                           │
│  Flask REST API │ Docker Container │ MLflow Tracking        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              User Interfaces                                 │
│  Plotly Dash Dashboard │ REST API Endpoints                 │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 1. Data Pipeline
- ✅ Automated preprocessing and cleaning
- ✅ 63 engineered features (demographic, clinical, utilization, medication, lab)
- ✅ Stratified train/val/test splits (70/15/15)

### 2. Machine Learning Models
- ✅ Multi-class classification (3 classes)
- ✅ Multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM
- ✅ Class imbalance handling with balanced weights
- ✅ Comprehensive evaluation metrics

### 3. Model Interpretability
- ✅ SHAP analysis for feature importance
- ✅ Individual prediction explanations
- ✅ Top risk factors identification

### 4. Interactive Dashboard
- ✅ Real-time KPI metrics
- ✅ Interactive patient filtering
- ✅ Risk distribution visualizations
- ✅ Confusion matrix and ROC curves
- ✅ Sortable patient risk table

### 5. REST API
- ✅ Single and batch predictions
- ✅ Model metadata endpoint
- ✅ Health check endpoint
- ✅ CORS enabled for web integration

### 6. MLOps & Deployment
- ✅ Docker containerization
- ✅ Docker Compose orchestration
- ✅ MLflow experiment tracking
- ✅ Data drift detection
- ✅ Performance monitoring

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd hospital-readmission-system

# Install dependencies
pip install -r requirements.txt

# Run Phase 1: Data Preparation
python3 run_phase1.py

# Run Phase 2: Model Training
python3 run_phase2.py

# Run Phase 3: Dashboard Validation
python3 run_phase3.py

# Run Phase 4: API Validation
python3 run_phase4.py
```

---

## 📊 Usage

### 1. Train Models

```bash
python3 run_phase2.py
```

This will:
- Train 4 models (Logistic, Random Forest, XGBoost, LightGBM)
- Evaluate on test set
- Save best model to `models/production/`
- Generate evaluation plots

### 2. Launch Dashboard

```bash
python3 run_dashboard.py
```

Access at: http://127.0.0.1:8050

### 3. Start API Server

```bash
python3 -m api.app
```

Access at: http://127.0.0.1:5000

### 4. Make Predictions

**Single Prediction:**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age_numeric": 65,
      "gender_male": 1,
      "time_in_hospital": 7,
      "number_inpatient": 2,
      "number_emergency": 1,
      "number_diagnoses": 8,
      "num_medications": 15
    }
  }'
```

**Response:**

```json
{
  "predicted_class": "<30",
  "risk_score": 85.3,
  "probabilities": {
    "NO": 0.147,
    "<30": 0.853,
    ">30": 0.000
  },
  "confidence": 0.853,
  "recommendation": "HIGH RISK: Schedule early follow-up within 7 days...",
  "top_risk_factors": [
    {"feature": "Previous inpatient visits", "value": 2.0, "impact": "High"},
    {"feature": "Length of stay (days)", "value": 7.0, "impact": "High"}
  ]
}
```

---

## 📈 Model Performance

### Best Model: Random Forest

| Metric | Value |
|--------|-------|
| **Accuracy** | 53.21% |
| **F1-Score (Macro)** | 0.443 |
| **F1-Score (Weighted)** | 0.531 |
| **ROC-AUC (OvR)** | 0.652 |
| **Precision (Macro)** | 0.443 |
| **Recall (Macro)** | 0.444 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NO | 0.57 | 0.67 | 0.61 | 7,981 |
| <30 | 0.19 | 0.05 | 0.08 | 1,703 |
| >30 | 0.57 | 0.63 | 0.60 | 5,332 |

### Top Predictive Features (SHAP Analysis)

1. **number_inpatient** - Previous inpatient admissions
2. **time_in_hospital** - Length of current stay
3. **number_diagnoses** - Complexity of patient condition
4. **age_numeric** - Patient age
5. **num_medications** - Medication count
6. **number_emergency** - Previous emergency visits
7. **diabetes_med_count** - Diabetes medications
8. **discharged_to_home** - Discharge destination

---

## 📡 API Documentation

### Endpoints

#### `GET /`
API information and available endpoints

#### `GET /health`
Health check for monitoring

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### `GET /model_info`
Model metadata and performance metrics

#### `POST /predict`
Single patient prediction

**Request Body:**
```json
{
  "features": {
    "age_numeric": 65,
    "gender_male": 1,
    ...
  }
}
```

#### `POST /batch_predict`
Batch predictions for multiple patients

**Request Body:**
```json
{
  "patients": [
    {"features": {...}},
    {"features": {...}}
  ]
}
```

See `docs/API_DOCUMENTATION.md` for detailed API specifications.

---

## 📊 Dashboard

The interactive Plotly Dash dashboard provides:

- **KPI Cards**: Total patients, risk distribution, model accuracy
- **Cost Impact**: Predicted readmissions and potential savings
- **Interactive Filters**: Gender, risk threshold
- **Visualizations**:
  - Risk distribution (bar & pie charts)
  - Risk score histogram
  - Confusion matrix heatmap
- **Patient Table**: Sortable, filterable, color-coded by risk level

**Launch:** `python3 run_dashboard.py`

**Access:** http://127.0.0.1:8050

---

## 🐳 Deployment

### Docker

**Build Image:**
```bash
docker build -t hospital-readmission-api .
```

**Run Container:**
```bash
docker run -p 5000:5000 hospital-readmission-api
```

### Docker Compose

**Start Services:**
```bash
docker-compose up -d
```

**Stop Services:**
```bash
docker-compose down
```

### Production Considerations

- Use production WSGI server (gunicorn/uwsgi)
- Enable HTTPS/TLS
- Implement authentication/authorization
- Set up logging and monitoring
- Configure auto-scaling
- Use managed database for predictions log

---

## 📁 Project Structure

```
hospital-readmission-system/
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Processed train/val/test splits
│   └── feature_schema.json     # Feature documentation
├── notebooks/
│   └── 01_eda.ipynb           # Exploratory data analysis
├── src/
│   ├── data_preprocessing.py   # Data cleaning
│   ├── feature_engineering.py  # Feature creation
│   ├── data_splitting.py       # Train/val/test splits
│   ├── mlflow_utils.py         # Experiment tracking
│   ├── models/
│   │   ├── train.py           # Model training
│   │   ├── evaluation.py      # Model evaluation
│   │   └── explainability.py  # SHAP analysis
│   └── monitoring/
│       └── drift_detector.py  # Drift detection
├── api/
│   ├── app.py                 # Flask REST API
│   └── inference.py           # Prediction service
├── dashboard/
│   ├── app.py                 # Plotly Dash app
│   ├── utils.py               # Dashboard utilities
│   └── assets/
│       └── styles.css         # Custom styling
├── models/
│   └── production/
│       ├── best_model.pkl     # Trained model
│       └── model_metadata.json # Model info
├── outputs/
│   ├── evaluation/            # Evaluation plots
│   └── shap/                  # SHAP explanations
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies
├── run_phase1.py              # Data pipeline script
├── run_phase2.py              # Training script
├── run_phase3.py              # Dashboard validation
├── run_phase4.py              # API validation
└── README.md                   # This file
```

---

## 🛠️ Technologies Used

### Core ML Stack
- **Python 3.9+**: Programming language
- **scikit-learn**: ML algorithms and pipelines
- **XGBoost & LightGBM**: Gradient boosting
- **imbalanced-learn**: SMOTE for class imbalance
- **pandas & numpy**: Data manipulation
- **SHAP**: Model interpretability

### Visualization & Dashboard
- **Plotly**: Interactive visualizations
- **Dash**: Web dashboard framework
- **Dash Bootstrap**: UI components
- **matplotlib & seaborn**: Static plots

### API & Deployment
- **Flask**: REST API framework
- **Flask-CORS**: Cross-origin requests
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration

### MLOps & Monitoring
- **MLflow**: Experiment tracking & model registry
- **scipy**: Statistical tests for drift detection
- **joblib**: Model serialization

---

## 🔮 Future Enhancements

### Short Term
- [ ] SHAP integration in dashboard (real-time explanations)
- [ ] Additional models: Neural Networks, Ensemble stacking
- [ ] Hyperparameter tuning with Optuna/Ray Tune
- [ ] A/B testing framework

### Medium Term
- [ ] Real-time prediction streaming (Kafka/RabbitMQ)
- [ ] Automated retraining pipeline
- [ ] Model versioning and rollback
- [ ] Alert system for drift/degradation

### Long Term
- [ ] Integration with EHR systems (HL7/FHIR)
- [ ] Multi-hospital deployment
- [ ] Fairness audit and bias mitigation
- [ ] Causal inference analysis

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio.com]
- LinkedIn: [linkedin.com/in/yourprofile]
- GitHub: [github.com/yourusername]

---

## 🙏 Acknowledgments

- Dataset: UCI Machine Learning Repository - Diabetes 130-US Hospitals
- Research: Based on best practices from healthcare ML literature
- Community: Open-source ML and healthcare analytics community

---

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through:
- Email: your.email@example.com
- LinkedIn: [Your LinkedIn Profile]
- GitHub Issues: [Repository Issues Page]

---

**Built with ❤️ for better healthcare outcomes**

