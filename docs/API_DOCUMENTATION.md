# API Documentation

## Hospital Readmission Prediction REST API

Base URL: `http://localhost:5000`

---

## Endpoints

### 1. Root Endpoint

**GET /**

Returns API information and available endpoints.

**Response:**
```json
{
  "service": "Hospital Readmission Prediction API",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "GET /": "API information",
    "GET /health": "Health check",
    "GET /model_info": "Model metadata",
    "POST /predict": "Single patient prediction",
    "POST /batch_predict": "Batch predictions"
  }
}
```

---

### 2. Health Check

**GET /health**

Check API and model health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Service is running",
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Service unavailable (model not loaded)

---

### 3. Model Information

**GET /model_info**

Get model metadata and performance metrics.

**Response:**
```json
{
  "model_name": "random_forest",
  "model_type": "random_forest",
  "n_features": 63,
  "classes": ["NO", "<30", ">30"],
  "metrics": {
    "accuracy": 0.5321,
    "f1_macro": 0.4428,
    "f1_weighted": 0.5308,
    "roc_auc_ovr": 0.6523,
    "precision_macro": 0.4427,
    "recall_macro": 0.4441
  }
}
```

---

### 4. Single Prediction

**POST /predict**

Make readmission risk prediction for a single patient.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "features": {
    "age_numeric": 65,
    "gender_male": 1,
    "gender_female": 0,
    "time_in_hospital": 7,
    "number_inpatient": 2,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_diagnoses": 8,
    "num_medications": 15,
    "num_procedures": 2,
    "num_lab_procedures": 50,
    "has_prev_inpatient": 1,
    "has_prev_emergency": 1,
    "emergency_admission": 1,
    "discharged_to_home": 1,
    "diabetes_med_count": 2,
    "medication_changed": 1
    // ... additional features
  }
}
```

**Response:**
```json
{
  "predicted_class": "<30",
  "predicted_class_id": 1,
  "risk_score": 85.3,
  "probabilities": {
    "NO": 0.147,
    "<30": 0.853,
    ">30": 0.000
  },
  "confidence": 0.853,
  "top_risk_factors": [
    {
      "feature": "Previous inpatient visits",
      "value": 2.0,
      "impact": "High"
    },
    {
      "feature": "Length of stay (days)",
      "value": 7.0,
      "impact": "High"
    }
  ],
  "recommendation": "HIGH RISK: Schedule early follow-up within 7 days. Consider discharge planning and medication reconciliation.",
  "timestamp": "2024-01-15T10:30:00.123456",
  "model": "random_forest"
}
```

**Status Codes:**
- `200`: Successful prediction
- `400`: Invalid request format
- `500`: Internal server error
- `503`: Service unavailable

---

### 5. Batch Prediction

**POST /batch_predict**

Make predictions for multiple patients in a single request.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
  "patients": [
    {
      "features": {
        "age_numeric": 65,
        "gender_male": 1,
        "time_in_hospital": 7,
        // ... all features
      }
    },
    {
      "features": {
        "age_numeric": 50,
        "gender_female": 1,
        "time_in_hospital": 3,
        // ... all features
      }
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_class": "<30",
      "risk_score": 85.3,
      "probabilities": {...},
      "top_risk_factors": [...]
    },
    {
      "predicted_class": "NO",
      "risk_score": 25.1,
      "probabilities": {...},
      "top_risk_factors": [...]
    }
  ],
  "count": 2,
  "timestamp": "2024-01-15T10:30:00.123456",
  "model": "random_forest"
}
```

---

## Feature Requirements

### Required Features (63 total)

The model expects all 63 engineered features. Missing features will be automatically set to 0.

**Demographic Features:**
- `age_numeric` (float): Age in years
- `gender_male` (0/1): Binary indicator
- `gender_female` (0/1): Binary indicator
- `race_*` (0/1): One-hot encoded race categories

**Clinical Features:**
- `time_in_hospital` (int): Days in hospital
- `number_diagnoses` (int): Count of diagnoses
- `number_inpatient` (int): Previous inpatient visits
- `number_outpatient` (int): Previous outpatient visits
- `number_emergency` (int): Previous emergency visits
- `has_diabetes_complication` (0/1): Binary
- `has_circulatory` (0/1): Binary
- `has_respiratory` (0/1): Binary

**Medication Features:**
- `num_medications` (int): Total medication count
- `diabetes_med_count` (int): Diabetes medication count
- `medication_changed` (0/1): Binary indicator
- `insulin_prescribed` (0/1): Binary indicator

**Lab & Procedure Features:**
- `num_lab_procedures` (int): Number of lab tests
- `num_procedures` (int): Number of procedures
- `glucose_high` (0/1): Binary indicator
- `a1c_high` (0/1): Binary indicator

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid request format",
  "expected": {
    "features": {...}
  }
}
```

### 500 Internal Server Error
```json
{
  "error": "Error message describing the issue"
}
```

### 503 Service Unavailable
```json
{
  "error": "Prediction service not available"
}
```

---

## Usage Examples

### cURL Examples

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Model Info:**
```bash
curl http://localhost:5000/model_info
```

**Single Prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "age_numeric": 65,
      "gender_male": 1,
      "gender_female": 0,
      "time_in_hospital": 7,
      "number_inpatient": 2,
      "number_emergency": 1,
      "number_diagnoses": 8,
      "num_medications": 15
    }
  }'
```

### Python Example

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Patient data
patient_data = {
    "features": {
        "age_numeric": 65,
        "gender_male": 1,
        "time_in_hospital": 7,
        "number_inpatient": 2,
        "number_emergency": 1,
        "number_diagnoses": 8,
        "num_medications": 15,
        # ... add all other features
    }
}

# Make request
response = requests.post(url, json=patient_data)

# Parse response
if response.status_code == 200:
    result = response.json()
    print(f"Predicted Risk: {result['predicted_class']}")
    print(f"Risk Score: {result['risk_score']:.1f}%")
    print(f"Recommendation: {result['recommendation']}")
else:
    print(f"Error: {response.json()['error']}")
```

### JavaScript Example

```javascript
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    features: {
      age_numeric: 65,
      gender_male: 1,
      time_in_hospital: 7,
      number_inpatient: 2,
      number_emergency: 1,
      number_diagnoses: 8,
      num_medications: 15,
      // ... add all other features
    }
  })
})
.then(response => response.json())
.then(data => {
  console.log('Predicted Risk:', data.predicted_class);
  console.log('Risk Score:', data.risk_score);
  console.log('Recommendation:', data.recommendation);
})
.catch(error => console.error('Error:', error));
```

---

## Rate Limiting

Currently, there are no rate limits implemented. For production deployment, consider implementing:
- Rate limiting per IP address
- Authentication tokens
- API quotas per user/organization

---

## Versioning

API Version: `1.0.0`

Future versions will maintain backward compatibility or be versioned in the URL path (e.g., `/v2/predict`).

---

## Support

For issues, questions, or feature requests:
- Email: support@example.com
- GitHub Issues: [repository/issues]
- Documentation: [docs.example.com]

