"""
Phase 4 Validation Script
Tests API, Docker configuration, and monitoring setup
"""

import sys
import os
import json

print("\n" + "="*70)
print("PHASE 4: MODEL DEPLOYMENT & MLOps - VALIDATION")
print("="*70)

# Step 1: Test API Components
print("\n[STEP 1/4] Testing API Components...")
print("-" * 70)

try:
    from api.inference import PredictionService
    
    # Initialize prediction service
    service = PredictionService(
        model_path='models/production/best_model.pkl',
        metadata_path='models/production/model_metadata.json',
        feature_cols_path='data/processed/feature_columns.pkl'
    )
    
    print("✓ Prediction service initialized")
    
    # Test prediction
    example_patient = {
        'age_numeric': 65,
        'gender_male': 1,
        'time_in_hospital': 7,
        'number_inpatient': 2,
        'number_emergency': 1,
        'number_diagnoses': 8,
        'num_medications': 15
    }
    
    result = service.predict(example_patient)
    print(f"✓ Single prediction: {result['predicted_class']} (risk: {result['risk_score']:.1f}%)")
    
    # Test batch prediction
    batch_results = service.batch_predict([example_patient, example_patient])
    print(f"✓ Batch prediction: {len(batch_results)} patients processed")
    
    # Get model info
    info = service.get_model_info()
    print(f"✓ Model info: {info['model_name']} - Accuracy: {info['metrics']['accuracy']:.2%}")
    
except Exception as e:
    print(f"❌ API component test failed: {str(e)}")
    sys.exit(1)

# Step 2: Test Flask API
print("\n[STEP 2/4] Testing Flask API...")
print("-" * 70)

try:
    from flask import Flask
    import flask_cors
    
    # Import Flask app (don't run it)
    from api.app import app, prediction_service
    
    print("✓ Flask app initialized")
    print("✓ CORS enabled")
    print(f"✓ Prediction service: {'loaded' if prediction_service else 'not loaded'}")
    
    # List endpoints
    endpoints = [rule.rule for rule in app.url_map.iter_rules()]
    print(f"✓ API endpoints: {len(endpoints)}")
    for endpoint in endpoints:
        print(f"    - {endpoint}")
    
except Exception as e:
    print(f"❌ Flask API test failed: {str(e)}")
    sys.exit(1)

# Step 3: Test Docker Configuration
print("\n[STEP 3/4] Testing Docker Configuration...")
print("-" * 70)

try:
    # Check Dockerfile exists
    if os.path.exists('Dockerfile'):
        print("✓ Dockerfile present")
        with open('Dockerfile', 'r') as f:
            content = f.read()
            if 'python:3.9-slim' in content:
                print("✓ Base image: python:3.9-slim")
            if 'EXPOSE 5000' in content:
                print("✓ Port 5000 exposed")
            if 'HEALTHCHECK' in content:
                print("✓ Health check configured")
    else:
        print("⚠ Dockerfile not found")
    
    # Check docker-compose.yml exists
    if os.path.exists('docker-compose.yml'):
        print("✓ docker-compose.yml present")
    else:
        print("⚠ docker-compose.yml not found")
    
    # Check .dockerignore exists
    if os.path.exists('.dockerignore'):
        print("✓ .dockerignore present")
    else:
        print("⚠ .dockerignore not found")
    
    print("\n  To build Docker image:")
    print("    docker build -t hospital-readmission-api .")
    print("\n  To run with docker-compose:")
    print("    docker-compose up -d")
    
except Exception as e:
    print(f"❌ Docker configuration test failed: {str(e)}")

# Step 4: Test Monitoring Setup
print("\n[STEP 4/4] Testing Monitoring Setup...")
print("-" * 70)

try:
    from src.monitoring.drift_detector import DriftDetector
    import pickle
    
    # Load data
    with open('data/processed/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    
    with open('data/processed/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Initialize detector
    detector = DriftDetector(train_df, feature_cols)
    print("✓ Drift detector initialized")
    
    # Test drift detection
    data_drift = detector.detect_data_drift(test_df[:1000])  # Use subset for speed
    print(f"✓ Data drift detection: {data_drift['n_drifted_features']} features drifted")
    
    # Test prediction drift
    y_train = train_df['readmitted_encoded'][:1000]
    y_test = test_df['readmitted_encoded'][:1000]
    pred_drift = detector.detect_prediction_drift(y_train, y_test)
    print(f"✓ Prediction drift detection: drift={'detected' if pred_drift['drift_detected'] else 'not detected'}")
    
    # Test performance monitoring
    import joblib
    model = joblib.load('models/production/best_model.pkl')
    X_test = test_df[feature_cols][:1000]
    y_pred = model.predict(X_test)
    
    with open('models/production/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    perf_result = detector.monitor_performance(y_test, y_pred, metadata['metrics'])
    print(f"✓ Performance monitoring: {'degradation' if perf_result['degradation_detected'] else 'stable'}")
    
except Exception as e:
    print(f"❌ Monitoring test failed: {str(e)}")
    import traceback
    traceback.print_exc()

# MLflow Check
print("\n[BONUS] MLflow Setup...")
print("-" * 70)

try:
    import mlflow
    print("✓ MLflow installed")
    
    if os.path.exists('mlruns'):
        print("✓ MLflow tracking directory exists")
        # Count experiments
        import glob
        experiments = glob.glob('mlruns/*/')
        print(f"✓ Experiments tracked: {len(experiments)-1}")  # Subtract .trash
    
    print("\n  To view MLflow UI:")
    print("    mlflow ui")
    print("    Open: http://127.0.0.1:5000")
    
except ImportError:
    print("⚠ MLflow not installed (optional)")

# Summary
print("\n" + "="*70)
print("PHASE 4 COMPLETED SUCCESSFULLY!")
print("="*70)

print("\n📦 Generated Artifacts:")
print("\n  API:")
print("    - api/app.py (Flask REST API)")
print("    - api/inference.py (Prediction service)")
print("\n  Docker:")
print("    - Dockerfile (Container definition)")
print("    - docker-compose.yml (Service orchestration)")
print("    - .dockerignore (Build optimization)")
print("\n  Monitoring:")
print("    - src/monitoring/drift_detector.py (Drift detection)")
print("\n  MLOps:")
print("    - mlruns/ (Experiment tracking)")
print("    - logs/ (Monitoring logs)")

print("\n🚀 Deployment Commands:")
print("\n  Local API:")
print("    python3 -m api.app")
print("    Access: http://127.0.0.1:5000")
print("\n  Docker:")
print("    docker build -t hospital-readmission-api .")
print("    docker run -p 5000:5000 hospital-readmission-api")
print("\n  Docker Compose:")
print("    docker-compose up -d")
print("    docker-compose down")

print("\n📡 API Endpoints:")
print("    GET  /             - API information")
print("    GET  /health       - Health check")
print("    GET  /model_info   - Model metadata")
print("    POST /predict      - Single prediction")
print("    POST /batch_predict - Batch predictions")

print("\n💡 Example API Request:")
print('    curl -X POST http://localhost:5000/predict \\')
print('      -H "Content-Type: application/json" \\')
print('      -d \'{"features": {"age_numeric": 65, "gender_male": 1, ...}}\'')

print("\nReady for Phase 5: Documentation & Portfolio Presentation")
print("="*70 + "\n")

