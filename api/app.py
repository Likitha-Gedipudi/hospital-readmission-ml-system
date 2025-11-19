"""
Flask REST API for Hospital Readmission Prediction
Provides endpoints for model inference and health checks
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.inference import PredictionService

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize prediction service
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(base_dir, 'models/production/best_model.pkl')
metadata_path = os.path.join(base_dir, 'models/production/model_metadata.json')
feature_cols_path = os.path.join(base_dir, 'data/processed/feature_columns.pkl')

try:
    prediction_service = PredictionService(model_path, metadata_path, feature_cols_path)
    logger.info("✓ Prediction service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize prediction service: {str(e)}")
    prediction_service = None


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API information"""
    return jsonify({
        'service': 'Hospital Readmission Prediction API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /model_info': 'Model metadata',
            'POST /predict': 'Single patient prediction',
            'POST /batch_predict': 'Batch predictions'
        },
        'documentation': 'See /health for system status and /model_info for model details'
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if prediction_service is None:
        return jsonify({
            'status': 'unhealthy',
            'message': 'Prediction service not initialized',
            'timestamp': datetime.now().isoformat()
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running',
        'model_loaded': True,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model metadata and performance metrics"""
    if prediction_service is None:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        info = prediction_service.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single patient prediction endpoint
    
    Expected JSON format:
    {
        "features": {
            "age_numeric": 65,
            "gender_male": 1,
            "time_in_hospital": 7,
            ...
        }
    }
    """
    if prediction_service is None:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'expected': {'features': {...}}
            }), 400
        
        features = data['features']
        
        # Make prediction
        result = prediction_service.predict(features)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['model'] = prediction_service.metadata['model_name']
        
        logger.info(f"Prediction made: {result['predicted_class']} (risk: {result['risk_score']:.1f}%)")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "patients": [
            {"features": {...}},
            {"features": {...}},
            ...
        ]
    }
    """
    if prediction_service is None:
        return jsonify({'error': 'Prediction service not available'}), 503
    
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'patients' not in data:
            return jsonify({
                'error': 'Invalid request format',
                'expected': {'patients': [{'features': {...}}, ...]}
            }), 400
        
        patients = data['patients']
        
        # Extract features from each patient
        features_list = [p['features'] for p in patients if 'features' in p]
        
        if not features_list:
            return jsonify({'error': 'No valid patient data provided'}), 400
        
        # Make predictions
        results = prediction_service.batch_predict(features_list)
        
        # Add metadata
        response = {
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat(),
            'model': prediction_service.metadata['model_name']
        }
        
        logger.info(f"Batch prediction made for {len(results)} patients")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*70)
    print("HOSPITAL READMISSION PREDICTION API")
    print("="*70)
    print(f"\n🚀 Starting Flask API server...")
    print(f"\n📡 API will be available at: http://127.0.0.1:5000")
    print(f"\nEndpoints:")
    print(f"  GET  /             - API information")
    print(f"  GET  /health       - Health check")
    print(f"  GET  /model_info   - Model metadata")
    print(f"  POST /predict      - Single prediction")
    print(f"  POST /batch_predict - Batch predictions")
    print(f"\n⚠  Press Ctrl+C to stop the server\n")
    print("="*70 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

