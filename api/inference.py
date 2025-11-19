"""
Inference Module
Handles model predictions and feature preprocessing
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, List, Any


class PredictionService:
    """Service for making readmission predictions"""
    
    def __init__(self, model_path: str, metadata_path: str, feature_cols_path: str):
        """
        Initialize prediction service
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata JSON
            feature_cols_path: Path to feature columns pickle
        """
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        import pickle
        with open(feature_cols_path, 'rb') as f:
            self.feature_cols = pickle.load(f)
        
        self.class_names = ['NO', '<30', '>30']
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single patient
        
        Args:
            features: Dictionary of patient features
        
        Returns:
            Dictionary containing prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all required features are present
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0  # Default value for missing features
        
        # Select only required features in correct order
        X = df[self.feature_cols]
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        
        # Get predicted class name
        predicted_class = self.class_names[prediction]
        
        # Calculate risk score (probability of readmission)
        risk_score = (probabilities[1] + probabilities[2]) * 100
        
        # Get top risk factors (simplified - in production would use SHAP)
        top_features = self._get_top_risk_factors(X, probabilities)
        
        result = {
            'predicted_class': predicted_class,
            'predicted_class_id': int(prediction),
            'risk_score': float(risk_score),
            'probabilities': {
                'NO': float(probabilities[0]),
                '<30': float(probabilities[1]),
                '>30': float(probabilities[2])
            },
            'confidence': float(probabilities.max()),
            'top_risk_factors': top_features,
            'recommendation': self._get_recommendation(predicted_class, risk_score)
        }
        
        return result
    
    def batch_predict(self, features_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple patients
        
        Args:
            features_list: List of feature dictionaries
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for features in features_list:
            try:
                result = self.predict(features)
                results.append(result)
            except Exception as e:
                results.append({'error': str(e)})
        
        return results
    
    def _get_top_risk_factors(self, X: pd.DataFrame, probabilities: np.ndarray, 
                              n: int = 5) -> List[Dict[str, Any]]:
        """Get top risk factors contributing to prediction"""
        # Simplified version - in production would use actual SHAP values
        feature_values = X.iloc[0].to_dict()
        
        # Define important features (from our feature engineering)
        important_features = [
            ('number_inpatient', 'Previous inpatient visits'),
            ('time_in_hospital', 'Length of stay (days)'),
            ('number_diagnoses', 'Number of diagnoses'),
            ('age_numeric', 'Age'),
            ('num_medications', 'Number of medications'),
            ('number_emergency', 'Previous emergency visits'),
            ('diabetes_med_count', 'Diabetes medications count'),
            ('medication_changed', 'Medication changed')
        ]
        
        risk_factors = []
        for feat, desc in important_features[:n]:
            if feat in feature_values:
                risk_factors.append({
                    'feature': desc,
                    'value': float(feature_values[feat]),
                    'impact': 'High' if feature_values[feat] > 0 else 'Low'
                })
        
        return risk_factors
    
    def _get_recommendation(self, predicted_class: str, risk_score: float) -> str:
        """Generate recommendation based on prediction"""
        if predicted_class == '<30':
            return "HIGH RISK: Schedule early follow-up within 7 days. Consider discharge planning and medication reconciliation."
        elif predicted_class == '>30':
            return "MEDIUM RISK: Schedule follow-up within 30 days. Provide discharge instructions and ensure medication adherence."
        else:
            return "LOW RISK: Standard discharge protocols. Routine follow-up as needed."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata information"""
        return {
            'model_name': self.metadata['model_name'],
            'model_type': self.metadata['model_type'],
            'n_features': self.metadata['n_features'],
            'classes': self.metadata['classes'],
            'metrics': self.metadata['metrics']
        }


if __name__ == "__main__":
    # Example usage
    service = PredictionService(
        model_path='../models/production/best_model.pkl',
        metadata_path='../models/production/model_metadata.json',
        feature_cols_path='../data/processed/feature_columns.pkl'
    )
    
    # Example patient
    example_patient = {
        'age_numeric': 65,
        'gender_male': 1,
        'gender_female': 0,
        'time_in_hospital': 7,
        'number_inpatient': 2,
        'number_outpatient': 0,
        'number_emergency': 1,
        'number_diagnoses': 8,
        'num_medications': 15,
        'num_procedures': 2,
        'num_lab_procedures': 50
    }
    
    result = service.predict(example_patient)
    print(json.dumps(result, indent=2))

