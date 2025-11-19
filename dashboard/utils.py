"""
Dashboard Utilities
Helper functions for loading data and generating predictions
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import json
from typing import Dict, Tuple


def load_data_and_model():
    """Load test data, model, and generate predictions"""
    import os
    
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load test data
    with open(os.path.join(base_dir, 'data/processed/test.pkl'), 'rb') as f:
        test_df = pickle.load(f)
    
    # Load feature columns
    with open(os.path.join(base_dir, 'data/processed/feature_columns.pkl'), 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Load best model
    model = joblib.load(os.path.join(base_dir, 'models/production/best_model.pkl'))
    
    # Load model metadata
    with open(os.path.join(base_dir, 'models/production/model_metadata.json'), 'r') as f:
        model_metadata = json.load(f)
    
    # Generate predictions
    X_test = test_df[feature_cols]
    y_test = test_df['readmitted_encoded']
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Create results dataframe
    results_df = test_df.copy()
    results_df['predicted_class'] = predictions
    results_df['predicted_label'] = results_df['predicted_class'].map({
        0: 'NO', 1: '<30', 2: '>30'
    })
    results_df['true_label'] = results_df['readmitted_encoded'].map({
        0: 'NO', 1: '<30', 2: '>30'
    })
    results_df['prob_no'] = probabilities[:, 0]
    results_df['prob_under_30'] = probabilities[:, 1]
    results_df['prob_over_30'] = probabilities[:, 2]
    results_df['max_prob'] = probabilities.max(axis=1)
    
    # Add risk score (probability of readmission)
    results_df['risk_score'] = (probabilities[:, 1] + probabilities[:, 2]) * 100
    
    # Add patient ID
    results_df['patient_id'] = results_df['patient_nbr'] if 'patient_nbr' in results_df.columns else range(len(results_df))
    
    return results_df, model, model_metadata, feature_cols


def get_kpi_metrics(df: pd.DataFrame) -> Dict:
    """Calculate KPI metrics"""
    total_patients = len(df)
    high_risk = (df['predicted_class'] == 1).sum()
    low_risk = (df['predicted_class'] == 0).sum()
    medium_risk = (df['predicted_class'] == 2).sum()
    
    high_risk_pct = (high_risk / total_patients) * 100
    low_risk_pct = (low_risk / total_patients) * 100
    medium_risk_pct = (medium_risk / total_patients) * 100
    
    # Accuracy
    accuracy = (df['predicted_class'] == df['readmitted_encoded']).mean() * 100
    
    return {
        'total_patients': total_patients,
        'high_risk': high_risk,
        'low_risk': low_risk,
        'medium_risk': medium_risk,
        'high_risk_pct': high_risk_pct,
        'low_risk_pct': low_risk_pct,
        'medium_risk_pct': medium_risk_pct,
        'accuracy': accuracy
    }


def get_top_risk_factors(df: pd.DataFrame, feature_cols: list, n: int = 3) -> Dict:
    """Get top risk factors for each prediction"""
    # For now, return generic top features
    # In production, this would use SHAP values
    top_features = [
        'number_inpatient',
        'time_in_hospital',
        'number_diagnoses',
        'age_numeric',
        'num_medications',
        'number_emergency',
        'diabetes_med_count',
        'discharged_to_home'
    ]
    
    risk_factors = {}
    for idx in df.index:
        factors = []
        for feat in top_features[:n]:
            if feat in df.columns:
                value = df.loc[idx, feat]
                factors.append(f"{feat}={value:.1f}")
        risk_factors[idx] = ', '.join(factors)
    
    return risk_factors


def filter_dataframe(df: pd.DataFrame, 
                     age_range: Tuple[int, int] = None,
                     gender: str = None,
                     risk_threshold: float = None) -> pd.DataFrame:
    """Filter dataframe based on criteria"""
    filtered = df.copy()
    
    if age_range:
        if 'age_numeric' in filtered.columns:
            filtered = filtered[
                (filtered['age_numeric'] >= age_range[0]) & 
                (filtered['age_numeric'] <= age_range[1])
            ]
    
    if gender and gender != 'All':
        if 'gender' in filtered.columns:
            filtered = filtered[filtered['gender'] == gender]
    
    if risk_threshold is not None:
        filtered = filtered[filtered['risk_score'] >= risk_threshold]
    
    return filtered


def calculate_cost_savings(df: pd.DataFrame, 
                          cost_per_readmission: float = 12000,
                          prevention_rate: float = 0.25) -> Dict:
    """Calculate potential cost savings from preventing readmissions"""
    
    # Number of predicted readmissions
    predicted_readmissions = ((df['predicted_class'] == 1) | (df['predicted_class'] == 2)).sum()
    
    # Potential prevented readmissions (assuming 25% prevention rate)
    prevented = int(predicted_readmissions * prevention_rate)
    
    # Cost savings
    savings = prevented * cost_per_readmission
    
    return {
        'predicted_readmissions': predicted_readmissions,
        'prevented_readmissions': prevented,
        'cost_savings': savings
    }

