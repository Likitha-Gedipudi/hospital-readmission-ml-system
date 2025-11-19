"""
Drift Detection Module
Monitors data drift, prediction drift, and model performance degradation
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detect data and prediction drift"""
    
    def __init__(self, reference_data: pd.DataFrame, feature_cols: List[str]):
        """
        Initialize drift detector
        
        Args:
            reference_data: Reference (training) data
            feature_cols: List of feature column names
        """
        self.reference_data = reference_data[feature_cols]
        self.feature_cols = feature_cols
        
        # Calculate reference statistics
        self.reference_stats = self._calculate_stats(self.reference_data)
        
    def _calculate_stats(self, data: pd.DataFrame) -> Dict:
        """Calculate statistics for data"""
        stats_dict = {}
        for col in self.feature_cols:
            if col in data.columns:
                stats_dict[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median()
                }
        return stats_dict
    
    def detect_data_drift(self, current_data: pd.DataFrame, 
                         threshold: float = 0.05) -> Dict:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            current_data: Current data to check for drift
            threshold: P-value threshold for drift detection
        
        Returns:
            Dictionary with drift results
        """
        logger.info("Detecting data drift...")
        
        drift_detected = []
        drift_scores = {}
        
        for col in self.feature_cols:
            if col not in current_data.columns:
                continue
            
            # Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(
                self.reference_data[col].dropna(),
                current_data[col].dropna()
            )
            
            drift_scores[col] = {
                'ks_statistic': float(ks_stat),
                'p_value': float(p_value),
                'drift_detected': p_value < threshold
            }
            
            if p_value < threshold:
                drift_detected.append(col)
        
        result = {
            'drift_detected': len(drift_detected) > 0,
            'n_drifted_features': len(drift_detected),
            'drifted_features': drift_detected,
            'drift_scores': drift_scores,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        if drift_detected:
            logger.warning(f"Drift detected in {len(drift_detected)} features: {drift_detected[:5]}")
        else:
            logger.info("No significant data drift detected")
        
        return result
    
    def detect_prediction_drift(self, reference_predictions: np.ndarray,
                               current_predictions: np.ndarray,
                               threshold: float = 0.05) -> Dict:
        """
        Detect drift in prediction distributions
        
        Args:
            reference_predictions: Reference predictions
            current_predictions: Current predictions
            threshold: P-value threshold
        
        Returns:
            Dictionary with drift results
        """
        logger.info("Detecting prediction drift...")
        
        # Chi-square test for categorical predictions
        ref_counts = pd.Series(reference_predictions).value_counts()
        curr_counts = pd.Series(current_predictions).value_counts()
        
        # Ensure same categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_dist = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_dist = [curr_counts.get(cat, 0) for cat in all_categories]
        
        chi2_stat, p_value = stats.chisquare(curr_dist, f_exp=ref_dist)
        
        result = {
            'drift_detected': p_value < threshold,
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'reference_distribution': ref_counts.to_dict(),
            'current_distribution': curr_counts.to_dict(),
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        if result['drift_detected']:
            logger.warning(f"Prediction drift detected (p={p_value:.4f})")
        else:
            logger.info("No significant prediction drift detected")
        
        return result
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray,
                     bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI)
        
        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Small change
        PSI >= 0.2: Significant change (action required)
        
        Args:
            reference: Reference data
            current: Current data
            bins: Number of bins for discretization
        
        Returns:
            PSI value
        """
        # Create bins based on reference data
        breakpoints = np.histogram(reference, bins=bins)[1]
        
        # Calculate distributions
        ref_percents = np.histogram(reference, bins=breakpoints)[0] / len(reference)
        curr_percents = np.histogram(current, bins=breakpoints)[0] / len(current)
        
        # Avoid division by zero
        ref_percents = np.where(ref_percents == 0, 0.0001, ref_percents)
        curr_percents = np.where(curr_percents == 0, 0.0001, curr_percents)
        
        # Calculate PSI
        psi = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        
        return float(psi)
    
    def monitor_performance(self, y_true: np.ndarray, y_pred: np.ndarray,
                           baseline_metrics: Dict, threshold: float = 0.05) -> Dict:
        """
        Monitor model performance degradation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            baseline_metrics: Baseline performance metrics
            threshold: Acceptable degradation threshold
        
        Returns:
            Dictionary with performance monitoring results
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        logger.info("Monitoring model performance...")
        
        # Calculate current metrics
        current_accuracy = accuracy_score(y_true, y_pred)
        current_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Compare with baseline
        accuracy_change = current_accuracy - baseline_metrics.get('accuracy', 0)
        f1_change = current_f1 - baseline_metrics.get('f1_macro', 0)
        
        degradation_detected = (
            accuracy_change < -threshold or
            f1_change < -threshold
        )
        
        result = {
            'degradation_detected': degradation_detected,
            'current_metrics': {
                'accuracy': float(current_accuracy),
                'f1_macro': float(current_f1)
            },
            'baseline_metrics': baseline_metrics,
            'changes': {
                'accuracy_change': float(accuracy_change),
                'f1_change': float(f1_change)
            },
            'threshold': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        if degradation_detected:
            logger.warning(f"Performance degradation detected!")
            logger.warning(f"Accuracy change: {accuracy_change:+.4f}")
            logger.warning(f"F1 change: {f1_change:+.4f}")
        else:
            logger.info("No significant performance degradation")
        
        return result
    
    def generate_monitoring_report(self, data_drift_result: Dict,
                                  prediction_drift_result: Dict,
                                  performance_result: Dict) -> Dict:
        """Generate comprehensive monitoring report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'alerts': [],
            'data_drift': data_drift_result,
            'prediction_drift': prediction_drift_result,
            'performance': performance_result
        }
        
        # Check for alerts
        if data_drift_result.get('drift_detected'):
            report['overall_status'] = 'warning'
            report['alerts'].append('Data drift detected')
        
        if prediction_drift_result.get('drift_detected'):
            report['overall_status'] = 'warning'
            report['alerts'].append('Prediction drift detected')
        
        if performance_result.get('degradation_detected'):
            report['overall_status'] = 'critical'
            report['alerts'].append('Performance degradation detected')
        
        return report
    
    def save_report(self, report: Dict, output_path: str = 'logs/monitoring_report.json'):
        """Save monitoring report to file"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Monitoring report saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load data
    with open('../../data/processed/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    
    with open('../../data/processed/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    
    with open('../../data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    # Initialize detector
    detector = DriftDetector(train_df, feature_cols)
    
    # Detect drift
    data_drift = detector.detect_data_drift(test_df)
    print(f"\nData Drift: {data_drift['drift_detected']}")
    print(f"Drifted features: {len(data_drift['drifted_features'])}")
    
    print("\n✓ Drift detection complete!")

