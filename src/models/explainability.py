"""
Model Explainability Module
Uses SHAP for model interpretation and explanation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Any
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


class ModelExplainer:
    """Generate SHAP explanations for models"""
    
    def __init__(self, model, feature_names: List[str], class_names: List[str] = None):
        """
        Initialize explainer
        
        Args:
            model: Trained model pipeline
            feature_names: List of feature names
            class_names: List of class names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library is required. Install with: pip install shap")
        
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or ['NO', '<30', '>30']
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background: pd.DataFrame, max_samples: int = 100):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background data for SHAP
            max_samples: Maximum samples to use for background
        """
        logger.info("Creating SHAP explainer...")
        
        # Sample background data if too large
        if len(X_background) > max_samples:
            X_background = X_background.sample(n=max_samples, random_state=42)
        
        # Transform background data through pipeline (scaling)
        if hasattr(self.model, 'named_steps'):
            # Get the classifier from pipeline
            if 'classifier' in self.model.named_steps:
                classifier = self.model.named_steps['classifier']
                # Transform data through preprocessing steps
                X_transformed = X_background.copy()
                for step_name, step in self.model.named_steps.items():
                    if step_name != 'classifier' and step_name != 'smote':
                        X_transformed = step.transform(X_transformed)
                
                # Create explainer for the classifier
                try:
                    self.explainer = shap.TreeExplainer(classifier)
                    logger.info("✓ TreeExplainer created")
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {str(e)}, trying KernelExplainer...")
                    # Fallback to KernelExplainer
                    def predict_fn(x):
                        return self.model.predict_proba(
                            pd.DataFrame(x, columns=self.feature_names)
                        )
                    self.explainer = shap.KernelExplainer(predict_fn, X_background)
                    logger.info("✓ KernelExplainer created")
            else:
                # Use full pipeline
                def predict_fn(x):
                    return self.model.predict_proba(
                        pd.DataFrame(x, columns=self.feature_names)
                    )
                self.explainer = shap.KernelExplainer(predict_fn, X_background)
                logger.info("✓ KernelExplainer created")
        else:
            # Model is not a pipeline
            try:
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("✓ TreeExplainer created")
            except:
                def predict_fn(x):
                    return self.model.predict_proba(
                        pd.DataFrame(x, columns=self.feature_names)
                    )
                self.explainer = shap.KernelExplainer(predict_fn, X_background)
                logger.info("✓ KernelExplainer created")
    
    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 500):
        """
        Compute SHAP values
        
        Args:
            X: Data to explain
            max_samples: Maximum samples to explain
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer() first.")
        
        logger.info(f"Computing SHAP values for {min(len(X), max_samples)} samples...")
        
        # Sample if too large
        if len(X) > max_samples:
            X = X.sample(n=max_samples, random_state=42)
        
        # Compute SHAP values
        if hasattr(self.model, 'named_steps') and 'classifier' in self.model.named_steps:
            # Transform data through preprocessing
            X_transformed = X.copy()
            for step_name, step in self.model.named_steps.items():
                if step_name != 'classifier' and step_name != 'smote':
                    X_transformed = step.transform(X_transformed)
            
            if isinstance(self.explainer, shap.TreeExplainer):
                self.shap_values = self.explainer.shap_values(X_transformed)
            else:
                self.shap_values = self.explainer.shap_values(X)
        else:
            self.shap_values = self.explainer.shap_values(X)
        
        logger.info("✓ SHAP values computed")
        return self.shap_values
    
    def plot_summary(self, save_path: str = None):
        """Plot SHAP summary (beeswarm) plot"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values() first.")
        
        logger.info("Creating SHAP summary plot...")
        
        # For multi-class, plot each class
        if isinstance(self.shap_values, list):
            n_classes = len(self.shap_values)
            fig, axes = plt.subplots(1, n_classes, figsize=(8*n_classes, 6))
            if n_classes == 1:
                axes = [axes]
            
            for class_idx in range(n_classes):
                plt.sca(axes[class_idx])
                shap.summary_plot(
                    self.shap_values[class_idx],
                    features=None,
                    feature_names=self.feature_names,
                    show=False,
                    max_display=15
                )
                axes[class_idx].set_title(f'SHAP Summary - Class: {self.class_names[class_idx]}',
                                         fontweight='bold')
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                self.shap_values,
                feature_names=self.feature_names,
                show=False,
                max_display=15
            )
            plt.title('SHAP Summary Plot', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, save_path: str = None):
        """Plot SHAP feature importance (bar plot)"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        logger.info("Creating SHAP feature importance plot...")
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            # Multi-class: average across classes
            shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            shap_mean = np.abs(self.shap_values).mean(axis=0)
        
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_mean
        }).sort_values('importance', ascending=False)
        
        # Plot top 20
        top_n = min(20, len(importance_df))
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importance_df['importance'].head(top_n).values)
        plt.yticks(range(top_n), importance_df['feature'].head(top_n).values)
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.title('Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance saved to {save_path}")
        
        plt.show()
        
        return importance_df
    
    def explain_prediction(self, X_sample: pd.DataFrame, sample_idx: int = 0,
                          class_idx: int = None, save_path: str = None):
        """
        Create waterfall plot for individual prediction
        
        Args:
            X_sample: Sample data
            sample_idx: Index of sample to explain
            class_idx: Class to explain (for multi-class)
            save_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        logger.info(f"Explaining prediction for sample {sample_idx}...")
        
        # Get prediction
        prediction = self.model.predict(X_sample.iloc[[sample_idx]])[0]
        proba = self.model.predict_proba(X_sample.iloc[[sample_idx]])[0]
        
        logger.info(f"Predicted class: {self.class_names[prediction]} (probability: {proba[prediction]:.3f})")
        
        # Create waterfall plot
        if isinstance(self.shap_values, list):
            if class_idx is None:
                class_idx = prediction
            shap_values_sample = self.shap_values[class_idx][sample_idx]
            class_name = self.class_names[class_idx]
        else:
            shap_values_sample = self.shap_values[sample_idx]
            class_name = self.class_names[prediction]
        
        # Create explanation object
        if hasattr(self.explainer, 'expected_value'):
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                expected_value = self.explainer.expected_value[class_idx if class_idx is not None else 0]
            else:
                expected_value = self.explainer.expected_value
        else:
            expected_value = 0
        
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_sample,
                base_values=expected_value,
                data=X_sample.iloc[sample_idx].values,
                feature_names=self.feature_names
            ),
            show=False,
            max_display=15
        )
        plt.title(f'SHAP Explanation - Predicted: {class_name}', fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation saved to {save_path}")
        
        plt.show()
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # Calculate mean absolute SHAP values
        if isinstance(self.shap_values, list):
            shap_mean = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
        else:
            shap_mean = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': shap_mean
        }).sort_values('importance', ascending=False).head(n)
        
        return importance_df
    
    def save_shap_values(self, output_path: str):
        """Save SHAP values to disk"""
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        np.save(output_path, self.shap_values)
        logger.info(f"SHAP values saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load data
    with open('data/processed/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    X_test = test_df[feature_cols]
    
    # Load best model
    model = joblib.load('models/production/best_model.pkl')
    
    # Create explainer
    explainer = ModelExplainer(model, feature_cols)
    explainer.create_explainer(X_test, max_samples=100)
    explainer.compute_shap_values(X_test, max_samples=500)
    
    # Generate plots
    explainer.plot_summary(save_path='outputs/shap/shap_summary.png')
    importance_df = explainer.plot_feature_importance(save_path='outputs/shap/feature_importance.png')
    
    print("\n✓ SHAP analysis complete!")
    print("\nTop 10 Features:")
    print(importance_df.head(10))

