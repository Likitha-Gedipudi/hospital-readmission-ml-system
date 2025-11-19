"""
Model Evaluation Module
Evaluates multi-class classification models with comprehensive metrics
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Tuple, Any
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate machine learning models"""
    
    def __init__(self, class_names: list = None):
        """
        Initialize evaluator
        
        Args:
            class_names: List of class names ['NO', '<30', '>30']
        """
        self.class_names = class_names or ['NO', '<30', '>30']
        self.results = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model pipeline
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # ROC-AUC (One-vs-Rest)
        try:
            roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            roc_auc_weighted = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {str(e)}")
            roc_auc_ovr = 0.0
            roc_auc_weighted = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names, 
                                      output_dict=True, zero_division=0)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'roc_auc_ovr': roc_auc_ovr,
            'roc_auc_weighted': roc_auc_weighted,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        self.results[model_name] = results
        
        logger.info(f"✓ {model_name} - Accuracy: {accuracy:.4f}, F1 (macro): {f1_macro:.4f}, AUC: {roc_auc_ovr:.4f}")
        return results
    
    def evaluate_all_models(self, models: Dict, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Dict[str, Dict]:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary of all results
        """
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATING ALL MODELS")
        logger.info(f"{'='*70}")
        logger.info(f"Test set size: {len(X_test)}")
        
        for model_name, model in models.items():
            try:
                self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        logger.info(f"\n✓ Evaluated {len(self.results)} models")
        return self.results
    
    def print_summary(self):
        """Print evaluation summary"""
        if not self.results:
            logger.warning("No results to display")
            return
        
        print("\n" + "="*90)
        print("MODEL EVALUATION SUMMARY")
        print("="*90)
        
        # Create comparison table
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1 (Macro)': f"{results['f1_macro']:.4f}",
                'F1 (Weighted)': f"{results['f1_weighted']:.4f}",
                'ROC-AUC': f"{results['roc_auc_ovr']:.4f}",
                'Precision (Macro)': f"{results['precision_macro']:.4f}",
                'Recall (Macro)': f"{results['recall_macro']:.4f}"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        print("="*90)
        
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_macro'])
        print(f"\n🏆 Best Model (by F1-Macro): {best_model[0]}")
        print(f"   F1-Macro: {best_model[1]['f1_macro']:.4f}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   ROC-AUC: {best_model[1]['roc_auc_ovr']:.4f}")
        print("="*90 + "\n")
    
    def plot_confusion_matrix(self, model_name: str, save_path: str = None):
        """Plot confusion matrix for a model"""
        if model_name not in self.results:
            logger.error(f"Model {model_name} not found in results")
            return
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def plot_all_confusion_matrices(self, save_dir: str = 'outputs'):
        """Plot confusion matrices for all models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        n_models = len(self.results)
        if n_models == 0:
            return
        
        # Calculate grid size
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = np.array(results['confusion_matrix'])
            ax = axes[idx]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=ax, cbar_kws={'label': 'Count'})
            ax.set_title(f'{model_name}\nF1: {results["f1_macro"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        save_path = f'{save_dir}/all_confusion_matrices.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"All confusion matrices saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, models: Dict, X_test: pd.DataFrame, 
                       y_test: pd.Series, save_path: str = None):
        """Plot ROC curves for all models (One-vs-Rest)"""
        n_classes = len(self.class_names)
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
        
        fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))
        
        for class_idx in range(n_classes):
            ax = axes[class_idx]
            
            for model_name, model in models.items():
                if model_name not in self.results:
                    continue
                
                y_pred_proba = np.array(self.results[model_name]['probabilities'])
                
                fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_pred_proba[:, class_idx])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})', linewidth=2)
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - Class: {self.class_names[class_idx]}', fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.close()
    
    def save_results(self, output_path: str = 'outputs/evaluation_results.json'):
        """Save evaluation results to JSON"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for model_name, results in self.results.items():
            results_json[model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in results.items()
                if k not in ['predictions', 'probabilities']  # Exclude large arrays
            }
        
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
    
    def get_best_model_name(self) -> str:
        """Get name of best performing model"""
        if not self.results:
            return None
        
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_macro'])
        return best_model[0]


if __name__ == "__main__":
    # Example usage
    import pickle
    import joblib
    
    # Load data
    with open('data/processed/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    X_test = test_df[feature_cols]
    y_test = test_df['readmitted_encoded']
    
    # Load models
    import os
    models = {}
    for model_file in os.listdir('models'):
        if model_file.endswith('_model.pkl'):
            model_name = model_file.replace('_model.pkl', '')
            models[model_name] = joblib.load(f'models/{model_file}')
    
    # Evaluate
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_all_models(models, X_test, y_test)
    evaluator.print_summary()
    evaluator.save_results()
    
    print("\n✓ Model evaluation complete!")

