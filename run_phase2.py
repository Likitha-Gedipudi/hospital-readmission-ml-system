"""
Phase 2 Execution Script
Trains models, evaluates performance, generates SHAP explanations, and tracks with MLflow
"""

import sys
import os
import pickle
import logging

# Add src to path
sys.path.append('src')
sys.path.append('src/models')

from train import ModelTrainer
from evaluation import ModelEvaluator
from explainability import ModelExplainer, SHAP_AVAILABLE
from mlflow_utils import MLflowTracker, log_model_comparison
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SKIP_SHAP = True  # Set to False to enable SHAP analysis (takes 5-15 minutes)
SKIP_MLFLOW = True  # Set to False to enable MLflow logging


def main():
    """Execute Phase 2: Model Development"""
    
    print("\n" + "="*70)
    print("PHASE 2: MODEL DEVELOPMENT")
    print("="*70)
    
    # Create output directories
    os.makedirs('models/production', exist_ok=True)
    os.makedirs('outputs/shap', exist_ok=True)
    os.makedirs('outputs/evaluation', exist_ok=True)
    
    # Step 1: Load Data
    print("\n[STEP 1/5] Loading Data...")
    print("-" * 70)
    
    with open('data/processed/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    
    with open('data/processed/val.pkl', 'rb') as f:
        val_df = pickle.load(f)
    
    with open('data/processed/test.pkl', 'rb') as f:
        test_df = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    X_train = train_df[feature_cols]
    y_train = train_df['readmitted_encoded']
    
    X_val = val_df[feature_cols]
    y_val = val_df['readmitted_encoded']
    
    X_test = test_df[feature_cols]
    y_test = test_df['readmitted_encoded']
    
    print(f"✓ Data loaded successfully")
    print(f"  Train: {len(X_train)} samples, {len(feature_cols)} features")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Step 2: Initialize MLflow
    print("\n[STEP 2/5] Initializing MLflow...")
    print("-" * 70)
    
    if SKIP_MLFLOW:
        print("⚠ MLflow tracking skipped (SKIP_MLFLOW=True)")
        mlflow_tracker = None
    else:
        mlflow_tracker = MLflowTracker(experiment_name="Hospital_Readmission")
        if mlflow_tracker.enabled:
            print("✓ MLflow tracking initialized")
        else:
            print("⚠ MLflow not available, continuing without tracking")
    
    # Step 3: Train Models
    print("\n[STEP 3/5] Training Models...")
    print("-" * 70)
    
    trainer = ModelTrainer(random_state=42)
    models = trainer.train_all_models(X_train, y_train, use_smote=False)
    
    # Save all models
    trainer.save_all_models('models')
    print(f"\n✓ Trained and saved {len(models)} models")
    
    # Step 4: Evaluate Models
    print("\n[STEP 4/5] Evaluating Models...")
    print("-" * 70)
    
    evaluator = ModelEvaluator(class_names=['NO', '<30', '>30'])
    results = evaluator.evaluate_all_models(models, X_test, y_test)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results('outputs/evaluation/evaluation_results.json')
    
    # Generate plots
    evaluator.plot_all_confusion_matrices('outputs/evaluation')
    
    try:
        evaluator.plot_roc_curves(models, X_test, y_test, 
                                 save_path='outputs/evaluation/roc_curves.png')
    except Exception as e:
        logger.warning(f"Could not generate ROC curves: {str(e)}")
    
    # Get best model
    best_model_name = evaluator.get_best_model_name()
    best_model = models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Macro: {results[best_model_name]['f1_macro']:.4f}")
    print(f"   Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print(f"   ROC-AUC: {results[best_model_name]['roc_auc_ovr']:.4f}")
    
    # Save best model to production
    joblib.dump(best_model, 'models/production/best_model.pkl')
    print(f"\n✓ Best model saved to models/production/best_model.pkl")
    
    # Save model metadata
    import json
    metadata = {
        'model_name': best_model_name,
        'model_type': best_model_name.split('_')[0],
        'metrics': {
            'accuracy': results[best_model_name]['accuracy'],
            'f1_macro': results[best_model_name]['f1_macro'],
            'f1_weighted': results[best_model_name]['f1_weighted'],
            'roc_auc_ovr': results[best_model_name]['roc_auc_ovr'],
            'precision_macro': results[best_model_name]['precision_macro'],
            'recall_macro': results[best_model_name]['recall_macro']
        },
        'n_features': len(feature_cols),
        'classes': ['NO', '<30', '>30']
    }
    
    with open('models/production/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✓ Model metadata saved")
    
    # Step 5: SHAP Explainability
    print("\n[STEP 5/5] Generating SHAP Explanations...")
    print("-" * 70)
    
    if SKIP_SHAP:
        print("⚠ SHAP analysis skipped for faster execution (SKIP_SHAP=True)")
        print("  To enable SHAP, set SKIP_SHAP=False in run_phase2.py and re-run")
    elif SHAP_AVAILABLE:
        try:
            explainer = ModelExplainer(best_model, feature_cols, class_names=['NO', '<30', '>30'])
            explainer.create_explainer(X_train, max_samples=100)
            explainer.compute_shap_values(X_test, max_samples=500)
            
            # Generate plots
            try:
                explainer.plot_summary(save_path='outputs/shap/shap_summary.png')
            except Exception as e:
                logger.warning(f"Could not generate SHAP summary: {str(e)}")
            
            try:
                importance_df = explainer.plot_feature_importance(
                    save_path='outputs/shap/feature_importance.png'
                )
                
                # Save top features
                importance_df.to_csv('outputs/shap/top_features.csv', index=False)
                
                print("\n✓ SHAP explanations generated")
                print("\nTop 10 Most Important Features:")
                for idx, row in importance_df.head(10).iterrows():
                    print(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
            except Exception as e:
                logger.warning(f"Could not generate feature importance: {str(e)}")
            
            # Explain a few sample predictions
            try:
                # High-risk patient
                high_risk_idx = (y_test == 1).idxmax()
                if high_risk_idx in X_test.index:
                    sample_idx = X_test.index.get_loc(high_risk_idx)
                    explainer.explain_prediction(
                        X_test, 
                        sample_idx=sample_idx,
                        save_path='outputs/shap/example_high_risk.png'
                    )
            except Exception as e:
                logger.warning(f"Could not generate example explanation: {str(e)}")
            
            # Save SHAP values
            try:
                explainer.save_shap_values('outputs/shap/shap_values.npy')
            except Exception as e:
                logger.warning(f"Could not save SHAP values: {str(e)}")
                
        except Exception as e:
            logger.error(f"SHAP analysis failed: {str(e)}")
            print("⚠ SHAP analysis failed, continuing...")
    else:
        print("⚠ SHAP not available, skipping explainability analysis")
    
    # MLflow Logging
    if mlflow_tracker and mlflow_tracker.enabled:
        print("\n[BONUS] Logging to MLflow...")
        print("-" * 70)
        
        try:
            # Log each model
            for model_name, model in models.items():
                params = {
                    'model_type': model_name,
                    'n_features': len(feature_cols),
                    'use_smote': 'smote' in model_name
                }
                
                metrics = {
                    'accuracy': results[model_name]['accuracy'],
                    'f1_macro': results[model_name]['f1_macro'],
                    'f1_weighted': results[model_name]['f1_weighted'],
                    'roc_auc_ovr': results[model_name]['roc_auc_ovr'],
                    'precision_macro': results[model_name]['precision_macro'],
                    'recall_macro': results[model_name]['recall_macro']
                }
                
                mlflow_tracker.log_training_run(
                    model_name=model_name,
                    model=model,
                    params=params,
                    metrics=metrics,
                    artifacts=['outputs/evaluation/evaluation_results.json']
                )
            
            # Log model comparison
            log_model_comparison(mlflow_tracker, results)
            
            print("✓ Logged all runs to MLflow")
        except Exception as e:
            logger.error(f"MLflow logging failed: {str(e)}")
    
    # Final Summary
    print("\n" + "="*70)
    print("PHASE 2 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Artifacts:")
    print("  Models:")
    print("    - models/logistic_model.pkl")
    print("    - models/random_forest_model.pkl")
    if 'xgboost' in models:
        print("    - models/xgboost_model.pkl")
    if 'lightgbm' in models:
        print("    - models/lightgbm_model.pkl")
    print("    - models/production/best_model.pkl")
    print("    - models/production/model_metadata.json")
    print("\n  Evaluation:")
    print("    - outputs/evaluation/evaluation_results.json")
    print("    - outputs/evaluation/all_confusion_matrices.png")
    print("    - outputs/evaluation/roc_curves.png")
    print("\n  Explainability:")
    if SHAP_AVAILABLE:
        print("    - outputs/shap/shap_summary.png")
        print("    - outputs/shap/feature_importance.png")
        print("    - outputs/shap/top_features.csv")
    else:
        print("    - (SHAP not available)")
    
    print("\nModel Performance Summary:")
    for model_name, result in results.items():
        print(f"  {model_name}:")
        print(f"    Accuracy: {result['accuracy']:.4f} | F1-Macro: {result['f1_macro']:.4f} | AUC: {result['roc_auc_ovr']:.4f}")
    
    print("\nReady for Phase 3: Dashboard Development")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in Phase 2 execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

