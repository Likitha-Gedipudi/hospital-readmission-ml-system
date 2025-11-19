"""
MLflow Utilities Module
Handles experiment tracking and model registry with MLflow
"""

import logging
from typing import Dict, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Install with: pip install mlflow")


class MLflowTracker:
    """Track experiments and models with MLflow"""
    
    def __init__(self, experiment_name: str = "Hospital_Readmission", 
                 tracking_uri: str = None):
        """
        Initialize MLflow tracker
        
        Args:
            experiment_name: Name of MLflow experiment
            tracking_uri: URI for MLflow tracking server
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Tracking disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created MLflow experiment: {experiment_name}")
        except:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name: str = None):
        """Start MLflow run"""
        if not self.enabled:
            return None
        
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def end_run(self):
        """End MLflow run"""
        if not self.enabled:
            return
        
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if not self.enabled:
            return
        
        for key, value in params.items():
            try:
                mlflow.log_param(key, value)
            except Exception as e:
                logger.warning(f"Could not log param {key}: {str(e)}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        if not self.enabled:
            return
        
        for key, value in metrics.items():
            try:
                mlflow.log_metric(key, value, step=step)
            except Exception as e:
                logger.warning(f"Could not log metric {key}: {str(e)}")
    
    def log_artifact(self, artifact_path: str):
        """Log artifact file"""
        if not self.enabled:
            return
        
        if os.path.exists(artifact_path):
            mlflow.log_artifact(artifact_path)
            logger.info(f"Logged artifact: {artifact_path}")
        else:
            logger.warning(f"Artifact not found: {artifact_path}")
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log sklearn model"""
        if not self.enabled:
            return
        
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info(f"Logged model to {artifact_path}")
    
    def register_model(self, model_uri: str, model_name: str):
        """Register model in MLflow Model Registry"""
        if not self.enabled:
            return None
        
        try:
            model_version = mlflow.register_model(model_uri, model_name)
            logger.info(f"Registered model {model_name} version {model_version.version}")
            return model_version
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            return None
    
    def transition_model_stage(self, model_name: str, version: int, stage: str):
        """
        Transition model to a stage
        
        Args:
            model_name: Name of registered model
            version: Model version number
            stage: Stage name ('Staging', 'Production', 'Archived')
        """
        if not self.enabled:
            return
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Error transitioning model: {str(e)}")
    
    def log_training_run(self, model_name: str, model, params: Dict, 
                        metrics: Dict, artifacts: list = None):
        """
        Complete training run logging
        
        Args:
            model_name: Name of the model
            model: Trained model
            params: Model parameters
            metrics: Performance metrics
            artifacts: List of artifact paths to log
        """
        if not self.enabled:
            logger.info(f"MLflow disabled. Would log: {model_name}")
            return
        
        with mlflow.start_run(run_name=model_name):
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model
            self.log_model(model, artifact_path="model")
            
            # Log artifacts
            if artifacts:
                for artifact_path in artifacts:
                    self.log_artifact(artifact_path)
            
            # Add tags
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("framework", "scikit-learn")
            
            logger.info(f"✓ Logged training run for {model_name}")


def log_model_comparison(tracker: MLflowTracker, results: Dict[str, Dict]):
    """
    Log comparison of multiple models
    
    Args:
        tracker: MLflowTracker instance
        results: Dictionary of model results from evaluation
    """
    if not tracker.enabled:
        return
    
    with mlflow.start_run(run_name="model_comparison"):
        # Log all model metrics for comparison
        for model_name, metrics in results.items():
            mlflow.log_metric(f"{model_name}_accuracy", metrics['accuracy'])
            mlflow.log_metric(f"{model_name}_f1_macro", metrics['f1_macro'])
            mlflow.log_metric(f"{model_name}_f1_weighted", metrics['f1_weighted'])
            mlflow.log_metric(f"{model_name}_roc_auc", metrics['roc_auc_ovr'])
        
        # Find and log best model
        best_model = max(results.items(), key=lambda x: x[1]['f1_macro'])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_f1_macro", best_model[1]['f1_macro'])
        
        logger.info("✓ Logged model comparison")


if __name__ == "__main__":
    # Example usage
    tracker = MLflowTracker(experiment_name="Hospital_Readmission_Test")
    
    # Simulate training run
    with mlflow.start_run(run_name="test_model"):
        tracker.log_params({
            "model_type": "xgboost",
            "max_depth": 6,
            "learning_rate": 0.1
        })
        
        tracker.log_metrics({
            "accuracy": 0.85,
            "f1_macro": 0.72,
            "roc_auc": 0.78
        })
        
        print("✓ MLflow tracking test complete")

