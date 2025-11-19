"""
Model Training Module
Trains multiple machine learning models for hospital readmission prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
from typing import Dict, Tuple, Any
import joblib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")


class ModelTrainer:
    """Train multiple models for readmission prediction"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.training_times = {}
        self.scaler = StandardScaler()
        
    def get_logistic_regression(self, use_smote: bool = True) -> Pipeline:
        """Create logistic regression pipeline"""
        if use_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
                ('classifier', LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(
                    multi_class='multinomial',
                    solver='lbfgs',
                    max_iter=1000,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ])
        return pipeline
    
    def get_random_forest(self, use_smote: bool = False) -> Pipeline:
        """Create random forest pipeline"""
        if use_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                ))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=0
                ))
            ])
        return pipeline
    
    def get_xgboost(self, use_smote: bool = False) -> Pipeline:
        """Create XGBoost pipeline"""
        if not XGBOOST_AVAILABLE:
            logger.error("XGBoost not installed")
            return None
        
        if use_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
                ('classifier', xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                ))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbosity=0
                ))
            ])
        return pipeline
    
    def get_lightgbm(self, use_smote: bool = False) -> Pipeline:
        """Create LightGBM pipeline"""
        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not installed")
            return None
        
        if use_smote:
            pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=self.random_state, k_neighbors=5)),
                ('classifier', lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ))
            ])
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1,
                    verbose=-1
                ))
            ])
        return pipeline
    
    def train_model(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                   use_smote: bool = False) -> Pipeline:
        """
        Train a specific model
        
        Args:
            model_name: Name of model ('logistic', 'random_forest', 'xgboost', 'lightgbm')
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE for balancing
        
        Returns:
            Trained pipeline
        """
        logger.info(f"Training {model_name} (SMOTE: {use_smote})...")
        
        # Get model
        if model_name == 'logistic':
            model = self.get_logistic_regression(use_smote)
        elif model_name == 'random_forest':
            model = self.get_random_forest(use_smote)
        elif model_name == 'xgboost':
            model = self.get_xgboost(use_smote)
        elif model_name == 'lightgbm':
            model = self.get_lightgbm(use_smote)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        if model is None:
            return None
        
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Store
        key = f"{model_name}_smote" if use_smote else model_name
        self.models[key] = model
        self.training_times[key] = training_time
        
        logger.info(f"✓ {model_name} trained in {training_time:.2f} seconds")
        return model
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        use_smote: bool = False) -> Dict[str, Pipeline]:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE
        
        Returns:
            Dictionary of trained models
        """
        logger.info(f"\n{'='*70}")
        logger.info("TRAINING ALL MODELS")
        logger.info(f"{'='*70}")
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Number of features: {X_train.shape[1]}")
        logger.info(f"Class distribution: {y_train.value_counts().sort_index().to_dict()}")
        logger.info(f"Using SMOTE: {use_smote}")
        
        models_to_train = ['logistic', 'random_forest']
        
        if XGBOOST_AVAILABLE:
            models_to_train.append('xgboost')
        if LIGHTGBM_AVAILABLE:
            models_to_train.append('lightgbm')
        
        for model_name in models_to_train:
            try:
                self.train_model(model_name, X_train, y_train, use_smote)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
        
        logger.info(f"\n✓ Trained {len(self.models)} models successfully")
        return self.models
    
    def save_model(self, model_name: str, output_path: str):
        """Save a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], output_path)
        logger.info(f"Model {model_name} saved to {output_path}")
    
    def save_all_models(self, output_dir: str = 'models'):
        """Save all trained models"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            output_path = f"{output_dir}/{model_name}_model.pkl"
            joblib.dump(model, output_path)
            logger.info(f"✓ Saved {model_name} to {output_path}")
    
    def get_model(self, model_name: str) -> Pipeline:
        """Get a trained model"""
        return self.models.get(model_name)


if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load data
    with open('data/processed/train.pkl', 'rb') as f:
        train_df = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    
    X_train = train_df[feature_cols]
    y_train = train_df['readmitted_encoded']
    
    # Train models
    trainer = ModelTrainer(random_state=42)
    models = trainer.train_all_models(X_train, y_train, use_smote=False)
    
    # Save models
    trainer.save_all_models('models')
    
    print("\n✓ Model training complete!")

