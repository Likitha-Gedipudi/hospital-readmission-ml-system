"""
Data Splitting Module
Splits data into train, validation, and test sets with stratification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train/val/test sets"""
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'readmitted_encoded',
                 test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        """
        Initialize data splitter
        
        Args:
            data: DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.data = data
        self.target_col = target_col
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
    def split_data(self, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into stratified train, validation, and test sets
        
        Args:
            feature_columns: List of feature column names to include
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test sets...")
        
        # Ensure target column exists
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        # Select features and target
        X = self.data[feature_columns].copy()
        y = self.data[self.target_col].copy()
        
        # Keep track of indices for joining back
        X['original_index'] = self.data.index
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: separate train and validation
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        # Add target back to dataframes
        X_train[self.target_col] = y_train
        X_val[self.target_col] = y_val
        X_test[self.target_col] = y_test
        
        logger.info(f"Train set size: {len(X_train)} ({len(X_train)/len(self.data)*100:.1f}%)")
        logger.info(f"Val set size: {len(X_val)} ({len(X_val)/len(self.data)*100:.1f}%)")
        logger.info(f"Test set size: {len(X_test)} ({len(X_test)/len(self.data)*100:.1f}%)")
        
        # Log class distribution
        logger.info("\nClass distribution in splits:")
        logger.info(f"Train: {y_train.value_counts().sort_index().to_dict()}")
        logger.info(f"Val: {y_val.value_counts().sort_index().to_dict()}")
        logger.info(f"Test: {y_test.value_counts().sort_index().to_dict()}")
        
        return X_train, X_val, X_test
    
    def save_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    output_dir: str = 'data/processed'):
        """Save train, val, test splits to disk"""
        logger.info(f"Saving data splits to {output_dir}...")
        
        train_df.to_pickle(f'{output_dir}/train.pkl')
        val_df.to_pickle(f'{output_dir}/val.pkl')
        test_df.to_pickle(f'{output_dir}/test.pkl')
        
        logger.info("Data splits saved successfully")
    
    def save_feature_schema(self, feature_columns: List[str], output_path: str = 'data/feature_schema.json'):
        """Save feature schema for reference"""
        schema = {
            'feature_columns': feature_columns,
            'n_features': len(feature_columns),
            'target_column': self.target_col,
            'target_classes': {
                0: 'NO',
                1: '<30',
                2: '>30'
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(schema, f, indent=2)
        
        logger.info(f"Feature schema saved to {output_path}")


def main():
    """Main execution function"""
    import pickle
    
    # Load featured data
    logger.info("Loading featured data...")
    with open('data/processed/featured_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    with open('data/processed/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    # Initialize splitter
    splitter = DataSplitter(data, target_col='readmitted_encoded')
    
    # Split data
    train_df, val_df, test_df = splitter.split_data(feature_columns)
    
    # Save splits
    splitter.save_splits(train_df, val_df, test_df)
    
    # Save schema
    splitter.save_feature_schema(feature_columns)
    
    logger.info("\nData splitting complete!")
    
    # Print summary
    print("\n" + "="*50)
    print("DATA SPLITTING SUMMARY")
    print("="*50)
    print(f"Total samples: {len(data)}")
    print(f"Total features: {len(feature_columns)}")
    print(f"\nTrain: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print("="*50)


if __name__ == "__main__":
    main()

