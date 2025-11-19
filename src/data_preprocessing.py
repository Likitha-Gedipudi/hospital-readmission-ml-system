"""
Data Preprocessing Module
Handles data cleaning, missing value imputation, and initial transformations
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocesses hospital readmission data"""
    
    def __init__(self, data_path: str, mapping_path: str):
        """
        Initialize preprocessor with data paths
        
        Args:
            data_path: Path to diabetic_data.csv
            mapping_path: Path to IDS_mapping.csv
        """
        self.data_path = data_path
        self.mapping_path = mapping_path
        self.data = None
        self.mappings = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV files"""
        logger.info(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
        
        # Load mappings
        logger.info(f"Loading ID mappings from {self.mapping_path}")
        self._load_mappings()
        
        return self.data
    
    def _load_mappings(self):
        """Parse the IDS_mapping.csv file to extract mappings"""
        with open(self.mapping_path, 'r') as f:
            lines = f.readlines()
        
        self.mappings = {}
        current_map = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            
            key, value = parts[0], parts[1]
            
            # Check if this is a header line
            if value == 'description':
                current_map = key
                self.mappings[current_map] = {}
            elif current_map and key:
                self.mappings[current_map][int(key) if key.isdigit() else key] = value
        
        logger.info(f"Loaded mappings for: {list(self.mappings.keys())}")
    
    def remove_invalid_admissions(self) -> pd.DataFrame:
        """Remove invalid admission types and deceased patients"""
        initial_count = len(self.data)
        
        # Remove expired patients (discharge_disposition_id: 11, 19, 20, 21)
        expired_ids = [11, 19, 20, 21]
        self.data = self.data[~self.data['discharge_disposition_id'].isin(expired_ids)]
        
        # Remove newborn admissions (admission_type_id: 4)
        self.data = self.data[self.data['admission_type_id'] != 4]
        
        logger.info(f"Removed {initial_count - len(self.data)} invalid admissions")
        logger.info(f"Remaining records: {len(self.data)}")
        
        return self.data
    
    def handle_missing_values(self) -> pd.DataFrame:
        """Handle missing values and '?' placeholders"""
        # Replace '?' with NaN
        self.data = self.data.replace('?', np.nan)
        
        # Drop columns with >95% missing values
        missing_pct = (self.data.isna().sum() / len(self.data) * 100).sort_values(ascending=False)
        logger.info("Missing value percentages:")
        logger.info(missing_pct[missing_pct > 0].head(10))
        
        # Drop weight (97% missing) and payer_code (40% missing)
        columns_to_drop = ['weight', 'payer_code']
        self.data = self.data.drop(columns=columns_to_drop, errors='ignore')
        logger.info(f"Dropped columns: {columns_to_drop}")
        
        # Impute medical_specialty with 'Unknown'
        if 'medical_specialty' in self.data.columns:
            self.data['medical_specialty'].fillna('Unknown', inplace=True)
        
        # Impute diagnosis columns with 'Unknown'
        for col in ['diag_1', 'diag_2', 'diag_3']:
            if col in self.data.columns:
                self.data[col].fillna('Unknown', inplace=True)
        
        # Impute race with 'Unknown'
        if 'race' in self.data.columns:
            self.data['race'].fillna('Unknown', inplace=True)
        
        logger.info("Missing value imputation completed")
        return self.data
    
    def encode_target_variable(self) -> pd.DataFrame:
        """Encode readmitted variable: 0=NO, 1=<30, 2=>30"""
        if 'readmitted' in self.data.columns:
            target_mapping = {'NO': 0, '<30': 1, '>30': 2}
            self.data['readmitted_encoded'] = self.data['readmitted'].map(target_mapping)
            
            # Check for any unmapped values
            if self.data['readmitted_encoded'].isna().any():
                logger.warning(f"Found unmapped readmitted values: {self.data['readmitted'].unique()}")
            
            logger.info("Target variable encoding:")
            logger.info(self.data['readmitted_encoded'].value_counts().sort_index())
        
        return self.data
    
    def map_categorical_ids(self) -> pd.DataFrame:
        """Map ID columns to descriptive labels"""
        if self.mappings:
            # Map admission_type_id
            if 'admission_type_id' in self.data.columns and 'admission_type_id' in self.mappings:
                self.data['admission_type'] = self.data['admission_type_id'].map(self.mappings['admission_type_id'])
                self.data['admission_type'].fillna('Unknown', inplace=True)
            
            # Map discharge_disposition_id
            if 'discharge_disposition_id' in self.data.columns and 'discharge_disposition_id' in self.mappings:
                self.data['discharge_disposition'] = self.data['discharge_disposition_id'].map(
                    self.mappings['discharge_disposition_id']
                )
                self.data['discharge_disposition'].fillna('Unknown', inplace=True)
            
            # Map admission_source_id
            if 'admission_source_id' in self.data.columns and 'admission_source_id' in self.mappings:
                self.data['admission_source'] = self.data['admission_source_id'].map(
                    self.mappings['admission_source_id']
                )
                self.data['admission_source'].fillna('Unknown', inplace=True)
        
        logger.info("Categorical ID mapping completed")
        return self.data
    
    def get_clean_data(self) -> pd.DataFrame:
        """Get the cleaned dataset"""
        return self.data
    
    def preprocess(self) -> pd.DataFrame:
        """Execute full preprocessing pipeline"""
        logger.info("Starting full preprocessing pipeline...")
        
        self.load_data()
        self.remove_invalid_admissions()
        self.handle_missing_values()
        self.map_categorical_ids()
        self.encode_target_variable()
        
        logger.info(f"Preprocessing complete. Final dataset shape: {self.data.shape}")
        return self.data
    
    def get_preprocessing_summary(self) -> Dict:
        """Get summary statistics of preprocessing"""
        if self.data is None:
            return {}
        
        return {
            'total_records': len(self.data),
            'total_features': len(self.data.columns),
            'target_distribution': self.data['readmitted_encoded'].value_counts().to_dict() if 'readmitted_encoded' in self.data.columns else {},
            'missing_values': self.data.isna().sum().sum(),
            'numeric_features': len(self.data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(self.data.select_dtypes(include=['object']).columns)
        }


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(
        data_path='data/raw/diabetic_data.csv',
        mapping_path='data/raw/IDS_mapping.csv'
    )
    
    cleaned_data = preprocessor.preprocess()
    print("\nPreprocessing Summary:")
    print(preprocessor.get_preprocessing_summary())
    
    # Save cleaned data
    cleaned_data.to_pickle('data/processed/cleaned_data.pkl')
    print("\nCleaned data saved to data/processed/cleaned_data.pkl")

