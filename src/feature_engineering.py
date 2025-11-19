"""
Feature Engineering Module
Creates features from raw data for hospital readmission prediction
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer features for readmission prediction"""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            data: Preprocessed DataFrame
        """
        self.data = data.copy()
        self.feature_names = []
        
    def create_demographic_features(self) -> pd.DataFrame:
        """Create demographic features"""
        logger.info("Creating demographic features...")
        
        # Age: Extract numeric value from age ranges
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        self.data['age_numeric'] = self.data['age'].map(age_mapping)
        
        # Age groups
        self.data['age_group_young'] = (self.data['age_numeric'] < 40).astype(int)
        self.data['age_group_middle'] = ((self.data['age_numeric'] >= 40) & (self.data['age_numeric'] < 65)).astype(int)
        self.data['age_group_senior'] = (self.data['age_numeric'] >= 65).astype(int)
        
        # Gender encoding
        self.data['gender_male'] = (self.data['gender'] == 'Male').astype(int)
        self.data['gender_female'] = (self.data['gender'] == 'Female').astype(int)
        
        # Race encoding (one-hot)
        race_dummies = pd.get_dummies(self.data['race'], prefix='race', drop_first=False)
        self.data = pd.concat([self.data, race_dummies], axis=1)
        
        logger.info("Demographic features created")
        return self.data
    
    def create_clinical_features(self) -> pd.DataFrame:
        """Create clinical and diagnosis features"""
        logger.info("Creating clinical features...")
        
        # Diagnosis categories based on ICD-9 codes
        def categorize_diagnosis(diag):
            """Categorize diagnosis codes into major groups"""
            if pd.isna(diag) or diag == 'Unknown':
                return 'Unknown'
            
            diag_str = str(diag)
            
            # Try to extract numeric part
            try:
                if diag_str.startswith('V') or diag_str.startswith('E'):
                    return 'Other'
                
                code = float(diag_str)
                
                if 390 <= code < 460 or code == 785:
                    return 'Circulatory'
                elif 460 <= code < 520 or code == 786:
                    return 'Respiratory'
                elif 520 <= code < 580 or code == 787:
                    return 'Digestive'
                elif 250 <= code < 251:
                    return 'Diabetes'
                elif 800 <= code < 1000:
                    return 'Injury'
                elif 710 <= code < 740:
                    return 'Musculoskeletal'
                elif 580 <= code < 630 or code == 788:
                    return 'Genitourinary'
                elif 140 <= code < 240:
                    return 'Neoplasms'
                else:
                    return 'Other'
            except:
                return 'Other'
        
        # Categorize primary, secondary, and tertiary diagnoses
        self.data['diag_1_category'] = self.data['diag_1'].apply(categorize_diagnosis)
        self.data['diag_2_category'] = self.data['diag_2'].apply(categorize_diagnosis)
        self.data['diag_3_category'] = self.data['diag_3'].apply(categorize_diagnosis)
        
        # One-hot encode primary diagnosis
        diag_1_dummies = pd.get_dummies(self.data['diag_1_category'], prefix='primary_diag')
        self.data = pd.concat([self.data, diag_1_dummies], axis=1)
        
        # Comorbidity flags
        self.data['has_diabetes_complication'] = (self.data['diag_1_category'] == 'Diabetes').astype(int)
        self.data['has_circulatory'] = (
            (self.data['diag_1_category'] == 'Circulatory') |
            (self.data['diag_2_category'] == 'Circulatory') |
            (self.data['diag_3_category'] == 'Circulatory')
        ).astype(int)
        self.data['has_respiratory'] = (
            (self.data['diag_1_category'] == 'Respiratory') |
            (self.data['diag_2_category'] == 'Respiratory') |
            (self.data['diag_3_category'] == 'Respiratory')
        ).astype(int)
        
        # Number of diagnoses is already in the dataset
        # Ensure it's numeric
        if 'number_diagnoses' in self.data.columns:
            self.data['number_diagnoses'] = pd.to_numeric(self.data['number_diagnoses'], errors='coerce')
        
        logger.info("Clinical features created")
        return self.data
    
    def create_utilization_features(self) -> pd.DataFrame:
        """Create healthcare utilization features"""
        logger.info("Creating utilization features...")
        
        # Time in hospital is already present
        if 'time_in_hospital' in self.data.columns:
            self.data['time_in_hospital'] = pd.to_numeric(self.data['time_in_hospital'], errors='coerce')
            self.data['long_stay'] = (self.data['time_in_hospital'] > 7).astype(int)
        
        # Previous admissions
        if 'number_inpatient' in self.data.columns:
            self.data['number_inpatient'] = pd.to_numeric(self.data['number_inpatient'], errors='coerce')
            self.data['has_prev_inpatient'] = (self.data['number_inpatient'] > 0).astype(int)
            self.data['frequent_inpatient'] = (self.data['number_inpatient'] >= 2).astype(int)
        
        if 'number_outpatient' in self.data.columns:
            self.data['number_outpatient'] = pd.to_numeric(self.data['number_outpatient'], errors='coerce')
            self.data['has_prev_outpatient'] = (self.data['number_outpatient'] > 0).astype(int)
        
        if 'number_emergency' in self.data.columns:
            self.data['number_emergency'] = pd.to_numeric(self.data['number_emergency'], errors='coerce')
            self.data['has_prev_emergency'] = (self.data['number_emergency'] > 0).astype(int)
        
        # Total previous encounters
        self.data['total_prev_encounters'] = (
            self.data['number_inpatient'].fillna(0) +
            self.data['number_outpatient'].fillna(0) +
            self.data['number_emergency'].fillna(0)
        )
        
        # Admission type features
        if 'admission_type_id' in self.data.columns:
            self.data['emergency_admission'] = (self.data['admission_type_id'] == 1).astype(int)
            self.data['urgent_admission'] = (self.data['admission_type_id'] == 2).astype(int)
            self.data['elective_admission'] = (self.data['admission_type_id'] == 3).astype(int)
        
        # Discharge disposition features
        if 'discharge_disposition_id' in self.data.columns:
            self.data['discharged_to_home'] = (self.data['discharge_disposition_id'] == 1).astype(int)
            self.data['transferred'] = (
                self.data['discharge_disposition_id'].isin([2, 3, 4, 5, 10, 22, 23, 24, 27, 28, 29])
            ).astype(int)
        
        logger.info("Utilization features created")
        return self.data
    
    def create_medication_features(self) -> pd.DataFrame:
        """Create medication-related features"""
        logger.info("Creating medication features...")
        
        # Medication columns
        med_columns = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
            'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
            'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
            'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
            'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        # Count diabetes medications used
        diabetes_med_count = 0
        for med in med_columns:
            if med in self.data.columns:
                # Medication is changed if value is not 'No' or 'Steady'
                med_changed = ~self.data[med].isin(['No', 'Steady'])
                diabetes_med_count += med_changed.astype(int)
        
        self.data['diabetes_med_count'] = diabetes_med_count
        
        # Medication change indicator
        if 'change' in self.data.columns:
            self.data['medication_changed'] = (self.data['change'] == 'Ch').astype(int)
        
        # Diabetes medication indicator
        if 'diabetesMed' in self.data.columns:
            self.data['on_diabetes_med'] = (self.data['diabetesMed'] == 'Yes').astype(int)
        
        # Insulin usage
        if 'insulin' in self.data.columns:
            self.data['insulin_prescribed'] = (~self.data['insulin'].isin(['No'])).astype(int)
            self.data['insulin_up'] = (self.data['insulin'] == 'Up').astype(int)
            self.data['insulin_down'] = (self.data['insulin'] == 'Down').astype(int)
        
        # Metformin usage
        if 'metformin' in self.data.columns:
            self.data['metformin_prescribed'] = (~self.data['metformin'].isin(['No'])).astype(int)
        
        # Total number of medications
        if 'num_medications' in self.data.columns:
            self.data['num_medications'] = pd.to_numeric(self.data['num_medications'], errors='coerce')
            self.data['high_medication_count'] = (self.data['num_medications'] > 15).astype(int)
        
        logger.info("Medication features created")
        return self.data
    
    def create_lab_features(self) -> pd.DataFrame:
        """Create lab and procedure features"""
        logger.info("Creating lab features...")
        
        # Number of lab procedures
        if 'num_lab_procedures' in self.data.columns:
            self.data['num_lab_procedures'] = pd.to_numeric(self.data['num_lab_procedures'], errors='coerce')
            self.data['high_lab_count'] = (self.data['num_lab_procedures'] > 50).astype(int)
        
        # Number of procedures
        if 'num_procedures' in self.data.columns:
            self.data['num_procedures'] = pd.to_numeric(self.data['num_procedures'], errors='coerce')
            self.data['had_procedures'] = (self.data['num_procedures'] > 0).astype(int)
        
        # Glucose serum test result
        if 'max_glu_serum' in self.data.columns:
            self.data['glucose_high'] = (self.data['max_glu_serum'] == '>200').astype(int)
            self.data['glucose_normal'] = (self.data['max_glu_serum'] == 'Norm').astype(int)
            self.data['glucose_tested'] = (~self.data['max_glu_serum'].isin(['None'])).astype(int)
        
        # A1C test result
        if 'A1Cresult' in self.data.columns:
            self.data['a1c_high'] = (self.data['A1Cresult'] == '>8').astype(int)
            self.data['a1c_normal'] = (self.data['A1Cresult'] == 'Norm').astype(int)
            self.data['a1c_tested'] = (~self.data['A1Cresult'].isin(['None'])).astype(int)
        
        logger.info("Lab features created")
        return self.data
    
    def create_interaction_features(self) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        # Age × previous inpatient admissions
        if 'age_numeric' in self.data.columns and 'number_inpatient' in self.data.columns:
            self.data['age_x_inpatient'] = self.data['age_numeric'] * self.data['number_inpatient'].fillna(0)
        
        # Time in hospital × number of procedures
        if 'time_in_hospital' in self.data.columns and 'num_procedures' in self.data.columns:
            self.data['stay_x_procedures'] = self.data['time_in_hospital'] * self.data['num_procedures'].fillna(0)
        
        # Number of medications × medication change
        if 'num_medications' in self.data.columns and 'medication_changed' in self.data.columns:
            self.data['med_count_x_change'] = self.data['num_medications'].fillna(0) * self.data['medication_changed']
        
        logger.info("Interaction features created")
        return self.data
    
    def get_feature_columns(self, exclude_original: bool = True) -> List[str]:
        """
        Get list of engineered feature columns
        
        Args:
            exclude_original: If True, exclude original raw columns
        """
        # Core engineered features
        core_features = [
            'age_numeric', 'age_group_young', 'age_group_middle', 'age_group_senior',
            'gender_male', 'gender_female',
            'time_in_hospital', 'long_stay',
            'number_inpatient', 'number_outpatient', 'number_emergency',
            'has_prev_inpatient', 'frequent_inpatient', 'has_prev_outpatient', 'has_prev_emergency',
            'total_prev_encounters',
            'emergency_admission', 'urgent_admission', 'elective_admission',
            'discharged_to_home', 'transferred',
            'num_medications', 'num_procedures', 'num_lab_procedures',
            'high_medication_count', 'high_lab_count', 'had_procedures',
            'number_diagnoses',
            'diabetes_med_count', 'medication_changed', 'on_diabetes_med',
            'insulin_prescribed', 'insulin_up', 'insulin_down', 'metformin_prescribed',
            'glucose_high', 'glucose_normal', 'glucose_tested',
            'a1c_high', 'a1c_normal', 'a1c_tested',
            'has_diabetes_complication', 'has_circulatory', 'has_respiratory',
            'age_x_inpatient', 'stay_x_procedures', 'med_count_x_change'
        ]
        
        # Add race dummies
        race_cols = [col for col in self.data.columns if col.startswith('race_')]
        core_features.extend(race_cols)
        
        # Add diagnosis dummies
        diag_cols = [col for col in self.data.columns if col.startswith('primary_diag_')]
        core_features.extend(diag_cols)
        
        # Filter to only columns that exist in data
        feature_columns = [col for col in core_features if col in self.data.columns]
        
        logger.info(f"Total engineered features: {len(feature_columns)}")
        return feature_columns
    
    def engineer_all_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute full feature engineering pipeline
        
        Returns:
            Tuple of (data with features, list of feature column names)
        """
        logger.info("Starting feature engineering pipeline...")
        
        self.create_demographic_features()
        self.create_clinical_features()
        self.create_utilization_features()
        self.create_medication_features()
        self.create_lab_features()
        self.create_interaction_features()
        
        feature_columns = self.get_feature_columns()
        
        logger.info(f"Feature engineering complete. Created {len(feature_columns)} features")
        return self.data, feature_columns


if __name__ == "__main__":
    # Example usage
    import pickle
    
    # Load preprocessed data
    with open('data/processed/cleaned_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    # Engineer features
    engineer = FeatureEngineer(data)
    featured_data, feature_cols = engineer.engineer_all_features()
    
    print(f"\nEngineered {len(feature_cols)} features")
    print(f"Final dataset shape: {featured_data.shape}")
    print(f"\nSample features: {feature_cols[:10]}")
    
    # Save featured data
    featured_data.to_pickle('data/processed/featured_data.pkl')
    with open('data/processed/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("\nFeatured data saved to data/processed/featured_data.pkl")

