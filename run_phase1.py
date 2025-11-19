"""
Phase 1 Execution Script
Runs data preprocessing, feature engineering, and data splitting
"""

import sys
import os

# Add src to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from data_splitting import DataSplitter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Execute Phase 1: Data preparation"""
    
    print("\n" + "="*70)
    print("PHASE 1: DATA COLLECTION & PREPARATION")
    print("="*70)
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1/3] Data Preprocessing...")
    print("-" * 70)
    
    preprocessor = DataPreprocessor(
        data_path='data/raw/diabetic_data.csv',
        mapping_path='data/raw/IDS_mapping.csv'
    )
    
    cleaned_data = preprocessor.preprocess()
    
    # Save cleaned data
    cleaned_data.to_pickle('data/processed/cleaned_data.pkl')
    print(f"\n✓ Cleaned data saved to data/processed/cleaned_data.pkl")
    
    # Print summary
    summary = preprocessor.get_preprocessing_summary()
    print(f"\nPreprocessing Summary:")
    print(f"  - Total records: {summary['total_records']:,}")
    print(f"  - Total features: {summary['total_features']}")
    print(f"  - Target distribution: {summary['target_distribution']}")
    
    # Step 2: Feature Engineering
    print("\n[STEP 2/3] Feature Engineering...")
    print("-" * 70)
    
    engineer = FeatureEngineer(cleaned_data)
    featured_data, feature_cols = engineer.engineer_all_features()
    
    # Save featured data and feature columns
    featured_data.to_pickle('data/processed/featured_data.pkl')
    
    import pickle
    with open('data/processed/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n✓ Featured data saved to data/processed/featured_data.pkl")
    print(f"✓ Feature columns saved to data/processed/feature_columns.pkl")
    print(f"\nFeature Engineering Summary:")
    print(f"  - Engineered {len(feature_cols)} features")
    print(f"  - Final dataset shape: {featured_data.shape}")
    print(f"  - Sample features: {feature_cols[:10]}")
    
    # Step 3: Data Splitting
    print("\n[STEP 3/3] Data Splitting...")
    print("-" * 70)
    
    splitter = DataSplitter(featured_data, target_col='readmitted_encoded')
    train_df, val_df, test_df = splitter.split_data(feature_cols)
    
    # Save splits
    splitter.save_splits(train_df, val_df, test_df)
    splitter.save_feature_schema(feature_cols)
    
    print(f"\n✓ Train/Val/Test splits saved to data/processed/")
    print(f"✓ Feature schema saved to data/feature_schema.json")
    
    print(f"\nData Splitting Summary:")
    print(f"  - Train: {len(train_df):,} samples ({len(train_df)/len(featured_data)*100:.1f}%)")
    print(f"  - Val: {len(val_df):,} samples ({len(val_df)/len(featured_data)*100:.1f}%)")
    print(f"  - Test: {len(test_df):,} samples ({len(test_df)/len(featured_data)*100:.1f}%)")
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Files:")
    print("  1. data/processed/cleaned_data.pkl")
    print("  2. data/processed/featured_data.pkl")
    print("  3. data/processed/feature_columns.pkl")
    print("  4. data/processed/train.pkl")
    print("  5. data/processed/val.pkl")
    print("  6. data/processed/test.pkl")
    print("  7. data/feature_schema.json")
    print("\nReady for Phase 2: Model Development")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error in Phase 1 execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

