import pandas as pd
import numpy as np

def select_features(df):
    """Select and engineer features"""
    print("\n" + "="*60)
    print("FEATURE SELECTION & ENGINEERING")
    print("="*60)
    
    # Select numeric features that are most relevant
    feature_cols = [
        'LotArea',
        'YearBuilt',
        '1stFlrSF',
        '2ndFlrSF',
        'BsmtFinSF1',
        'GrLivArea',
        'GarageCars',
        'TotRmsAbvGrd'
    ]
    
    # Create new engineered feature
    df['HasBasement'] = (df['BsmtFinSF1'] > 0).astype(int)
    feature_cols.append('HasBasement')
    
    print(f"\nSelected features ({len(feature_cols)} total):")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i}. {feat}")
    
    # Prepare X and y
    X = df[feature_cols].copy()
    y = df['SalePrice'].copy()
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    
    print(f"\nFeature statistics:")
    print(X.describe())
    
    return X, y, feature_cols