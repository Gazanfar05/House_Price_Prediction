import pandas as pd
import numpy as np

def clean_data(df):
    """Clean and preprocess data"""
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    # Remove rows with missing target variable
    print(f"\nTarget variable (SalePrice) - Missing: {df['SalePrice'].isnull().sum()}")
    df = df.dropna(subset=['SalePrice'])
    print(f"Rows after removing missing SalePrice: {len(df)}")
    
    # Fill numeric missing values with median
    print("\nFilling numeric missing values with median:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna(df[col].median(), inplace=True)
            print(f"  ✓ {col}: filled {missing_count} values")
    
    # Fill categorical missing values with mode
    print("\nFilling categorical missing values with mode:")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            print(f"  ✓ {col}: filled {missing_count} values")
    
    print(f"\nTotal missing values after cleaning: {df.isnull().sum().sum()}")
    print(f"✓ Data cleaning completed!")
    
    return df