import pandas as pd
import numpy as np

def load_data(filepath):
    """Load housing dataset from CSV"""
    print("Loading housing dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def explore_data(df):
    """Explore and display dataset information"""
    print("\n" + "="*60)
    print("DATA EXPLORATION")
    print("="*60)
    
    print(f"\nFirst 5 rows:\n{df.head()}")
    
    print(f"\n\nDataset Info:")
    print(df.info())
    
    print(f"\n\nMissing values:\n{df.isnull().sum().sort_values(ascending=False)}")
    
    print(f"\n\nBasic Statistics:\n{df.describe()}")
    
    print(f"\n\nData Types:\n{df.dtypes}")
    
    return df