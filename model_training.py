import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"\nTotal samples: {len(X)}")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train Linear Regression model"""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✓ Model trained successfully!")
    print(f"Number of features: {len(model.coef_)}")
    print(f"Model intercept: ${model.intercept_:,.2f}")
    
    return model