import joblib
import json

def save_model(model, filename='house_price_model.pkl'):
    """Save trained model to disk"""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    joblib.dump(model, filename)
    print(f"\n✓ Model saved as: '{filename}'")

def load_model(filename='house_price_model.pkl'):
    """Load model from disk"""
    try:
        model = joblib.load(filename)
        print(f"✓ Model loaded from: '{filename}'")
        return model
    except FileNotFoundError:
        print(f"ERROR: Model file '{filename}' not found!")
        return None

def save_model_info(model, feature_cols, metrics, filename='model_info.json'):
    """Save model information as JSON"""
    info = {
        'features': feature_cols,
        'intercept': float(model.intercept_),
        'coefficients': {feat: float(coef) for feat, coef in zip(feature_cols, model.coef_)},
        'metrics': {
            'test_rmse': float(metrics['test_rmse']),
            'test_r2': float(metrics['test_r2']),
            'test_mae': float(metrics['test_mae'])
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"✓ Model info saved as: '{filename}'")