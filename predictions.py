import numpy as np

def make_predictions(model, X_test, y_test):
    """Make and display example predictions"""
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    # Select diverse sample indices
    sample_indices = [0, len(X_test)//4, len(X_test)//2, 3*len(X_test)//4, -1]
    
    print("\nSample Predictions:")
    print("-" * 90)
    print(f"{'#':<3} {'Actual Price':<15} {'Predicted Price':<18} "
          f"{'Error ($)':<15} {'Error (%)':<12}")
    print("-" * 90)
    
    total_error = 0
    for i, idx in enumerate(sample_indices, 1):
        actual = y_test.iloc[idx]
        predicted = y_pred[idx]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        total_error += error_pct
        
        print(f"{i:<3} ${actual:>13,.0f} ${predicted:>16,.0f} "
              f"${error:>13,.0f} {error_pct:>10.2f}%")
    
    avg_error = total_error / len(sample_indices)
    print("-" * 90)
    print(f"Average Error: {avg_error:.2f}%")
    
    return y_pred

def predict_custom_house(model, feature_cols, values_dict):
    """Make prediction for custom house"""
    print("\n" + "="*60)
    print("CUSTOM HOUSE PRICE PREDICTION")
    print("="*60)
    
    try:
        # Create feature array in correct order
        features = np.array([values_dict[col] for col in feature_cols])
        
        print(f"\nInput Features:")
        for col, val in zip(feature_cols, features):
            print(f"  {col}: {val}")
        
        # Make prediction
        predicted_price = model.predict([features])[0]
        
        print(f"\n✓ Predicted House Price: ${predicted_price:,.2f}")
        
        return predicted_price
    
    except KeyError as e:
        print(f"ERROR: Missing required feature: {e}")
        print(f"Required features: {feature_cols}")
        return None