import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate Mean Absolute Error
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
    # Display results
    print("\nPerformance Metrics:")
    print("-" * 60)
    print(f"{'Metric':<25} {'Train':<20} {'Test':<20}")
    print("-" * 60)
    print(f"{'RMSE (Root Mean Sq Error)':<25} ${train_rmse:>18,.2f} ${test_rmse:>18,.2f}")
    print(f"{'R² Score':<25} {train_r2:>20.4f} {test_r2:>20.4f}")
    print(f"{'MAE (Mean Absolute Error)':<25} ${train_mae:>18,.2f} ${test_mae:>18,.2f}")
    print("-" * 60)
    
    # Interpretation
    print("\nInterpretation:")
    print(f"  • Train R²: {train_r2:.4f} - Model explains {train_r2*100:.2f}% of training variance")
    print(f"  • Test R²: {test_r2:.4f} - Model explains {test_r2*100:.2f}% of test variance")
    print(f"  • Test RMSE: ${test_rmse:,.2f} - Average prediction error on test set")
    
    metrics = {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }
    
    return metrics

def interpret_coefficients(model, feature_cols):
    """Interpret and display model coefficients"""
    print("\n" + "="*60)
    print("COEFFICIENT INTERPRETATION")
    print("="*60)
    
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_,
    })
    
    coef_df['Impact'] = coef_df['Coefficient'].apply(
        lambda x: f"↑ Increases by ${abs(x):,.0f}" if x > 0 else f"↓ Decreases by ${abs(x):,.0f}"
    )
    coef_df['Abs_Value'] = np.abs(coef_df['Coefficient'])
    coef_df_sorted = coef_df.sort_values('Abs_Value', ascending=False)
    
    print("\nFeature Coefficients (sorted by impact):")
    print("-" * 80)
    print(f"{'Feature':<20} {'Coefficient':>15} {'Impact':>45}")
    print("-" * 80)
    
    for idx, row in coef_df_sorted.iterrows():
        print(f"{row['Feature']:<20} {row['Coefficient']:>15,.2f} {row['Impact']:>45}")
    
    print("-" * 80)
    print(f"\nIntercept (Base Price): ${model.intercept_:,.2f}")
    
    print("\nInterpretation Examples:")
    top_feature = coef_df_sorted.iloc[0]
    print(f"  • {top_feature['Feature']}: Each unit increase "
          f"{'increases' if top_feature['Coefficient'] > 0 else 'decreases'} "
          f"price by ${abs(top_feature['Coefficient']):,.2f}")
    
    return coef_df_sorted