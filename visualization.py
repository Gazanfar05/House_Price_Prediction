import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = '#f8f9fa'

def plot_evaluation(y_test, y_pred, test_r2, test_rmse):
    """Create comprehensive evaluation visualizations"""
    print("\n" + "="*60)
    print("CREATING ADVANCED VISUALIZATIONS")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # Color scheme
    colors_gradient = ['#667eea', '#764ba2', '#f093fb']
    
    # 1. Actual vs Predicted (Top Left - Large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    scatter = ax1.scatter(y_test, y_pred, alpha=0.6, s=100, c=y_test, 
                         cmap='viridis', edgecolors='k', linewidth=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=3, label='Perfect Prediction', alpha=0.7)
    ax1.set_xlabel('Actual Price ($)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Predicted Price ($)', fontsize=13, fontweight='bold')
    ax1.set_title(f'Actual vs Predicted Prices\n(R² = {test_r2:.4f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Actual Price ($)', fontsize=11, fontweight='bold')
    
    # 2. Residuals Distribution (Top Right)
    ax2 = fig.add_subplot(gs[0, 2])
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=30, edgecolor='black', color='#667eea', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Residuals ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Distribution', fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Error Metrics (Middle Right)
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.axis('off')
    
    errors = np.abs(y_test - y_pred)
    error_pct = (errors / y_test) * 100
    
    metrics_text = f"""
    MODEL PERFORMANCE METRICS
    {'='*35}
    
    Test RMSE:        ${test_rmse:,.2f}
    Mean Error:       ${errors.mean():,.2f}
    Median Error:     ${np.median(errors):,.2f}
    Std Dev Error:    ${errors.std():,.2f}
    
    Mean % Error:     {error_pct.mean():.2f}%
    Max % Error:      {error_pct.max():.2f}%
    Min % Error:      {error_pct.min():.2f}%
    
    Predictions within:
    ±$10K:  {(errors <= 10000).sum()} ({(errors <= 10000).sum()/len(errors)*100:.1f}%)
    ±$20K:  {(errors <= 20000).sum()} ({(errors <= 20000).sum()/len(errors)*100:.1f}%)
    ±$30K:  {(errors <= 30000).sum()} ({(errors <= 30000).sum()/len(errors)*100:.1f}%)
    """
    
    ax3.text(0.05, 0.95, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    # 4. Residuals vs Predicted (Bottom Left)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(y_pred, residuals, alpha=0.6, s=80, c='#764ba2', edgecolors='k', linewidth=0.5)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Predicted Price ($)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Residuals ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold', pad=10)
    ax4.grid(alpha=0.3)
    
    # 5. Absolute Error Distribution (Bottom Middle)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(errors, bins=30, edgecolor='black', color='#f093fb', alpha=0.7)
    ax5.axvline(x=errors.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: ${errors.mean():,.0f}')
    ax5.set_xlabel('Absolute Error ($)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Absolute Error Distribution', fontsize=12, fontweight='bold', pad=10)
    ax5.legend(fontsize=10)
    ax5.grid(alpha=0.3, axis='y')
    
    # 6. Error Percentage Distribution (Bottom Right)
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.hist(error_pct, bins=30, edgecolor='black', color='#667eea', alpha=0.7)
    ax6.axvline(x=error_pct.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {error_pct.mean():.1f}%')
    ax6.set_xlabel('Error Percentage (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title('Error % Distribution', fontsize=12, fontweight='bold', pad=10)
    ax6.legend(fontsize=10)
    ax6.grid(alpha=0.3, axis='y')
    
    # Main title
    fig.suptitle('House Price Prediction Model - Comprehensive Evaluation', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print("✓ Comprehensive evaluation visualization saved: 'model_evaluation.png'")
    plt.close()

def plot_rmse_comparison(y_train, y_train_pred, y_test, y_test_pred, train_rmse, test_rmse):
    """Create dedicated RMSE comparison plot"""
    print("Creating RMSE comparison chart...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('RMSE Analysis - Training vs Testing', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. RMSE Comparison Bar Chart
    ax1 = axes[0, 0]
    rmse_values = [train_rmse, test_rmse]
    colors = ['#667eea', '#f093fb']
    bars = ax1.bar(['Train RMSE', 'Test RMSE'], rmse_values, color=colors, 
                    edgecolor='black', linewidth=2, alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax1.set_ylabel('RMSE ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Train vs Test RMSE', fontsize=13, fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(rmse_values) * 1.2)
    
    # 2. Train Set Errors
    ax2 = axes[0, 1]
    train_errors = np.abs(y_train - y_train_pred)
    ax2.hist(train_errors, bins=40, edgecolor='black', color='#667eea', alpha=0.7)
    ax2.axvline(x=train_errors.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean Error: ${train_errors.mean():,.0f}')
    ax2.set_xlabel('Absolute Error ($)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'Training Set Error Distribution (RMSE: ${train_rmse:,.0f})', 
                  fontsize=12, fontweight='bold', pad=10)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Test Set Errors
    ax3 = axes[1, 0]
    test_errors = np.abs(y_test - y_test_pred)
    ax3.hist(test_errors, bins=40, edgecolor='black', color='#f093fb', alpha=0.7)
    ax3.axvline(x=test_errors.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean Error: ${test_errors.mean():,.0f}')
    ax3.set_xlabel('Absolute Error ($)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title(f'Test Set Error Distribution (RMSE: ${test_rmse:,.0f})', 
                  fontsize=12, fontweight='bold', pad=10)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Error Statistics Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [
        ['Metric', 'Train', 'Test'],
        ['Mean Error', f"${train_errors.mean():,.0f}", f"${test_errors.mean():,.0f}"],
        ['Median Error', f"${np.median(train_errors):,.0f}", f"${np.median(test_errors):,.0f}"],
        ['Std Dev', f"${train_errors.std():,.0f}", f"${test_errors.std():,.0f}"],
        ['Min Error', f"${train_errors.min():,.0f}", f"${test_errors.min():,.0f}"],
        ['Max Error', f"${train_errors.max():,.0f}", f"${test_errors.max():,.0f}"],
        ['RMSE', f"${train_rmse:,.0f}", f"${test_rmse:,.0f}"],
    ]
    
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#667eea')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            table[(i, j)].set_text_props(weight='bold')
    
    plt.savefig('rmse_analysis.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print("✓ RMSE analysis chart saved: 'rmse_analysis.png'")
    plt.close()

def plot_coefficients(coef_df):
    """Plot feature coefficients"""
    print("Creating coefficients visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['#28a745' if x > 0 else '#dc3545' for x in coef_df['Coefficient']]
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, 
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, coef_df['Coefficient'])):
        ax.text(val, bar.get_y() + bar.get_height()/2, f'  ${val:,.0f}',
               ha='left' if val > 0 else 'right', va='center', 
               fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Coefficient Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Feature Coefficients Impact on House Price', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('coefficients.png', dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print("✓ Coefficients plot saved: 'coefficients.png'")
    plt.close()