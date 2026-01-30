"""
📊 MODEL EVALUATION - Regression Metrics Deep Dive
===============================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. Mean Absolute Error (MAE) - intuitive error metric
2. Mean Squared Error (MSE) - penalizes large errors
3. Root Mean Squared Error (RMSE) - in original units
4. R² Score (Coefficient of Determination) - variance explained
5. Residual analysis and diagnostics
6. When to use which metric
7. Cross-validation for robust evaluation

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "R-squared explained"
   https://www.youtube.com/watch?v=2AQKmw14mHM
   BEST explanation of R²!

⭐ StatQuest: "Machine Learning Fundamentals: Cross Validation"
   https://www.youtube.com/watch?v=fSytzGwwBVw
   Why cross-validation matters

📚 Krish Naik: "Performance Metrics for Regression"
   Comprehensive overview

TIME: 60-75 minutes
DIFFICULTY: Intermediate
PREREQUISITES: linear_regression_intro.py

KEY CONCEPTS:
------------
- MAE: Mean Absolute Error
- MSE: Mean Squared Error
- RMSE: Root Mean Squared Error
- R²: Coefficient of Determination
- Residuals: Actual - Predicted
- Baseline Model: Simple benchmark
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'model_evaluation'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📊 MODEL EVALUATION - Regression Metrics")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: GENERATE SAMPLE DATA
# ============================================================================

print("=" * 80)
print("SECTION 1: Sample Data and Predictions")
print("=" * 80)
print()

# Generate sample data
np.random.seed(42)
n_samples = 100

# True relationship: y = 3 + 2x + noise
X = np.random.uniform(0, 10, n_samples)
y_true = 3 + 2 * X + np.random.normal(0, 2, n_samples)

# Simulate model predictions (imperfect)
y_pred = 3.5 + 1.9 * X + np.random.normal(0, 0.5, n_samples)

print(f"Generated {n_samples} data points")
print(f"   True values range: [{y_true.min():.2f}, {y_true.max():.2f}]")
print(f"   Predicted values range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
print()

# Show some examples
print("SAMPLE PREDICTIONS:")
print("-" * 70)
print(f"{'Actual':<12} {'Predicted':<12} {'Error':<12} {'Squared Error'}")
print("-" * 70)
for i in range(5):
    error = y_true[i] - y_pred[i]
    sq_error = error ** 2
    print(f"{y_true[i]:<12.2f} {y_pred[i]:<12.2f} {error:<12.2f} {sq_error:<12.2f}")
print()

# ============================================================================
# SECTION 2: MEAN ABSOLUTE ERROR (MAE)
# ============================================================================

print("=" * 80)
print("SECTION 2: Mean Absolute Error (MAE)")
print("=" * 80)
print()

print("FORMULA:")
print("   MAE = (1/n) × Σ|yᵢ - ŷᵢ|")
print()
print("WHAT IT MEANS:")
print("   Average absolute difference between actual and predicted")
print("   In the SAME UNITS as your target variable")
print()

# Calculate MAE manually
errors = y_true - y_pred
absolute_errors = np.abs(errors)
mae = np.mean(absolute_errors)

print("CALCULATION:")
print("-" * 70)
print(f"1. Calculate errors: y_true - y_pred")
print(f"   Example: {y_true[0]:.2f} - {y_pred[0]:.2f} = {errors[0]:.2f}")
print()
print(f"2. Take absolute values: |error|")
print(f"   Example: |{errors[0]:.2f}| = {absolute_errors[0]:.2f}")
print()
print(f"3. Average all absolute errors")
print(f"   MAE = {mae:.3f}")
print()

print("INTERPRETATION:")
print(f"   On average, predictions are off by {mae:.3f} units")
print(f"   If predicting house prices in $100k, error = ${mae*100:.0f}k")
print()

print("PROS:")
print("   ✓ Easy to understand (average error)")
print("   ✓ Same units as target")
print("   ✓ Robust to outliers (no squaring)")
print()

print("CONS:")
print("   ✗ Treats all errors equally")
print("   ✗ Not differentiable at zero (calculus issue)")
print()

# ============================================================================
# SECTION 3: MEAN SQUARED ERROR (MSE)
# ============================================================================

print("=" * 80)
print("SECTION 3: Mean Squared Error (MSE)")
print("=" * 80)
print()

print("FORMULA:")
print("   MSE = (1/n) × Σ(yᵢ - ŷᵢ)²")
print()
print("WHAT IT MEANS:")
print("   Average SQUARED difference between actual and predicted")
print("   Units are SQUARED (e.g., dollars²)")
print()

# Calculate MSE manually
squared_errors = errors ** 2
mse = np.mean(squared_errors)

print("CALCULATION:")
print("-" * 70)
print(f"1. Calculate errors: y_true - y_pred")
print(f"   Example: {y_true[0]:.2f} - {y_pred[0]:.2f} = {errors[0]:.2f}")
print()
print(f"2. Square each error")
print(f"   Example: ({errors[0]:.2f})² = {squared_errors[0]:.2f}")
print()
print(f"3. Average all squared errors")
print(f"   MSE = {mse:.3f}")
print()

print("WHY SQUARE?")
print("-" * 70)
print("   • Makes all errors positive")
print("   • PENALIZES LARGE ERRORS MORE")
print()
print("   Example:")
print(f"      Error = 1  → Squared = 1")
print(f"      Error = 2  → Squared = 4  (2× error = 4× penalty)")
print(f"      Error = 10 → Squared = 100 (10× error = 100× penalty!)")
print()

print("INTERPRETATION:")
print(f"   MSE = {mse:.3f} (squared units)")
print("   Hard to interpret directly because of squared units")
print()

print("PROS:")
print("   ✓ Heavily penalizes large errors")
print("   ✓ Differentiable (good for optimization)")
print("   ✓ Common in machine learning")
print()

print("CONS:")
print("   ✗ Squared units (hard to interpret)")
print("   ✗ Sensitive to outliers")
print()

# ============================================================================
# SECTION 4: ROOT MEAN SQUARED ERROR (RMSE)
# ============================================================================

print("=" * 80)
print("SECTION 4: Root Mean Squared Error (RMSE)")
print("=" * 80)
print()

print("FORMULA:")
print("   RMSE = √MSE = √[(1/n) × Σ(yᵢ - ŷᵢ)²]")
print()
print("WHAT IT MEANS:")
print("   Square root of MSE")
print("   Back to ORIGINAL UNITS (like MAE)")
print("   Still penalizes large errors (from MSE)")
print()

# Calculate RMSE
rmse = np.sqrt(mse)

print("CALCULATION:")
print("-" * 70)
print(f"1. Calculate MSE = {mse:.3f}")
print(f"2. Take square root: √{mse:.3f} = {rmse:.3f}")
print()

print("INTERPRETATION:")
print(f"   RMSE = {rmse:.3f}")
print(f"   On average, predictions are off by {rmse:.3f} units")
print(f"   (But large errors contribute more to this value)")
print()

print("MAE vs RMSE COMPARISON:")
print("-" * 70)
print(f"   MAE:  {mae:.3f}")
print(f"   RMSE: {rmse:.3f}")
print()
print(f"   RMSE > MAE: {rmse > mae}")
if rmse > mae:
    print("   → Some large errors are present")
    print("   → RMSE is more sensitive to outliers")
print()

print("WHEN TO USE:")
print("-" * 70)
print("   Use MAE when:")
print("      • All errors equally important")
print("      • Want robust metric (outliers present)")
print("      • Easy interpretation needed")
print()
print("   Use RMSE when:")
print("      • Large errors are especially bad")
print("      • Standard metric in competition/paper")
print("      • Optimization-friendly needed")
print()

# ============================================================================
# SECTION 5: R² SCORE (COEFFICIENT OF DETERMINATION)
# ============================================================================

print("=" * 80)
print("SECTION 5: R² Score - Variance Explained")
print("=" * 80)
print()

print("FORMULA:")
print("   R² = 1 - (SS_res / SS_tot)")
print()
print("Where:")
print("   SS_res = Σ(yᵢ - ŷᵢ)²  (residual sum of squares)")
print("   SS_tot = Σ(yᵢ - ȳ)²   (total sum of squares)")
print()

# Calculate R² manually
y_mean = np.mean(y_true)

# Sum of squared residuals (model errors)
ss_res = np.sum((y_true - y_pred) ** 2)

# Total sum of squares (variance in data)
ss_tot = np.sum((y_true - y_mean) ** 2)

# R²
r2 = 1 - (ss_res / ss_tot)

print("STEP-BY-STEP CALCULATION:")
print("-" * 70)
print(f"1. Calculate mean of y: ȳ = {y_mean:.3f}")
print()
print(f"2. SS_res (residual sum of squares):")
print(f"   = Σ(actual - predicted)²")
print(f"   = {ss_res:.3f}")
print(f"   (How much error our model makes)")
print()
print(f"3. SS_tot (total sum of squares):")
print(f"   = Σ(actual - mean)²")
print(f"   = {ss_tot:.3f}")
print(f"   (Total variance in data)")
print()
print(f"4. R² = 1 - ({ss_res:.3f} / {ss_tot:.3f})")
print(f"   R² = {r2:.4f}")
print()

print("INTERPRETATION:")
print("-" * 70)
print(f"   R² = {r2:.4f}")
print()
print("   MEANING:")
if r2 >= 0:
    print(f"      • Model explains {r2*100:.1f}% of variance in target")
    print(f"      • {(1-r2)*100:.1f}% of variance is unexplained")
else:
    print(f"      • R² is NEGATIVE - model is worse than baseline!")
print()

print("   R² SCALE:")
print("      1.00 = Perfect predictions (100% variance explained)")
print("      0.90 = Excellent (90% variance explained)")
print("      0.70 = Good (70% variance explained)")
print("      0.50 = Moderate (50% variance explained)")
print("      0.00 = Baseline (just predicting mean)")
print("     <0.00 = Worse than baseline!")
print()

print("BASELINE MODEL COMPARISON:")
print("-" * 70)
print("What if we just predicted the mean for everything?")
print()

# Baseline predictions (always predict mean)
baseline_pred = np.full_like(y_true, y_mean)
baseline_mse = np.mean((y_true - baseline_pred) ** 2)
baseline_r2 = 1 - (baseline_mse / ss_tot)

print(f"   Baseline (predict mean): R² = {baseline_r2:.4f}")
print(f"   Our model:              R² = {r2:.4f}")
print()

if r2 > baseline_r2:
    improvement = ((r2 - baseline_r2) / (1 - baseline_r2)) * 100
    print(f"   ✓ Our model is {improvement:.1f}% better than baseline!")
else:
    print(f"   ✗ Our model is worse than baseline!")
print()

print("PROS:")
print("   ✓ Scale-independent (always between -∞ and 1)")
print("   ✓ Intuitive (percentage of variance explained)")
print("   ✓ Comparable across different problems")
print()

print("CONS:")
print("   ✗ Can be misleading with outliers")
print("   ✗ Not in original units")
print("   ✗ Doesn't tell you magnitude of errors")
print()

# ============================================================================
# SECTION 6: RESIDUAL ANALYSIS
# ============================================================================

print("=" * 80)
print("SECTION 6: Residual Analysis")
print("=" * 80)
print()

print("RESIDUALS = Actual - Predicted")
print()

# Calculate residuals
residuals = y_true - y_pred

print("RESIDUAL STATISTICS:")
print("-" * 70)
print(f"   Mean:              {np.mean(residuals):.4f} (should be ≈ 0)")
print(f"   Std Dev:           {np.std(residuals):.4f}")
print(f"   Min (overpredict): {np.min(residuals):.4f}")
print(f"   Max (underpredict): {np.max(residuals):.4f}")
print()

print("WHAT TO LOOK FOR IN RESIDUALS:")
print("-" * 70)
print()
print("1. MEAN ≈ 0:")
if abs(np.mean(residuals)) < 0.1:
    print("   ✓ Mean is close to zero - no systematic bias")
else:
    print("   ⚠ Mean is NOT zero - systematic bias present")
print()

print("2. RANDOM SCATTER:")
print("   • Should have no patterns")
print("   • Should be evenly distributed")
print("   • Check residual plot (see visualization)")
print()

print("3. CONSTANT VARIANCE (Homoscedasticity):")
print("   • Spread should be similar across predictions")
print("   • If not: heteroscedasticity (variance changes)")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regression Metrics: Complete Guide', fontsize=16, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
ax1.scatter(y_true, y_pred, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
ax1.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
ax1.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Add metrics text
metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
        fontsize=11, verticalalignment='top', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

# Plot 2: Residual Plot
ax2 = axes[0, 1]
ax2.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
ax2.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Add interpretation
ax2.text(0.05, 0.95, 'Good model:\n• Random scatter\n• Centered at 0\n• No patterns',
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Plot 3: Error Distribution
ax3 = axes[1, 0]
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero error')
ax3.axvline(x=np.mean(residuals), color='g', linestyle='--', linewidth=2,
           label=f'Mean = {np.mean(residuals):.3f}')
ax3.set_xlabel('Residual Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Metric comparison
ax4 = axes[1, 1]
ax4.axis('off')

comparison_text = """
METRIC COMPARISON AND USAGE
═══════════════════════════════════════════════════════════════

MAE (Mean Absolute Error)
├─ Formula: (1/n) × Σ|actual - predicted|
├─ Units: Same as target
├─ When to use: All errors equally important
└─ Robust to outliers

MSE (Mean Squared Error)
├─ Formula: (1/n) × Σ(actual - predicted)²
├─ Units: Squared
├─ When to use: Large errors are especially bad
└─ Sensitive to outliers

RMSE (Root Mean Squared Error)
├─ Formula: √MSE
├─ Units: Same as target
├─ When to use: Standard metric, penalize large errors
└─ More sensitive than MAE

R² Score (Coefficient of Determination)
├─ Formula: 1 - (SS_res / SS_tot)
├─ Range: (-∞, 1], where 1 is perfect
├─ Interpretation: % of variance explained
└─ Scale-independent

CHOOSING THE RIGHT METRIC:
───────────────────────────────────────────────────────────────

• Report ALL metrics (MAE, RMSE, R²)
• Use RMSE for competitions/papers (standard)
• Use MAE for easy interpretation
• Use R² to compare different models
• Always look at residual plots!

BASELINE COMPARISON:
───────────────────────────────────────────────────────────────

Always compare to a baseline model:
• Baseline: Predict mean → R² = 0
• Your model should beat this!

═══════════════════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, comparison_text.strip(),
        transform=ax4.transAxes,
        fontsize=8,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_metrics_complete_guide.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_metrics_complete_guide.png")
plt.close()

print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("📊 SUMMARY: Regression Metrics")
print("=" * 80)
print()

print("✓ METRICS CALCULATED:")
print()
print(f"   MAE:  {mae:.3f}  (average absolute error)")
print(f"   MSE:  {mse:.3f}  (average squared error)")
print(f"   RMSE: {rmse:.3f}  (root mean squared error)")
print(f"   R²:   {r2:.3f}  ({r2*100:.1f}% variance explained)")
print()

print("✓ KEY TAKEAWAYS:")
print()
print("   1. MULTIPLE METRICS:")
print("      • Always report MAE, RMSE, and R²")
print("      • Each tells you something different")
print("      • No single 'best' metric")
print()

print("   2. INTERPRETATION:")
print(f"      • MAE = {mae:.3f}: Average error magnitude")
print(f"      • RMSE = {rmse:.3f}: Penalizes large errors")
print(f"      • R² = {r2:.3f}: {r2*100:.1f}% variance explained")
print()

print("   3. RESIDUAL ANALYSIS:")
print("      • Check for patterns in residuals")
print("      • Should be randomly scattered")
print("      • Mean should be near zero")
print()

print("   4. BASELINE COMPARISON:")
print("      • Always compare to simple baseline")
print("      • Baseline R² = 0 (predicting mean)")
print("      • Your model should do better!")
print()

print("=" * 80)
print("📊 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
