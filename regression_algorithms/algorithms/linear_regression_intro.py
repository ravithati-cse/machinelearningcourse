"""
📈 LINEAR REGRESSION - The Foundation of Machine Learning

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What linear regression is and when to use it
2. The regression equation: ŷ = β₀ + β₁x
3. Cost functions and Mean Squared Error (MSE)
4. How to find the best fit line (normal equation)
5. Making predictions with linear regression
6. Interpreting coefficients (what do β₀ and β₁ mean?)
7. Using scikit-learn for linear regression

This is where ALL the math comes together!

================================================================================
📺 RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
⭐ ABSOLUTE MUST WATCH:
   - StatQuest: "Linear Regression, Clearly Explained!!!"
     https://www.youtube.com/watch?v=nk2CQITm_eo
     (The BEST introduction to linear regression - watch this first!)

   - StatQuest: "Linear Models Pt.1 - Linear Regression"
     https://www.youtube.com/watch?v=PaFPbb66DxQ
     (Goes deeper into the math)

Also Highly Recommended:
   - 3Blue1Brown: "Neural Networks Chapter 2" (Gradient Descent)
     https://www.youtube.com/watch?v=IHZwWFHWa-w

   - Khan Academy: "Introduction to residuals and least squares"
     https://www.youtube.com/watch?v=yMgFHbjbAW8

================================================================================
OVERVIEW
================================================================================
Linear regression is the FOUNDATION of machine learning!

It answers the question: "What's the relationship between X and Y?"
- X = input/feature (e.g., house size)
- Y = output/target (e.g., house price)

We find the BEST LINE that describes this relationship.

All the math you learned comes together here:
- Algebra: y = mx + b (the line equation)
- Statistics: correlation, mean, variance (finding the best fit)
- Calculus: derivatives, minimizing error (gradient descent)
- Linear Algebra: vectors, dot products (making predictions)
- Probability: normal distribution (understanding residuals)

Let's build a complete linear regression model from scratch!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/regression/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("📈 LINEAR REGRESSION - Making Predictions with Lines")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE PROBLEM - FINDING RELATIONSHIPS IN DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: The Problem - Understanding Relationships")
print("=" * 80)
print()

print("THE FUNDAMENTAL QUESTION:")
print("-" * 70)
print("Given data about X (input) and Y (output), can we predict Y from X?")
print()
print("Examples:")
print("  • X = hours studied, Y = test score")
print("  • X = house size, Y = house price")
print("  • X = advertising spend, Y = sales")
print("  • X = temperature, Y = ice cream sales")
print()

print("EXAMPLE DATASET: House Prices")
print("-" * 70)

# Generate sample data
np.random.seed(42)
sizes = np.array([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600])
# True relationship: price = 150 * size + 50000 + noise
prices = 150 * sizes + 50000 + np.random.normal(0, 15000, len(sizes))

print(f"{'House Size (sqft)':<20} {'Price ($)':<15}")
print("-" * 40)
for size, price in zip(sizes, prices):
    print(f"{size:<20} ${price:<14,.0f}")

print()
print("QUESTION: If a house is 1500 sqft, what price should we predict?")
print()
print("To answer this, we need to:")
print("  1. Find the RELATIONSHIP between size and price")
print("  2. Express it as an EQUATION")
print("  3. Use the equation to PREDICT new prices")
print()
print("This is exactly what linear regression does!")
print()

# ============================================================================
# SECTION 2: THE REGRESSION EQUATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: The Regression Equation")
print("=" * 80)
print()

print("LINEAR REGRESSION EQUATION:")
print("-" * 70)
print("  ŷ = β₀ + β₁x")
print()
print("Where:")
print("  • ŷ (y-hat) = PREDICTED value")
print("  • x = INPUT (feature, independent variable)")
print("  • β₀ (beta-zero) = INTERCEPT")
print("    → Value of y when x = 0")
print("    → Where the line crosses the y-axis")
print("  • β₁ (beta-one) = SLOPE")
print("    → How much y changes when x increases by 1")
print("    → Rate of change")
print()

print("THIS IS THE SAME AS y = mx + b from algebra!")
print("  • β₁ is the slope (m)")
print("  • β₀ is the intercept (b)")
print("  • ŷ is our prediction")
print()

print("FOR OUR HOUSE PRICE EXAMPLE:")
print("-" * 70)
print("  ŷ = β₀ + β₁ × size")
print()
print("If we find β₀ = 50,000 and β₁ = 150:")
print("  ŷ = 50,000 + 150 × size")
print()
print("INTERPRETATION:")
print(f"  • β₀ = 50,000: Base price (even for 0 sqft - not realistic!)")
print(f"  • β₁ = 150: Each additional sqft adds $150 to price")
print()
print("PREDICTION for 1500 sqft house:")
print("  ŷ = 50,000 + 150 × 1500")
print("  ŷ = 50,000 + 225,000")
print("  ŷ = $275,000")
print()

# ============================================================================
# SECTION 3: COST FUNCTION - MEASURING ERROR
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: The Cost Function - How Wrong Are We?")
print("=" * 80)
print()

print("THE PROBLEM:")
print("-" * 70)
print("There are INFINITE possible lines (infinite choices of β₀ and β₁)")
print("We need to find the BEST line!")
print()
print("How do we measure 'best'?")
print("→ The line that makes the SMALLEST ERRORS!")
print()

print("RESIDUALS (ERRORS):")
print("-" * 70)
print("For each data point:")
print("  • yᵢ = actual value (what we observed)")
print("  • ŷᵢ = predicted value (what our line predicts)")
print("  • Residual = yᵢ - ŷᵢ (the error)")
print()

print("Example calculations:")
actual_sample = prices[:3]
sizes_sample = sizes[:3]

# Make simple predictions with β₀=50000, β₁=150
predicted_sample = 50000 + 150 * sizes_sample

print(f"{'Size':<10} {'Actual':<15} {'Predicted':<15} {'Residual':<15}")
print("-" * 60)
for size, actual, pred in zip(sizes_sample, actual_sample, predicted_sample):
    residual = actual - pred
    print(f"{size:<10} ${actual:<14,.0f} ${pred:<14,.0f} ${residual:<14,.0f}")

print()

print("MEAN SQUARED ERROR (MSE) - The Cost Function:")
print("-" * 70)
print("MSE = (1/n) × Σ(yᵢ - ŷᵢ)²")
print()
print("In words:")
print("  1. For each point: calculate (actual - predicted)²")
print("  2. Add up all the squared errors")
print("  3. Divide by the number of points (n)")
print()
print("Why SQUARE the errors?")
print("  • Makes all errors positive (can't cancel out)")
print("  • Penalizes BIG errors more (2² = 4, but 4² = 16!)")
print("  • Mathematically convenient for optimization")
print()

# Calculate MSE manually
residuals = actual_sample - predicted_sample
squared_errors = residuals ** 2
mse = np.mean(squared_errors)

print("Calculating MSE for our sample:")
print(f"  Squared errors: {np.array2string(squared_errors, precision=0)}")
print(f"  MSE = {mse:,.0f}")
print()

print("GOAL OF LINEAR REGRESSION:")
print("  Find β₀ and β₁ that MINIMIZE MSE!")
print("  → Smallest average squared error")
print("  → Best fit line!")
print()

# ============================================================================
# SECTION 4: FINDING THE BEST LINE - THE NORMAL EQUATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Finding the Best Fit Line")
print("=" * 80)
print()

print("TWO METHODS TO FIND BEST β₀ AND β₁:")
print("-" * 70)
print("1. GRADIENT DESCENT (calculus - what you learned!)")
print("   • Start with random β₀, β₁")
print("   • Calculate derivatives")
print("   • Update: β = β - α × derivative")
print("   • Repeat until MSE is minimized")
print()
print("2. NORMAL EQUATION (closed-form solution - we'll use this!)")
print("   • Direct mathematical formula")
print("   • Gives exact answer in one step")
print("   • Uses statistics: mean, covariance, variance")
print()

print("NORMAL EQUATION FORMULAS:")
print("-" * 70)
print("  β₁ = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²")
print()
print("  β₀ = ȳ - β₁x̄")
print()
print("Where:")
print("  • x̄ = mean of x")
print("  • ȳ = mean of y")
print("  • The numerator of β₁ is COVARIANCE!")
print("  • The denominator of β₁ is VARIANCE!")
print()
print("Remember from statistics?")
print("  → β₁ = Cov(x, y) / Var(x)")
print("  → This is how correlation and regression connect!")
print()

print("CALCULATING β₁ AND β₀ MANUALLY:")
print("-" * 70)

# Manual calculation
x_mean = np.mean(sizes)
y_mean = np.mean(prices)

print(f"Step 1: Calculate means")
print(f"  x̄ (mean size) = {x_mean:.1f} sqft")
print(f"  ȳ (mean price) = ${y_mean:,.0f}")
print()

# Calculate deviations
x_deviations = sizes - x_mean
y_deviations = prices - y_mean

print("Step 2: Calculate deviations from mean")
print(f"  (xᵢ - x̄) for first few: {np.array2string(x_deviations[:3], precision=1)}")
print(f"  (yᵢ - ȳ) for first few: {np.array2string(y_deviations[:3], precision=0)}")
print()

# Calculate β₁ (slope)
numerator = np.sum(x_deviations * y_deviations)
denominator = np.sum(x_deviations ** 2)
beta_1 = numerator / denominator

print("Step 3: Calculate β₁ (slope)")
print(f"  Numerator (covariance × n) = Σ(xᵢ-x̄)(yᵢ-ȳ) = {numerator:,.0f}")
print(f"  Denominator (variance × n) = Σ(xᵢ-x̄)² = {denominator:,.0f}")
print(f"  β₁ = {numerator:,.0f} / {denominator:,.0f} = {beta_1:.2f}")
print()

# Calculate β₀ (intercept)
beta_0 = y_mean - beta_1 * x_mean

print("Step 4: Calculate β₀ (intercept)")
print(f"  β₀ = ȳ - β₁x̄")
print(f"  β₀ = {y_mean:,.0f} - {beta_1:.2f} × {x_mean:.1f}")
print(f"  β₀ = {beta_0:,.0f}")
print()

print("✅ OUR EQUATION:")
print(f"  ŷ = {beta_0:,.0f} + {beta_1:.2f} × size")
print()

print("INTERPRETATION:")
print(f"  • Base price (β₀): ${beta_0:,.0f}")
print(f"  • Price per sqft (β₁): ${beta_1:.2f}")
print(f"  • For each additional sqft, price increases by ${beta_1:.2f}")
print()

# ============================================================================
# SECTION 5: MAKING PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Making Predictions with Our Model")
print("=" * 80)
print()

print("Now that we have β₀ and β₁, we can predict ANY house price!")
print()

# Make predictions
test_sizes = [1500, 1750, 2100]

print(f"{'House Size':<15} {'Prediction Calculation':<40} {'Predicted Price'}")
print("-" * 75)
for test_size in test_sizes:
    prediction = beta_0 + beta_1 * test_size
    calc_str = f"{beta_0:,.0f} + {beta_1:.2f} × {test_size}"
    print(f"{test_size} sqft{' ':<7} {calc_str:<40} ${prediction:,.0f}")

print()

print("HOW GOOD ARE OUR PREDICTIONS?")
print("-" * 70)

# Calculate MSE for our model
predictions = beta_0 + beta_1 * sizes
residuals = prices - predictions
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)

print("Calculating error metrics:")
print(f"  MSE (Mean Squared Error) = ${mse:,.0f}")
print(f"  RMSE (Root MSE) = ${rmse:,.0f}")
print()
print("RMSE interpretation:")
print(f"  Our predictions are off by about ${rmse:,.0f} on average")
print()

# Calculate R²
ss_total = np.sum((prices - y_mean) ** 2)
ss_residual = np.sum(residuals ** 2)
r_squared = 1 - (ss_residual / ss_total)

print("R² Score (Coefficient of Determination):")
print(f"  R² = {r_squared:.4f} ({r_squared*100:.2f}%)")
print()
print("R² interpretation:")
print(f"  Our model explains {r_squared*100:.2f}% of the variance in prices!")
print(f"  {(1-r_squared)*100:.2f}% is due to other factors or noise")
print()

# ============================================================================
# VISUALIZATION 1: The Regression Line
# ============================================================================
print("📊 Generating Visualization 1: The Best Fit Line...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('📈 LINEAR REGRESSION: Finding the Best Fit Line',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Scatter plot with regression line
ax = axes[0, 0]

# Scatter plot of data
ax.scatter(sizes, prices, color='blue', s=100, alpha=0.6, edgecolor='black', label='Actual data', zorder=5)

# Plot regression line
x_line = np.linspace(sizes.min(), sizes.max(), 100)
y_line = beta_0 + beta_1 * x_line
ax.plot(x_line, y_line, 'r-', linewidth=3, label=f'ŷ = {beta_0:,.0f} + {beta_1:.1f}x', zorder=3)

# Plot residuals as vertical lines
for size, price, pred in zip(sizes, prices, predictions):
    ax.plot([size, size], [price, pred], 'g--', linewidth=1, alpha=0.5)

ax.set_xlabel('House Size (sqft)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Linear Regression: Best Fit Line', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Annotate one point
sample_idx = 5
ax.annotate(f'Actual: ${prices[sample_idx]:,.0f}\nPredicted: ${predictions[sample_idx]:,.0f}\nError: ${residuals[sample_idx]:,.0f}',
            xy=(sizes[sample_idx], prices[sample_idx]),
            xytext=(sizes[sample_idx] - 300, prices[sample_idx] + 40000),
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Plot 2: Residual plot
ax = axes[0, 1]

ax.scatter(predictions, residuals, color='purple', s=100, alpha=0.6, edgecolor='black')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero error line')

ax.set_xlabel('Predicted Price ($)', fontsize=11)
ax.set_ylabel('Residual ($)', fontsize=11)
ax.set_title('Residual Plot\n(Should look random!)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(predictions.mean(), residuals.max() * 0.8,
        'Good: Points scattered randomly\n→ No patterns\n→ Model fits well',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 3: Actual vs Predicted
ax = axes[1, 0]

ax.scatter(prices, predictions, color='orange', s=100, alpha=0.6, edgecolor='black', label='Our predictions')

# Perfect prediction line
min_val = min(prices.min(), predictions.min())
max_val = max(prices.max(), predictions.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect predictions')

ax.set_xlabel('Actual Price ($)', fontsize=11)
ax.set_ylabel('Predicted Price ($)', fontsize=11)
ax.set_title(f'Actual vs Predicted (R² = {r_squared:.3f})', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(prices.mean(), predictions.max(),
        'Points near red line → Good predictions\nPoints far from line → Poor predictions',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 4: Model summary
ax = axes[1, 1]
ax.text(0.5, 0.95, 'LINEAR REGRESSION SUMMARY', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

summary = [
    "📊 EQUATION:",
    f"   ŷ = {beta_0:,.0f} + {beta_1:.2f} × size",
    "",
    "📏 COEFFICIENTS:",
    f"   β₀ (intercept) = ${beta_0:,.0f}",
    f"   β₁ (slope) = ${beta_1:.2f} per sqft",
    "",
    "📈 INTERPRETATION:",
    f"   • Base price: ${beta_0:,.0f}",
    f"   • Each sqft adds: ${beta_1:.2f}",
    "",
    "✅ PERFORMANCE:",
    f"   • MSE = ${mse:,.0f}",
    f"   • RMSE = ${rmse:,.0f}",
    f"   • R² = {r_squared:.4f} ({r_squared*100:.1f}%)",
    "",
    "🎯 WHAT THIS MEANS:",
    f"   Our model explains {r_squared*100:.1f}% of",
    "   price variation!",
    f"   Typical error: ±${rmse:,.0f}",
    "",
    "💡 EXAMPLE PREDICTION:",
    "   For 1500 sqft house:",
    f"   ŷ = {beta_0:,.0f} + {beta_1:.2f}×1500",
    f"   ŷ = ${beta_0 + beta_1*1500:,.0f}"
]

y_pos = 0.87
for line in summary:
    if line.startswith(('📊', '📏', '📈', '✅', '🎯', '💡')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.036

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_linear_regression_basics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 01_linear_regression_basics.png")
print()

# ============================================================================
# SECTION 6: USING SCIKIT-LEARN
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Linear Regression with Scikit-Learn")
print("=" * 80)
print()

print("We did it manually to understand the math!")
print("Now let's use scikit-learn - the standard ML library")
print()

print("SCIKIT-LEARN CODE:")
print("-" * 70)

# Reshape data for sklearn (needs 2D array)
X = sizes.reshape(-1, 1)  # Features (2D)
y = prices  # Target (1D)

print("Step 1: Prepare data")
print(f"  X shape: {X.shape} (10 samples, 1 feature)")
print(f"  y shape: {y.shape} (10 values)")
print()

# Create and fit model
model = LinearRegression()
model.fit(X, y)

print("Step 2: Create and train model")
print("  model = LinearRegression()")
print("  model.fit(X, y)")
print()

print("Step 3: Extract coefficients")
sklearn_beta_0 = model.intercept_
sklearn_beta_1 = model.coef_[0]

print(f"  Intercept (β₀) = ${sklearn_beta_0:,.2f}")
print(f"  Slope (β₁) = ${sklearn_beta_1:.2f}")
print()

print("COMPARING OUR MANUAL CALCULATION vs SCIKIT-LEARN:")
print("-" * 70)
print(f"{'Parameter':<20} {'Manual':<20} {'Scikit-Learn':<20} {'Match?'}")
print("-" * 75)
print(f"{'β₀ (intercept)':<20} ${beta_0:<19,.2f} ${sklearn_beta_0:<19,.2f} {'✅' if abs(beta_0 - sklearn_beta_0) < 1 else '❌'}")
print(f"{'β₁ (slope)':<20} ${beta_1:<19,.2f} ${sklearn_beta_1:<19,.2f} {'✅' if abs(beta_1 - sklearn_beta_1) < 0.01 else '❌'}")
print()
print("They match! Our manual calculation was correct! 🎉")
print()

# Make predictions with sklearn
sklearn_predictions = model.predict(X)

print("Step 4: Make predictions")
print("  predictions = model.predict(X)")
print()
print(f"{'Size':<10} {'Actual':<15} {'Predicted':<15} {'Error'}")
print("-" * 55)
for size, actual, pred in zip(sizes[:5], prices[:5], sklearn_predictions[:5]):
    error = actual - pred
    print(f"{size:<10} ${actual:<14,.0f} ${pred:<14,.0f} ${error:,.0f}")
print("...")
print()

# Evaluate
sklearn_mse = mean_squared_error(y, sklearn_predictions)
sklearn_r2 = r2_score(y, sklearn_predictions)

print("Step 5: Evaluate model")
print(f"  MSE = ${sklearn_mse:,.0f}")
print(f"  R² = {sklearn_r2:.4f}")
print()

# ============================================================================
# SECTION 7: ASSUMPTIONS AND WHEN TO USE LINEAR REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: When to Use Linear Regression")
print("=" * 80)
print()

print("LINEAR REGRESSION WORKS BEST WHEN:")
print("-" * 70)
print("✅ 1. LINEAR RELATIONSHIP:")
print("     The relationship between X and Y is approximately a straight line")
print()
print("✅ 2. INDEPENDENCE:")
print("     Data points are independent (one doesn't affect another)")
print()
print("✅ 3. NORMAL RESIDUALS:")
print("     Errors are normally distributed (bell curve)")
print()
print("✅ 4. CONSTANT VARIANCE (Homoscedasticity):")
print("     Errors have similar spread across all values of X")
print()

print("WHEN NOT TO USE LINEAR REGRESSION:")
print("-" * 70)
print("❌ Non-linear relationship (curve, not line)")
print("❌ Categorical target (use classification instead)")
print("❌ Extreme outliers dominating the fit")
print("❌ Time series with trends/seasonality (need special methods)")
print()

print("HOW TO CHECK ASSUMPTIONS:")
print("-" * 70)
print("1. Plot scatter plot → Should see roughly linear pattern")
print("2. Plot residuals → Should look random, no patterns")
print("3. Plot histogram of residuals → Should look normal (bell curve)")
print("4. Check for outliers → Points far from the line")
print()

# ============================================================================
# VISUALIZATION 2: Complete Workflow
# ============================================================================
print("📊 Generating Visualization 2: Complete Workflow...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('📈 COMPLETE LINEAR REGRESSION WORKFLOW',
             fontsize=16, fontweight='bold')

# Plot 1: Raw data
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(sizes, prices, color='blue', s=80, alpha=0.6, edgecolor='black')
ax1.set_xlabel('Size (sqft)', fontsize=10)
ax1.set_ylabel('Price ($)', fontsize=10)
ax1.set_title('Step 1: Collect Data', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Find best line
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(sizes, prices, color='blue', s=80, alpha=0.6, edgecolor='black')
ax2.plot(x_line, y_line, 'r-', linewidth=3)
ax2.set_xlabel('Size (sqft)', fontsize=10)
ax2.set_ylabel('Price ($)', fontsize=10)
ax2.set_title('Step 2: Fit Line (Minimize MSE)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(sizes.mean(), prices.max(), f'ŷ = {beta_0:,.0f} + {beta_1:.1f}x',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Plot 3: Make predictions
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(sizes, prices, color='blue', s=60, alpha=0.4, label='Training data')
ax3.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.5)

# New points to predict
new_sizes = np.array([1300, 1900, 2300])
new_predictions = beta_0 + beta_1 * new_sizes
ax3.scatter(new_sizes, new_predictions, color='green', s=150, marker='*',
            edgecolor='black', linewidth=2, label='Predictions', zorder=5)

for ns, np_val in zip(new_sizes, new_predictions):
    ax3.annotate(f'${np_val:,.0f}', xy=(ns, np_val), xytext=(ns, np_val + 20000),
                fontsize=8, ha='center', fontweight='bold')

ax3.set_xlabel('Size (sqft)', fontsize=10)
ax3.set_ylabel('Price ($)', fontsize=10)
ax3.set_title('Step 3: Make Predictions', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals histogram
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(residuals, bins=7, color='lightblue', edgecolor='black', alpha=0.7)
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax4.set_xlabel('Residual ($)', fontsize=10)
ax4.set_ylabel('Frequency', fontsize=10)
ax4.set_title('Step 4a: Check Residuals\n(Should be normal)', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Residuals vs fitted
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(predictions, residuals, color='purple', s=80, alpha=0.6, edgecolor='black')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Fitted values ($)', fontsize=10)
ax5.set_ylabel('Residual ($)', fontsize=10)
ax5.set_title('Step 4b: Check Residuals\n(Should be random)', fontsize=10, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Q-Q plot (normal check)
ax6 = fig.add_subplot(gs[1, 2])
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=ax6)
ax6.set_title('Step 4c: Normal Q-Q Plot\n(Check normality)', fontsize=10, fontweight='bold')
ax6.grid(True, alpha=0.3)

# Plot 7: Code example
ax7 = fig.add_subplot(gs[2, :])
code_text = """
SCIKIT-LEARN CODE:

# 1. Import
from sklearn.linear_model import LinearRegression

# 2. Prepare data
X = sizes.reshape(-1, 1)  # 2D array
y = prices

# 3. Create and train
model = LinearRegression()
model.fit(X, y)

# 4. Get coefficients
β₀ = model.intercept_
β₁ = model.coef_[0]

# 5. Make predictions
predictions = model.predict(X)

# 6. Evaluate
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

That's it! Just 6 steps to build a complete linear regression model!
"""

ax7.text(0.05, 0.95, code_text, transform=ax7.transAxes,
         fontsize=9, family='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
ax7.axis('off')

plt.savefig(f'{VISUAL_DIR}02_complete_workflow.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 02_complete_workflow.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: Linear Regression Complete!")
print("=" * 80)
print()

print("🎯 WHAT WE LEARNED:")
print("-" * 70)
print("1. LINEAR REGRESSION EQUATION:")
print("   ŷ = β₀ + β₁x")
print("   • β₀ = intercept (base value)")
print("   • β₁ = slope (rate of change)")
print()

print("2. COST FUNCTION (MSE):")
print("   MSE = (1/n) × Σ(yᵢ - ŷᵢ)²")
print("   • Measures average squared error")
print("   • Goal: minimize MSE")
print()

print("3. FINDING BEST FIT LINE:")
print("   • Normal equation (closed-form)")
print("   • Or gradient descent (iterative)")
print("   • Both find β₀ and β₁ that minimize MSE")
print()

print("4. MAKING PREDICTIONS:")
print("   • Plug in x value")
print("   • Calculate ŷ = β₀ + β₁x")
print("   • That's your prediction!")
print()

print("5. EVALUATING MODEL:")
print("   • MSE: average squared error")
print("   • RMSE: √MSE (same units as y)")
print("   • R²: % of variance explained")
print()

print("🤖 USING SCIKIT-LEARN:")
print("-" * 70)
print("  from sklearn.linear_model import LinearRegression")
print("  model = LinearRegression()")
print("  model.fit(X, y)")
print("  predictions = model.predict(X)")
print()

print("📊 KEY INSIGHTS:")
print("-" * 70)
print("  • Linear regression finds the best straight line through data")
print("  • 'Best' means minimizing prediction errors (MSE)")
print("  • Coefficients have clear interpretations")
print("  • Works great when relationship is linear!")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 01_linear_regression_basics.png")
print("✅ 02_complete_workflow.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review visualizations - understand the complete workflow")
print("   2. Watch StatQuest video on linear regression (absolute must!)")
print("   3. Try with your own data - change the house sizes and prices!")
print("   4. Next: algorithms/multiple_regression.py (multiple features)")
print()

print("💡 REMEMBER:")
print("   All of machine learning builds on this foundation!")
print("   • Neural networks = stacked linear regressions (+ nonlinearity)")
print("   • Decision trees = piece-wise linear regressions")
print("   • Everything connects back to finding patterns in data!")
print()

print("=" * 80)
print("🎉 LINEAR REGRESSION MASTERED!")
print("   You now understand how ML models learn from data!")
print("=" * 80)
