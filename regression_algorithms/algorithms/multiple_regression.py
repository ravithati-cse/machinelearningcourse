"""
📊 MULTIPLE LINEAR REGRESSION - Using Multiple Features to Predict

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. How to use MULTIPLE features (not just one) for predictions
2. The matrix form of linear regression
3. Feature importance and coefficient interpretation
4. Multicollinearity (when features correlate with each other)
5. How to use scikit-learn for multiple regression
6. Real-world applications with multiple features

This extends simple linear regression to the real world!

================================================================================
📺 RECOMMENDED VIDEOS
================================================================================
⭐ MUST WATCH:
   - StatQuest: "Multiple Regression"
     https://www.youtube.com/watch?v=zITIFTsivN8
     (Clear explanation of multiple features)

   - StatQuest: "R-squared and Adjusted R-squared"
     https://www.youtube.com/watch?v=bMccdk8EdGo

Also Recommended:
   - ritvikmath: "Multiple Linear Regression from Scratch"
     https://www.youtube.com/watch?v=J_LnPL3Qg70
     (Python-focused with matrix operations)

================================================================================
OVERVIEW
================================================================================
Real-world predictions need MORE THAN ONE feature!

Simple regression:  Price = f(size)
Multiple regression: Price = f(size, bedrooms, age, location, ...)

This is where LINEAR ALGEBRA comes in!
- Data stored as MATRICES
- Predictions using DOT PRODUCTS
- Coefficients found using MATRIX OPERATIONS

Everything you learned comes together here!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/regression/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("📊 MULTIPLE LINEAR REGRESSION")
print("   Using Multiple Features for Better Predictions")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: FROM ONE FEATURE TO MANY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Why We Need Multiple Features")
print("=" * 80)
print()

print("THE LIMITATION OF SIMPLE REGRESSION:")
print("-" * 70)
print("Using only ONE feature limits our predictions!")
print()
print("Example: Predicting house prices")
print("  Simple regression:  Price = f(size)")
print("  But houses with same size can have VERY different prices!")
print()
print("Why? Because price also depends on:")
print("  • Number of bedrooms")
print("  • Age of house")
print("  • Location")
print("  • Condition")
print("  • School district")
print("  • ... and more!")
print()

print("THE SOLUTION: MULTIPLE LINEAR REGRESSION")
print("-" * 70)
print("Use MULTIPLE features to make better predictions!")
print()

# Generate sample data with multiple features
np.random.seed(42)
n_samples = 100

# Features
sizes = np.random.normal(1700, 400, n_samples)
bedrooms = np.random.randint(2, 6, n_samples)
ages = np.random.normal(15, 10, n_samples)

# True relationship with all three features
prices = (150 * sizes +
          15000 * bedrooms +
          -2000 * ages +
          50000 +
          np.random.normal(0, 20000, n_samples))

print("SAMPLE DATASET:")
print(f"{'Size (sqft)':<15} {'Bedrooms':<12} {'Age (yrs)':<12} {'Price ($)'}")
print("-" * 60)
for i in range(5):
    print(f"{sizes[i]:<15.0f} {bedrooms[i]:<12} {ages[i]:<12.1f} ${prices[i]:,.0f}")
print("...")
print(f"Total samples: {n_samples}")
print()

# ============================================================================
# SECTION 2: THE MULTIPLE REGRESSION EQUATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: The Multiple Regression Equation")
print("=" * 80)
print()

print("MATHEMATICAL FORM:")
print("-" * 70)
print("  ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃ + ... + βₙxₙ")
print()
print("Where:")
print("  • ŷ = predicted value")
print("  • β₀ = intercept (base value)")
print("  • β₁, β₂, β₃, ... = coefficients (one for each feature)")
print("  • x₁, x₂, x₃, ... = features (input variables)")
print()

print("FOR OUR HOUSE PRICE EXAMPLE:")
print("-" * 70)
print("  ŷ = β₀ + β₁×size + β₂×bedrooms + β₃×age")
print()
print("Each coefficient tells us:")
print("  • β₁: How much price changes per sqft (holding others constant)")
print("  • β₂: How much price changes per bedroom (holding others constant)")
print("  • β₃: How much price changes per year of age (holding others constant)")
print()

print("MATRIX FORM (The Elegant Way):")
print("-" * 70)
print("  ŷ = Xβ")
print()
print("Where:")
print("  • X = matrix of all features (n samples × m features)")
print("  • β = vector of coefficients (m + 1 values, including β₀)")
print("  • ŷ = vector of predictions (n values)")
print()
print("This is a DOT PRODUCT! Remember linear algebra?")
print()

# ============================================================================
# SECTION 3: BUILDING THE MODEL (WITH SCIKIT-LEARN)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Building a Multiple Regression Model")
print("=" * 80)
print()

print("PREPARING THE DATA:")
print("-" * 70)

# Create feature matrix X
X = np.column_stack([sizes, bedrooms, ages])
y = prices

print(f"Feature matrix X shape: {X.shape}")
print(f"  {X.shape[0]} samples (houses)")
print(f"  {X.shape[1]} features (size, bedrooms, age)")
print()
print("First few rows of X:")
print(f"{'Size':<12} {'Bedrooms':<12} {'Age':<12}")
print("-" * 40)
for i in range(5):
    print(f"{X[i,0]:<12.0f} {X[i,1]:<12.0f} {X[i,2]:<12.1f}")
print("...")
print()

print("Target vector y shape:", y.shape)
print(f"  {y.shape[0]} price values")
print()

print("TRAINING THE MODEL:")
print("-" * 70)
print("Code:")
print("  model = LinearRegression()")
print("  model.fit(X, y)")
print()

# Train model
model = LinearRegression()
model.fit(X, y)

print("✅ Model trained!")
print()

print("EXTRACTING COEFFICIENTS:")
print("-" * 70)

beta_0 = model.intercept_
beta_coeffs = model.coef_

print(f"β₀ (intercept) = ${beta_0:,.2f}")
print()
print("Feature coefficients:")
feature_names = ['Size (sqft)', 'Bedrooms', 'Age (years)']
for name, coef in zip(feature_names, beta_coeffs):
    print(f"  β ({name:<15}) = ${coef:>10,.2f}")
print()

print("OUR EQUATION:")
print("-" * 70)
print(f"ŷ = {beta_0:,.0f} + {beta_coeffs[0]:.2f}×size + {beta_coeffs[1]:.2f}×bedrooms + {beta_coeffs[2]:.2f}×age")
print()

print("INTERPRETING COEFFICIENTS:")
print("-" * 70)
print(f"• β₀ = ${beta_0:,.0f}")
print(f"  Base price (theoretical price for a 0 sqft, 0 bedroom, 0 year old house)")
print()
print(f"• β₁ = ${beta_coeffs[0]:.2f} per sqft")
print(f"  Each additional sqft adds ${beta_coeffs[0]:.2f} to price")
print(f"  (holding bedrooms and age constant)")
print()
print(f"• β₂ = ${beta_coeffs[1]:,.2f} per bedroom")
print(f"  Each additional bedroom adds ${beta_coeffs[1]:,.2f} to price")
print(f"  (holding size and age constant)")
print()
print(f"• β₃ = ${beta_coeffs[2]:.2f} per year")
print(f"  Each additional year {'reduces' if beta_coeffs[2] < 0 else 'adds'} ${abs(beta_coeffs[2]):.2f} to price")
print(f"  (holding size and bedrooms constant)")
print()

# ============================================================================
# SECTION 4: MAKING PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Making Predictions")
print("=" * 80)
print()

print("EXAMPLE PREDICTIONS:")
print("-" * 70)

# Make predictions
predictions = model.predict(X)

print(f"{'Size':<10} {'Beds':<6} {'Age':<8} {'Actual':<15} {'Predicted':<15} {'Error'}")
print("-" * 75)
for i in range(5):
    error = prices[i] - predictions[i]
    print(f"{X[i,0]:<10.0f} {X[i,1]:<6.0f} {X[i,2]:<8.1f} ${prices[i]:<14,.0f} ${predictions[i]:<14,.0f} ${error:,.0f}")
print("...")
print()

print("PREDICTION FOR A NEW HOUSE:")
print("-" * 70)

# New house to predict
new_house = np.array([[2000, 4, 8]])  # 2000 sqft, 4 bedrooms, 8 years old

print("House specs:")
print(f"  Size: 2000 sqft")
print(f"  Bedrooms: 4")
print(f"  Age: 8 years")
print()

# Manual calculation
manual_prediction = (beta_0 +
                    beta_coeffs[0] * new_house[0,0] +
                    beta_coeffs[1] * new_house[0,1] +
                    beta_coeffs[2] * new_house[0,2])

print("Manual calculation:")
print(f"  ŷ = {beta_0:,.0f} + {beta_coeffs[0]:.2f}×2000 + {beta_coeffs[1]:,.2f}×4 + {beta_coeffs[2]:.2f}×8")
print(f"  ŷ = {beta_0:,.0f} + {beta_coeffs[0]*2000:,.0f} + {beta_coeffs[1]*4:,.0f} + {beta_coeffs[2]*8:,.0f}")
print(f"  ŷ = ${manual_prediction:,.0f}")
print()

# Using model
model_prediction = model.predict(new_house)[0]
print(f"Using model.predict(): ${model_prediction:,.0f}")
print()

# ============================================================================
# SECTION 5: EVALUATING THE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Model Performance")
print("=" * 80)
print()

# Calculate metrics
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y, predictions)

print("PERFORMANCE METRICS:")
print("-" * 70)
print(f"Mean Squared Error (MSE):  ${mse:,.0f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.0f}")
print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
print()

print("INTERPRETATION:")
print("-" * 70)
print(f"• Our predictions are off by about ${rmse:,.0f} on average (RMSE)")
print(f"• Our model explains {r2*100:.2f}% of the variance in prices (R²)")
print(f"• {(1-r2)*100:.2f}% is due to factors we didn't measure or randomness")
print()

# Compare with simple regression (just size)
X_simple = sizes.reshape(-1, 1)
model_simple = LinearRegression()
model_simple.fit(X_simple, y)
predictions_simple = model_simple.predict(X_simple)
r2_simple = r2_score(y, predictions_simple)
rmse_simple = np.sqrt(mean_squared_error(y, predictions_simple))

print("COMPARISON: Multiple vs Simple Regression")
print("-" * 70)
print(f"{'Metric':<20} {'Simple (size only)':<20} {'Multiple (3 features)':<20} {'Improvement'}")
print("-" * 85)
print(f"{'R²':<20} {r2_simple:<20.4f} {r2:<20.4f} {'+' if r2 > r2_simple else ''}{(r2 - r2_simple):.4f}")
print(f"{'RMSE':<20} ${rmse_simple:<19,.0f} ${rmse:<19,.0f} ${-(rmse - rmse_simple):,.0f}")
print()
print(f"✅ Multiple regression is {'better' if r2 > r2_simple else 'not better'}!")
print(f"   Using 3 features improved R² by {(r2 - r2_simple):.4f}")
print()

# ============================================================================
# SECTION 6: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Which Features Matter Most?")
print("=" * 80)
print()

print("COMPARING COEFFICIENT MAGNITUDES:")
print("-" * 70)
print("⚠️  WARNING: Can't directly compare coefficients if features have different scales!")
print()
print("Current coefficients:")
for name, coef in zip(feature_names, beta_coeffs):
    print(f"  {name:<20}: ${coef:>12,.2f}")
print()
print("Problem: Size has larger coefficient, but is it actually more important?")
print("  • Size ranges from ~1000-2500 sqft")
print("  • Bedrooms ranges from 2-5")
print("  • Different scales!")
print()

print("SOLUTION: STANDARDIZE FEATURES")
print("-" * 70)
print("Convert all features to same scale (mean=0, std=1)")
print()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train on standardized data
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)

scaled_coeffs = model_scaled.coef_

print("Standardized coefficients (comparable!):")
print(f"{'Feature':<20} {'Coefficient':<15} {'Importance Rank'}")
print("-" * 60)

# Sort by absolute value
coef_importance = sorted(zip(feature_names, scaled_coeffs, abs(scaled_coeffs)),
                         key=lambda x: x[2], reverse=True)

for i, (name, coef, abs_coef) in enumerate(coef_importance, 1):
    print(f"{name:<20} {coef:>14,.2f} #{i}")

print()
print("INTERPRETATION:")
print("-" * 70)
most_important = coef_importance[0][0]
print(f"• {most_important} is the most important feature")
print(f"• Ranking based on how much each standardized feature affects price")
print(f"• Larger absolute value = more important")
print()

# ============================================================================
# VISUALIZATION 1: Multiple Regression Results
# ============================================================================
print("📊 Generating Visualization 1: Multiple Regression Results...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('📊 MULTIPLE LINEAR REGRESSION: Using Multiple Features',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y, predictions, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
min_val = min(y.min(), predictions.min())
max_val = max(y.max(), predictions.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect predictions')
ax.set_xlabel('Actual Price ($)', fontsize=10)
ax.set_ylabel('Predicted Price ($)', fontsize=10)
ax.set_title(f'Actual vs Predicted\n(R² = {r2:.3f})', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Residual plot
ax = axes[0, 1]
residuals = y - predictions
ax.scatter(predictions, residuals, alpha=0.6, s=50, edgecolor='black', linewidth=0.5)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Price ($)', fontsize=10)
ax.set_ylabel('Residual ($)', fontsize=10)
ax.set_title('Residual Plot\n(Should be random)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Feature coefficients
ax = axes[0, 2]
colors = ['green' if c > 0 else 'red' for c in beta_coeffs]
bars = ax.bar(feature_names, beta_coeffs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylabel('Coefficient Value', fontsize=10)
ax.set_title('Feature Coefficients\n(Impact on Price)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${height:,.0f}' if abs(height) > 1000 else f'${height:.2f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=8, fontweight='bold')

# Plot 4: Standardized coefficients (importance)
ax = axes[1, 0]
colors_scaled = ['green' if c > 0 else 'red' for c in scaled_coeffs]
bars = ax.bar(feature_names, scaled_coeffs, color=colors_scaled, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=1)
ax.set_ylabel('Standardized Coefficient', fontsize=10)
ax.set_title('Feature Importance\n(Standardized Scale)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:,.0f}',
            ha='center', va='bottom' if height > 0 else 'top',
            fontsize=9, fontweight='bold')

# Plot 5: Comparison with simple regression
ax = axes[1, 1]
metrics = ['R²', 'RMSE']
simple_vals = [r2_simple, rmse_simple/1000]  # RMSE in thousands for scale
multiple_vals = [r2, rmse/1000]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x_pos - width/2, simple_vals, width, label='Simple (1 feature)',
               color='lightblue', edgecolor='black')
bars2 = ax.bar(x_pos + width/2, multiple_vals, width, label='Multiple (3 features)',
               color='lightgreen', edgecolor='black')

ax.set_ylabel('Value', fontsize=10)
ax.set_title('Simple vs Multiple Regression', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom',
                fontsize=8)

# Plot 6: Summary
ax = axes[1, 2]
ax.text(0.5, 0.95, 'MODEL SUMMARY', fontsize=11, fontweight='bold',
        ha='center', transform=ax.transAxes)

summary_text = [
    "📊 EQUATION:",
    f"ŷ = {beta_0:,.0f}",
    f"  + {beta_coeffs[0]:.1f} × size",
    f"  + {beta_coeffs[1]:,.0f} × bedrooms",
    f"  + {beta_coeffs[2]:.1f} × age",
    "",
    "📈 PERFORMANCE:",
    f"R² = {r2:.4f} ({r2*100:.1f}%)",
    f"RMSE = ${rmse:,.0f}",
    "",
    "🎯 MOST IMPORTANT:",
    f"1. {coef_importance[0][0]}",
    f"2. {coef_importance[1][0]}",
    f"3. {coef_importance[2][0]}",
    "",
    "✅ IMPROVEMENT:",
    f"Multiple regression improved",
    f"R² by {(r2-r2_simple):.4f} compared",
    "to using size alone!"
]

y_pos = 0.87
for line in summary_text:
    if line.startswith(('📊', '📈', '🎯', '✅')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.043

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_multiple_regression.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 03_multiple_regression.png")
print()

# ============================================================================
# VISUALIZATION 2: 3D Regression Plane (for 2 features)
# ============================================================================
print("📊 Generating Visualization 2: 3D Regression Plane...")

# Train model with just 2 features for 3D visualization
X_2feat = X[:, :2]  # Just size and bedrooms
model_2feat = LinearRegression()
model_2feat.fit(X_2feat, y)

fig = plt.figure(figsize=(14, 10))

# Plot 1: 3D scatter and regression plane
ax1 = fig.add_subplot(221, projection='3d')

# Scatter plot
ax1.scatter(X[:,0], X[:,1], y, c=y, cmap='viridis', marker='o', s=50, alpha=0.6)

# Create mesh for regression plane
size_range = np.linspace(X[:,0].min(), X[:,0].max(), 20)
bed_range = np.linspace(X[:,1].min(), X[:,1].max(), 20)
size_mesh, bed_mesh = np.meshgrid(size_range, bed_range)
X_mesh = np.column_stack([size_mesh.ravel(), bed_mesh.ravel()])
price_mesh = model_2feat.predict(X_mesh).reshape(size_mesh.shape)

# Plot plane
ax1.plot_surface(size_mesh, bed_mesh, price_mesh, alpha=0.3, color='red')

ax1.set_xlabel('Size (sqft)', fontsize=9)
ax1.set_ylabel('Bedrooms', fontsize=9)
ax1.set_zlabel('Price ($)', fontsize=9)
ax1.set_title('3D Regression Plane\n(2 features)', fontsize=11, fontweight='bold')

# Plot 2: View from different angle
ax2 = fig.add_subplot(222, projection='3d')
ax2.scatter(X[:,0], X[:,1], y, c=y, cmap='viridis', marker='o', s=50, alpha=0.6)
ax2.plot_surface(size_mesh, bed_mesh, price_mesh, alpha=0.3, color='red')
ax2.set_xlabel('Size (sqft)', fontsize=9)
ax2.set_ylabel('Bedrooms', fontsize=9)
ax2.set_zlabel('Price ($)', fontsize=9)
ax2.set_title('3D Plane (Different Angle)', fontsize=11, fontweight='bold')
ax2.view_init(elev=20, azim=45)

# Plot 3: Correlation matrix
ax3 = fig.add_subplot(223)

# Create DataFrame for correlation
df = pd.DataFrame(X, columns=feature_names)
df['Price'] = y
corr_matrix = df.corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, ax=ax3, cbar_kws={'label': 'Correlation'})
ax3.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold')

# Plot 4: Explanation
ax4 = fig.add_subplot(224)
ax4.text(0.5, 0.9, '3D VISUALIZATION EXPLANATION', fontsize=11, fontweight='bold',
         ha='center', transform=ax4.transAxes)

explanation = [
    "📊 What you're seeing:",
    "",
    "• Each dot = one house",
    "• X-axis = Size",
    "• Y-axis = Bedrooms",
    "• Z-axis (height) = Price",
    "",
    "• Red plane = Our regression model",
    "• Plane tries to be close to all dots",
    "",
    "🎯 In 2D regression:",
    "   We fit a LINE through points",
    "",
    "🎯 In 3D regression (2 features):",
    "   We fit a PLANE through points",
    "",
    "🎯 With 3+ features:",
    "   We fit a HYPERPLANE",
    "   (can't visualize, but same idea!)",
    "",
    "💡 The math is the same:",
    "   Minimize distance from",
    "   points to plane/hyperplane"
]

y_pos = 0.78
for line in explanation:
    if line.startswith(('📊', '🎯', '💡', '•')):
        weight = 'bold'
        size = 9
    else:
        weight = 'normal'
        size = 8.5
    ax4.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax4.transAxes,
             family='monospace', fontweight=weight)
    y_pos -= 0.038

ax4.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}04_3d_regression_plane.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 04_3d_regression_plane.png")
print()

# ============================================================================
# SECTION 7: MULTICOLLINEARITY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Multicollinearity - When Features Correlate")
print("=" * 80)
print()

print("WHAT IS MULTICOLLINEARITY?")
print("-" * 70)
print("When two or more features are highly correlated with each other")
print()
print("Example problems:")
print("  • Size and Number of Rooms (larger houses have more rooms)")
print("  • Age and Condition (older houses often in worse condition)")
print("  • Income and Education (higher education → higher income)")
print()

print("WHY IS IT A PROBLEM?")
print("-" * 70)
print("• Hard to determine individual feature importance")
print("• Coefficients become unstable (small data changes → big coef changes)")
print("• Difficult to interpret results")
print()

# Check correlation in our data
print("CHECKING OUR DATA:")
print("-" * 70)
print("Correlation matrix:")
print(corr_matrix.round(2))
print()

# Find high correlations
high_corr_threshold = 0.7
print(f"Looking for correlations > {high_corr_threshold}:")
high_corr_found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > high_corr_threshold and corr_matrix.columns[i] != 'Price':
            print(f"  • {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_val:.2f}")
            high_corr_found = True

if not high_corr_found:
    print("  ✅ No high multicollinearity detected!")
    print("     Our features are relatively independent")
print()

print("SOLUTIONS TO MULTICOLLINEARITY:")
print("-" * 70)
print("1. Remove one of the correlated features")
print("2. Combine correlated features into one")
print("3. Use regularization (Ridge/Lasso regression)")
print("4. Collect more data")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: Multiple Linear Regression")
print("=" * 80)
print()

print("🎯 WHAT WE LEARNED:")
print("-" * 70)
print("1. MULTIPLE FEATURES:")
print("   ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ")
print("   • Use multiple features for better predictions")
print("   • Each feature gets its own coefficient")
print()

print("2. MATRIX FORM:")
print("   ŷ = Xβ")
print("   • Elegant representation using linear algebra")
print("   • Enables efficient computation")
print()

print("3. COEFFICIENT INTERPRETATION:")
print("   • Each β tells us feature's effect holding others constant")
print("   • Standardize features to compare importance")
print()

print("4. FEATURE IMPORTANCE:")
print("   • Standardized coefficients show relative importance")
print("   • Larger absolute value = more important")
print()

print("5. MULTICOLLINEARITY:")
print("   • Features correlating with each other")
print("   • Check correlation matrix")
print("   • Remove or combine correlated features")
print()

print("🤖 SCIKIT-LEARN CODE:")
print("-" * 70)
print("  # Prepare data")
print("  X = np.column_stack([feature1, feature2, feature3])")
print("  y = target")
print()
print("  # Train")
print("  model = LinearRegression()")
print("  model.fit(X, y)")
print()
print("  # Get coefficients")
print("  intercept = model.intercept_")
print("  coefficients = model.coef_")
print()
print("  # Predict")
print("  predictions = model.predict(X)")
print()

print("📊 KEY INSIGHTS:")
print("-" * 70)
print(f"  • Using {X.shape[1]} features instead of 1 improved R² by {(r2-r2_simple):.4f}")
print(f"  • Most important feature: {coef_importance[0][0]}")
print(f"  • Model explains {r2*100:.1f}% of price variance")
print("  • More features = better predictions (usually!)")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 03_multiple_regression.png")
print("✅ 04_3d_regression_plane.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review 3D visualization - see how the plane fits!")
print("   2. Check correlation matrix - understand feature relationships")
print("   3. Try adding your own features to the model")
print("   4. Next: examples/simple_examples.py (practice with real examples)")
print()

print("💡 REMEMBER:")
print("   Multiple regression is simple regression + more features!")
print("   Same principles, just using linear algebra for efficiency.")
print()

print("=" * 80)
print("🎉 MULTIPLE REGRESSION COMPLETE!")
print("   You can now use multiple features for predictions!")
print("=" * 80)
