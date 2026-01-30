"""
💡 SIMPLE REGRESSION EXAMPLES - Practice Makes Perfect!

================================================================================
LEARNING OBJECTIVES
================================================================================
Practice linear regression with 3-4 fun, real-world examples:
1. Perfect linear relationship (temperature conversion)
2. Noisy linear relationship (advertising & sales)
3. Strong correlation (study hours & test scores)
4. Interpret coefficients and make predictions

Each example is self-contained and demonstrates key concepts!

================================================================================
📺 RECOMMENDED VIDEOS
================================================================================
   - Krish Naik: "Linear Regression Practical Implementation"
     https://www.youtube.com/watch?v=UZPfbG0jHKY

   - freeCodeCamp: "Linear Regression - Machine Learning"
     https://www.youtube.com/watch?v=5yfh5cf4-0w

================================================================================
OVERVIEW
================================================================================
Learn by doing! We'll work through multiple examples:
- Generate data
- Visualize relationships
- Build regression models
- Interpret results
- Make predictions

Each example teaches something new!
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import os

# Setup visualization directory
VISUAL_DIR = '../visuals/regression/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("💡 SIMPLE REGRESSION EXAMPLES")
print("   Learning Through Practice")
print("=" * 80)
print()

# ============================================================================
# EXAMPLE 1: PERFECT LINEAR RELATIONSHIP
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 1: Temperature Conversion (Perfect Relationship)")
print("=" * 80)
print()

print("SCENARIO:")
print("-" * 70)
print("Convert Celsius to Fahrenheit")
print("Formula: F = (9/5) × C + 32")
print()
print("This is PERFECT linear relationship (no noise!)")
print()

# Generate data
celsius = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
fahrenheit = (9/5) * celsius + 32

print("DATA:")
print(f"{'Celsius':<10} {'Fahrenheit':<12}")
print("-" * 25)
for c, f in zip(celsius, fahrenheit):
    print(f"{c:<10} {f:<12.1f}")
print()

# Build model
X = celsius.reshape(-1, 1)
y = fahrenheit

model_1 = LinearRegression()
model_1.fit(X, y)

beta_0 = model_1.intercept_
beta_1 = model_1.coef_[0]

print("MODEL RESULTS:")
print("-" * 70)
print(f"  β₀ (intercept) = {beta_0:.10f}")
print(f"  β₁ (slope) = {beta_1:.10f}")
print()
print(f"  Equation: F = {beta_0:.2f} + {beta_1:.4f} × C")
print()
print("COMPARE TO TRUE FORMULA:")
print(f"  True: F = 32 + (9/5) × C = 32 + 1.8 × C")
print(f"  Model: F = {beta_0:.2f} + {beta_1:.4f} × C")
print()

# Evaluate
predictions_1 = model_1.predict(X)
r2_1 = r2_score(y, predictions_1)
rmse_1 = np.sqrt(mean_squared_error(y, predictions_1))

print(f"  R² = {r2_1:.10f} (perfect fit!)")
print(f"  RMSE = {rmse_1:.10f} (essentially zero!)")
print()

print("KEY LESSON:")
print("  When there's a perfect linear relationship (no noise),")
print("  linear regression finds the EXACT equation!")
print()

# ============================================================================
# EXAMPLE 2: ADVERTISING & SALES (WITH NOISE)
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Advertising Spend vs Sales (Real-World with Noise)")
print("=" * 80)
print()

print("SCENARIO:")
print("-" * 70)
print("A company tracks advertising spend vs sales")
print("Relationship exists, but it's NOT perfect (other factors affect sales!)")
print()

# Generate data
np.random.seed(42)
ad_spend = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 10  # $10k units
# True relationship: sales = 3 * ad_spend + 20 + noise
sales = 3 * ad_spend + 20 + np.random.normal(0, 5, len(ad_spend))  # $1k units

print("DATA ($ in thousands):")
print(f"{'Ad Spend ($k)':<15} {'Sales ($k)':<12}")
print("-" * 30)
for ad, sale in zip(ad_spend, sales):
    print(f"${ad:<14.0f} ${sale:<11.1f}")
print()

# Build model
X = ad_spend.reshape(-1, 1)
y = sales

model_2 = LinearRegression()
model_2.fit(X, y)

beta_0_2 = model_2.intercept_
beta_1_2 = model_2.coef_[0]

print("MODEL RESULTS:")
print("-" * 70)
print(f"  Equation: Sales = {beta_0_2:.2f} + {beta_1_2:.3f} × AdSpend")
print()
print("INTERPRETATION:")
print(f"  • Base sales (no advertising): ${beta_0_2:.2f}k")
print(f"  • For each $1k in ads, sales increase by ${beta_1_2:.3f}k")
print(f"  • ROI: ${beta_1_2:.2f} in sales for every $1 in ads")
print()

# Evaluate
predictions_2 = model_2.predict(X)
r2_2 = r2_score(y, predictions_2)
rmse_2 = np.sqrt(mean_squared_error(y, predictions_2))

print(f"  R² = {r2_2:.4f} ({r2_2*100:.2f}% of variance explained)")
print(f"  RMSE = ${rmse_2:.2f}k (typical prediction error)")
print()

# Make a prediction
new_ad_spend = np.array([[150]])  # $150k
predicted_sales = model_2.predict(new_ad_spend)[0]

print("PREDICTION:")
print(f"  If we spend ${new_ad_spend[0,0]:.0f}k on advertising:")
print(f"  Predicted sales = ${beta_0_2:.2f}k + {beta_1_2:.3f} × {new_ad_spend[0,0]:.0f}")
print(f"                  = ${predicted_sales:.2f}k")
print()

print("KEY LESSON:")
print("  Real data has noise! R² < 1.0 is normal.")
print("  We can still make useful predictions despite imperfect fit.")
print()

# ============================================================================
# EXAMPLE 3: STUDY HOURS & TEST SCORES
# ============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Study Hours vs Test Scores")
print("=" * 80)
print()

print("SCENARIO:")
print("-" * 70)
print("Students track study hours and their test scores")
print("Does more studying lead to higher scores?")
print()

# Generate data
np.random.seed(123)
study_hours = np.random.uniform(0.5, 10, 30)
# True relationship: score = 5 * hours + 50 + noise
test_scores = 5 * study_hours + 50 + np.random.normal(0, 6, len(study_hours))
# Cap scores at 100
test_scores = np.minimum(test_scores, 100)

print("DATA (first 10 students):")
print(f"{'Study Hours':<15} {'Test Score':<12}")
print("-" * 30)
for hours, score in zip(study_hours[:10], test_scores[:10]):
    print(f"{hours:<15.1f} {score:<12.1f}")
print("...")
print(f"Total students: {len(study_hours)}")
print()

# Build model
X = study_hours.reshape(-1, 1)
y = test_scores

model_3 = LinearRegression()
model_3.fit(X, y)

beta_0_3 = model_3.intercept_
beta_1_3 = model_3.coef_[0]

print("MODEL RESULTS:")
print("-" * 70)
print(f"  Equation: Score = {beta_0_3:.2f} + {beta_1_3:.3f} × Hours")
print()
print("INTERPRETATION:")
print(f"  • Base score (no studying): {beta_0_3:.2f} points")
print(f"  • Each hour of study adds {beta_1_3:.3f} points to score")
print(f"  • Study 5 hours: predicted score = {beta_0_3 + beta_1_3*5:.1f}")
print(f"  • Study 10 hours: predicted score = {beta_0_3 + beta_1_3*10:.1f}")
print()

# Evaluate
predictions_3 = model_3.predict(X)
r2_3 = r2_score(y, predictions_3)
rmse_3 = np.sqrt(mean_squared_error(y, predictions_3))

print(f"  R² = {r2_3:.4f} ({r2_3*100:.2f}% of variance explained)")
print(f"  RMSE = {rmse_3:.2f} points (typical prediction error)")
print()

# Calculate correlation
correlation = np.corrcoef(study_hours, test_scores)[0, 1]
print(f"  Correlation: r = {correlation:.4f}")
print(f"  This is a {'strong' if abs(correlation) > 0.7 else 'moderate' if abs(correlation) > 0.4 else 'weak'} positive correlation")
print()

print("KEY LESSON:")
print(f"  Studying DOES help! R² = {r2_3:.3f} means {r2_3*100:.1f}% of score")
print("  variation is explained by study hours.")
print(f"  But {(1-r2_3)*100:.1f}% depends on other factors (prior knowledge,")
print("  test difficulty, sleep, etc.)")
print()

# ============================================================================
# VISUALIZATION: ALL THREE EXAMPLES
# ============================================================================
print("📊 Generating Visualization: All Examples...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('💡 SIMPLE REGRESSION EXAMPLES: Learning Through Practice',
             fontsize=16, fontweight='bold', y=0.995)

# Example 1: Temperature
ax = axes[0, 0]
ax.scatter(celsius, fahrenheit, s=100, color='blue', edgecolor='black', linewidth=1.5,
           label='Data', zorder=5)
x_line = np.linspace(celsius.min(), celsius.max(), 100)
y_line = model_1.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, 'r-', linewidth=3, label=f'ŷ = {beta_0:.1f} + {beta_1:.2f}x')

ax.set_xlabel('Celsius (°C)', fontsize=10)
ax.set_ylabel('Fahrenheit (°F)', fontsize=10)
ax.set_title(f'Example 1: Temperature Conversion\n(R² = {r2_1:.4f} - Perfect!)',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(50, 100, f'Each 1°C adds {beta_1:.2f}°F',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Example 1: Residuals
ax = axes[1, 0]
residuals_1 = fahrenheit - predictions_1
ax.scatter(predictions_1, residuals_1, s=80, color='purple', edgecolor='black', linewidth=1)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Fahrenheit', fontsize=10)
ax.set_ylabel('Residual', fontsize=10)
ax.set_title('Residuals (Nearly Zero!)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Example 2: Advertising
ax = axes[0, 1]
ax.scatter(ad_spend, sales, s=100, color='green', edgecolor='black', linewidth=1.5,
           label='Data', zorder=5)
x_line = np.linspace(ad_spend.min(), ad_spend.max(), 100)
y_line = model_2.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, 'r-', linewidth=3,
        label=f'ŷ = {beta_0_2:.1f} + {beta_1_2:.2f}x')

ax.set_xlabel('Ad Spend ($k)', fontsize=10)
ax.set_ylabel('Sales ($k)', fontsize=10)
ax.set_title(f'Example 2: Advertising & Sales\n(R² = {r2_2:.3f})',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(60, 40, f'ROI: ${beta_1_2:.2f} sales per $1 ad',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Example 2: Residuals
ax = axes[1, 1]
residuals_2 = sales - predictions_2
ax.scatter(predictions_2, residuals_2, s=80, color='green', edgecolor='black', linewidth=1)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Sales ($k)', fontsize=10)
ax.set_ylabel('Residual ($k)', fontsize=10)
ax.set_title('Residuals (Random Pattern Good!)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# Example 3: Study hours
ax = axes[0, 2]
ax.scatter(study_hours, test_scores, s=80, color='orange', edgecolor='black', linewidth=1,
           alpha=0.7, label='Students', zorder=5)
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
y_line = model_3.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, 'r-', linewidth=3,
        label=f'ŷ = {beta_0_3:.1f} + {beta_1_3:.1f}x')

ax.set_xlabel('Study Hours', fontsize=10)
ax.set_ylabel('Test Score', fontsize=10)
ax.set_title(f'Example 3: Study Hours & Scores\n(R² = {r2_3:.3f})',
             fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(5, 90, f'+{beta_1_3:.1f} pts per hour',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Example 3: Residuals
ax = axes[1, 2]
residuals_3 = test_scores - predictions_3
ax.scatter(predictions_3, residuals_3, s=80, color='orange', edgecolor='black',
           linewidth=1, alpha=0.7)
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Score', fontsize=10)
ax.set_ylabel('Residual (points)', fontsize=10)
ax.set_title('Residuals (Some Scatter)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}05_simple_examples.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 05_simple_examples.png")
print()

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON: All Three Examples")
print("=" * 80)
print()

print(f"{'Example':<25} {'R²':<10} {'RMSE':<15} {'Interpretation'}")
print("-" * 80)
print(f"{'1. Temperature':<25} {r2_1:<10.4f} {rmse_1:<15.6f} {'Perfect fit'}")
print(f"{'2. Advertising':<25} {r2_2:<10.4f} {rmse_2:<15.2f} {'Strong relationship'}")
print(f"{'3. Study Hours':<25} {r2_3:<10.4f} {rmse_3:<15.2f} {'Moderate relationship'}")
print()

print("KEY INSIGHTS:")
print("-" * 70)
print("1. PERFECT RELATIONSHIP (R² ≈ 1.0):")
print("   • No noise, exact formula")
print("   • RMSE ≈ 0")
print("   • Rare in real world!")
print()

print("2. STRONG RELATIONSHIP (R² > 0.8):")
print("   • Most variance explained")
print("   • Useful predictions")
print("   • Some unexplained variation")
print()

print("3. MODERATE RELATIONSHIP (R² = 0.5-0.8):")
print("   • Significant relationship exists")
print("   • Predictions have higher error")
print("   • Many other factors at play")
print()

print("WHEN IS R² 'GOOD ENOUGH'?")
print("-" * 70)
print("  • Depends on your field and goals!")
print("  • Social sciences: R² > 0.3 often considered decent")
print("  • Physical sciences: R² > 0.9 expected")
print("  • Business: R² > 0.6 usually useful")
print("  • Focus on: Is it useful for decisions?")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: What We Learned from These Examples")
print("=" * 80)
print()

print("🎯 KEY LESSONS:")
print("-" * 70)
print("1. LINEAR REGRESSION WORKS:")
print("   • Found correct relationship in all cases")
print("   • Even with noise, identified the pattern")
print()

print("2. R² TELLS THE STORY:")
print("   • R² = 1.0: Perfect fit (temperature)")
print(f"   • R² = {r2_2:.3f}: Strong fit (advertising)")
print(f"   • R² = {r2_3:.3f}: Moderate fit (study hours)")
print()

print("3. RESIDUALS MATTER:")
print("   • Should look random (no patterns)")
print("   • Centered around zero")
print("   • Constant variance across predictions")
print()

print("4. COEFFICIENTS ARE INTERPRETABLE:")
print("   • β₁ = rate of change")
print("   • Real-world meaning!")
print("   • Helps make decisions")
print()

print("💡 PRACTICE TIP:")
print("-" * 70)
print("Try modifying the code:")
print("  • Change the noise levels")
print("  • Create your own examples")
print("  • See how R² changes")
print("  • Experiment with different relationships")
print()

print("=" * 80)
print("📁 Visualization saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 05_simple_examples.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Study the visualizations - compare the three examples")
print("   2. Modify the code - try your own data")
print("   3. Next: examples/data_exploration.py (learn EDA)")
print()

print("=" * 80)
print("🎉 SIMPLE EXAMPLES COMPLETE!")
print("   You've practiced regression on 3 different scenarios!")
print("=" * 80)
