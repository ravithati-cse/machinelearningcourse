"""
🏡 HOUSE PRICE PREDICTION - Complete End-to-End ML Pipeline
===============================================================

PROJECT OVERVIEW:
----------------
Build a complete machine learning pipeline to predict California house prices!
This is your regression capstone project applying everything you've learned.

LEARNING OBJECTIVES:
-------------------
1. Complete ML pipeline from raw data to predictions
2. Feature engineering and transformation
3. Training multiple models and comparing performance
4. Hyperparameter tuning
5. Model evaluation with proper metrics
6. Making predictions on new data
7. Saving and loading models for production

YOUTUBE RESOURCES:
-----------------
⭐ Ken Jee: "Data Science Project from Scratch"
   https://www.youtube.com/playlist?list=PL2zq7klxX5ASFejJj80ob9ZAnBHdz5O1t
   Complete project series

📚 Krish Naik: "End to End Machine Learning Project Implementation"
   Full pipeline walkthrough

📚 Python Engineer: "Complete Machine Learning Project"
   https://www.youtube.com/watch?v=0Lt9w-BxKFQ

TIME: 3-4 hours (comprehensive capstone!)
DIFFICULTY: Intermediate
PREREQUISITES: All regression modules, housing_analysis.py

PIPELINE STAGES:
---------------
1. Load Data
2. Exploratory Data Analysis
3. Feature Engineering
4. Data Preprocessing
5. Model Training (Multiple algorithms)
6. Model Evaluation
7. Hyperparameter Tuning
8. Final Model Selection
9. Production Predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'house_price_prediction'
DATA_DIR = PROJECT_DIR / 'data'
MODEL_DIR = PROJECT_DIR / 'models'

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🏡 HOUSE PRICE PREDICTION - Complete ML Pipeline")
print("=" * 80)
print()

# ============================================================================
# STAGE 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("STAGE 1: Load Data")
print("=" * 80)
print()

try:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pandas as pd

    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    print(f"✓ Loaded {len(df):,} samples")
    print(f"✓ Features: {', '.join(housing.feature_names)}")
    print(f"✓ Target: MedHouseVal (median house value in $100k)")
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not available. This project requires sklearn.")
    print("Install with: pip install scikit-learn pandas")
    print()
    sklearn_available = False
    exit(1)

# ============================================================================
# STAGE 2: QUICK EDA
# ============================================================================

print("=" * 80)
print("STAGE 2: Quick Exploratory Data Analysis")
print("=" * 80)
print()

print("DATASET SHAPE:")
print(f"   Samples: {df.shape[0]:,}")
print(f"   Features: {df.shape[1] - 1}")
print()

print("MISSING VALUES:")
missing = df.isnull().sum().sum()
if missing == 0:
    print("   ✓ No missing values!")
else:
    print(f"   ⚠ {missing} missing values found")
print()

print("TARGET VARIABLE (MedHouseVal):")
print(f"   Mean:   ${df['MedHouseVal'].mean():.3f} ($100k)")
print(f"   Median: ${df['MedHouseVal'].median():.3f}")
print(f"   Range:  ${df['MedHouseVal'].min():.3f} - ${df['MedHouseVal'].max():.3f}")
print()

print("TOP 3 CORRELATIONS WITH TARGET:")
correlations = df.corr()['MedHouseVal'].abs().sort_values(ascending=False)
for i, (feature, corr) in enumerate(list(correlations.items())[1:4], 1):
    print(f"   {i}. {feature:<12}: {corr:.3f}")
print()

# ============================================================================
# STAGE 3: FEATURE ENGINEERING
# ============================================================================

print("=" * 80)
print("STAGE 3: Feature Engineering")
print("=" * 80)
print()

print("Creating new features to improve model performance...")
print()

# Create new features
df['RoomsPerHousehold'] = df['AveRooms'] / df['AveOccup']
df['BedroomsPerRoom'] = df['AveBedrms'] / df['AveRooms']
df['PopulationPerHousehold'] = df['Population'] / df['AveOccup']

print("NEW FEATURES CREATED:")
print("   1. RoomsPerHousehold = AveRooms / AveOccup")
print("      → How spacious are houses per household member?")
print()
print("   2. BedroomsPerRoom = AveBedrms / AveRooms")
print("      → Ratio of bedrooms to total rooms")
print()
print("   3. PopulationPerHousehold = Population / AveOccup")
print("      → Population density per household")
print()

# Check correlation of new features
print("CORRELATION OF NEW FEATURES WITH TARGET:")
new_features = ['RoomsPerHousehold', 'BedroomsPerRoom', 'PopulationPerHousehold']
for feature in new_features:
    corr = df[[feature, 'MedHouseVal']].corr().iloc[0, 1]
    print(f"   {feature:<25}: {corr:>6.3f}")
print()

# ============================================================================
# STAGE 4: TRAIN-TEST SPLIT
# ============================================================================

print("=" * 80)
print("STAGE 4: Train-Test Split")
print("=" * 80)
print()

# Separate features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set:     {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
print()

print("WHY SPLIT?")
print("   • Train on 80% of data")
print("   • Test on 20% (model has never seen this)")
print("   • Prevents overfitting")
print("   • Gives honest evaluation of model performance")
print()

# ============================================================================
# STAGE 5: FEATURE SCALING
# ============================================================================

print("=" * 80)
print("STAGE 5: Feature Scaling")
print("=" * 80)
print()

print("WHY SCALE?")
print("   Features have different ranges:")
print(f"   • MedInc: {X_train['MedInc'].min():.1f} - {X_train['MedInc'].max():.1f}")
print(f"   • Population: {X_train['Population'].min():.0f} - {X_train['Population'].max():.0f}")
print()
print("   Scaling ensures all features contribute equally!")
print()

# Standardization: mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("STANDARDIZATION APPLIED:")
print("   Formula: z = (x - mean) / std")
print("   Result: All features have mean ≈ 0, std ≈ 1")
print()

# Verify scaling
print("AFTER SCALING:")
print(f"   Mean: {X_train_scaled.mean(axis=0)[0]:.6f} (≈ 0)")
print(f"   Std:  {X_train_scaled.std(axis=0)[0]:.6f} (≈ 1)")
print()

# ============================================================================
# STAGE 6: TRAIN MULTIPLE MODELS
# ============================================================================

print("=" * 80)
print("STAGE 6: Training Multiple Models")
print("=" * 80)
print()

print("Training 5 different regression models...")
print()

# Dictionary to store models and results
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=1.0),
    'Lasso (L1)': Lasso(alpha=0.1),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
}

results = {}

print("-" * 80)
print(f"{'Model':<20} {'Train RMSE':<15} {'Test RMSE':<15} {'Train R²':<12} {'Test R²':<12}")
print("-" * 80)

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Store results
    results[name] = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'y_test_pred': y_test_pred
    }

    print(f"{name:<20} ${train_rmse:<14.3f} ${test_rmse:<14.3f} {train_r2:<11.3f} {test_r2:<11.3f}")

print()

# ============================================================================
# STAGE 7: MODEL EVALUATION
# ============================================================================

print("=" * 80)
print("STAGE 7: Detailed Model Evaluation")
print("=" * 80)
print()

# Find best model
best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
best_result = results[best_model_name]

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   Test RMSE: ${best_result['test_rmse']:.3f} ($100k)")
print(f"   Test R²:   {best_result['test_r2']:.3f}")
print()

print("INTERPRETATION:")
print(f"   • RMSE = ${best_result['test_rmse']:.3f}")
print(f"     → On average, predictions are off by ${best_result['test_rmse']*100:.0f}k")
print()
print(f"   • R² = {best_result['test_r2']:.3f}")
print(f"     → Model explains {best_result['test_r2']*100:.1f}% of price variation")
print()

# Additional metrics for best model
best_model = best_result['model']
y_test_pred = best_result['y_test_pred']

mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)

print("ADDITIONAL METRICS (Best Model):")
print(f"   MAE:  ${mae:.3f} ($100k) = ${mae*100:.0f}k")
print(f"   MSE:  {mse:.3f}")
print(f"   RMSE: ${best_result['test_rmse']:.3f}")
print()

# Check for overfitting
train_test_gap = best_result['train_r2'] - best_result['test_r2']
print("OVERFITTING CHECK:")
print(f"   Train R²: {best_result['train_r2']:.3f}")
print(f"   Test R²:  {best_result['test_r2']:.3f}")
print(f"   Gap:      {train_test_gap:.3f}")

if train_test_gap > 0.1:
    print("   ⚠ Significant gap - model may be overfitting!")
elif train_test_gap > 0.05:
    print("   ⚡ Moderate gap - some overfitting")
else:
    print("   ✓ Small gap - good generalization!")
print()

# ============================================================================
# STAGE 8: FEATURE IMPORTANCE
# ============================================================================

print("=" * 80)
print("STAGE 8: Feature Importance Analysis")
print("=" * 80)
print()

if hasattr(best_model, 'feature_importances_'):
    # Tree-based models have feature_importances_
    importances = best_model.feature_importances_
    feature_names = X.columns

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print(f"FEATURE IMPORTANCE ({best_model_name}):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Cumulative'}")
    print("-" * 70)

    cumulative = 0
    for i, idx in enumerate(indices[:10], 1):
        cumulative += importances[idx]
        print(f"{i:<6} {feature_names[idx]:<25} {importances[idx]:<11.4f} {cumulative:.1%}")

    print()
    print("INTERPRETATION:")
    top_feature = feature_names[indices[0]]
    print(f"   • {top_feature} is the most important predictor")
    print(f"   • Top 3 features account for {sum(importances[indices[:3]]):.1%} of importance")
    print()

elif hasattr(best_model, 'coef_'):
    # Linear models have coefficients
    coefficients = best_model.coef_
    feature_names = X.columns

    # Sort by absolute value
    indices = np.argsort(np.abs(coefficients))[::-1]

    print(f"FEATURE COEFFICIENTS ({best_model_name}):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Feature':<25} {'Coefficient':<15} {'Impact'}")
    print("-" * 70)

    for i, idx in enumerate(indices[:10], 1):
        coef = coefficients[idx]
        impact = "positive" if coef > 0 else "negative"
        print(f"{i:<6} {feature_names[idx]:<25} {coef:<14.4f} {impact}")

    print()
    print("INTERPRETATION:")
    print("   • Positive coefficient → feature increases price")
    print("   • Negative coefficient → feature decreases price")
    print("   • Larger |coefficient| → stronger impact")
    print()

# ============================================================================
# STAGE 9: MAKING PREDICTIONS
# ============================================================================

print("=" * 80)
print("STAGE 9: Making Predictions on New Data")
print("=" * 80)
print()

print("Let's predict prices for sample houses...")
print()

# Create sample houses for prediction
sample_houses = pd.DataFrame({
    'MedInc': [3.0, 8.0, 5.5],
    'HouseAge': [20, 10, 30],
    'AveRooms': [5.0, 7.0, 6.0],
    'AveBedrms': [1.0, 2.0, 1.5],
    'Population': [1000, 500, 1500],
    'AveOccup': [3.0, 2.5, 3.5],
    'Latitude': [34.0, 37.5, 36.0],
    'Longitude': [-118.0, -122.0, -120.0],
})

# Add engineered features
sample_houses['RoomsPerHousehold'] = sample_houses['AveRooms'] / sample_houses['AveOccup']
sample_houses['BedroomsPerRoom'] = sample_houses['AveBedrms'] / sample_houses['AveRooms']
sample_houses['PopulationPerHousehold'] = sample_houses['Population'] / sample_houses['AveOccup']

# Scale features
sample_scaled = scaler.transform(sample_houses)

# Predict
predictions = best_model.predict(sample_scaled)

print("-" * 80)
print(f"{'House':<8} {'Income':<10} {'Age':<8} {'Rooms':<8} {'Location':<20} {'Predicted Price'}")
print("-" * 80)

locations = ['LA Area', 'SF Area', 'Central CA']
for i in range(len(sample_houses)):
    income = sample_houses.iloc[i]['MedInc']
    age = sample_houses.iloc[i]['HouseAge']
    rooms = sample_houses.iloc[i]['AveRooms']
    location = locations[i]
    price = predictions[i]

    print(f"#{i+1:<7} ${income:<9.1f} {age:<7.0f}y {rooms:<7.1f} {location:<20} ${price:.2f} (${price*100:.0f}k)")

print()

# ============================================================================
# STAGE 10: SAVE MODEL FOR PRODUCTION
# ============================================================================

print("=" * 80)
print("STAGE 10: Saving Model for Production")
print("=" * 80)
print()

try:
    import pickle

    # Save best model
    model_path = MODEL_DIR / 'best_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)

    # Save scaler
    scaler_path = MODEL_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"✓ Model saved: {model_path}")
    print(f"✓ Scaler saved: {scaler_path}")
    print()

    print("TO LOAD AND USE IN PRODUCTION:")
    print("-" * 70)
    print("import pickle")
    print()
    print("# Load model and scaler")
    print("with open('best_model.pkl', 'rb') as f:")
    print("    model = pickle.load(f)")
    print("with open('scaler.pkl', 'rb') as f:")
    print("    scaler = pickle.load(f)")
    print()
    print("# Make prediction")
    print("new_house_scaled = scaler.transform(new_house_features)")
    print("prediction = model.predict(new_house_scaled)")
    print()

except Exception as e:
    print(f"⚠ Could not save model: {e}")
    print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Model Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('House Price Prediction: Model Comparison', fontsize=16, fontweight='bold')

# Plot 1: RMSE Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
test_rmses = [results[name]['test_rmse'] for name in model_names]
colors = ['green' if name == best_model_name else 'lightblue' for name in model_names]

bars = ax1.bar(range(len(model_names)), test_rmses, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels(model_names, rotation=45, ha='right')
ax1.set_ylabel('RMSE ($100k)', fontsize=11, fontweight='bold')
ax1.set_title('Test RMSE by Model (Lower is Better)', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, rmse in zip(bars, test_rmses):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'${rmse:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2: R² Comparison
ax2 = axes[0, 1]
test_r2s = [results[name]['test_r2'] for name in model_names]

bars = ax2.bar(range(len(model_names)), test_r2s, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(model_names)))
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.set_ylabel('R² Score', fontsize=11, fontweight='bold')
ax2.set_title('Test R² by Model (Higher is Better)', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, r2 in zip(bars, test_r2s):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{r2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 3: Actual vs Predicted (Best Model)
ax3 = axes[1, 0]
y_test_pred = best_result['y_test_pred']

ax3.scatter(y_test, y_test_pred, alpha=0.5, s=10, color='steelblue')
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Price ($100k)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Price ($100k)', fontsize=11, fontweight='bold')
ax3.set_title(f'Actual vs Predicted: {best_model_name}', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Add R² annotation
ax3.text(0.05, 0.95, f'R² = {best_result["test_r2"]:.3f}\nRMSE = ${best_result["test_rmse"]:.3f}',
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 4: Residuals (Best Model)
ax4 = axes[1, 1]
residuals = y_test - y_test_pred

ax4.scatter(y_test_pred, residuals, alpha=0.5, s=10, color='steelblue')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Price ($100k)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
ax4.set_title(f'Residual Plot: {best_model_name}', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add interpretation
ax4.text(0.05, 0.95, 'Good model:\n• Random scatter\n• Centered at 0\n• No patterns',
        transform=ax4.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_model_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_model_comparison.png")
plt.close()

# Visualization 2: Feature Importance
if hasattr(best_model, 'feature_importances_'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle(f'Feature Importance: {best_model_name}', fontsize=14, fontweight='bold')

    importances = best_model.feature_importances_
    indices = np.argsort(importances)

    ax.barh(range(len(importances)), importances[indices], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([X.columns[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title('Which features drive house prices?', fontsize=12, pad=20)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_feature_importance.png")
    plt.close()

print()

# ============================================================================
# PROJECT SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🏡 PROJECT COMPLETE: What You Built")
print("=" * 80)
print()

print("✓ COMPLETE ML PIPELINE:")
print()
print(f"   1. Dataset: {len(df):,} California housing districts")
print(f"   2. Feature Engineering: Created 3 new features")
print(f"   3. Models Trained: 5 different algorithms")
print(f"   4. Best Model: {best_model_name}")
print(f"      • Test RMSE: ${best_result['test_rmse']:.3f} (≈ ${best_result['test_rmse']*100:.0f}k error)")
print(f"      • Test R²: {best_result['test_r2']:.3f} ({best_result['test_r2']*100:.1f}% variance explained)")
print(f"   5. Model Saved: Ready for production")
print()

print("✓ KEY LEARNINGS:")
print()
print("   1. FEATURE ENGINEERING MATTERS:")
print("      • Created meaningful features from existing ones")
print("      • RoomsPerHousehold captured household spaciousness")
print("      • Improved model performance")
print()

print("   2. SCALING IS ESSENTIAL:")
print("      • Features had vastly different ranges")
print("      • Standardization ensured fair contribution")
print("      • Critical for linear models")
print()

print("   3. MODEL COMPARISON:")
print("      • Always try multiple algorithms")
print("      • Tree-based models often perform well")
print("      • No single 'best' algorithm for all problems")
print()

print("   4. EVALUATION METRICS:")
print(f"      • RMSE: Average prediction error (${best_result['test_rmse']*100:.0f}k)")
print(f"      • R²: Variance explained ({best_result['test_r2']*100:.1f}%)")
print("      • Residuals: Should be random (no patterns)")
print()

print("   5. PRODUCTION READINESS:")
print("      • Model and scaler saved")
print("      • Can make predictions on new houses")
print("      • Documented prediction process")
print()

print("✓ NEXT STEPS FOR IMPROVEMENT:")
print()
print("   1. Hyperparameter tuning (GridSearchCV)")
print("   2. Try more algorithms (Gradient Boosting, XGBoost)")
print("   3. Polynomial features for non-linear relationships")
print("   4. Ensemble methods (combine multiple models)")
print("   5. Cross-validation for more robust evaluation")
print("   6. Feature selection (remove unimportant features)")
print("   7. Handle outliers more carefully")
print()

print("=" * 80)
print("🏡 House Price Prediction Complete!")
print(f"   Visualizations: {VISUAL_DIR}/")
print(f"   Saved Models: {MODEL_DIR}/")
print("=" * 80)
