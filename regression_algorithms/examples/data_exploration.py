"""
🔍 DATA EXPLORATION - Understanding Your Data Before Modeling

================================================================================
LEARNING OBJECTIVES
================================================================================
Master Exploratory Data Analysis (EDA):
1. Load and inspect data
2. Calculate statistical summaries
3. Visualize distributions
4. Find correlations between features
5. Detect outliers
6. Prepare data for modeling

EDA is 80% of the work in real ML projects!

================================================================================
📺 RECOMMENDED VIDEOS
================================================================================
⭐ MUST WATCH:
   - Keith Galli: "Complete Python Pandas Data Science Tutorial"
     https://www.youtube.com/watch?v=vmEHCJofslg

   - Krish Naik: "Exploratory Data Analysis"
     https://www.youtube.com/watch?v=fHFOANOHwh8

   - Ken Jee: "Data Science Project from Scratch"
     https://www.youtube.com/watch?v=MpF9HENQjDo

================================================================================
OVERVIEW
================================================================================
Before building ANY model, you must understand your data!

Questions to answer:
- What does the data look like?
- Are there missing values?
- What's the distribution of each feature?
- How do features relate to each other?
- Are there outliers?
- What patterns exist?

Let's explore a real dataset step by step!
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Setup visualization directory
VISUAL_DIR = '../visuals/regression/'
os.makedirs(VISUAL_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

print("=" * 80)
print("🔍 EXPLORATORY DATA ANALYSIS (EDA)")
print("   Understanding Data Before Modeling")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: LOAD AND INSPECT DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Load and Initial Inspection")
print("=" * 80)
print()

print("GENERATING SAMPLE DATASET:")
print("-" * 70)
print("Simulating house price data with multiple features...")
print()

# Generate realistic house data
np.random.seed(42)
n_samples = 200

# Generate features with realistic relationships
sizes = np.random.normal(1800, 500, n_samples).clip(800, 4000)
bedrooms = np.random.choice([2, 3, 4, 5], n_samples, p=[0.1, 0.4, 0.4, 0.1])
bathrooms = bedrooms * 0.75 + np.random.normal(0, 0.3, n_samples)
bathrooms = np.round(bathrooms * 2) / 2  # Round to nearest 0.5
bathrooms = bathrooms.clip(1, 5)
age = np.random.exponential(15, n_samples).clip(0, 50)
garage = np.random.choice([0, 1, 2, 3], n_samples, p=[0.1, 0.3, 0.5, 0.1])
location_score = np.random.normal(7, 2, n_samples).clip(1, 10)

# Generate prices with relationships
prices = (150 * sizes +
          25000 * bedrooms +
          15000 * bathrooms +
          -1500 * age +
          8000 * garage +
          10000 * location_score +
          np.random.normal(0, 30000, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'Size_sqft': sizes,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Age_years': age,
    'Garage_spaces': garage,
    'Location_score': location_score,
    'Price': prices
})

# Add a few missing values for realism
missing_indices = np.random.choice(n_samples, 5, replace=False)
df.loc[missing_indices, 'Garage_spaces'] = np.nan

print("✅ Dataset created!")
print()

print("STEP 1: BASIC INFO")
print("-" * 70)
print(f"Dataset shape: {df.shape}")
print(f"  • {df.shape[0]} samples (houses)")
print(f"  • {df.shape[1]} columns (6 features + 1 target)")
print()

print("Column names and types:")
print(df.dtypes)
print()

print("STEP 2: FIRST FEW ROWS")
print("-" * 70)
print(df.head(10))
print()

print("STEP 3: BASIC STATISTICS")
print("-" * 70)
print(df.describe().round(2))
print()

print("KEY OBSERVATIONS:")
print("-" * 70)
print(f"  • Sizes range from {df['Size_sqft'].min():.0f} to {df['Size_sqft'].max():.0f} sqft")
print(f"  • Median size: {df['Size_sqft'].median():.0f} sqft")
print(f"  • Prices range from ${df['Price'].min():,.0f} to ${df['Price'].max():,.0f}")
print(f"  • Median price: ${df['Price'].median():,.0f}")
print(f"  • Average age: {df['Age_years'].mean():.1f} years")
print()

# ============================================================================
# SECTION 2: MISSING VALUES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Handling Missing Values")
print("=" * 80)
print()

print("CHECKING FOR MISSING VALUES:")
print("-" * 70)

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

print(f"{'Column':<20} {'Missing Count':<15} {'Percentage'}")
print("-" * 55)
for col in df.columns:
    print(f"{col:<20} {missing_count[col]:<15} {missing_percent[col]:.2f}%")

print()

if df.isnull().sum().sum() > 0:
    print("HANDLING MISSING VALUES:")
    print("-" * 70)
    print("Options:")
    print("  1. Remove rows with missing values (if few)")
    print("  2. Fill with mean/median (numerical)")
    print("  3. Fill with mode (categorical)")
    print("  4. Advanced: imputation techniques")
    print()

    print("For this dataset:")
    print(f"  • Garage_spaces: {missing_count['Garage_spaces']} missing ({missing_percent['Garage_spaces']:.1f}%)")
    print("  • Strategy: Fill with median (most common approach)")
    print()

    # Fill missing values
    df['Garage_spaces'].fillna(df['Garage_spaces'].median(), inplace=True)
    print("✅ Missing values handled!")
else:
    print("✅ No missing values detected!")

print()

# ============================================================================
# SECTION 3: DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Feature Distributions")
print("=" * 80)
print()

print("ANALYZING DISTRIBUTIONS:")
print("-" * 70)

# Analyze each numerical column
numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print(f"  Skewness: {df[col].skew():.2f}")

    if abs(df[col].skew()) < 0.5:
        dist_type = "Symmetric (Normal-ish)"
    elif df[col].skew() > 0:
        dist_type = "Right-skewed (tail on right)"
    else:
        dist_type = "Left-skewed (tail on left)"
    print(f"  Distribution: {dist_type}")

print()

# ============================================================================
# SECTION 4: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Correlation Analysis")
print("=" * 80)
print()

print("CALCULATING CORRELATIONS WITH PRICE:")
print("-" * 70)

# Calculate correlations with target
correlations = df.corr()['Price'].sort_values(ascending=False)

print(f"{'Feature':<20} {'Correlation with Price':<25} {'Strength'}")
print("-" * 70)
for feature, corr in correlations.items():
    if feature != 'Price':
        if abs(corr) > 0.7:
            strength = "Strong"
        elif abs(corr) > 0.4:
            strength = "Moderate"
        elif abs(corr) > 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"

        print(f"{feature:<20} {corr:>8.3f} {' ':<14} {strength}")

print()

print("INTERPRETATION:")
print("-" * 70)
print("Correlation ranges from -1 to +1:")
print("  •  +1: Perfect positive (both increase together)")
print("  •   0: No linear relationship")
print("  •  -1: Perfect negative (one increases, other decreases)")
print()
print("  • |r| > 0.7: Strong correlation")
print("  • |r| = 0.4-0.7: Moderate correlation")
print("  • |r| < 0.4: Weak correlation")
print()

# Feature correlations with each other
print("FEATURE INTERCORRELATIONS:")
print("-" * 70)
print("(Checking for multicollinearity)")
print()

feature_cols = [col for col in df.columns if col != 'Price']
feature_corr = df[feature_cols].corr()

# Find high correlations
high_corr_pairs = []
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        corr_val = feature_corr.iloc[i, j]
        if abs(corr_val) > 0.7:
            high_corr_pairs.append((feature_corr.columns[i],
                                   feature_corr.columns[j],
                                   corr_val))

if high_corr_pairs:
    print("High correlations found:")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"  • {feat1} ↔ {feat2}: r = {corr:.3f}")
    print()
    print("⚠️  Warning: High correlations may cause multicollinearity!")
else:
    print("✅ No concerning multicollinearity detected")

print()

# ============================================================================
# SECTION 5: OUTLIER DETECTION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Outlier Detection")
print("=" * 80)
print()

print("DETECTING OUTLIERS (Z-score method):")
print("-" * 70)
print("Outlier if |z-score| > 3 (beyond 3 standard deviations)")
print()

# Calculate z-scores
z_scores = np.abs(stats.zscore(df[numerical_cols]))
outliers = (z_scores > 3).any(axis=1)
n_outliers = outliers.sum()

print(f"Found {n_outliers} potential outliers ({n_outliers/len(df)*100:.1f}% of data)")
print()

if n_outliers > 0:
    print("Outlier details:")
    print(df[outliers].head())
    print()

    print("WHAT TO DO WITH OUTLIERS:")
    print("-" * 70)
    print("Options:")
    print("  1. Keep them (if they're valid data points)")
    print("  2. Remove them (if they're errors)")
    print("  3. Cap them (winsorization)")
    print("  4. Transform data (log transform)")
    print()
    print("For this analysis: We'll keep them (they seem valid)")
print()

# ============================================================================
# VISUALIZATION: COMPREHENSIVE EDA
# ============================================================================
print("📊 Generating Comprehensive EDA Visualization...")

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
fig.suptitle('🔍 EXPLORATORY DATA ANALYSIS: Complete Overview',
             fontsize=16, fontweight='bold')

# 1. Price distribution
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(df['Price'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(df['Price'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(df['Price'].median(), color='green', linestyle='--', linewidth=2, label='Median')
ax1.set_xlabel('Price ($)', fontsize=10)
ax1.set_ylabel('Frequency', fontsize=10)
ax1.set_title('Target Distribution: House Prices', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# 2. Size distribution
ax2 = fig.add_subplot(gs[0, 2:])
ax2.hist(df['Size_sqft'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Size (sqft)', fontsize=10)
ax2.set_ylabel('Frequency', fontsize=10)
ax2.set_title('Feature Distribution: House Size', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# 3-6. Box plots for numerical features
numerical_features = ['Size_sqft', 'Age_years', 'Bathrooms', 'Location_score']
colors_box = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']

for idx, (feat, color) in enumerate(zip(numerical_features, colors_box)):
    ax = fig.add_subplot(gs[1, idx])
    bp = ax.boxplot(df[feat], patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
    ax.set_ylabel(feat, fontsize=9)
    ax.set_title(f'{feat}\n(Check outliers)', fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

# 7. Correlation heatmap
ax7 = fig.add_subplot(gs[2, :2])
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={'label': 'Correlation'},
            ax=ax7)
ax7.set_title('Correlation Heatmap', fontsize=11, fontweight='bold')

# 8. Price vs Size scatter
ax8 = fig.add_subplot(gs[2, 2:])
scatter = ax8.scatter(df['Size_sqft'], df['Price'], c=df['Bedrooms'],
                     cmap='viridis', s=50, alpha=0.6, edgecolor='black', linewidth=0.5)
ax8.set_xlabel('Size (sqft)', fontsize=10)
ax8.set_ylabel('Price ($)', fontsize=10)
ax8.set_title('Price vs Size (colored by Bedrooms)', fontsize=11, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax8)
cbar.set_label('Bedrooms', fontsize=9)
ax8.grid(True, alpha=0.3)

# 9. Bedrooms distribution (bar chart)
ax9 = fig.add_subplot(gs[3, 0])
bedroom_counts = df['Bedrooms'].value_counts().sort_index()
ax9.bar(bedroom_counts.index, bedroom_counts.values, color='coral', edgecolor='black', alpha=0.7)
ax9.set_xlabel('Bedrooms', fontsize=10)
ax9.set_ylabel('Count', fontsize=10)
ax9.set_title('Bedroom Distribution', fontsize=10, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

# 10. Price vs Age
ax10 = fig.add_subplot(gs[3, 1])
ax10.scatter(df['Age_years'], df['Price'], alpha=0.5, s=40, edgecolor='black', linewidth=0.5)
ax10.set_xlabel('Age (years)', fontsize=10)
ax10.set_ylabel('Price ($)', fontsize=10)
ax10.set_title('Price vs Age', fontsize=10, fontweight='bold')
ax10.grid(True, alpha=0.3)

# 11. Location score vs Price
ax11 = fig.add_subplot(gs[3, 2])
ax11.scatter(df['Location_score'], df['Price'], alpha=0.5, s=40,
             edgecolor='black', linewidth=0.5, color='green')
ax11.set_xlabel('Location Score', fontsize=10)
ax11.set_ylabel('Price ($)', fontsize=10)
ax11.set_title('Price vs Location', fontsize=10, fontweight='bold')
ax11.grid(True, alpha=0.3)

# 12. Summary statistics
ax12 = fig.add_subplot(gs[3, 3])
ax12.text(0.5, 0.95, 'KEY FINDINGS', fontsize=11, fontweight='bold',
          ha='center', transform=ax12.transAxes)

summary_text = [
    f"📊 Dataset:",
    f"  • {len(df)} houses",
    f"  • {df.shape[1]-1} features",
    "",
    f"💰 Price:",
    f"  • Mean: ${df['Price'].mean():,.0f}",
    f"  • Range: ${df['Price'].min():,.0f}",
    f"           to ${df['Price'].max():,.0f}",
    "",
    f"🔗 Strongest correlations:",
    f"  1. {correlations.index[1]}: {correlations.iloc[1]:.2f}",
    f"  2. {correlations.index[2]}: {correlations.iloc[2]:.2f}",
    "",
    f"⚠️  Outliers: {n_outliers}",
    "",
    f"✅ Data quality: Good",
    f"   Ready for modeling!"
]

y_pos = 0.85
for line in summary_text:
    if line.startswith(('📊', '💰', '🔗', '⚠️', '✅')):
        weight = 'bold'
        size = 9
    else:
        weight = 'normal'
        size = 8
    ax12.text(0.5, y_pos, line, fontsize=size, ha='center',
              transform=ax12.transAxes, family='monospace', fontweight=weight)
    y_pos -= 0.052

ax12.axis('off')

plt.savefig(f'{VISUAL_DIR}06_data_exploration.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 06_data_exploration.png")
print()

# ============================================================================
# FINAL EDA CHECKLIST
# ============================================================================
print("\n" + "=" * 80)
print("✅ EDA CHECKLIST - What We Covered")
print("=" * 80)
print()

checklist = [
    ("Load data", "✅"),
    ("Check shape and types", "✅"),
    ("View first few rows", "✅"),
    ("Calculate statistics", "✅"),
    ("Check for missing values", "✅"),
    ("Handle missing values", "✅"),
    ("Analyze distributions", "✅"),
    ("Calculate correlations", "✅"),
    ("Check for multicollinearity", "✅"),
    ("Detect outliers", "✅"),
    ("Create visualizations", "✅"),
    ("Document findings", "✅")
]

print(f"{'Task':<35} {'Status'}")
print("-" * 45)
for task, status in checklist:
    print(f"{task:<35} {status}")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: EDA Complete!")
print("=" * 80)
print()

print("🎯 KEY FINDINGS:")
print("-" * 70)
print(f"1. Dataset has {len(df)} samples with {df.shape[1]-1} features")
print(f"2. Strongest predictor: {correlations.index[1]} (r = {correlations.iloc[1]:.3f})")
print(f"3. Price range: ${df['Price'].min():,.0f} to ${df['Price'].max():,.0f}")
print(f"4. {n_outliers} potential outliers detected ({n_outliers/len(df)*100:.1f}%)")
print("5. No severe multicollinearity issues")
print("6. Data quality: Good, ready for modeling")
print()

print("💡 NEXT STEPS FOR MODELING:")
print("-" * 70)
print("1. Feature selection based on correlations")
print("2. Consider feature engineering:")
print("   • Price per sqft = Price / Size")
print("   • Age category (new, old, very old)")
print("3. Handle outliers if needed")
print("4. Split data (train/test)")
print("5. Build and evaluate model")
print()

print("📊 EDA BEST PRACTICES:")
print("-" * 70)
print("✅ Always explore data BEFORE modeling")
print("✅ Check for missing values and outliers")
print("✅ Understand feature distributions")
print("✅ Identify correlations")
print("✅ Document your findings")
print("✅ Visualize, visualize, visualize!")
print()

print("=" * 80)
print("📁 Visualization saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 06_data_exploration.png")
print("=" * 80)
print()

print("🎓 WHAT'S NEXT:")
print("   1. Review the comprehensive EDA visualization")
print("   2. Practice EDA on your own datasets")
print("   3. Next: examples/model_evaluation.py (learn all metrics)")
print()

print("=" * 80)
print("🎉 DATA EXPLORATION MASTERED!")
print("   You now know how to analyze data like a pro!")
print("=" * 80)
