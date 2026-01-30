"""
🏠 HOUSING DATA ANALYSIS - Complete Exploratory Data Analysis
===============================================================

PROJECT OVERVIEW:
----------------
Comprehensive exploratory data analysis (EDA) on the California Housing dataset.
Learn how to explore, visualize, and understand real-world data before modeling.

LEARNING OBJECTIVES:
-------------------
1. Loading and inspecting real datasets
2. Statistical summaries and distributions
3. Identifying missing values and outliers
4. Correlation analysis and feature relationships
5. Geographical visualization
6. Data quality assessment
7. Feature engineering insights

YOUTUBE RESOURCES:
-----------------
⭐ Ken Jee: "Data Science Project from Scratch - Part 2: EDA"
   https://www.youtube.com/watch?v=QWgg4w1SpJ8
   Real-world EDA walkthrough

📚 Keith Galli: "Complete Python Pandas Data Science Tutorial"
   https://www.youtube.com/watch?v=vmEHCJofslg
   Data manipulation and analysis

📚 Krish Naik: "Exploratory Data Analysis"
   Comprehensive EDA techniques

TIME: 2-3 hours
DIFFICULTY: Beginner-Intermediate
PREREQUISITES: 02_statistics_fundamentals.py, data_exploration.py

DATASET: California Housing (1990 Census)
-----------------------------------------
- 20,640 California districts
- Median house prices
- Features: location, house age, rooms, bedrooms, population, etc.
- Built into scikit-learn!
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import pandas
try:
    import pandas as pd
    pandas_available = True
except ImportError:
    pandas_available = False
    print("⚠ Pandas not available")

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'housing_analysis'
DATA_DIR = PROJECT_DIR / 'data'

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🏠 CALIFORNIA HOUSING DATA ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: LOAD AND INSPECT DATA
# ============================================================================

print("=" * 80)
print("SECTION 1: Loading and Initial Inspection")
print("=" * 80)
print()

try:
    from sklearn.datasets import fetch_california_housing

    if not pandas_available:
        raise ImportError("Pandas required for sklearn dataset")

    print("Loading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)

    # Get data and target
    df = housing.frame

    print(f"✓ Loaded {len(df)} housing districts")
    print()

    print("Dataset Description:")
    print("-" * 70)
    print(housing.DESCR[:500] + "...")
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not available. Creating sample dataset...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Create dict first, then convert to DataFrame if pandas available
    data_dict = {
        'MedInc': np.random.uniform(1, 15, n_samples),
        'HouseAge': np.random.uniform(1, 52, n_samples),
        'AveRooms': np.random.uniform(2, 10, n_samples),
        'AveBedrms': np.random.uniform(0.5, 3, n_samples),
        'Population': np.random.uniform(100, 3000, n_samples),
        'AveOccup': np.random.uniform(1, 10, n_samples),
        'Latitude': np.random.uniform(32, 42, n_samples),
        'Longitude': np.random.uniform(-125, -114, n_samples),
    }

    # Create target with realistic relationship
    data_dict['MedHouseVal'] = (
        2.5 * data_dict['MedInc'] +
        0.01 * data_dict['HouseAge'] +
        0.3 * data_dict['AveRooms'] +
        np.random.normal(0, 0.5, n_samples)
    )

    if pandas_available:
        df = pd.DataFrame(data_dict)
    else:
        # Use numpy structured array as fallback
        print("⚠ Creating numpy-based dataset (limited functionality)")
        df = type('SimpleDataFrame', (), {})()  # Simple object
        for key, val in data_dict.items():
            setattr(df, key, val)
        df.columns = list(data_dict.keys())

        # Add simple methods
        def get_column(col_name):
            return getattr(df, col_name)
        df.__getitem__ = lambda self, key: getattr(self, key) if isinstance(key, str) else None

    sklearn_available = False

# Basic info
print("DATASET OVERVIEW:")
print("-" * 70)
print(f"Number of samples: {len(df):,}")
print(f"Number of features: {len(df.columns) - 1}")
print(f"Target variable: MedHouseVal (median house value)")
print()

print("FEATURES:")
print("-" * 70)
feature_descriptions = {
    'MedInc': 'Median income in block group (tens of thousands)',
    'HouseAge': 'Median house age in block group',
    'AveRooms': 'Average number of rooms per household',
    'AveBedrms': 'Average number of bedrooms per household',
    'Population': 'Block group population',
    'AveOccup': 'Average number of household members',
    'Latitude': 'Block group latitude',
    'Longitude': 'Block group longitude',
}

for feature, description in feature_descriptions.items():
    if feature in df.columns:
        print(f"   • {feature:<12} - {description}")

print()

# First few rows
print("FIRST 5 SAMPLES:")
print("-" * 70)
print(df.head().to_string())
print()

# ============================================================================
# SECTION 2: STATISTICAL SUMMARY
# ============================================================================

print("=" * 80)
print("SECTION 2: Statistical Summary")
print("=" * 80)
print()

print("DESCRIPTIVE STATISTICS:")
print("-" * 70)
print(df.describe().to_string())
print()

print("KEY OBSERVATIONS:")
print("-" * 70)

# Median house value analysis
if 'MedHouseVal' in df.columns:
    median_price = df['MedHouseVal'].median()
    mean_price = df['MedHouseVal'].mean()
    min_price = df['MedHouseVal'].min()
    max_price = df['MedHouseVal'].max()

    print(f"Median House Value:")
    print(f"   • Mean:   ${mean_price:.2f} (hundreds of thousands)")
    print(f"   • Median: ${median_price:.2f}")
    print(f"   • Range:  ${min_price:.2f} - ${max_price:.2f}")
    print(f"   • Note: Values are in $100,000s (1990 dollars)")
    print()

# Income analysis
if 'MedInc' in df.columns:
    print(f"Median Income:")
    print(f"   • Mean:   ${df['MedInc'].mean():.2f} (tens of thousands)")
    print(f"   • Median: ${df['MedInc'].median():.2f}")
    print(f"   • Range:  ${df['MedInc'].min():.2f} - ${df['MedInc'].max():.2f}")
    print()

# House characteristics
if 'AveRooms' in df.columns and 'AveBedrms' in df.columns:
    print(f"House Characteristics:")
    print(f"   • Average rooms: {df['AveRooms'].mean():.2f}")
    print(f"   • Average bedrooms: {df['AveBedrms'].mean():.2f}")
    print(f"   • Rooms/Bedroom ratio: {(df['AveRooms']/df['AveBedrms']).mean():.2f}")
    print()

# ============================================================================
# SECTION 3: MISSING VALUES AND DATA QUALITY
# ============================================================================

print("=" * 80)
print("SECTION 3: Missing Values and Data Quality")
print("=" * 80)
print()

# Check for missing values
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

print("MISSING VALUES:")
print("-" * 70)
if missing_counts.sum() == 0:
    print("✓ No missing values found!")
else:
    print(f"{'Feature':<15} {'Missing':<10} {'Percentage'}")
    print("-" * 70)
    for feature in missing_counts.index:
        if missing_counts[feature] > 0:
            print(f"{feature:<15} {missing_counts[feature]:<10} {missing_percent[feature]:.2f}%")

print()

# Check for duplicates
n_duplicates = df.duplicated().sum()
print(f"Duplicate rows: {n_duplicates}")
print()

# Data types
print("DATA TYPES:")
print("-" * 70)
for col in df.columns:
    print(f"   {col:<15} - {df[col].dtype}")
print()

# ============================================================================
# SECTION 4: DISTRIBUTIONS
# ============================================================================

print("=" * 80)
print("SECTION 4: Distribution Analysis")
print("=" * 80)
print()

print("DISTRIBUTION CHARACTERISTICS:")
print("-" * 70)

for col in ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']:
    if col in df.columns:
        data = df[col]

        # Calculate skewness manually
        mean = data.mean()
        std = data.std()
        n = len(data)
        skew = ((data - mean) ** 3).sum() / (n * std ** 3)

        print(f"{col}:")
        print(f"   Mean: {mean:.2f}, Std: {std:.2f}")
        print(f"   Skewness: {skew:.2f}", end="")

        if abs(skew) < 0.5:
            print(" (Approximately symmetric)")
        elif skew > 0:
            print(" (Right-skewed: long tail on right)")
        else:
            print(" (Left-skewed: long tail on left)")
        print()

# ============================================================================
# SECTION 5: OUTLIER DETECTION
# ============================================================================

print("=" * 80)
print("SECTION 5: Outlier Detection")
print("=" * 80)
print()

print("Using IQR (Interquartile Range) method:")
print("   Outliers are values < Q1 - 1.5×IQR  OR  > Q3 + 1.5×IQR")
print()

for col in ['MedInc', 'AveRooms', 'Population', 'MedHouseVal']:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df)) * 100

        print(f"{col}:")
        print(f"   Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
        print(f"   Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"   Outliers: {outliers} ({outlier_pct:.1f}%)")
        print()

# ============================================================================
# SECTION 6: CORRELATION ANALYSIS
# ============================================================================

print("=" * 80)
print("SECTION 6: Correlation Analysis")
print("=" * 80)
print()

# Calculate correlation matrix
corr_matrix = df.corr()

print("CORRELATION WITH TARGET (MedHouseVal):")
print("-" * 70)
if 'MedHouseVal' in corr_matrix.columns:
    correlations = corr_matrix['MedHouseVal'].sort_values(ascending=False)

    for feature, corr in correlations.items():
        if feature != 'MedHouseVal':
            strength = ""
            if abs(corr) > 0.7:
                strength = "STRONG"
            elif abs(corr) > 0.4:
                strength = "MODERATE"
            elif abs(corr) > 0.2:
                strength = "WEAK"
            else:
                strength = "VERY WEAK"

            direction = "positive" if corr > 0 else "negative"
            print(f"   {feature:<12}: {corr:>6.3f}  ({strength} {direction})")

print()

print("INTERPRETATION:")
print("-" * 70)
print("Correlation coefficient ranges from -1 to +1:")
print("   • +1: Perfect positive correlation (both increase together)")
print("   •  0: No linear correlation")
print("   • -1: Perfect negative correlation (one increases, other decreases)")
print()
print("Strength interpretation:")
print("   • |r| > 0.7: Strong correlation")
print("   • |r| > 0.4: Moderate correlation")
print("   • |r| > 0.2: Weak correlation")
print("   • |r| ≤ 0.2: Very weak/no correlation")
print()

# ============================================================================
# SECTION 7: FEATURE RELATIONSHIPS
# ============================================================================

print("=" * 80)
print("SECTION 7: Key Feature Relationships")
print("=" * 80)
print()

if 'MedInc' in df.columns and 'MedHouseVal' in df.columns:
    # Income vs Price
    low_income = df[df['MedInc'] < 3]
    high_income = df[df['MedInc'] >= 8]

    print("INCOME IMPACT:")
    print("-" * 70)
    print(f"Low income areas (< $30k):")
    print(f"   Average house value: ${low_income['MedHouseVal'].mean():.2f}")
    print()
    print(f"High income areas (≥ $80k):")
    print(f"   Average house value: ${high_income['MedHouseVal'].mean():.2f}")
    print()

if 'HouseAge' in df.columns and 'MedHouseVal' in df.columns:
    # Age vs Price
    new_houses = df[df['HouseAge'] < 10]
    old_houses = df[df['HouseAge'] >= 40]

    print("AGE IMPACT:")
    print("-" * 70)
    print(f"New houses (< 10 years):")
    print(f"   Average house value: ${new_houses['MedHouseVal'].mean():.2f}")
    print()
    print(f"Old houses (≥ 40 years):")
    print(f"   Average house value: ${old_houses['MedHouseVal'].mean():.2f}")
    print()

if 'AveRooms' in df.columns and 'MedHouseVal' in df.columns:
    # Rooms vs Price
    small_houses = df[df['AveRooms'] < 5]
    large_houses = df[df['AveRooms'] >= 7]

    print("SIZE IMPACT:")
    print("-" * 70)
    print(f"Smaller houses (< 5 rooms avg):")
    print(f"   Average house value: ${small_houses['MedHouseVal'].mean():.2f}")
    print()
    print(f"Larger houses (≥ 7 rooms avg):")
    print(f"   Average house value: ${large_houses['MedHouseVal'].mean():.2f}")
    print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('California Housing: Distribution Analysis', fontsize=16, fontweight='bold')

# Select key features for visualization
viz_features = ['MedInc', 'HouseAge', 'AveRooms', 'Population', 'MedHouseVal']
viz_features = [f for f in viz_features if f in df.columns][:6]

for idx, feature in enumerate(viz_features):
    ax = axes[idx // 3, idx % 3]

    # Histogram
    ax.hist(df[feature], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.set_xlabel(feature, fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add mean and median lines
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend(fontsize=9)

# Hide extra subplot if needed
if len(viz_features) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_distributions.png")
plt.close()

# Visualization 2: Correlation heatmap
if len(df.columns) > 1:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    # Calculate correlation
    corr = df.corr()

    # Plot heatmap
    im = ax.imshow(corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(corr.columns, fontsize=10)

    # Add correlation values
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='bold')

    ax.set_title('Darker red = strong positive, Darker blue = strong negative',
                fontsize=11, pad=20)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_correlation_heatmap.png")
    plt.close()

# Visualization 3: Scatter plots (key relationships)
if 'MedInc' in df.columns and 'MedHouseVal' in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Feature vs House Value Relationships', fontsize=16, fontweight='bold')

    relationships = [
        ('MedInc', 'Median Income (tens of thousands $)'),
        ('HouseAge', 'House Age (years)'),
        ('AveRooms', 'Average Rooms'),
        ('Population', 'Population')
    ]

    for idx, (feature, label) in enumerate(relationships):
        if feature in df.columns:
            ax = axes[idx // 2, idx % 2]

            # Scatter plot with alpha for density
            ax.scatter(df[feature], df['MedHouseVal'], alpha=0.3, s=10, color='steelblue')
            ax.set_xlabel(label, fontsize=11, fontweight='bold')
            ax.set_ylabel('Median House Value ($100k)', fontsize=11, fontweight='bold')

            # Calculate and display correlation
            corr = df[[feature, 'MedHouseVal']].corr().iloc[0, 1]
            ax.set_title(f'{feature} vs Price (correlation: {corr:.3f})',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/03_scatter_relationships.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/03_scatter_relationships.png")
    plt.close()

# Visualization 4: Box plots (outlier visualization)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Box Plots: Outlier Detection', fontsize=16, fontweight='bold')

for idx, feature in enumerate(viz_features):
    ax = axes[idx // 3, idx % 3]

    # Box plot
    bp = ax.boxplot([df[feature].dropna()], labels=[feature], patch_artist=True)

    # Color the box
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_ylabel('Value', fontsize=11, fontweight='bold')
    ax.set_title(f'{feature} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    Q1 = df[feature].quantile(0.25)
    median = df[feature].median()
    Q3 = df[feature].quantile(0.75)

    stats_text = f'Q1: {Q1:.2f}\nMedian: {median:.2f}\nQ3: {Q3:.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Hide extra subplot if needed
if len(viz_features) < 6:
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/04_boxplots_outliers.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/04_boxplots_outliers.png")
plt.close()

# Visualization 5: Geographic map (if lat/lon available)
if 'Latitude' in df.columns and 'Longitude' in df.columns and 'MedHouseVal' in df.columns:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.suptitle('California Housing: Geographic Distribution', fontsize=14, fontweight='bold')

    # Scatter plot colored by price
    scatter = ax.scatter(df['Longitude'], df['Latitude'],
                        c=df['MedHouseVal'], cmap='YlOrRd',
                        s=20, alpha=0.6, edgecolors='none')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title('House Values by Location (darker = more expensive)', fontsize=12, pad=20)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Median House Value ($100k)', fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/05_geographic_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/05_geographic_distribution.png")
    plt.close()

print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🏠 ANALYSIS SUMMARY: Key Findings")
print("=" * 80)
print()

print("✓ DATASET OVERVIEW:")
print(f"   • {len(df):,} housing districts analyzed")
print(f"   • {len(df.columns)-1} features + 1 target variable")
print(f"   • No missing values" if missing_counts.sum() == 0 else f"   • {missing_counts.sum()} missing values")
print()

if 'MedHouseVal' in df.columns and 'MedInc' in df.columns:
    print("✓ KEY INSIGHTS:")
    print()
    strongest_corr = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False).iloc[1]
    strongest_feature = corr_matrix['MedHouseVal'].abs().sort_values(ascending=False).index[1]
    print(f"   1. STRONGEST PREDICTOR: {strongest_feature}")
    print(f"      • Correlation with price: {corr_matrix['MedHouseVal'][strongest_feature]:.3f}")
    print(f"      • This will be our most important feature for modeling")
    print()

    print(f"   2. PRICE RANGE:")
    print(f"      • Median: ${df['MedHouseVal'].median():.2f} ($100k)")
    print(f"      • Most houses are in ${df['MedHouseVal'].quantile(0.25):.2f} - ${df['MedHouseVal'].quantile(0.75):.2f} range")
    print()

    print(f"   3. OUTLIERS DETECTED:")
    for col in ['AveRooms', 'Population']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > 0:
                print(f"      • {col}: {outliers} outliers ({outliers/len(df)*100:.1f}%)")
    print()

print("✓ RECOMMENDATIONS FOR MODELING:")
print()
print("   1. FEATURE ENGINEERING:")
print("      • Create rooms_per_person = AveRooms / AveOccup")
print("      • Create bedrooms_ratio = AveBedrms / AveRooms")
print("      • Consider polynomial features for MedInc")
print()

print("   2. DATA PREPROCESSING:")
if 'Population' in df.columns:
    outlier_count = ((df['Population'] < df['Population'].quantile(0.25) - 1.5*(df['Population'].quantile(0.75)-df['Population'].quantile(0.25))) |
                    (df['Population'] > df['Population'].quantile(0.75) + 1.5*(df['Population'].quantile(0.75)-df['Population'].quantile(0.25)))).sum()
    if outlier_count > len(df) * 0.05:
        print(f"      • Handle outliers in Population ({outlier_count} detected)")
print("      • Feature scaling (standardization) - different ranges")
print("      • Consider log transformation for skewed features")
print()

print("   3. MODELING STRATEGY:")
print("      • Start with Linear Regression (strong linear relationships)")
print("      • Try Polynomial Regression for non-linear patterns")
print("      • Consider regularization (Ridge/Lasso) if overfitting")
print()

print("=" * 80)
print("🏠 Housing Analysis Complete!")
print(f"   Check visualizations: {VISUAL_DIR}/")
print("=" * 80)
