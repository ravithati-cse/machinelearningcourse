"""
STATISTICS FUNDAMENTALS FOR MACHINE LEARNING
============================================

Statistics is the FOUNDATION of understanding data and machine learning!
This module teaches essential statistical concepts with lots of visuals and examples.

LEARNING OBJECTIVES:
-------------------
1. Calculate and interpret mean, median, and mode
2. Understand variance and standard deviation
3. Master covariance and correlation
4. See how statistics apply to machine learning
5. Visualize distributions and relationships

📺 RECOMMENDED VIDEOS (watch for deeper understanding):
----------------------------------------------------
⭐ StatQuest: "Mean, Median, Mode and Range"
   https://www.youtube.com/watch?v=h8EYEJ32oQ8

⭐ Khan Academy: "Variance and Standard Deviation"
   https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/variance-standard-deviation-sample/v/statistics-sample-variance

⭐ StatQuest: "Covariance and Correlation"
   https://www.youtube.com/watch?v=qtaqvPAeEJY

⭐ zedstatistics: "Understanding Correlation"
   https://www.youtube.com/watch?v=11c9cs6WpJU

VISUAL-FIRST APPROACH:
---------------------
Run this file to generate 7+ visualizations in visuals/02_statistics/
Study the images to build intuition before diving into formulas!
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Visual styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

# Create output directory
VISUAL_DIR = '../visuals/02_statistics/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 70)
print("STATISTICS FUNDAMENTALS: The Language of Data")
print("=" * 70)
print()

# ============================================================================
# SECTION 1: MEASURES OF CENTRAL TENDENCY
# ============================================================================

print("SECTION 1: Measures of Central Tendency")
print("-" * 70)
print("(Where is the 'center' of our data?)")
print()

# Sample dataset
data = [2, 4, 4, 4, 5, 5, 7, 9]
print(f"Sample data: {data}")
print()

# MEAN (Average)
mean = sum(data) / len(data)
print("1. MEAN (μ - pronounced 'mu')")
print("   Formula: μ = (Σx) / n")
print("   Sum all values and divide by count")
print(f"   Calculation: ({' + '.join(map(str, data))}) / {len(data)}")
print(f"   Mean = {mean}")
print("   📌 Use when: Data is symmetric and has no extreme outliers")
print()

# MEDIAN (Middle value)
sorted_data = sorted(data)
n = len(sorted_data)
if n % 2 == 0:
    median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
else:
    median = sorted_data[n//2]

print("2. MEDIAN (Middle Value)")
print("   Formula: Middle value when data is sorted")
print(f"   Sorted data: {sorted_data}")
print(f"   Median = {median}")
print("   📌 Use when: Data has outliers or is skewed")
print()

# MODE (Most frequent)
from collections import Counter
mode_dict = Counter(data)
mode = mode_dict.most_common(1)[0][0]
print("3. MODE (Most Frequent)")
print(f"   Most common value: {mode} (appears {mode_dict[mode]} times)")
print("   📌 Use when: Working with categorical or discrete data")
print()

# ============================================================================
# VISUALIZATION 1: MEAN VS MEDIAN VS MODE
# ============================================================================

print("Creating Visualization: Mean vs Median vs Mode...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📊 MEASURES OF CENTRAL TENDENCY: Mean, Median, Mode',
             fontsize=18, fontweight='bold')

# Dataset 1: Symmetric (mean ≈ median)
ax1 = axes[0, 0]
symmetric_data = np.random.normal(50, 10, 1000)
ax1.hist(symmetric_data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
mean_sym = np.mean(symmetric_data)
median_sym = np.median(symmetric_data)
ax1.axvline(mean_sym, color='red', linewidth=3, label=f'Mean = {mean_sym:.1f}')
ax1.axvline(median_sym, color='green', linewidth=3, linestyle='--', label=f'Median = {median_sym:.1f}')
ax1.set_title('SYMMETRIC DATA: Mean ≈ Median', fontweight='bold', fontsize=14)
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')
ax1.legend(fontsize=11)
ax1.text(0.5, 0.95, 'Normal distribution\nMean and median are nearly equal',
         transform=ax1.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Dataset 2: Right-skewed (mean > median)
ax2 = axes[0, 1]
skewed_data = np.random.exponential(20, 1000)
ax2.hist(skewed_data, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
mean_skew = np.mean(skewed_data)
median_skew = np.median(skewed_data)
ax2.axvline(mean_skew, color='red', linewidth=3, label=f'Mean = {mean_skew:.1f}')
ax2.axvline(median_skew, color='green', linewidth=3, linestyle='--', label=f'Median = {median_skew:.1f}')
ax2.set_title('RIGHT-SKEWED DATA: Mean > Median', fontweight='bold', fontsize=14)
ax2.set_xlabel('Value')
ax2.set_ylabel('Frequency')
ax2.legend(fontsize=11)
ax2.text(0.5, 0.95, 'Mean pulled by high outliers\nMedian more representative',
         transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Dataset 3: With outliers
ax3 = axes[1, 0]
normal_points = np.random.normal(50, 5, 95)
outliers = [90, 95, 100, 105, 110]
data_with_outliers = np.concatenate([normal_points, outliers])
ax3.hist(data_with_outliers, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
mean_out = np.mean(data_with_outliers)
median_out = np.median(data_with_outliers)
ax3.axvline(mean_out, color='red', linewidth=3, label=f'Mean = {mean_out:.1f}')
ax3.axvline(median_out, color='green', linewidth=3, linestyle='--', label=f'Median = {median_out:.1f}')
ax3.set_title('DATA WITH OUTLIERS: Median Robust', fontweight='bold', fontsize=14)
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.legend(fontsize=11)
ax3.text(0.5, 0.95, 'Outliers strongly affect mean\nMedian stays stable',
         transform=ax3.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Dataset 4: Multimodal (mode different from mean/median)
ax4 = axes[1, 1]
mode_data = np.concatenate([np.random.normal(30, 5, 500),
                            np.random.normal(70, 5, 500)])
ax4.hist(mode_data, bins=30, edgecolor='black', alpha=0.7, color='plum')
mean_mode = np.mean(mode_data)
median_mode = np.median(mode_data)
ax4.axvline(mean_mode, color='red', linewidth=3, label=f'Mean = {mean_mode:.1f}')
ax4.axvline(median_mode, color='green', linewidth=3, linestyle='--', label=f'Median = {median_mode:.1f}')
ax4.axvline(30, color='blue', linewidth=3, linestyle=':', label='Mode 1 ≈ 30')
ax4.axvline(70, color='blue', linewidth=3, linestyle=':', label='Mode 2 ≈ 70')
ax4.set_title('BIMODAL DATA: Two Peaks', fontweight='bold', fontsize=14)
ax4.set_xlabel('Value')
ax4.set_ylabel('Frequency')
ax4.legend(fontsize=10)
ax4.text(0.5, 0.95, 'Two distinct groups\nModes show the peaks',
         transform=ax4.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_mean_median_mode_comparison.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}01_mean_median_mode_comparison.png")
plt.close()

# ============================================================================
# SECTION 2: MEASURES OF SPREAD (VARIABILITY)
# ============================================================================

print()
print("SECTION 2: Measures of Spread")
print("-" * 70)
print("(How spread out is our data?)")
print()

data = [2, 4, 4, 4, 5, 5, 7, 9]
mean = np.mean(data)

# VARIANCE
print("1. VARIANCE (σ² - pronounced 'sigma squared')")
print("   Formula: σ² = Σ(x - μ)² / n")
print("   Steps:")
print(f"   a) Calculate mean: μ = {mean}")
print("   b) Find deviations from mean:")
for x in data:
    print(f"      {x} - {mean} = {x - mean}")
print("   c) Square each deviation (make all positive):")
squared_devs = [(x - mean)**2 for x in data]
for x, sq in zip(data, squared_devs):
    print(f"      ({x} - {mean})² = {sq:.2f}")
print(f"   d) Average the squared deviations:")
variance = np.var(data)
print(f"      Variance = {variance:.2f}")
print("   📌 Why square? Makes all values positive and penalizes large deviations")
print()

# STANDARD DEVIATION
std = np.std(data)
print("2. STANDARD DEVIATION (σ - pronounced 'sigma')")
print("   Formula: σ = √variance")
print(f"   Standard Deviation = √{variance:.2f} = {std:.2f}")
print("   📌 Same units as original data (easier to interpret)")
print("   📌 68% of data within 1σ, 95% within 2σ, 99.7% within 3σ")
print()

# ============================================================================
# VISUALIZATION 2: VARIANCE AND STANDARD DEVIATION
# ============================================================================

print("Creating Visualization: Variance and Standard Deviation...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('📏 MEASURES OF SPREAD: Variance & Standard Deviation',
             fontsize=18, fontweight='bold')

# Plot 1: Visual explanation of variance calculation
ax1 = axes[0, 0]
sample_data = [2, 4, 4, 4, 5, 5, 7, 9]
sample_mean = np.mean(sample_data)
x_pos = range(len(sample_data))

ax1.bar(x_pos, sample_data, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axhline(sample_mean, color='red', linewidth=2, linestyle='--',
            label=f'Mean = {sample_mean:.1f}')

# Draw deviations
for i, val in enumerate(sample_data):
    deviation = val - sample_mean
    ax1.plot([i, i], [sample_mean, val], 'r-', linewidth=2, alpha=0.5)
    ax1.text(i, val + 0.3, f'{deviation:+.1f}',
             ha='center', fontsize=9, fontweight='bold')

ax1.set_title('Step 1: Deviations from Mean', fontweight='bold', fontsize=14)
ax1.set_xlabel('Data Point Index')
ax1.set_ylabel('Value')
ax1.legend()
ax1.text(0.5, 0.02, 'Red lines show deviation from mean',
         transform=ax1.transAxes, ha='center', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 2: Squared deviations
ax2 = axes[0, 1]
squared_devs = [(x - sample_mean)**2 for x in sample_data]
ax2.bar(x_pos, squared_devs, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.axhline(np.mean(squared_devs), color='darkred', linewidth=2, linestyle='--',
            label=f'Variance = {np.mean(squared_devs):.2f}')
ax2.set_title('Step 2: Squared Deviations', fontweight='bold', fontsize=14)
ax2.set_xlabel('Data Point Index')
ax2.set_ylabel('Squared Deviation')
ax2.legend()
ax2.text(0.5, 0.02, 'Squaring makes all values positive',
         transform=ax2.transAxes, ha='center', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 3: Low variance vs high variance
ax3 = axes[1, 0]
low_var = np.random.normal(50, 5, 1000)
high_var = np.random.normal(50, 15, 1000)
ax3.hist(low_var, bins=30, alpha=0.6, color='green', label=f'Low Spread (σ=5)', edgecolor='black')
ax3.hist(high_var, bins=30, alpha=0.6, color='red', label=f'High Spread (σ=15)', edgecolor='black')
ax3.axvline(50, color='black', linewidth=2, linestyle='--', label='Same Mean = 50')
ax3.set_title('LOW vs HIGH Variability', fontweight='bold', fontsize=14)
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.text(0.5, 0.95, 'Same mean, different spread!',
         transform=ax3.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 4: 68-95-99.7 Rule (Normal Distribution)
ax4 = axes[1, 1]
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, 0, 1)
ax4.plot(x, y, 'b-', linewidth=2, label='Normal Distribution')

# Shade regions
ax4.fill_between(x, 0, y, where=(x >= -1) & (x <= 1),
                 alpha=0.3, color='green', label='68% within 1σ')
ax4.fill_between(x, 0, y, where=(x >= -2) & (x <= 2) & ((x < -1) | (x > 1)),
                 alpha=0.3, color='yellow', label='95% within 2σ')
ax4.fill_between(x, 0, y, where=(x >= -3) & (x <= 3) & ((x < -2) | (x > 2)),
                 alpha=0.3, color='red', label='99.7% within 3σ')

ax4.axvline(0, color='black', linewidth=2, linestyle='--', label='Mean (μ)')
ax4.axvline(1, color='green', linewidth=1, linestyle='--')
ax4.axvline(-1, color='green', linewidth=1, linestyle='--')
ax4.axvline(2, color='orange', linewidth=1, linestyle='--')
ax4.axvline(-2, color='orange', linewidth=1, linestyle='--')

ax4.set_title('68-95-99.7 Rule (Empirical Rule)', fontweight='bold', fontsize=14)
ax4.set_xlabel('Standard Deviations from Mean')
ax4.set_ylabel('Probability Density')
ax4.legend(fontsize=10)
ax4.text(0, 0.45, '68%', ha='center', fontsize=14, fontweight='bold')
ax4.text(0, 0.35, '95%', ha='center', fontsize=14, fontweight='bold')
ax4.text(0, 0.25, '99.7%', ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_variance_std_deviation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}02_variance_std_deviation.png")
plt.close()

# ============================================================================
# SECTION 3: COVARIANCE AND CORRELATION
# ============================================================================

print()
print("SECTION 3: Relationships Between Variables")
print("-" * 70)
print()

# Generate sample data
np.random.seed(42)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_positive = 2 * x + np.random.normal(0, 2, 10)  # Positive relationship
y_negative = -2 * x + np.random.normal(0, 2, 10) + 20  # Negative relationship

# COVARIANCE
print("1. COVARIANCE")
print("   Formula: cov(X,Y) = Σ(x - μx)(y - μy) / n")
print("   Measures: How two variables change together")
print()

cov_matrix = np.cov(x, y_positive)
covariance = cov_matrix[0, 1]

print("   Step-by-step for positive relationship:")
print(f"   X values: {x[:5]}... (mean = {np.mean(x):.1f})")
print(f"   Y values: {np.array2string(y_positive[:5], precision=1)}... (mean = {np.mean(y_positive):.1f})")
print(f"   Covariance = {covariance:.2f}")
print("   📌 Positive cov: Variables move together (both increase/decrease)")
print("   📌 Negative cov: Variables move opposite (one up, other down)")
print("   📌 Problem: Units depend on scale (hard to interpret)")
print()

# CORRELATION
print("2. CORRELATION (r - Pearson's r)")
print("   Formula: r = cov(X,Y) / (σx × σy)")
print("   Standardized covariance: Always between -1 and +1")
print()

correlation = np.corrcoef(x, y_positive)[0, 1]
print(f"   Correlation = {correlation:.3f}")
print()
print("   Interpretation:")
print("   r = +1: Perfect positive relationship")
print("   r = +0.7 to +1: Strong positive")
print("   r = +0.3 to +0.7: Moderate positive")
print("   r = -0.3 to +0.3: Weak/No relationship")
print("   r = -0.7 to -0.3: Moderate negative")
print("   r = -1 to -0.7: Strong negative")
print("   r = -1: Perfect negative relationship")
print()

# ============================================================================
# VISUALIZATION 3: CORRELATION EXAMPLES
# ============================================================================

print("Creating Visualization: Correlation Examples...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('📉 CORRELATION: Understanding Relationships Between Variables',
             fontsize=18, fontweight='bold')

# Generate data for different correlations
np.random.seed(42)
n = 100
x_base = np.random.normal(50, 10, n)

correlations = [0.95, 0.6, 0.0, -0.6, -0.95]
titles = ['Strong Positive (r≈+0.95)', 'Moderate Positive (r≈+0.6)',
          'No Correlation (r≈0)', 'Moderate Negative (r≈-0.6)',
          'Strong Negative (r≈-0.95)']

for idx, (ax, target_corr, title) in enumerate(zip(axes.flat[:5], correlations, titles)):
    # Generate correlated data
    noise = np.random.normal(0, 1, n)
    y = target_corr * x_base + (1 - abs(target_corr)) * 30 * noise + 50

    # Calculate actual correlation
    actual_corr = np.corrcoef(x_base, y)[0, 1]

    # Plot
    ax.scatter(x_base, y, alpha=0.6, s=50, edgecolors='black')

    # Add regression line
    z = np.polyfit(x_base, y, 1)
    p = np.poly1d(z)
    ax.plot(x_base, p(x_base), "r--", linewidth=2, label='Trend line')

    ax.set_title(f'{title}\nActual r = {actual_corr:.3f}',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('X Variable')
    ax.set_ylabel('Y Variable')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add interpretation
    if actual_corr > 0.7:
        interp = "Strong positive:\nX↑ → Y↑"
    elif actual_corr > 0.3:
        interp = "Moderate positive:\nX tends ↑ → Y tends ↑"
    elif actual_corr > -0.3:
        interp = "Weak/No relationship:\nX and Y independent"
    elif actual_corr > -0.7:
        interp = "Moderate negative:\nX tends ↑ → Y tends ↓"
    else:
        interp = "Strong negative:\nX↑ → Y↓"

    ax.text(0.05, 0.95, interp, transform=ax.transAxes,
           fontsize=9, va='top', bbox=dict(boxstyle='round',
           facecolor='lightyellow', alpha=0.8))

# Sixth plot: Correlation matrix heatmap example
ax6 = axes.flat[5]
# Create sample dataset with multiple variables
np.random.seed(42)
data_matrix = np.random.multivariate_normal(
    [0, 0, 0, 0],
    [[1, 0.8, 0.3, -0.5],
     [0.8, 1, 0.4, -0.6],
     [0.3, 0.4, 1, -0.2],
     [-0.5, -0.6, -0.2, 1]],
    100
)

corr_matrix = np.corrcoef(data_matrix.T)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, square=True, ax=ax6,
            cbar_kws={'label': 'Correlation Coefficient'})
ax6.set_title('Correlation Matrix Heatmap', fontweight='bold', fontsize=12)
ax6.set_xticklabels(['Var1', 'Var2', 'Var3', 'Var4'])
ax6.set_yticklabels(['Var1', 'Var2', 'Var3', 'Var4'])

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_correlation_examples.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}03_correlation_examples.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: STATISTICS CHEAT SHEET
# ============================================================================

print("Creating Infographic: Statistics Cheat Sheet...")

fig = plt.figure(figsize=(16, 20))
fig.suptitle('📊 STATISTICS CHEAT SHEET FOR MACHINE LEARNING',
             fontsize=22, fontweight='bold', y=0.98)

# Create grid
gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)

# 1. Mean
ax1 = fig.add_subplot(gs[0, 0])
ax1.axis('off')
ax1.text(0.5, 0.9, 'MEAN (μ)', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FF6B6B', alpha=0.7))
ax1.text(0.5, 0.7, 'μ = Σx / n', ha='center', fontsize=14, family='monospace')
ax1.text(0.5, 0.5, 'Sum all values, divide by count', ha='center', fontsize=11)
ax1.text(0.5, 0.3, 'Example: [2,4,6,8]\nμ = (2+4+6+8)/4 = 5',
         ha='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax1.text(0.5, 0.05, '✓ Sensitive to outliers\n✗ Not robust',
         ha='center', fontsize=9, style='italic')

# 2. Median
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis('off')
ax2.text(0.5, 0.9, 'MEDIAN', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#4ECDC4', alpha=0.7))
ax2.text(0.5, 0.7, 'Middle value when sorted', ha='center', fontsize=14)
ax2.text(0.5, 0.5, 'Sort data, pick the center', ha='center', fontsize=11)
ax2.text(0.5, 0.3, 'Example: [2,4,6,8]\nSorted: [2,4,6,8]\nMedian = (4+6)/2 = 5',
         ha='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax2.text(0.5, 0.05, '✓ Robust to outliers\n✓ Better for skewed data',
         ha='center', fontsize=9, style='italic')

# 3. Variance
ax3 = fig.add_subplot(gs[1, 0])
ax3.axis('off')
ax3.text(0.5, 0.9, 'VARIANCE (σ²)', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#95E1D3', alpha=0.7))
ax3.text(0.5, 0.7, 'σ² = Σ(x - μ)² / n', ha='center', fontsize=14, family='monospace')
ax3.text(0.5, 0.5, 'Average squared distance from mean', ha='center', fontsize=11)
ax3.text(0.5, 0.3, 'Steps:\n1. Find mean\n2. Subtract mean from each value\n3. Square results\n4. Average them',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax3.text(0.5, 0.05, '✓ Measures spread\n✗ Units are squared',
         ha='center', fontsize=9, style='italic')

# 4. Standard Deviation
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')
ax4.text(0.5, 0.9, 'STD DEVIATION (σ)', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#F38181', alpha=0.7))
ax4.text(0.5, 0.7, 'σ = √(variance)', ha='center', fontsize=14, family='monospace')
ax4.text(0.5, 0.5, 'Square root of variance', ha='center', fontsize=11)
ax4.text(0.5, 0.3, 'Same units as original data\nEasier to interpret!',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax4.text(0.5, 0.05, '✓ Same units as data\n✓ 68-95-99.7 rule applies',
         ha='center', fontsize=9, style='italic')

# 5. Covariance
ax5 = fig.add_subplot(gs[2, 0])
ax5.axis('off')
ax5.text(0.5, 0.9, 'COVARIANCE', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#AA96DA', alpha=0.7))
ax5.text(0.5, 0.7, 'cov(X,Y) = Σ(x-μx)(y-μy) / n', ha='center', fontsize=13, family='monospace')
ax5.text(0.5, 0.5, 'How two variables change together', ha='center', fontsize=11)
ax5.text(0.5, 0.3, 'Positive: Move together\nNegative: Move opposite\nZero: Independent',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax5.text(0.5, 0.05, '✗ Scale-dependent\n✗ Hard to interpret',
         ha='center', fontsize=9, style='italic')

# 6. Correlation
ax6 = fig.add_subplot(gs[2, 1])
ax6.axis('off')
ax6.text(0.5, 0.9, 'CORRELATION (r)', ha='center', fontsize=16, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='#FCBAD3', alpha=0.7))
ax6.text(0.5, 0.7, 'r = cov(X,Y) / (σx × σy)', ha='center', fontsize=13, family='monospace')
ax6.text(0.5, 0.5, 'Standardized covariance (-1 to +1)', ha='center', fontsize=11)
ax6.text(0.5, 0.3, '+1: Perfect positive\n0: No relationship\n-1: Perfect negative',
         ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
ax6.text(0.5, 0.05, '✓ Scale-independent\n✓ Easy to interpret',
         ha='center', fontsize=9, style='italic')

# 7-10: Quick reference guides
interpretations = [
    ("WHEN TO USE EACH", "Mean: Symmetric data\nMedian: Skewed/outliers\nMode: Categories\nStd Dev: Understand spread"),
    ("ML APPLICATIONS", "Mean: Centering data\nVariance: Feature scaling\nCorrelation: Feature selection\nStd: Normalization"),
    ("QUICK CHECKS", "Mean = Median? → Symmetric\nMean > Median? → Right skew\nMean < Median? → Left skew\nHigh σ? → Spread out"),
    ("FORMULAS AT A GLANCE", "μ = Σx/n\nσ² = Σ(x-μ)²/n\nσ = √σ²\nr = cov(X,Y)/(σx×σy)")
]

for idx, (title, content) in enumerate(interpretations):
    row = 3 + idx // 2
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])
    ax.axis('off')
    ax.text(0.5, 0.7, title, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.5, 0.3, content, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.savefig(f'{VISUAL_DIR}04_statistics_cheat_sheet.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}04_statistics_cheat_sheet.png")
plt.close()

# ============================================================================
# SECTION 4: WHY THIS MATTERS FOR MACHINE LEARNING
# ============================================================================

print()
print("SECTION 4: Connection to Machine Learning")
print("-" * 70)
print()
print("🎯 WHY STATISTICS MATTER FOR ML:")
print()
print("1. DATA EXPLORATION")
print("   • Mean/Median: Understand center of your data")
print("   • Std Dev: Understand spread and outliers")
print("   • Use before training any model!")
print()
print("2. FEATURE ENGINEERING")
print("   • Standardization: (x - μ) / σ")
print("   • Makes features comparable")
print("   • Many algorithms require this!")
print()
print("3. FEATURE SELECTION")
print("   • Correlation: Which features relate to target?")
print("   • Remove highly correlated features (multicollinearity)")
print("   • Keep most informative features")
print()
print("4. MODEL EVALUATION")
print("   • Mean Squared Error uses variance concepts")
print("   • R² Score compares model variance to data variance")
print("   • Understanding stats = Understanding performance")
print()

# ============================================================================
# PRACTICAL EXAMPLE
# ============================================================================

print()
print("SECTION 5: Practical Example - House Prices")
print("-" * 70)
print()

# Generate sample house data
np.random.seed(42)
sizes = np.random.normal(1500, 300, 50)
prices = 100000 + 150 * sizes + np.random.normal(0, 20000, 50)

print("Sample: House sizes and prices")
print(f"Size (sqft) - first 5: {sizes[:5].astype(int)}")
print(f"Price ($) - first 5: {prices[:5].astype(int)}")
print()

print("STATISTICS:")
print(f"Mean size: {np.mean(sizes):.0f} sqft")
print(f"Median size: {np.median(sizes):.0f} sqft")
print(f"Std dev size: {np.std(sizes):.0f} sqft")
print()
print(f"Mean price: ${np.mean(prices):,.0f}")
print(f"Median price: ${np.median(prices):,.0f}")
print(f"Std dev price: ${np.std(prices):,.0f}")
print()

correlation = np.corrcoef(sizes, prices)[0, 1]
print(f"Correlation between size and price: {correlation:.3f}")
print(f"Interpretation: Strong positive relationship!")
print(f"  → Larger houses tend to cost more")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 70)
print("✅ SUMMARY: What You Learned")
print("=" * 70)
print()
print("CENTRAL TENDENCY:")
print("  • Mean: Average value (Σx / n)")
print("  • Median: Middle value (robust to outliers)")
print("  • Mode: Most frequent value")
print()
print("SPREAD:")
print("  • Variance: Average squared deviation (σ²)")
print("  • Standard Deviation: Square root of variance (σ)")
print("  • 68-95-99.7 Rule for normal distributions")
print()
print("RELATIONSHIPS:")
print("  • Covariance: How variables move together")
print("  • Correlation: Standardized covariance (-1 to +1)")
print()
print("📊 Visual files created in:", VISUAL_DIR)
print("   - 01_mean_median_mode_comparison.png")
print("   - 02_variance_std_deviation.png")
print("   - 03_correlation_examples.png")
print("   - 04_statistics_cheat_sheet.png")
print()
print("🎓 NEXT STEPS:")
print("   1. Review the visualizations")
print("   2. Watch the StatQuest videos (best visual explanations!)")
print("   3. Practice calculating these by hand with small datasets")
print("   4. Move on to 03_intro_to_derivatives.py")
print()
print("=" * 70)
