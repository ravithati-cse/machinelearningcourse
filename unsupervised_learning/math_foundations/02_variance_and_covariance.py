"""
📊 VARIANCE AND COVARIANCE - The Math That Powers PCA

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. Variance: how much a single variable spreads around its mean
2. Covariance: how two variables move TOGETHER (positive, negative, zero)
3. The difference between covariance and correlation, and when to use each
4. The covariance matrix: capturing ALL pairwise relationships in one structure
5. Why the covariance matrix is the CORE input to PCA
6. How to read a covariance matrix and what its values tell you
7. Visualizing scatter plots to build geometric intuition for covariance

================================================================================
RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
ABSOLUTE MUST WATCH:
   - StatQuest: "PCA Step-by-Step"
     https://www.youtube.com/watch?v=FgakZw6K1QQ
     (Best visual explanation of variance and PCA together)

   - StatQuest: "Covariance, Clearly Explained!"
     https://www.youtube.com/watch?v=qtaqvPAeEJY
     (Perfect companion to this module)

Also Recommended:
   - 3Blue1Brown: "Visualizing the covariance matrix"
     https://www.youtube.com/watch?v=PDE43rQsANw
     (Beautiful geometric intuition for covariance)

================================================================================
OVERVIEW
================================================================================
The Bridge to PCA:
- PCA (Principal Component Analysis) finds the directions of MAXIMUM VARIANCE
- To find those directions, it needs to know the variance of EVERY feature
  AND how every pair of features is related
- All of this lives in ONE matrix: the COVARIANCE MATRIX

Think of it this way:
  Variance   = "How widely does height vary across people?"
  Covariance = "Do tall people also tend to be heavier?"
  Covariance Matrix = captures ALL such relationships at once

Master variance and covariance and PCA becomes easy!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/02_variance_covariance/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("VARIANCE AND COVARIANCE")
print("   Understanding Spread and Relationships in Data")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: VARIANCE — HOW SPREAD OUT IS ONE VARIABLE?
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Variance — Measuring the Spread of a Single Variable")
print("=" * 80)
print()

print("INTUITION: Imagine measuring the height of people in two cities.")
print()
print("  City A (athletes):   heights = [165, 172, 178, 185, 190, 196, 203]")
print("  City B (one school): heights = [174, 175, 175, 176, 176, 177, 177]")
print()
print("Both have roughly the same mean, but City A has much MORE spread!")
print("VARIANCE captures this 'spreadness'.")
print()

heights_A = np.array([165, 172, 178, 185, 190, 196, 203], dtype=float)
heights_B = np.array([174, 175, 175, 176, 176, 177, 177], dtype=float)

print("FORMULA:")
print("-" * 70)
print("  Population variance:  Var(X) = (1/N)   * sum((xi - mean)^2)")
print("  Sample variance:      Var(X) = (1/(N-1)) * sum((xi - mean)^2)")
print()
print("  We use (N-1) for SAMPLE data to get an unbiased estimate")
print("  (called Bessel's correction — numpy uses ddof=1 for sample variance)")
print()

def variance_scratch(x, ddof=1):
    """
    Compute variance from scratch.
    ddof=0: population variance (divide by N)
    ddof=1: sample variance (divide by N-1, default for unbiased estimate)
    """
    x = np.array(x, dtype=float)
    mean = np.mean(x)
    squared_diffs = (x - mean) ** 2
    return np.sum(squared_diffs) / (len(x) - ddof)

def std_scratch(x, ddof=1):
    """Standard deviation = sqrt(variance)."""
    return np.sqrt(variance_scratch(x, ddof=ddof))

# Step by step for City A
print("STEP-BY-STEP FOR CITY A:")
print("-" * 70)
mean_A = np.mean(heights_A)
print(f"  Data:     {heights_A}")
print(f"  Mean:     {mean_A:.2f} cm")
deviations_A = heights_A - mean_A
print(f"  Deviations from mean: {deviations_A}")
sq_devs_A = deviations_A ** 2
print(f"  Squared deviations:   {sq_devs_A}")
print(f"  Sum of squared devs:  {np.sum(sq_devs_A):.2f}")
N_A = len(heights_A)
var_A_sample = np.sum(sq_devs_A) / (N_A - 1)
print(f"  Sample variance = {np.sum(sq_devs_A):.2f} / {N_A-1} = {var_A_sample:.4f}")
print(f"  Std deviation   = sqrt({var_A_sample:.4f}) = {np.sqrt(var_A_sample):.4f} cm")
print()

print("COMPARISON:")
print("-" * 70)
var_A = variance_scratch(heights_A)
var_B = variance_scratch(heights_B)
std_A = std_scratch(heights_A)
std_B = std_scratch(heights_B)

print(f"{'Statistic':<30} {'City A (athletes)':<22} {'City B (school)'}")
print("-" * 75)
print(f"{'Mean (cm)':<30} {np.mean(heights_A):<22.2f} {np.mean(heights_B):.2f}")
print(f"{'Variance (cm^2)':<30} {var_A:<22.2f} {var_B:.2f}")
print(f"{'Std Deviation (cm)':<30} {std_A:<22.2f} {std_B:.2f}")
print()
print(f"City A variance is {var_A/var_B:.1f}x larger -> much more spread out!")
print()

# Verify with numpy
print("NUMPY VERIFICATION:")
print("-" * 70)
print(f"  np.var(heights_A, ddof=1)  = {np.var(heights_A, ddof=1):.4f}  (our answer: {var_A:.4f})")
print(f"  np.std(heights_A, ddof=1)  = {np.std(heights_A, ddof=1):.4f}  (our answer: {std_A:.4f})")
print()
print("  ddof=1 -> sample variance (use this for real data!)")
print("  ddof=0 -> population variance (only if you have ALL data)")
print()

print("KEY INSIGHT FOR PCA:")
print("-" * 70)
print("  PCA wants to find directions of MAXIMUM VARIANCE.")
print("  City A's heights vary a lot -> more 'information' captured.")
print("  City B's heights barely vary -> nearly a constant, boring!")
print("  PCA would weight height from City A much more than City B.")
print()

# ============================================================================
# SECTION 2: COVARIANCE — HOW TWO VARIABLES MOVE TOGETHER
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Covariance — Do Two Variables Move Together?")
print("=" * 80)
print()

print("SCENARIO: You measure both height and weight for 10 people.")
print()
print("  Do tall people tend to weigh more? (Positive relationship)")
print("  Do they weigh less? (Negative relationship)")
print("  No relationship at all? (Zero covariance)")
print()

print("FORMULA:")
print("-" * 70)
print("  Cov(X, Y) = (1/(N-1)) * sum((xi - mean_X) * (yi - mean_Y))")
print()
print("  When X is above its mean AND Y is above its mean -> positive product -> positive Cov")
print("  When X is above mean but Y is below mean        -> negative product -> negative Cov")
print("  Mixed signals cancel out                         ->                    near-zero Cov")
print()

def covariance_scratch(x, y, ddof=1):
    """
    Compute covariance between two variables from scratch.
    Positive: they move together
    Negative: they move oppositely
    Near zero: no linear relationship
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    products = (x - mean_x) * (y - mean_y)
    return np.sum(products) / (len(x) - ddof)

# Three scenarios
np.random.seed(42)
n_people = 30

# Scenario 1: Strong positive covariance (height and weight)
height = np.random.randn(n_people) * 10 + 170
weight = height * 0.5 + np.random.randn(n_people) * 5 - 15

# Scenario 2: Strong negative covariance (hours studying vs. errors made)
study_hours = np.random.uniform(0, 10, n_people)
errors_made = -3 * study_hours + np.random.randn(n_people) * 2 + 25

# Scenario 3: Near zero covariance (height vs. test score)
test_score = np.random.randn(n_people) * 10 + 70

print("THREE SCENARIOS:")
print("-" * 70)

scenarios = [
    ("Height vs Weight",          height,       weight,       "Positive"),
    ("Study Hours vs Errors",     study_hours,  errors_made,  "Negative"),
    ("Height vs Test Score",      height,       test_score,   "Near zero"),
]

for name, X, Y, expected in scenarios:
    cov = covariance_scratch(X, Y)
    print(f"  {name:<30}  Cov = {cov:>8.2f}  ({expected})")

print()
print("INTERPRETATION:")
print("-" * 70)
print("  Large positive Cov: both variables rise/fall together")
print("  Large negative Cov: when one rises, the other falls")
print("  Near zero Cov:      no consistent linear relationship")
print()
print("IMPORTANT: Covariance units depend on the units of X and Y!")
print("  height (cm) * weight (kg) -> Cov in 'cm*kg' — hard to interpret!")
print("  That's why we use CORRELATION (next section).")
print()

# Step by step
print("STEP-BY-STEP (Height vs Weight, first 5 people):")
print("-" * 70)
mean_h = np.mean(height)
mean_w = np.mean(weight)
print(f"  Mean height = {mean_h:.2f} cm,  Mean weight = {mean_w:.2f} kg")
print()
print(f"  {'Person':<8} {'Height':<10} {'h - mean_h':<14} {'Weight':<10} {'w - mean_w':<14} {'Product'}")
print(f"  {'-'*8} {'-'*10} {'-'*14} {'-'*10} {'-'*14} {'-'*10}")
for i in range(5):
    h_dev = height[i] - mean_h
    w_dev = weight[i] - mean_w
    prod  = h_dev * w_dev
    print(f"  {i+1:<8} {height[i]:<10.2f} {h_dev:<14.2f} {weight[i]:<10.2f} {w_dev:<14.2f} {prod:.2f}")
print(f"  ... (all {n_people} people)")
cov_hw = covariance_scratch(height, weight)
print(f"  Cov(H, W) = {cov_hw:.4f}")
print()

# ============================================================================
# SECTION 3: CORRELATION — THE STANDARDIZED COVARIANCE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Correlation — Covariance Made Interpretable")
print("=" * 80)
print()

print("PROBLEM WITH COVARIANCE:")
print("  Cov(Height_cm, Weight_kg) = large number in 'cm * kg' units")
print("  Cov(Height_m,  Weight_g)  = tiny number (same data, different units!)")
print("  Covariance is NOT scale-invariant -> hard to compare across datasets")
print()

print("SOLUTION: Pearson Correlation Coefficient (r)")
print("-" * 70)
print("  r(X, Y) = Cov(X, Y) / (Std(X) * Std(Y))")
print()
print("  This normalizes by the standard deviations, making r:")
print("    * Always between -1 and +1")
print("    * Unit-free (no cm*kg nonsense)")
print("    * Directly comparable across different datasets")
print()

def correlation_scratch(x, y):
    """
    Pearson correlation coefficient from scratch.
    = covariance / (std_x * std_y)
    Always in [-1, +1]
    """
    cov = covariance_scratch(x, y)
    std_x = std_scratch(x)
    std_y = std_scratch(y)
    return cov / (std_x * std_y)

print("COMPARISON (same three scenarios):")
print("-" * 70)
print(f"  {'Scenario':<35} {'Cov':<12} {'Corr (r)':<12} {'Interpretation'}")
print("-" * 85)
for name, X, Y, expected in scenarios:
    cov = covariance_scratch(X, Y)
    r = correlation_scratch(X, Y)
    if abs(r) > 0.7:
        interp = "Strong"
    elif abs(r) > 0.4:
        interp = "Moderate"
    else:
        interp = "Weak"
    sign = "positive" if r > 0 else "negative"
    print(f"  {name:<35} {cov:<12.2f} {r:<12.4f} {interp} {sign}")
print()

print("CORRELATION GUIDE:")
print("-" * 70)
print("  |r| = 1.0        Perfect linear relationship")
print("  0.7 < |r| < 1.0  Strong linear relationship")
print("  0.4 < |r| < 0.7  Moderate linear relationship")
print("  0.2 < |r| < 0.4  Weak linear relationship")
print("  |r| < 0.2        Very weak or no linear relationship")
print()

# Numpy verification
print("NUMPY VERIFICATION:")
print("-" * 70)
cov_matrix_np = np.cov(height, weight, ddof=1)
corr_matrix_np = np.corrcoef(height, weight)
print(f"  np.cov(height, weight)[0,1]      = {cov_matrix_np[0,1]:.4f}")
print(f"  Our covariance_scratch(h,w)      = {covariance_scratch(height, weight):.4f}")
print(f"  np.corrcoef(height, weight)[0,1] = {corr_matrix_np[0,1]:.4f}")
print(f"  Our correlation_scratch(h,w)     = {correlation_scratch(height, weight):.4f}")
print()
print("  THEY MATCH!")
print()

# ============================================================================
# SECTION 4: THE COVARIANCE MATRIX
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: The Covariance Matrix — All Relationships at Once")
print("=" * 80)
print()

print("For a dataset with P features, the covariance matrix is P x P.")
print()
print("  Entry [i, j] = Cov(feature_i, feature_j)")
print("  Entry [i, i] = Var(feature_i)  <- diagonal = individual variances!")
print()
print("KEY PROPERTIES:")
print("  * Symmetric:  Cov(X,Y) = Cov(Y,X)  ->  matrix is symmetric")
print("  * Diagonal:   variances of each feature (always positive)")
print("  * Off-diagonal: covariances (can be positive, negative, or zero)")
print()

# Create a 3-feature dataset
np.random.seed(99)
n = 100
feat1 = np.random.randn(n) * 5 + 10          # high variance
feat2 = 0.8 * feat1 + np.random.randn(n)     # correlated with feat1
feat3 = np.random.randn(n) * 2               # independent, low variance
X = np.column_stack([feat1, feat2, feat3])
feature_names = ['Feature1 (high var)', 'Feature2 (corr w/1)', 'Feature3 (low var)']

print("DATASET: 3 features, 100 samples")
print("-" * 70)
print(f"  Feature 1: high variance, independent")
print(f"  Feature 2: correlated with Feature 1")
print(f"  Feature 3: low variance, independent")
print()

def covariance_matrix_scratch(X, ddof=1):
    """
    Compute the full covariance matrix from scratch.
    X: (n_samples, n_features) array
    Returns: (n_features, n_features) symmetric covariance matrix
    """
    X = np.array(X, dtype=float)
    n, p = X.shape
    # Center the data (subtract column means)
    X_centered = X - np.mean(X, axis=0)
    # Cov matrix = X_centered.T @ X_centered / (n-1)
    return (X_centered.T @ X_centered) / (n - ddof)

C = covariance_matrix_scratch(X)
C_np = np.cov(X.T, ddof=1)

print("COVARIANCE MATRIX (3x3):")
print("-" * 70)
print(f"  {'':>25}", end="")
for name in feature_names:
    print(f"  {name[:12]:>14}", end="")
print()
for i, row_name in enumerate(feature_names):
    print(f"  {row_name[:25]:>25}", end="")
    for j in range(3):
        print(f"  {C[i,j]:>14.4f}", end="")
    print()
print()

print("READING THE MATRIX:")
print("-" * 70)
print(f"  Var(Feature1) = C[0,0] = {C[0,0]:.4f}  (large -> spread out)")
print(f"  Var(Feature2) = C[1,1] = {C[1,1]:.4f}  (large -> spread out)")
print(f"  Var(Feature3) = C[2,2] = {C[2,2]:.4f}  (small -> tightly clustered)")
print()
print(f"  Cov(F1, F2)   = C[0,1] = {C[0,1]:.4f}  (large positive -> strongly correlated!)")
print(f"  Cov(F1, F3)   = C[0,2] = {C[0,2]:.4f}  (near zero -> independent)")
print(f"  Cov(F2, F3)   = C[1,2] = {C[1,2]:.4f}  (near zero -> independent)")
print()
print(f"  numpy check: max diff from np.cov = {np.max(np.abs(C - C_np)):.2e}")
print()

print("THE CONNECTION TO PCA:")
print("-" * 70)
print("  PCA finds the eigenvectors of this covariance matrix.")
print("  Each eigenvector is a 'principal component' direction.")
print("  The eigenvalue tells you how much variance that direction captures.")
print("  (We'll compute this in Module 03!)")
print()
print("  Large eigenvalue -> that direction has lots of variance -> important!")
print("  Small eigenvalue -> that direction has little variance -> can discard!")
print()

# ============================================================================
# SECTION 5: VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Visualizations")
print("=" * 80)
print()
print("Generating Visualization 1: Variance — What 'Spread' Looks Like...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('VARIANCE AND COVARIANCE: The Math of Spread and Relationships',
             fontsize=14, fontweight='bold', y=1.01)

# ---- Row 1: Variance ----
ax = axes[0, 0]
ax.hist(heights_A, bins=8, color='steelblue', edgecolor='black', alpha=0.8, label='City A (athletes)')
ax.axvline(np.mean(heights_A), color='red', lw=2, label=f'Mean = {np.mean(heights_A):.1f}')
ax.axvline(np.mean(heights_A) - np.std(heights_A, ddof=1), color='orange', lw=2, ls='--',
           label=f'±1 Std = {np.std(heights_A, ddof=1):.1f} cm')
ax.axvline(np.mean(heights_A) + np.std(heights_A, ddof=1), color='orange', lw=2, ls='--')
ax.set_title(f'City A: High Variance\nVar={var_A:.1f}, Std={std_A:.1f} cm', fontsize=11, fontweight='bold')
ax.set_xlabel('Height (cm)'); ax.set_ylabel('Count')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(heights_B, bins=8, color='tomato', edgecolor='black', alpha=0.8, label='City B (school)')
ax.axvline(np.mean(heights_B), color='red', lw=2, label=f'Mean = {np.mean(heights_B):.1f}')
ax.axvline(np.mean(heights_B) - np.std(heights_B, ddof=1), color='orange', lw=2, ls='--',
           label=f'±1 Std = {np.std(heights_B, ddof=1):.1f} cm')
ax.axvline(np.mean(heights_B) + np.std(heights_B, ddof=1), color='orange', lw=2, ls='--')
ax.set_title(f'City B: Low Variance\nVar={var_B:.1f}, Std={std_B:.1f} cm', fontsize=11, fontweight='bold')
ax.set_xlabel('Height (cm)'); ax.set_ylabel('Count')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.text(0.5, 0.95, 'VARIANCE FORMULA', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)
lines = [
    "",
    "Var(X) = (1/(N-1)) * sum((xi - mean)^2)",
    "",
    "Step by step:",
    "  1. Find the mean: mean = sum(xi) / N",
    "  2. Deviation from mean: xi - mean",
    "  3. Square each deviation: (xi - mean)^2",
    "  4. Average them: sum / (N - 1)",
    "",
    "RESULT: always positive (squared!)",
    "",
    "Standard Deviation = sqrt(Variance)",
    "  Same units as original data",
    "  Easier to interpret than Variance",
    "",
    "City A: Var=111.2, Std=10.5 cm",
    "City B: Var=  0.9, Std= 1.0 cm",
    "",
    "City A is ~11x more spread out!",
    "",
    "PCA INSIGHT:",
    "  High variance = more information",
    "  PCA maximizes variance captured",
]
y = 0.88
for line in lines:
    bold = any(line.startswith(kw) for kw in ['Var(X)', 'Standard', 'City', 'PCA', 'Step', 'RESULT'])
    ax.text(0.05, y, line, fontsize=8.5, transform=ax.transAxes,
            family='monospace', fontweight='bold' if bold else 'normal')
    y -= 0.04
ax.axis('off')

# ---- Row 2: Covariance scatter plots ----
scatter_configs = [
    (height, weight,      'Height', 'Weight (kg)',       'Positive Covariance\n(Tall people tend to weigh more)',   'steelblue'),
    (study_hours, errors_made, 'Study Hours', 'Errors Made',  'Negative Covariance\n(More study -> fewer errors)',       'tomato'),
    (height, test_score,  'Height', 'Test Score',        'Near-Zero Covariance\n(Height does not predict score)',     'seagreen'),
]

for ax, (X_s, Y_s, xl, yl, title, color) in zip(axes[1], scatter_configs):
    cov_v = covariance_scratch(X_s, Y_s)
    r_v   = correlation_scratch(X_s, Y_s)
    ax.scatter(X_s, Y_s, alpha=0.6, s=50, c=color, edgecolors='white', linewidths=0.5)
    # Fit and plot a line to show the trend
    m, b = np.polyfit(X_s, Y_s, 1)
    x_line = np.linspace(X_s.min(), X_s.max(), 100)
    ax.plot(x_line, m * x_line + b, 'k-', lw=2, alpha=0.7, label='Trend line')
    # Mark means
    ax.axvline(np.mean(X_s), color='gray', lw=1, ls='--', alpha=0.5)
    ax.axhline(np.mean(Y_s), color='gray', lw=1, ls='--', alpha=0.5)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    ax.text(0.05, 0.93, f'Cov = {cov_v:.2f}\nr   = {r_v:.3f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_variance_covariance_intro.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_variance_covariance_intro.png")

# ============================================================================
# VISUALIZATION 2: Covariance matrix heatmap
# ============================================================================
print("Generating Visualization 2: Covariance Matrix Heatmap...")

# Create a richer 5-feature dataset
np.random.seed(7)
n = 200
f1 = np.random.randn(n) * 10      # large variance, independent
f2 = 0.9 * f1 + np.random.randn(n)  # strongly positively correlated with f1
f3 = -0.6 * f1 + np.random.randn(n) * 3  # negatively correlated with f1
f4 = np.random.randn(n) * 2       # small variance, independent
f5 = 0.7 * f4 + np.random.randn(n) * 0.5  # correlated with f4
X5 = np.column_stack([f1, f2, f3, f4, f5])
feat5_names = ['F1\n(large var)', 'F2\n(+corr F1)', 'F3\n(-corr F1)', 'F4\n(small var)', 'F5\n(+corr F4)']

C5 = covariance_matrix_scratch(X5)
# Also compute correlation matrix for comparison
std5 = np.sqrt(np.diag(C5))
R5   = C5 / np.outer(std5, std5)   # correlation matrix

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('COVARIANCE MATRIX vs CORRELATION MATRIX — Same Data, Different Scale',
             fontsize=13, fontweight='bold')

# --- Covariance matrix ---
ax = axes[0]
im = ax.imshow(C5, cmap='RdBu_r', aspect='auto',
               vmin=-np.abs(C5).max(), vmax=np.abs(C5).max())
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(feat5_names, fontsize=9)
ax.set_yticklabels(feat5_names, fontsize=9)
ax.set_title('Covariance Matrix\n(values depend on scale — hard to compare)', fontsize=11, fontweight='bold')
for i in range(5):
    for j in range(5):
        ax.text(j, i, f'{C5[i,j]:.1f}', ha='center', va='center',
                fontsize=8, fontweight='bold',
                color='white' if abs(C5[i,j]) > np.abs(C5).max() * 0.5 else 'black')

# --- Correlation matrix ---
ax = axes[1]
im = ax.imshow(R5, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(feat5_names, fontsize=9)
ax.set_yticklabels(feat5_names, fontsize=9)
ax.set_title('Correlation Matrix\n(values always in [-1, 1] — easy to compare)', fontsize=11, fontweight='bold')
for i in range(5):
    for j in range(5):
        color = 'white' if abs(R5[i,j]) > 0.5 else 'black'
        ax.text(j, i, f'{R5[i,j]:.2f}', ha='center', va='center',
                fontsize=9, fontweight='bold', color=color)

# --- Annotation panel ---
ax = axes[2]
ax.text(0.5, 0.97, 'HOW TO READ THESE MATRICES', fontsize=12,
        fontweight='bold', ha='center', transform=ax.transAxes)
notes = [
    "",
    "COVARIANCE MATRIX (left):",
    "  Diagonal = variance of each feature",
    "  Off-diag = covariance between pairs",
    "  Units: (units of X) * (units of Y)",
    "  Values hard to compare directly",
    "",
    "CORRELATION MATRIX (middle):",
    "  Diagonal = 1.0 (always!)",
    "  Off-diag = correlation r in [-1, 1]",
    "  Unit-free (standardized)",
    "  Easy to compare across datasets",
    "",
    "COLOR GUIDE:",
    "  Dark red  = strong positive relation",
    "  Dark blue = strong negative relation",
    "  White     = no relationship",
    "",
    "WHAT WE SEE:",
    "  F1-F2: strong positive (0.9x + noise)",
    "  F1-F3: strong negative (-0.6x)",
    "  F4-F5: moderate positive",
    "  F1-F4, F1-F5: near zero (independent)",
    "",
    "LINK TO PCA:",
    "  PCA decomposes the cov. matrix",
    "  Eigenvectors = principal components",
    "  Eigenvalues = variance explained",
    "  -> Module 03 covers this in detail!",
]
y = 0.92
for line in notes:
    bold = any(line.startswith(kw) for kw in ['COVARIANCE', 'CORRELATION', 'COLOR', 'WHAT', 'LINK', 'Diagonal', 'Off-diag'])
    ax.text(0.04, y, line, fontsize=8.5, transform=ax.transAxes,
            family='monospace', fontweight='bold' if bold else 'normal')
    y -= 0.038
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_covariance_matrix_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_covariance_matrix_heatmap.png")

# ============================================================================
# VISUALIZATION 3: Effect of changing covariance — scatter plot gallery
# ============================================================================
print("Generating Visualization 3: Scatter Plot Gallery by Correlation Strength...")

np.random.seed(42)
n_pts = 150
corr_values = [-0.95, -0.60, -0.20, 0.0, 0.20, 0.60, 0.95]
fig, axes = plt.subplots(1, 7, figsize=(26, 4))
fig.suptitle('HOW SCATTER PLOTS LOOK AT DIFFERENT CORRELATION STRENGTHS',
             fontsize=13, fontweight='bold')

for ax, r_target in zip(axes, corr_values):
    # Generate correlated data using Cholesky decomposition
    cov_matrix_target = np.array([[1.0, r_target], [r_target, 1.0]])
    try:
        L = np.linalg.cholesky(cov_matrix_target)
        z = np.random.randn(2, n_pts)
        xy = L @ z
        x_pts, y_pts = xy[0], xy[1]
    except np.linalg.LinAlgError:
        x_pts = np.random.randn(n_pts)
        y_pts = r_target * x_pts + np.sqrt(1 - r_target**2) * np.random.randn(n_pts)

    actual_r = correlation_scratch(x_pts, y_pts)
    color = 'tomato' if r_target < 0 else ('steelblue' if r_target > 0 else 'gray')
    ax.scatter(x_pts, y_pts, alpha=0.4, s=20, c=color)
    ax.set_title(f'r = {r_target:+.2f}', fontsize=11, fontweight='bold')
    ax.set_xlabel('X'); ax.set_ylabel('Y' if r_target == corr_values[0] else '')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.text(0.5, 0.03, f'actual r={actual_r:.2f}', transform=ax.transAxes,
            ha='center', fontsize=7, color='black',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_correlation_scatter_gallery.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_correlation_scatter_gallery.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Variance and Covariance")
print("=" * 80)
print()
print("WHAT WE LEARNED:")
print("-" * 70)
print()
print("1. VARIANCE:")
print("   Var(X) = (1/(N-1)) * sum((xi - mean)^2)")
print("   Measures spread of a SINGLE variable")
print("   Always >= 0 (it's a squared quantity)")
print("   numpy: np.var(x, ddof=1)")
print()
print("2. COVARIANCE:")
print("   Cov(X,Y) = (1/(N-1)) * sum((xi-mean_X)*(yi-mean_Y))")
print("   Measures how TWO variables move together")
print("   Positive: same direction | Negative: opposite | ~0: unrelated")
print("   numpy: np.cov(X, Y, ddof=1)  or  np.cov(X.T, ddof=1)")
print()
print("3. CORRELATION:")
print("   r = Cov(X,Y) / (Std(X) * Std(Y))")
print("   Standardized covariance — always in [-1, +1]")
print("   Unit-free and directly comparable")
print("   numpy: np.corrcoef(X, Y)")
print()
print("4. COVARIANCE MATRIX:")
print("   P x P matrix capturing ALL pairwise relationships")
print("   Diagonal = variances | Off-diagonal = covariances")
print("   Symmetric by definition")
print("   numpy: np.cov(X.T, ddof=1)")
print()
print("5. LINK TO PCA:")
print("   PCA eigenvectors = principal components (directions)")
print("   PCA eigenvalues  = variance captured by each direction")
print("   Coming up NEXT in Module 03!")
print()
print("=" * 80)
print("Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("  01_variance_covariance_intro.png")
print("  02_covariance_matrix_heatmap.png")
print("  03_correlation_scatter_gallery.png")
print("=" * 80)
print()
print("NEXT STEPS:")
print("  1. Open the scatter plot gallery — can you tell the correlation by eye?")
print("  2. Try np.cov() on your own data!")
print("  3. Next: 03_eigenvectors_for_pca.py")
print("     (How eigenvectors of the covariance matrix ARE the principal components)")
print()
print("=" * 80)
print("VARIANCE AND COVARIANCE MASTERED!")
print("   You understand the math that powers PCA!")
print("=" * 80)
