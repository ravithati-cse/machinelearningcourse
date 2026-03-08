"""
📏 DISTANCE METRICS - How Machines Measure "How Far Apart" Things Are

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. Why distance is the core concept in unsupervised learning (clustering, KNN, anomaly detection)
2. Euclidean distance: straight-line distance (the "as the crow flies" measure)
3. Manhattan distance: grid-based distance (the "city block" measure)
4. Cosine similarity: angle-based similarity (ignores magnitude, focuses on direction)
5. When to use each metric and how wrong choices can ruin your model
6. How to compute pairwise distances across an entire dataset
7. The effect of feature scaling on distance-based algorithms

================================================================================
RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
ABSOLUTE MUST WATCH:
   - StatQuest: "K-nearest neighbors, clearly explained"
     https://www.youtube.com/watch?v=HVXime0nQeI
     (Shows exactly why distance metrics matter in ML)

   - 3Blue1Brown: "Dot products and duality"
     https://www.youtube.com/watch?v=LyGKycYT2v0
     (The geometric intuition behind cosine similarity)

Also Recommended:
   - Sentdex: "Machine Learning with Python - K Means Clustering"
     https://www.youtube.com/watch?v=ikt0sny_ImY
     (Clustering in action — relies on distance metrics)

================================================================================
OVERVIEW
================================================================================
The Big Idea:
- Unsupervised learning finds STRUCTURE in data without labels
- "Structure" usually means: things that are SIMILAR should be grouped together
- But what does "similar" mean mathematically?

ANSWER: Distance (or similarity) metrics!

Every clustering algorithm, every nearest-neighbor search, every anomaly detector
needs to answer: "How close are these two data points?"

The three most important metrics:
  1. Euclidean  — straight-line distance (most common)
  2. Manhattan  — sum of absolute differences (robust to outliers)
  3. Cosine     — angle between vectors (great for text, ratings)

Master these and you understand the foundation of ALL distance-based ML!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/01_distance_metrics/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("DISTANCE METRICS")
print("   How Machines Measure Similarity Between Data Points")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHY DISTANCE MATTERS IN UNSUPERVISED LEARNING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Why Distance is the Foundation of Unsupervised Learning")
print("=" * 80)
print()

print("SCENARIO: You have 1000 customer records (age, income, spending_score)")
print("Goal: Group customers into segments WITHOUT any labels")
print()
print("How does the computer decide which customers belong together?")
print("  -> It measures how CLOSE they are in feature space!")
print()
print("Customer A: age=25, income=30k, spending=80")
print("Customer B: age=26, income=32k, spending=78")
print("Customer C: age=60, income=80k, spending=20")
print()
print("Intuition: A and B are CLOSE  ->  same cluster")
print("           A and C are FAR   ->  different clusters")
print()
print("But 'close' depends on HOW you measure distance!")
print("Different metrics give different answers — this is CRITICAL.")
print()
print("Three questions distance metrics must answer:")
print("  Q1: How far apart in straight-line space? (Euclidean)")
print("  Q2: How far if you can only move on a grid? (Manhattan)")
print("  Q3: How similar in direction, ignoring scale? (Cosine)")
print()

# ============================================================================
# SECTION 2: EUCLIDEAN DISTANCE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Euclidean Distance — The Straight-Line Ruler")
print("=" * 80)
print()

print("FORMULA (from scratch):")
print("-" * 70)
print("  For two points A = (a1, a2, ..., an) and B = (b1, b2, ..., bn):")
print()
print("  d(A, B) = sqrt( (a1-b1)^2 + (a2-b2)^2 + ... + (an-bn)^2 )")
print()
print("  This is just the Pythagorean theorem extended to N dimensions!")
print("  In 2D: d = sqrt( (x2-x1)^2 + (y2-y1)^2 )")
print()

def euclidean_distance(a, b):
    """
    Compute Euclidean distance between two points (numpy arrays).
    This is the straight-line, 'as the crow flies' distance.
    """
    diff = np.array(a) - np.array(b)
    return np.sqrt(np.sum(diff ** 2))

# Walk through a 2D example step by step
A = np.array([1.0, 2.0])
B = np.array([4.0, 6.0])

print("WORKED EXAMPLE (2D):")
print("-" * 70)
print(f"  Point A = {A}")
print(f"  Point B = {B}")
print()
print(f"  Step 1 — Differences:  (4-1, 6-2) = ({B[0]-A[0]}, {B[1]-A[1]})")
print(f"  Step 2 — Square each:  ({(B[0]-A[0])**2}, {(B[1]-A[1])**2})")
print(f"  Step 3 — Sum:          {(B[0]-A[0])**2 + (B[1]-A[1])**2}")
print(f"  Step 4 — Square root:  sqrt({(B[0]-A[0])**2 + (B[1]-A[1])**2}) = {euclidean_distance(A, B):.4f}")
print()
print(f"  Euclidean distance(A, B) = {euclidean_distance(A, B):.4f}")
print()

# Numpy shorthand
print("NUMPY SHORTHAND (what you'll use in practice):")
print("-" * 70)
dist_numpy = np.linalg.norm(B - A)
print(f"  np.linalg.norm(B - A) = {dist_numpy:.4f}")
print()

# 3D example
C = np.array([1.0, 3.0, 5.0])
D = np.array([4.0, 7.0, 1.0])
print(f"3D EXAMPLE: C={C}, D={D}")
print(f"  d(C, D) = sqrt({(D[0]-C[0])**2}+{(D[1]-C[1])**2}+{(D[2]-C[2])**2})")
print(f"           = sqrt({(D[0]-C[0])**2 + (D[1]-C[1])**2 + (D[2]-C[2])**2})")
print(f"           = {euclidean_distance(C, D):.4f}")
print()

print("WHEN TO USE EUCLIDEAN:")
print("-" * 70)
print("  GOOD for: Numerical features on similar scales (height, weight, age)")
print("            Low-dimensional data (< ~20 features)")
print("            When the 'blob' shape of clusters makes sense")
print()
print("  BAD for:  High-dimensional data (curse of dimensionality!)")
print("            Features on very different scales (income vs age)")
print("            Text data (word count vectors)")
print()
print("  WARNING: ALWAYS scale features before using Euclidean distance!")
print("  Unscaled: income=50000 dominates age=25 completely!")
print()

# ============================================================================
# SECTION 3: MANHATTAN DISTANCE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Manhattan Distance — The City Block Metric")
print("=" * 80)
print()

print("Imagine you are in New York City, on a grid of streets.")
print("You can't cut diagonally through buildings — you must walk")
print("along the blocks. That's Manhattan distance!")
print()
print("FORMULA:")
print("-" * 70)
print("  d(A, B) = |a1-b1| + |a2-b2| + ... + |an-bn|")
print()
print("  Also called: L1 norm, Taxicab distance, City-block distance")
print()

def manhattan_distance(a, b):
    """
    Compute Manhattan (L1) distance between two points.
    Sum of absolute differences along each dimension.
    """
    diff = np.array(a) - np.array(b)
    return np.sum(np.abs(diff))

print("WORKED EXAMPLE (2D):")
print("-" * 70)
print(f"  Point A = {A}")
print(f"  Point B = {B}")
print()
print(f"  Step 1 — Differences:     (4-1, 6-2) = ({B[0]-A[0]:.0f}, {B[1]-A[1]:.0f})")
print(f"  Step 2 — Absolute values: (|{B[0]-A[0]:.0f}|, |{B[1]-A[1]:.0f}|) = ({abs(B[0]-A[0]):.0f}, {abs(B[1]-A[1]):.0f})")
print(f"  Step 3 — Sum:             {abs(B[0]-A[0]):.0f} + {abs(B[1]-A[1]):.0f} = {manhattan_distance(A, B):.4f}")
print()
print(f"  Manhattan distance(A, B) = {manhattan_distance(A, B):.4f}")
print(f"  Euclidean distance(A, B) = {euclidean_distance(A, B):.4f}")
print()
print("  Notice: Manhattan >= Euclidean always! (Triangle inequality)")
print()

print("COMPARISON TABLE:")
print("-" * 70)
print(f"{'Metric':<20} {'Distance A-B':<18} {'Formula Used'}")
print("-" * 70)
print(f"{'Euclidean':<20} {euclidean_distance(A, B):<18.4f} sqrt(sum of squared diffs)")
print(f"{'Manhattan':<20} {manhattan_distance(A, B):<18.4f} sum of absolute diffs")
print()

print("WHEN TO USE MANHATTAN:")
print("-" * 70)
print("  GOOD for: Grid-like data (pixels, geographic blocks)")
print("            Data with many outliers (less sensitive than Euclidean)")
print("            High-dimensional sparse data")
print("            When all dimensions are equally important")
print()
print("  WHY less sensitive to outliers?")
print("  Euclidean squares differences -> huge outliers dominate")
print("  Manhattan takes absolute value -> outliers don't blow up as much")
print()

# Demonstrate outlier sensitivity
print("OUTLIER SENSITIVITY DEMO:")
print("-" * 70)
normal_A = np.array([0.0, 0.0])
normal_B = np.array([3.0, 4.0])
outlier_B = np.array([100.0, 4.0])   # x is an outlier

print(f"  Normal:  A={normal_A}, B={normal_B}")
print(f"    Euclidean: {euclidean_distance(normal_A, normal_B):.2f}  |  Manhattan: {manhattan_distance(normal_A, normal_B):.2f}")
print()
print(f"  Outlier: A={normal_A}, B={outlier_B}")
print(f"    Euclidean: {euclidean_distance(normal_A, outlier_B):.2f}  |  Manhattan: {manhattan_distance(normal_A, outlier_B):.2f}")
print()
ratio_euc = euclidean_distance(normal_A, outlier_B) / euclidean_distance(normal_A, normal_B)
ratio_man = manhattan_distance(normal_A, outlier_B) / manhattan_distance(normal_A, normal_B)
print(f"  Euclidean multiplied by: {ratio_euc:.1f}x  (very sensitive!)")
print(f"  Manhattan multiplied by: {ratio_man:.1f}x  (less sensitive)")
print()

# ============================================================================
# SECTION 4: COSINE SIMILARITY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Cosine Similarity — Measuring Angle, Not Distance")
print("=" * 80)
print()

print("A COMPLETELY DIFFERENT IDEA:")
print("-" * 70)
print("Euclidean and Manhattan measure how far points are from each other.")
print("Cosine similarity measures the ANGLE between two vectors.")
print()
print("  'Are these two vectors pointing in the same direction?'")
print()
print("MOTIVATION: Text Analysis")
print("  Document A: 'cat cat cat dog dog'    (3 cats, 2 dogs)")
print("  Document B: 'cat dog'                 (1 cat, 1 dog)")
print("  Document C: 'fish fish fish fish fish' (5 fish)")
print()
print("  A and B are about the same topic (cats and dogs)!")
print("  Euclidean: A and B look different (different word counts)")
print("  Cosine:    A and B are VERY similar (same direction in word-space)")
print()

print("FORMULA:")
print("-" * 70)
print("  cos_sim(A, B) = (A . B) / (||A|| * ||B||)")
print()
print("  Where:")
print("    A . B  = dot product   = sum(a_i * b_i)")
print("    ||A||  = L2 norm of A  = sqrt(sum(a_i^2))")
print()
print("  Result is between -1 and 1:")
print("    +1  = identical direction (maximally similar)")
print("     0  = perpendicular (no similarity)")
print("    -1  = opposite direction (maximally dissimilar)")
print()
print("  Often converted to 'Cosine Distance' = 1 - cosine_similarity")
print()

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two vectors.
    Ranges from -1 (opposite) to 1 (identical direction).
    """
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def cosine_distance(a, b):
    """Cosine distance = 1 - cosine similarity. Ranges from 0 to 2."""
    return 1.0 - cosine_similarity(a, b)

# Text example: word vectors (cat, dog, fish)
doc_A = np.array([3.0, 2.0, 0.0])  # 3 cats, 2 dogs, 0 fish
doc_B = np.array([1.0, 1.0, 0.0])  # 1 cat, 1 dog, 0 fish
doc_C = np.array([0.0, 0.0, 5.0])  # 0 cats, 0 dogs, 5 fish

print("TEXT DOCUMENT EXAMPLE (word counts: [cat, dog, fish]):")
print("-" * 70)
print(f"  Doc A: {doc_A}  (cat=3, dog=2, fish=0)")
print(f"  Doc B: {doc_B}  (cat=1, dog=1, fish=0)")
print(f"  Doc C: {doc_C}  (cat=0, dog=0, fish=5)")
print()
print(f"  cosine_similarity(A, B) = {cosine_similarity(doc_A, doc_B):.4f}  <- Same topic!")
print(f"  cosine_similarity(A, C) = {cosine_similarity(doc_A, doc_C):.4f}  <- Different topics!")
print(f"  cosine_similarity(B, C) = {cosine_similarity(doc_B, doc_C):.4f}  <- Different topics!")
print()
print(f"  euclidean_distance(A, B) = {euclidean_distance(doc_A, doc_B):.4f}  <- Looks large!")
print(f"  euclidean_distance(A, C) = {euclidean_distance(doc_A, doc_C):.4f}  <- Also large!")
print()
print("  Euclidean treats A and C as equally far as A and B!")
print("  Cosine CORRECTLY identifies A and B as similar (same topic).")
print()

# Step-by-step calculation
print("STEP-BY-STEP (A vs B):")
print("-" * 70)
dot = np.dot(doc_A, doc_B)
norm_a = np.linalg.norm(doc_A)
norm_b = np.linalg.norm(doc_B)
print(f"  A . B = (3*1) + (2*1) + (0*0) = {dot:.0f}")
print(f"  ||A|| = sqrt(3^2 + 2^2 + 0^2) = sqrt({3**2+2**2}) = {norm_a:.4f}")
print(f"  ||B|| = sqrt(1^2 + 1^2 + 0^2) = sqrt({1+1}) = {norm_b:.4f}")
print(f"  cos_sim = {dot:.0f} / ({norm_a:.4f} * {norm_b:.4f}) = {dot/(norm_a*norm_b):.4f}")
print()

print("WHEN TO USE COSINE SIMILARITY:")
print("-" * 70)
print("  GOOD for: Text data (TF-IDF vectors, word embeddings)")
print("            Recommendation systems (user preference vectors)")
print("            Any sparse, high-dimensional data")
print("            When you care about direction, not magnitude")
print()
print("  BAD for:  When magnitude matters (temperature: 10C vs 20C)")
print("            Data with negative values that have real meaning")
print()

# ============================================================================
# SECTION 5: PAIRWISE DISTANCE MATRIX FROM SCRATCH
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Pairwise Distance Matrix — Comparing All Points at Once")
print("=" * 80)
print()

print("In clustering, we often need the distance between EVERY pair of points.")
print("This is the PAIRWISE DISTANCE MATRIX (also called the distance matrix).")
print()
print("For N points -> N x N matrix where entry [i,j] = distance(point_i, point_j)")
print()
print("Properties:")
print("  * Diagonal = 0 (distance from a point to itself)")
print("  * Symmetric: d(i,j) = d(j,i) (distance is the same both ways)")
print()

def pairwise_distances_scratch(X, metric='euclidean'):
    """
    Compute full pairwise distance matrix from scratch.
    X: (n_samples, n_features) array
    Returns: (n_samples, n_samples) distance matrix
    """
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if metric == 'euclidean':
                D[i, j] = euclidean_distance(X[i], X[j])
            elif metric == 'manhattan':
                D[i, j] = manhattan_distance(X[i], X[j])
            elif metric == 'cosine':
                D[i, j] = cosine_distance(X[i], X[j])
    return D

# Small toy dataset: 5 points in 2D
np.random.seed(42)
toy_points = np.array([
    [1.0, 1.0],
    [1.5, 2.0],
    [5.0, 8.0],
    [8.0, 8.0],
    [9.0, 9.0],
])
labels = ['P1', 'P2', 'P3', 'P4', 'P5']

print("TOY DATASET (5 points in 2D):")
print("-" * 70)
for i, (pt, lab) in enumerate(zip(toy_points, labels)):
    print(f"  {lab}: ({pt[0]:.1f}, {pt[1]:.1f})")
print()

D_euc = pairwise_distances_scratch(toy_points, metric='euclidean')
D_man = pairwise_distances_scratch(toy_points, metric='manhattan')

print("EUCLIDEAN PAIRWISE DISTANCE MATRIX:")
print("-" * 70)
print(f"{'':>5}", end="")
for lab in labels:
    print(f"{lab:>8}", end="")
print()
for i, row_lab in enumerate(labels):
    print(f"{row_lab:>5}", end="")
    for j in range(len(labels)):
        print(f"{D_euc[i,j]:>8.2f}", end="")
    print()
print()

print("INTERPRETATION:")
print("  P1 and P2 are close (same cluster: bottom-left group)")
print("  P4 and P5 are close (same cluster: top-right group)")
print("  P3 sits between the two clusters")
print()

# ============================================================================
# SECTION 6: SCIKIT-LEARN VERSION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Using Scikit-Learn's pairwise_distances")
print("=" * 80)
print()

print("In production, use sklearn — it's vectorized and much faster!")
print()

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler

    # Verify our scratch version matches sklearn
    D_sklearn_euc = pairwise_distances(toy_points, metric='euclidean')
    D_sklearn_man = pairwise_distances(toy_points, metric='manhattan')
    D_sklearn_cos = pairwise_distances(toy_points, metric='cosine')

    max_diff = np.max(np.abs(D_sklearn_euc - D_euc))
    print(f"  Our scratch Euclidean vs sklearn: max difference = {max_diff:.2e} (essentially zero)")
    print("  Our implementation is CORRECT!")
    print()

    print("SKLEARN USAGE:")
    print("-" * 70)
    print("  from sklearn.metrics import pairwise_distances")
    print()
    print("  D_euc = pairwise_distances(X, metric='euclidean')")
    print("  D_man = pairwise_distances(X, metric='manhattan')")
    print("  D_cos = pairwise_distances(X, metric='cosine')")
    print()

    # Larger dataset to show scaling impact
    print("FEATURE SCALING DEMO (IMPORTANT!):")
    print("-" * 70)
    print("Creating 50 customer records: [age (0-60), income (0-100000)]")
    print()
    np.random.seed(7)
    n = 50
    age    = np.random.uniform(20, 80, n)
    income = np.random.uniform(20000, 100000, n)
    X_unscaled = np.column_stack([age, income])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)

    D_unscaled = pairwise_distances(X_unscaled, metric='euclidean')
    D_scaled   = pairwise_distances(X_scaled,   metric='euclidean')

    # Average distance to nearest neighbour
    np.fill_diagonal(D_unscaled, np.inf)
    np.fill_diagonal(D_scaled,   np.inf)
    avg_nn_unscaled = np.mean(np.min(D_unscaled, axis=1))
    avg_nn_scaled   = np.mean(np.min(D_scaled,   axis=1))

    print(f"  Without scaling — avg nearest-neighbour distance: {avg_nn_unscaled:.2f}")
    print(f"  With scaling    — avg nearest-neighbour distance: {avg_nn_scaled:.4f}")
    print()
    print("  Income (0-100k) would DOMINATE age (0-60) without scaling.")
    print("  ALWAYS scale before Euclidean/Manhattan distance!")
    print()

except ImportError:
    print("  sklearn not installed. Run: pip install scikit-learn")
    print("  (All concepts still apply — our from-scratch code works!)")

# ============================================================================
# VISUALIZATION 1: Distance metrics side by side in 2D
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Visualizations")
print("=" * 80)
print()
print("Generating Visualization 1: Distance Metrics in 2D...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('DISTANCE METRICS: Three Ways to Measure "How Far Apart"',
             fontsize=15, fontweight='bold', y=1.02)

origin = np.array([1.0, 1.0])
target = np.array([4.0, 5.0])

# --- Plot 1: Euclidean ---
ax = axes[0]
ax.scatter(*origin, s=200, c='steelblue', zorder=5, edgecolors='black', linewidths=1.5)
ax.scatter(*target, s=200, c='tomato',    zorder=5, edgecolors='black', linewidths=1.5)
ax.annotate('', xy=target, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5))
dist_euc = euclidean_distance(origin, target)
mid = (origin + target) / 2
ax.text(mid[0]+0.15, mid[1], f'd = {dist_euc:.2f}', fontsize=11,
        color='steelblue', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax.text(*origin + np.array([-0.1, -0.3]), 'A', fontsize=13, fontweight='bold')
ax.text(*target + np.array([0.1,  0.1]), 'B', fontsize=13, fontweight='bold')
ax.set_title('Euclidean Distance\n"Straight-line (crow flies)"', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6); ax.set_ylim(0, 7)
ax.text(0.5, 0.06, f'= sqrt((4-1)^2 + (5-1)^2) = {dist_euc:.2f}',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- Plot 2: Manhattan ---
ax = axes[1]
ax.scatter(*origin, s=200, c='steelblue', zorder=5, edgecolors='black', linewidths=1.5)
ax.scatter(*target, s=200, c='tomato',    zorder=5, edgecolors='black', linewidths=1.5)
# Draw the L-shaped path
corner = np.array([target[0], origin[1]])
ax.annotate('', xy=corner, xytext=origin,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.5))
ax.annotate('', xy=target, xytext=corner,
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=2.5))
ax.scatter(*corner, s=100, c='darkorange', zorder=4, marker='s')
dist_man = manhattan_distance(origin, target)
ax.text(2.5, origin[1]-0.4, f'|4-1|=3', fontsize=10, color='darkorange', fontweight='bold')
ax.text(target[0]+0.15, 3.0,  f'|5-1|=4', fontsize=10, color='darkorange', fontweight='bold')
ax.text(0.05, 0.93, f'Total = 3 + 4 = {dist_man:.0f}', transform=ax.transAxes,
        fontsize=11, color='darkorange', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.text(*origin + np.array([-0.1, -0.3]), 'A', fontsize=13, fontweight='bold')
ax.text(*target + np.array([0.1,  0.1]), 'B', fontsize=13, fontweight='bold')
ax.set_title('Manhattan Distance\n"City block (L-shaped path)"', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6); ax.set_ylim(0, 7)
ax.text(0.5, 0.06, f'= |4-1| + |5-1| = 3 + 4 = {dist_man:.2f}',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# --- Plot 3: Cosine ---
ax = axes[2]
v1 = np.array([3.0, 1.0])
v2 = np.array([2.0, 3.0])
v3 = np.array([6.0, 2.0])  # Same direction as v1, larger magnitude

origin_o = np.array([0.0, 0.0])
ax.annotate('', xy=v1, xytext=origin_o,
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5))
ax.annotate('', xy=v2, xytext=origin_o,
            arrowprops=dict(arrowstyle='->', color='tomato', lw=2.5))
ax.annotate('', xy=v3, xytext=origin_o,
            arrowprops=dict(arrowstyle='->', color='steelblue', lw=2.5, linestyle='dashed'))

cos_12 = cosine_similarity(v1, v2)
cos_13 = cosine_similarity(v1, v3)

ax.text(v1[0]+0.1, v1[1]+0.1, f'v1 = ({v1[0]:.0f},{v1[1]:.0f})', fontsize=10, color='steelblue', fontweight='bold')
ax.text(v2[0]-0.1, v2[1]+0.2, f'v2 = ({v2[0]:.0f},{v2[1]:.0f})', fontsize=10, color='tomato', fontweight='bold')
ax.text(v3[0]+0.1, v3[1]+0.1, f'v3 = ({v3[0]:.0f},{v3[1]:.0f})\n(same dir as v1)', fontsize=9,
        color='steelblue', alpha=0.7)

ax.text(0.05, 0.88, f'sim(v1, v2) = {cos_12:.3f}  (different direction)',
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
ax.text(0.05, 0.78, f'sim(v1, v3) = {cos_13:.3f}  (same direction!)',
        transform=ax.transAxes, fontsize=9, color='steelblue',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))

ax.set_title('Cosine Similarity\n"Angle between vectors"', fontsize=12, fontweight='bold')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 7.5); ax.set_ylim(-0.5, 4.5)
ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
ax.axvline(0, color='black', linewidth=0.8, alpha=0.5)
ax.text(0.5, 0.06, f'= (A.B) / (||A||*||B||) = cos(angle)',
        transform=ax.transAxes, ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_distance_metrics_2d.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_distance_metrics_2d.png")

# ============================================================================
# VISUALIZATION 2: Pairwise distance heatmaps
# ============================================================================
print("Generating Visualization 2: Pairwise Distance Heatmaps...")

# Generate a richer dataset: two clear clusters
np.random.seed(0)
cluster1 = np.random.randn(8, 2) * 0.8 + np.array([0.0, 0.0])
cluster2 = np.random.randn(7, 2) * 0.8 + np.array([6.0, 6.0])
X_clust = np.vstack([cluster1, cluster2])
n_clust = X_clust.shape[0]
point_labels = [f'C1-{i+1}' for i in range(8)] + [f'C2-{i+1}' for i in range(7)]

D_euc_c  = pairwise_distances_scratch(X_clust, metric='euclidean')
D_man_c  = pairwise_distances_scratch(X_clust, metric='manhattan')
D_cos_c  = pairwise_distances_scratch(X_clust, metric='cosine')

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('PAIRWISE DISTANCE HEATMAPS — Same Data, Three Metrics\n'
             '(Dark = Close, Light = Far; Block structure reveals clusters)',
             fontsize=13, fontweight='bold')

cmaps = ['Blues', 'Oranges', 'Greens']
titles = ['Euclidean Distance', 'Manhattan Distance', 'Cosine Distance']
Ds = [D_euc_c, D_man_c, D_cos_c]

for ax, D, cmap, title in zip(axes, Ds, cmaps, titles):
    im = ax.imshow(D, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_clust))
    ax.set_yticks(range(n_clust))
    ax.set_xticklabels(point_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(point_labels, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    # Draw cluster boundary lines
    ax.axhline(y=7.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
    ax.axvline(x=7.5, color='red', linewidth=2, linestyle='--', alpha=0.8)
    ax.text(3.5, -1.8, 'Cluster 1', ha='center', fontsize=9,
            color='red', fontweight='bold')
    ax.text(11.5, -1.8, 'Cluster 2', ha='center', fontsize=9,
            color='red', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_pairwise_distance_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_pairwise_distance_heatmap.png")

# ============================================================================
# VISUALIZATION 3: Effect of metric on "who is the nearest neighbour"
# ============================================================================
print("Generating Visualization 3: How Metric Choice Changes Nearest Neighbours...")

np.random.seed(3)
points = np.array([
    [2.0, 2.0],    # query point
    [2.5, 2.2],    # close Euclidean
    [1.5, 2.5],    # medium distance
    [5.0, 2.1],    # far on y-axis but close Manhattan in x
    [2.1, 5.0],    # similar direction (cosine)
    [6.0, 6.0],    # far
])
pt_names = ['Query', 'A', 'B', 'C', 'D', 'E']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('HOW METRIC CHOICE CHANGES "NEAREST NEIGHBOURS"',
             fontsize=13, fontweight='bold')

for ax_idx, (metric_name, metric_fn) in enumerate([
    ('Euclidean', euclidean_distance),
    ('Manhattan', manhattan_distance)
]):
    ax = axes[ax_idx]
    query = points[0]
    others = points[1:]
    dists = [metric_fn(query, p) for p in others]
    nearest_idx = np.argmin(dists)

    colors = ['gold'] + ['lightcoral' if i == nearest_idx else 'lightblue' for i in range(len(others))]
    for i, (pt, name, col) in enumerate(zip(points, pt_names, colors)):
        ax.scatter(*pt, s=300, c=col, edgecolors='black', linewidths=1.5, zorder=5)
        ax.text(pt[0]+0.1, pt[1]+0.15, name, fontsize=11, fontweight='bold')
        if i > 0:
            d = dists[i-1]
            ax.text(pt[0]+0.1, pt[1]-0.3, f'd={d:.2f}', fontsize=8, color='gray')

    # Highlight nearest neighbour
    nearest_pt = others[nearest_idx]
    ax.plot([query[0], nearest_pt[0]], [query[1], nearest_pt[1]],
            'r--', linewidth=2.5, label=f'Nearest: {pt_names[nearest_idx+1]}')

    ax.set_title(f'{metric_name} Distance\nNearest to Query: {pt_names[nearest_idx+1]}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')

    # Legend patches
    gold_patch = mpatches.Patch(color='gold', label='Query point')
    red_patch  = mpatches.Patch(color='lightcoral', label='Nearest neighbour')
    blue_patch = mpatches.Patch(color='lightblue', label='Other points')
    ax.legend(handles=[gold_patch, red_patch, blue_patch], fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_nearest_neighbour_by_metric.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_nearest_neighbour_by_metric.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Distance Metrics")
print("=" * 80)
print()
print("WHAT WE LEARNED:")
print("-" * 70)
print()
print("1. EUCLIDEAN DISTANCE:")
print("   Formula: sqrt(sum((ai - bi)^2))")
print("   Best for: Numerical data on similar scales")
print("   Warning:  Always standardize features first!")
print()
print("2. MANHATTAN DISTANCE:")
print("   Formula: sum(|ai - bi|)")
print("   Best for: Grid-like data, outlier-heavy datasets")
print("   Pro:      Less sensitive to extreme outliers")
print()
print("3. COSINE SIMILARITY:")
print("   Formula: (A . B) / (||A|| * ||B||)")
print("   Best for: Text, sparse high-dimensional data")
print("   Key idea: Measures angle, NOT magnitude")
print()
print("4. PAIRWISE DISTANCE MATRIX:")
print("   N x N matrix — distance between every pair of points")
print("   Foundation of hierarchical clustering, kernel methods")
print("   Use: sklearn.metrics.pairwise_distances(X, metric='...')")
print()
print("METRIC SELECTION CHEAT SHEET:")
print("-" * 70)
print(f"  {'Data Type':<30} {'Recommended Metric'}")
print(f"  {'-'*28} {'-'*20}")
print(f"  {'Numerical (scaled)':<30} {'Euclidean'}")
print(f"  {'Numerical (outliers present)':<30} {'Manhattan'}")
print(f"  {'Text / bag-of-words':<30} {'Cosine'}")
print(f"  {'User-item ratings':<30} {'Cosine'}")
print(f"  {'Geographic coordinates':<30} {'Euclidean or Haversine'}")
print(f"  {'Mixed types':<30} {'Gower distance (advanced)'}")
print()
print("=" * 80)
print("Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("  01_distance_metrics_2d.png")
print("  02_pairwise_distance_heatmap.png")
print("  03_nearest_neighbour_by_metric.png")
print("=" * 80)
print()
print("NEXT STEPS:")
print("  1. Study the heatmaps — notice the block structure = clusters!")
print("  2. Try: change the cluster separation and see how heatmaps change")
print("  3. Next module: 02_variance_and_covariance.py")
print("     (The math that powers PCA!)")
print()
print("=" * 80)
print("DISTANCE METRICS MASTERED!")
print("   You now know how machines measure 'how similar' things are!")
print("=" * 80)
