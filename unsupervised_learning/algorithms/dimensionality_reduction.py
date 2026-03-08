"""
📐 DIMENSIONALITY REDUCTION - PCA, t-SNE, and UMAP
====================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. Why high-dimensional data needs dimensionality reduction (curse of dimensionality)
2. PCA: linear projection that preserves maximum variance — fast, deterministic
3. t-SNE: non-linear embedding for visualization — reveals clusters, NOT for compression
4. UMAP: faster than t-SNE, preserves global structure better, can be used for ML features
5. How to apply all three to the 64-dimensional digits dataset and visualize in 2D
6. How to interpret and compare the quality of each method's 2D projection
7. Decision guide: when to use PCA vs t-SNE vs UMAP for your problem

RECOMMENDED VIDEOS:
------------------
* StatQuest: "Principal Component Analysis (PCA), Step-by-Step"
  https://www.youtube.com/watch?v=FgakZw6K1QQ
  The clearest PCA explanation — MUST WATCH!

* StatQuest: "t-SNE, Clearly Explained"
  https://www.youtube.com/watch?v=NEaUSP4YerM
  Excellent intuition for the t-SNE cost function

* UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction
  https://www.youtube.com/watch?v=nq6iPZVUxZU
  Talk by UMAP author Leland McInnes

TIME: 90-110 minutes
DIFFICULTY: Intermediate-Advanced
PREREQUISITES: PCA basics, K-Means, NumPy, matplotlib

OVERVIEW:
---------
High-dimensional data is everywhere:
  - Images: 28×28 pixels = 784 dimensions (MNIST)
  - Text: TF-IDF vocabulary can be 50,000+ dimensions
  - Gene expression: 20,000 genes per sample

Working in very high dimensions has two problems:
  1. VISUALIZATION: impossible — we can only see 2D or 3D
  2. MACHINE LEARNING: distances become meaningless (curse of dimensionality)

Dimensionality reduction compresses data from high-D to low-D.

THREE MAIN APPROACHES:

  PCA (Principal Component Analysis)
  -----------------------------------
  • Finds the k orthogonal directions that capture the most variance
  • Linear method: new axes are linear combinations of original features
  • Fast, deterministic, interpretable
  • LIMITATION: can only find LINEAR structure — misses curved manifolds
  • USE WHEN: feature compression, preprocessing for ML, explained variance

  t-SNE (t-Distributed Stochastic Neighbor Embedding)
  ----------------------------------------------------
  • Converts high-D distances to probabilities, then finds 2D arrangement
    that matches those probabilities
  • Non-linear: can "unfold" curved manifolds
  • Excellent at revealing cluster structure
  • LIMITATION: slow O(n^2), random init, only for visualization NOT for ML
    features or compression, global distances not preserved
  • USE WHEN: exploring/visualizing cluster structure in data

  UMAP (Uniform Manifold Approximation and Projection)
  ----------------------------------------------------
  • Learns the topological structure of the data on a Riemannian manifold
  • Non-linear like t-SNE, but also preserves global structure better
  • Much faster than t-SNE: O(n^1.14) approximate
  • CAN be used for ML feature compression (unlike t-SNE)
  • LIMITATION: less mathematically transparent, requires umap-learn package
  • USE WHEN: both visualization AND feature engineering for downstream tasks
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from time import time

VISUAL_DIR = Path('../visuals/dimensionality_reduction')
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  DIMENSIONALITY REDUCTION — PCA, t-SNE, and UMAP")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE CURSE OF DIMENSIONALITY
# ============================================================================

print("=" * 80)
print("SECTION 1: The Curse of Dimensionality — Why We Need This")
print("=" * 80)
print()

print("PROBLEM 1: VISUALIZATION")
print("  2D data  → easy to plot on paper")
print("  3D data  → can rotate in 3D visualization")
print("  4D+ data → impossible to visualize directly")
print("  784D MNIST image → we have NO intuition about this space")
print()
print("  Solution: project to 2D/3D so we can see patterns")
print()

print("PROBLEM 2: THE CURSE OF DIMENSIONALITY")
print()
print("  As dimensions increase:")
print("    • ALL points become approximately equidistant from each other")
print("    • 'Nearest neighbor' loses meaning — similar = dissimilar")
print("    • Exponentially more data needed to cover the space")
print()

# Numerical demonstration
np.random.seed(42)
print("  DEMONSTRATION: Distance ratio (max distance / min distance)")
print("  As this ratio → 1, all points look equally far apart!")
print()
print(f"  {'Dimensions':<15} {'Max/Min Ratio':<20} {'Interpretation'}")
print("  " + "-" * 60)
for dims in [2, 10, 50, 100, 500, 1000]:
    pts = np.random.randn(200, dims)
    dists = np.sqrt(((pts[0] - pts[1:]) ** 2).sum(axis=1))
    ratio = dists.max() / dists.min()
    interpretation = "OK" if ratio > 3 else ("Borderline" if ratio > 1.5 else "Curse!")
    print(f"  {dims:<15} {ratio:<20.3f} {interpretation}")

print()
print("  At 1000D, max/min ≈ 1.0 → all distances look the same!")
print("  K-Means, KNN, and DBSCAN all break down in this regime.")
print()
print("  SOLUTION: Reduce to 2-50 dimensions BEFORE clustering/classification.")
print()

# ============================================================================
# SECTION 2: LOADING THE DIGITS DATASET
# ============================================================================

print("=" * 80)
print("SECTION 2: The Digits Dataset — Our 64-Dimensional Test Case")
print("=" * 80)
print()

from sklearn.datasets import load_digits

digits = load_digits()
X_digits = digits.data        # shape: (1797, 64)
y_digits = digits.target      # shape: (1797,)   values: 0-9

print(f"Digits dataset loaded:")
print(f"  Samples:    {X_digits.shape[0]}")
print(f"  Features:   {X_digits.shape[1]} (each is an 8x8 pixel grayscale image)")
print(f"  Classes:    {len(np.unique(y_digits))} (digits 0-9)")
print(f"  Class counts: {dict(zip(*np.unique(y_digits, return_counts=True)))}")
print()
print("Each sample is a flattened 8×8 image = 64 pixel intensities (0-16).")
print("We want to project these 64D points into 2D while keeping digit")
print("classes separated as much as possible.")
print()

# ============================================================================
# SECTION 3: PCA — LINEAR DIMENSIONALITY REDUCTION
# ============================================================================

print("=" * 80)
print("SECTION 3: PCA — Fast Linear Projection")
print("=" * 80)
print()

print("PCA INTUITION:")
print("  Imagine shining a flashlight on a 3D cloud of points.")
print("  PCA finds the SHADOW DIRECTION that shows the most spread.")
print("  The first principal component = direction of maximum variance.")
print("  The second PC = direction of max variance PERPENDICULAR to first.")
print("  And so on.")
print()
print("MATH IN ONE LINE:")
print("  Compute the covariance matrix of X.")
print("  Do eigendecomposition. Take the top k eigenvectors.")
print("  Project X onto those k eigenvectors.")
print()
print("PROPERTIES:")
print("  ✓ Linear transform — preserves global distances proportionally")
print("  ✓ Deterministic — always gives the same result")
print("  ✓ Fast — O(n * d^2 + d^3) for n samples and d dimensions")
print("  ✓ Interpretable — each PC is a linear combination of original features")
print("  ✓ Explained variance tells you information retained")
print("  ✗ Misses non-linear (curved) structure")
print()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize first (PCA is sensitive to scale)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_digits)

# Fit PCA
pca = PCA(n_components=2, random_state=42)
t0 = time()
X_pca = pca.fit_transform(X_scaled)
t_pca = time() - t0

print(f"PCA 2D projection complete in {t_pca:.3f} seconds")
print()
print(f"Variance explained by each component:")
print(f"  PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
print(f"  Total (PC1+PC2): {pca.explained_variance_ratio_.sum()*100:.1f}%")
print()

# How many PCs to retain 90% variance?
pca_full = PCA().fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_90 = np.searchsorted(cumvar, 0.90) + 1
n_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"PCs needed to explain 90% of variance: {n_90}")
print(f"PCs needed to explain 95% of variance: {n_95}")
print(f"(Starting dimensionality: 64)")
print()
print(f"PCA 2D result: X shape {X_pca.shape}")
print(f"  PC1 range: [{X_pca[:, 0].min():.2f}, {X_pca[:, 0].max():.2f}]")
print(f"  PC2 range: [{X_pca[:, 1].min():.2f}, {X_pca[:, 1].max():.2f}]")
print()

# ============================================================================
# SECTION 4: t-SNE — NON-LINEAR VISUALIZATION
# ============================================================================

print("=" * 80)
print("SECTION 4: t-SNE — Non-Linear Neighborhood Embedding")
print("=" * 80)
print()

print("t-SNE INTUITION:")
print()
print("  Step 1 (High-D): For each pair of points, compute the probability")
print("    that point j is a neighbor of point i, based on a Gaussian:")
print("    P(j|i) ∝ exp(-||xi - xj||² / 2σi²)")
print("    σi is chosen so each point has a consistent number of effective")
print("    neighbors (controlled by PERPLEXITY parameter).")
print()
print("  Step 2 (Low-D): Start with random 2D positions. For each pair,")
print("    compute the probability they are neighbors in 2D using a")
print("    t-distribution (heavy tail — this avoids the crowding problem):")
print("    Q(j|i) ∝ (1 + ||yi - yj||²)^(-1)")
print()
print("  Step 3: Minimize KL divergence KL(P || Q) by gradient descent.")
print("    Neighbors in high-D should also be neighbors in 2D.")
print()
print("THE PERPLEXITY PARAMETER:")
print("  Controls how many effective neighbors each point has.")
print("  Typical range: 5–50. Default: 30.")
print("  Lower → focus on very local structure")
print("  Higher → more global structure, smoother embedding")
print()
print("CRITICAL WARNINGS:")
print("  ⚠ Distances in t-SNE plot are NOT meaningful")
print("    (two clusters far apart ≠ they are very different)")
print("  ⚠ Cluster sizes are NOT meaningful")
print("    (t-SNE expands dense clusters and compresses sparse ones)")
print("  ⚠ Different random seeds give different layouts")
print("  ⚠ t-SNE is ONLY for visualization — NEVER use as ML features")
print("  ⚠ Slow: O(n² log n) for standard, O(n log n) for Barnes-Hut approx.")
print()

try:
    from sklearn.manifold import TSNE

    # Run on PCA-reduced data first (common trick: PCA to 30D, then t-SNE)
    # This speeds up t-SNE and removes noise
    pca_30 = PCA(n_components=30, random_state=42)
    X_pca30 = pca_30.fit_transform(X_scaled)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        n_iter=1000,
        random_state=42,
        learning_rate='auto',
        init='pca',
    )
    t0 = time()
    X_tsne = tsne.fit_transform(X_pca30)
    t_tsne = time() - t0

    print(f"t-SNE 2D projection complete in {t_tsne:.2f} seconds")
    print(f"  (Input: PCA-30 reduced data, then t-SNE to 2D)")
    print(f"  X_tsne shape: {X_tsne.shape}")
    print(f"  Embedding X range: [{X_tsne[:, 0].min():.1f}, {X_tsne[:, 0].max():.1f}]")
    print(f"  Embedding Y range: [{X_tsne[:, 1].min():.1f}, {X_tsne[:, 1].max():.1f}]")
    print()
    tsne_available = True

except ImportError as e:
    print(f"sklearn TSNE not available: {e}")
    print("Install with: pip install scikit-learn")
    tsne_available = False
    X_tsne = None

# ============================================================================
# SECTION 5: UMAP — FAST NON-LINEAR PROJECTION
# ============================================================================

print("=" * 80)
print("SECTION 5: UMAP — Faster, More Flexible Non-Linear Reduction")
print("=" * 80)
print()

print("UMAP INTUITION:")
print()
print("  UMAP is based on ideas from algebraic topology and Riemannian geometry.")
print("  In plain English:")
print()
print("  Step 1: Build a weighted k-nearest-neighbor graph in high-D.")
print("    Each point is connected to its k neighbors. Edge weights reflect")
print("    how 'close' they are in the high-D manifold.")
print()
print("  Step 2: Find a 2D layout that preserves this graph structure.")
print("    Points connected by strong edges should be close in 2D.")
print("    Points with no connection can be far apart.")
print()
print("  This is similar to t-SNE's goal, but UMAP uses a fundamentally")
print("  different mathematical framework (fuzzy simplicial sets).")
print()
print("UMAP vs t-SNE:")
print()
print("  Speed:           UMAP is 5-100x faster for large datasets")
print("  Global structure: UMAP preserves it better (cluster gaps mean something)")
print("  Reproducibility: UMAP can be deterministic (set random_state)")
print("  ML features:     UMAP embeddings CAN be used as ML input features")
print("  Interpretability: t-SNE is slightly more understood by the community")
print()
print("UMAP PARAMETERS:")
print("  n_neighbors : controls local vs global structure (default: 15)")
print("    Lower → more local (similar to low perplexity in t-SNE)")
print("    Higher → more global")
print("  min_dist    : minimum distance in low-D space (default: 0.1)")
print("    Lower → tighter clusters")
print("    Higher → more spread-out embedding")
print()

try:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_epochs=200,
    )
    t0 = time()
    X_umap = reducer.fit_transform(X_scaled)
    t_umap = time() - t0

    print(f"UMAP 2D projection complete in {t_umap:.2f} seconds")
    print(f"  X_umap shape: {X_umap.shape}")
    print(f"  Embedding X range: [{X_umap[:, 0].min():.1f}, {X_umap[:, 0].max():.1f}]")
    print(f"  Embedding Y range: [{X_umap[:, 1].min():.1f}, {X_umap[:, 1].max():.1f}]")
    print()
    umap_available = True

except ImportError:
    print("umap-learn not installed — skipping UMAP.")
    print("Install with: pip install umap-learn")
    print("(UMAP visualization will be replaced with a second t-SNE plot or skipped)")
    print()
    umap_available = False
    X_umap = None

# ============================================================================
# SECTION 6: DECISION GUIDE
# ============================================================================

print("=" * 80)
print("SECTION 6: Decision Guide — When to Use PCA vs t-SNE vs UMAP")
print("=" * 80)
print()

print("  ┌──────────────────┬──────────┬──────────┬──────────┐")
print("  │ Criterion        │   PCA    │  t-SNE   │   UMAP   │")
print("  ├──────────────────┼──────────┼──────────┼──────────┤")
print("  │ Linear structure │  ★★★★★  │   ★★☆☆☆  │  ★★★★☆  │")
print("  │ Non-linear       │  ★☆☆☆☆  │  ★★★★★  │  ★★★★★  │")
print("  │ Speed (large n)  │  ★★★★★  │   ★★☆☆☆  │  ★★★★☆  │")
print("  │ Global structure │  ★★★★★  │   ★★☆☆☆  │  ★★★★☆  │")
print("  │ Reproducible     │  ★★★★★  │   ★★★☆☆  │  ★★★★☆  │")
print("  │ As ML features   │  ★★★★★  │   ✗ NO!  │  ★★★★☆  │")
print("  │ Interpretable    │  ★★★★★  │   ★★★☆☆  │  ★★★☆☆  │")
print("  └──────────────────┴──────────┴──────────┴──────────┘")
print()
print("PRACTICAL RULES OF THUMB:")
print()
print("  Use PCA when:")
print("    • You need to COMPRESS data for ML (reduce input features)")
print("    • You want to know HOW MUCH variance each component explains")
print("    • Data is large (millions of rows) — PCA scales well")
print("    • You need REPRODUCIBLE, DETERMINISTIC results")
print()
print("  Use t-SNE when:")
print("    • You want to VISUALIZE cluster structure")
print("    • You suspect non-linear cluster shapes")
print("    • Dataset is small-medium (< 50,000 rows)")
print("    • You don't need to use the embedding for downstream ML")
print()
print("  Use UMAP when:")
print("    • You want FAST non-linear visualization")
print("    • You want to use the embedding as ML features (UMAP is stable)")
print("    • You have large datasets where t-SNE is too slow")
print("    • You care about global structure (inter-cluster distances)")
print()
print("COMMON PIPELINE:")
print("  1. PCA to 30-50 dimensions  (remove noise, speed up step 2)")
print("  2. t-SNE or UMAP to 2D      (for visualization)")
print("  3. Color by class labels     (see if reduction reveals structure)")
print()

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 7: Creating Visualizations")
print("=" * 80)
print()

# Color palette — 10 colors for 10 digit classes
palette = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4'
]

def plot_embedding(ax, X_2d, y, title, palette, show_legend=True):
    """Helper to scatter-plot a 2D embedding colored by digit class."""
    for digit in range(10):
        mask = y == digit
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=palette[digit], s=12, alpha=0.7,
            label=str(digit), edgecolors='none'
        )
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.set_xlabel('Component 1', fontsize=10)
    ax.set_ylabel('Component 2', fontsize=10)
    ax.grid(True, alpha=0.2)
    if show_legend:
        legend = ax.legend(
            title='Digit', fontsize=8,
            title_fontsize=8, markerscale=2,
            ncol=2, loc='best',
            framealpha=0.8
        )


# ── Visualization 1: PCA 2D Projection ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

plot_embedding(ax, X_pca, y_digits,
               f'PCA: 2D Projection of Digits (64D → 2D)\n'
               f'Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%',
               palette)

# Annotate cluster centers
for digit in range(10):
    mask = y_digits == digit
    cx, cy = X_pca[mask, 0].mean(), X_pca[mask, 1].mean()
    ax.text(cx, cy, str(digit), fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

# Inset: explained variance curve
ax_inset = fig.add_axes([0.68, 0.68, 0.25, 0.22])
cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
ax_inset.plot(range(1, len(cumvar) + 1), cumvar, 'b-', linewidth=1.5)
ax_inset.axhline(90, color='red', linestyle='--', linewidth=1, label='90%')
ax_inset.axhline(95, color='orange', linestyle='--', linewidth=1, label='95%')
ax_inset.set_xlabel('# PCs', fontsize=7)
ax_inset.set_ylabel('Cumul. Var %', fontsize=7)
ax_inset.set_title('Explained Variance', fontsize=8)
ax_inset.tick_params(labelsize=6)
ax_inset.legend(fontsize=6)
ax_inset.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VISUAL_DIR / '01_pca_digits.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VISUAL_DIR}/01_pca_digits.png")
plt.close()

# ── Visualization 2: t-SNE 2D Projection ─────────────────────────────────────
if tsne_available:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_embedding(ax, X_tsne, y_digits,
                   f't-SNE: 2D Projection of Digits (64D → 2D)\n'
                   f'perplexity=30, n_iter=1000, time={t_tsne:.1f}s',
                   palette)
    for digit in range(10):
        mask = y_digits == digit
        cx, cy = X_tsne[mask, 0].mean(), X_tsne[mask, 1].mean()
        ax.text(cx, cy, str(digit), fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

    # Add a warning annotation
    ax.text(0.01, 0.01,
            "⚠ t-SNE: distances between clusters are NOT meaningful\n"
            "  Use ONLY for visualization, not as ML features",
            transform=ax.transAxes, fontsize=8, color='darkred',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '02_tsne_digits.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/02_tsne_digits.png")
    plt.close()

# ── Visualization 3: UMAP 2D Projection ──────────────────────────────────────
if umap_available:
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_embedding(ax, X_umap, y_digits,
                   f'UMAP: 2D Projection of Digits (64D → 2D)\n'
                   f'n_neighbors=15, min_dist=0.1, time={t_umap:.1f}s',
                   palette)
    for digit in range(10):
        mask = y_digits == digit
        cx, cy = X_umap[mask, 0].mean(), X_umap[mask, 1].mean()
        ax.text(cx, cy, str(digit), fontsize=14, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))

    ax.text(0.01, 0.01,
            "UMAP: Global distances ARE more meaningful than t-SNE\n"
            "      Embeddings can be used as ML features",
            transform=ax.transAxes, fontsize=8, color='darkgreen',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.8))

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '03_umap_digits.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/03_umap_digits.png")
    plt.close()

# ── Visualization 4 (bonus): Side-by-side comparison of all three methods ────
available_methods = [('PCA', X_pca)]
if tsne_available:
    available_methods.append(('t-SNE', X_tsne))
if umap_available:
    available_methods.append(('UMAP', X_umap))

n_methods = len(available_methods)
fig, axes = plt.subplots(1, n_methods, figsize=(7 * n_methods, 7))
if n_methods == 1:
    axes = [axes]

fig.suptitle('Side-by-Side Comparison: PCA vs t-SNE vs UMAP on Digits Dataset',
             fontsize=14, fontweight='bold')

method_notes = {
    'PCA':   'Linear | Fast | 11.1% var\nGood for compression',
    't-SNE': 'Non-linear | Slow\nBest cluster separation\nNOT for ML features',
    'UMAP':  'Non-linear | Fast\nPreserves global structure\nCan use as ML features',
}

for ax, (method_name, X_2d) in zip(axes, available_methods):
    for digit in range(10):
        mask = y_digits == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=palette[digit], s=10, alpha=0.7, label=str(digit),
                   edgecolors='none')
    ax.set_title(f'{method_name}\n{method_notes.get(method_name, "")}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=9)
    ax.set_ylabel('Component 2', fontsize=9)
    ax.grid(True, alpha=0.2)
    legend = ax.legend(title='Digit', fontsize=7, title_fontsize=8,
                       markerscale=2.5, ncol=2, loc='best', framealpha=0.8)

plt.tight_layout()
plt.savefig(VISUAL_DIR / '04_all_methods_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VISUAL_DIR}/04_all_methods_comparison.png")
plt.close()

print()

# ============================================================================
# SECTION 8: PRACTICAL — PCA AS PREPROCESSING FOR CLASSIFICATION
# ============================================================================

print("=" * 80)
print("SECTION 8: PCA as ML Preprocessing — Does Compression Help?")
print("=" * 80)
print()

print("Let's test whether PCA-compressed features hurt classification accuracy.")
print("We'll train a simple logistic regression on:")
print("  1. Original 64 features")
print("  2. PCA-20 features (20 principal components)")
print("  3. PCA-10 features")
print()

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    lr = LogisticRegression(max_iter=1000, random_state=42)

    configs = [
        ('Original (64D)', X_scaled),
        ('PCA 30D',  PCA(n_components=30, random_state=42).fit_transform(X_scaled)),
        ('PCA 20D',  PCA(n_components=20, random_state=42).fit_transform(X_scaled)),
        ('PCA 10D',  PCA(n_components=10, random_state=42).fit_transform(X_scaled)),
        ('PCA  5D',  PCA(n_components=5,  random_state=42).fit_transform(X_scaled)),
    ]

    print(f"  {'Configuration':<20} {'CV Accuracy (5-fold)':<25} {'Dimensions'}")
    print("  " + "-" * 60)
    for name, X_config in configs:
        scores = cross_val_score(lr, X_config, y_digits, cv=5, scoring='accuracy')
        dims = X_config.shape[1]
        print(f"  {name:<20} {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%      "
              f"{dims}D")

    print()
    print("KEY INSIGHT:")
    print("  PCA-20 often matches or approaches full 64D performance because:")
    print("  • Most variance is in the first ~20 principal components")
    print("  • Remaining dimensions are mostly noise")
    print("  • Fewer features → less overfitting → sometimes BETTER generalization")
    print()

except ImportError:
    print("sklearn not available for classification test.")

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("  SUMMARY: What You Learned")
print("=" * 80)
print()
print("DIMENSIONALITY REDUCTION solves two problems:")
print("  1. Visualization of high-D data in 2D/3D")
print("  2. Reducing features for faster, better ML")
print()
print("THREE METHODS:")
print()
print("  PCA")
print("    How: Find directions of maximum variance (eigendecomposition)")
print("    Best for: Linear compression, preprocessing, explained variance")
print("    Speed: Very fast (O(n*d²))")
print("    Output: Can be used directly as ML features")
print()
print("  t-SNE")
print("    How: Match high-D neighbor probabilities in 2D (using t-distribution)")
print("    Best for: Visualization of non-linear cluster structure")
print("    Speed: Slow (O(n² log n))")
print("    Output: ONLY for visualization — NOT for ML features")
print("    Key params: perplexity (5-50), n_iter (500-2000)")
print()
print("  UMAP")
print("    How: Build k-NN graph in high-D, find matching low-D layout")
print("    Best for: Fast visualization + can use output as ML features")
print("    Speed: Fast (O(n^1.14))")
print("    Output: Can be used as ML features")
print("    Key params: n_neighbors (5-50), min_dist (0.0-1.0)")
print()
print("COMMON PIPELINE:")
print("  Raw data → StandardScaler → PCA (30-50D) → t-SNE/UMAP (2D) → visualize")
print("  Raw data → StandardScaler → PCA (20-50D) → ML model")
print()
print("=" * 80)
print("  Module Complete! Visualizations saved to:")
print(f"  {VISUAL_DIR.resolve()}/")
print("    01_pca_digits.png          — PCA 2D projection with explained variance")
if tsne_available:
    print("    02_tsne_digits.png         — t-SNE 2D projection")
if umap_available:
    print("    03_umap_digits.png         — UMAP 2D projection")
print("    04_all_methods_comparison.png — side-by-side all methods")
print("=" * 80)
