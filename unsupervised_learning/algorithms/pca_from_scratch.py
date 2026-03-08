"""
PCA - PRINCIPAL COMPONENT ANALYSIS - Finding the Important Directions
======================================================================

LEARNING OBJECTIVES:
-------------------
After this module, you will understand:
1. Why dimensionality reduction matters: the curse of dimensionality
2. The PCA algorithm step by step: center → covariance → eigenvectors → project
3. How to implement PCA from scratch using only NumPy (eigendecomposition)
4. What explained variance ratio means and how to read a scree plot
5. How to use scikit-learn's PCA for production use
6. How to reconstruct (approximately) the original data from reduced dimensions
7. When PCA helps and when it does not (non-linear structure)

RECOMMENDED VIDEOS:
------------------
* StatQuest: "PCA main ideas in only 5 minutes!!!"
   https://www.youtube.com/watch?v=HMOI_lkzW08
   Best quick overview — start here!

* StatQuest: "StatQuest: PCA Step-by-Step"
   https://www.youtube.com/watch?v=FgakZw6K1QQ
   Detailed mathematical walkthrough — HIGHLY recommended!

* 3Blue1Brown: "Eigenvectors and eigenvalues"
   https://www.youtube.com/watch?v=PFDu9oVAE-g
   Understand what eigenvectors actually are visually

* Sentdex: "PCA from Scratch in Python"
   https://www.youtube.com/watch?v=kApPBm1YsqU

TIME: 90-120 minutes
DIFFICULTY: Intermediate
PREREQUISITES: Linear algebra foundations (vectors, matrices, dot products)
               K-Means clustering (or any Part 8 module)

OVERVIEW:
---------
PCA answers the question: "I have 50 features — most are redundant. What are
the 2-3 most informative directions I should look at?"

Key Intuition:
- Imagine a cloud of data points in 3D space
- PCA finds the axis along which the data spreads the MOST (1st component)
- Then the axis perpendicular to that with the NEXT most spread (2nd component)
- And so on — each component orthogonal to all previous ones
- We keep only the top few components and discard the rest
- Result: we go from 50D to 2D with minimal information loss!

Why this works:
- Variance = information
- The direction with maximum variance preserves the most information
- Directions with tiny variance are mostly noise anyway
- So we keep high-variance directions, drop low-variance ones
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Setup visualization directory
VISUAL_DIR = '../visuals/pca_from_scratch/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("PCA - PRINCIPAL COMPONENT ANALYSIS - Finding Important Directions")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE PROBLEM — TOO MANY FEATURES
# ============================================================================

print("=" * 80)
print("SECTION 1: The Problem — Too Many Features")
print("=" * 80)
print()

print("REAL-WORLD DATA IS HIGH-DIMENSIONAL:")
print("   - Medical records: 100+ clinical measurements per patient")
print("   - Text data: 50,000+ unique words as features")
print("   - Images: 64x64 pixels = 4,096 dimensions!")
print("   - Genomics: 20,000+ gene expression measurements")
print()

print("PROBLEMS WITH HIGH DIMENSIONS:")
print()
print("1. CURSE OF DIMENSIONALITY")
print("   - In high dimensions, all points become equally distant from each other")
print("   - Distance-based algorithms (KNN, K-Means) degrade or fail")
print("   - You need exponentially more data to cover high-dimensional space")
print()

print("2. MULTICOLLINEARITY")
print("   - Many features are correlated (petal length and petal width move together)")
print("   - Correlated features add noise but not real new information")
print("   - This hurts many models (especially linear ones)")
print()

print("3. VISUALIZATION IMPOSSIBILITY")
print("   - Humans can see 2D and 3D, not 50D")
print("   - PCA gives us a 2D 'map' of any high-dimensional dataset")
print()

print("4. COMPUTATIONAL COST")
print("   - More features = more parameters = slower training")
print("   - Reducing dimensions speeds up every downstream model")
print()

print("SOLUTION: DIMENSIONALITY REDUCTION")
print("   Find a lower-dimensional representation that preserves")
print("   as much information (variance) as possible.")
print()
print("PCA (Principal Component Analysis) is the most widely-used method.")
print()

# ============================================================================
# SECTION 2: THE PCA ALGORITHM — STEP BY STEP
# ============================================================================

print("=" * 80)
print("SECTION 2: The PCA Algorithm — Step by Step")
print("=" * 80)
print()

print("INPUT: Data matrix X of shape (n_samples, n_features)")
print()

print("STEP 1 — CENTER THE DATA")
print("   Subtract the mean of each feature from all values.")
print("   This shifts the data so the mean is at the origin.")
print("   Why: PCA measures variance around the mean.")
print()
print("   X_centered = X - mean(X, axis=0)")
print()

print("STEP 2 — COMPUTE THE COVARIANCE MATRIX")
print("   The covariance matrix C captures how features vary together.")
print("   C[i, j] = covariance of feature i and feature j")
print("   C[i, i] = variance of feature i")
print()
print("   C = (1 / (n-1)) × X_centered.T @ X_centered")
print("   Shape: (n_features, n_features)")
print()
print("   High C[i,j] means feature i and j are correlated.")
print("   PCA finds new axes that eliminate these correlations.")
print()

print("STEP 3 — EIGENDECOMPOSITION OF THE COVARIANCE MATRIX")
print("   C × v = λ × v")
print("   - v = eigenvector (a direction in feature space)")
print("   - λ = eigenvalue  (how much variance in direction v)")
print()
print("   The eigenvectors are the PRINCIPAL COMPONENTS.")
print("   The eigenvalues tell us the IMPORTANCE of each component.")
print()
print("   Key fact: eigenvectors of a covariance matrix are ORTHOGONAL")
print("   (perpendicular to each other) — no redundancy!")
print()

print("STEP 4 — SORT BY EIGENVALUE (descending)")
print("   The eigenvector with the largest eigenvalue is PC1 (most variance).")
print("   The eigenvector with the 2nd largest is PC2, and so on.")
print()

print("STEP 5 — SELECT TOP k COMPONENTS")
print("   Choose to keep the first k principal components.")
print("   The projection matrix W has shape (n_features, k).")
print()

print("STEP 6 — PROJECT THE DATA")
print("   X_reduced = X_centered @ W")
print("   Shape goes from (n_samples, n_features) → (n_samples, k)")
print()

print("INVERSE TRANSFORM (reconstruction / decompression):")
print("   X_reconstructed ≈ X_reduced @ W.T + mean")
print("   This is approximate — we lost some information by discarding components.")
print()

# ============================================================================
# SECTION 3: PCA FROM SCRATCH — NUMPY IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 3: PCA from Scratch — NumPy Implementation")
print("=" * 80)
print()

class PCA:
    """
    Principal Component Analysis implemented from scratch with NumPy.

    Parameters
    ----------
    n_components : int or None
        Number of principal components to keep.
        If None, keeps all components.

    Attributes (set after fit)
    --------------------------
    components_ : ndarray (n_components, n_features)
        Principal component directions (eigenvectors), rows are PCs.
    explained_variance_ : ndarray (n_components,)
        Variance explained by each component (eigenvalues).
    explained_variance_ratio_ : ndarray (n_components,)
        Fraction of total variance explained by each component.
    mean_ : ndarray (n_features,)
        Per-feature mean (used for centering).
    singular_values_ : ndarray (n_components,)
        Standard deviations along each component (sqrt of eigenvalues).
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None
        self.singular_values_ = None
        self._n_features = None

    def fit(self, X):
        """
        Compute principal components from data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape
        self._n_features = n_features

        # Step 1: Center the data
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute covariance matrix
        # Using (n-1) denominator = unbiased estimate
        cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

        # Step 3: Eigendecomposition
        # numpy.linalg.eigh is specialized for symmetric matrices (faster, numerically stable)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: Sort eigenvectors by eigenvalue in DESCENDING order
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]  # columns are eigenvectors

        # Clip tiny negatives to 0 (numerical noise in nearly-zero eigenvalues)
        eigenvalues = np.maximum(eigenvalues, 0)

        # Step 5: Keep top n_components
        n_keep = self.n_components if self.n_components is not None else n_features
        n_keep = min(n_keep, n_features, n_samples)

        self.explained_variance_ = eigenvalues[:n_keep]
        # Components stored as rows (each row is one PC direction)
        self.components_ = eigenvectors[:, :n_keep].T

        # Explained variance ratio
        total_variance = eigenvalues.sum()
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(n_keep)

        self.singular_values_ = np.sqrt(self.explained_variance_)
        return self

    def transform(self, X):
        """
        Project X onto the principal component axes.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_reduced : ndarray of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before transform().")
        X_centered = X - self.mean_
        # components_ is (n_components, n_features), so transpose to project
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        """Fit then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_reduced):
        """
        Reconstruct approximate original data from reduced representation.

        Parameters
        ----------
        X_reduced : ndarray of shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise RuntimeError("Call fit() before inverse_transform().")
        # Project back to original feature space and add the mean back
        return X_reduced @ self.components_ + self.mean_


print("PCA class defined! Here are the key methods:")
print("   fit(X)                — compute principal components from X")
print("   transform(X)          — project X onto PC axes (reduce dimensions)")
print("   fit_transform(X)      — fit and transform in one step")
print("   inverse_transform(X)  — reconstruct approximate original data")
print()
print("Key attributes after fit:")
print("   .components_              — PC directions (eigenvectors), shape (k, n_features)")
print("   .explained_variance_      — eigenvalues (variance per component)")
print("   .explained_variance_ratio_— fraction of variance each PC explains")
print("   .mean_                    — per-feature mean (subtracted before projecting)")
print()

# ============================================================================
# SECTION 4: 2D EXAMPLE — VISUALIZE THE PROJECTION
# ============================================================================

print("=" * 80)
print("SECTION 4: 2D Example — Seeing PCA in Action")
print("=" * 80)
print()

print("Let us start with a simple 2D dataset where we can SEE what is happening.")
print()

# Create correlated 2D data
np.random.seed(42)
n = 200
angle = np.pi / 5    # 36 degrees
cos_a, sin_a = np.cos(angle), np.sin(angle)

raw = np.random.randn(n, 2) * np.array([3.0, 0.7])
rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
X_2d = raw @ rotation.T
X_2d += np.array([2.0, 1.0])   # shift from origin

print(f"Created 2D dataset: {X_2d.shape}")
print(f"   Feature 1: mean={X_2d[:,0].mean():.2f}, std={X_2d[:,0].std():.2f}")
print(f"   Feature 2: mean={X_2d[:,1].mean():.2f}, std={X_2d[:,1].std():.2f}")
print(f"   Correlation: {np.corrcoef(X_2d[:,0], X_2d[:,1])[0,1]:.3f}")
print()

# Fit PCA keeping 2 components (all)
pca_2d = PCA(n_components=2)
pca_2d.fit(X_2d)

print("PCA Results (2D → 2 components):")
print("-" * 60)
for i, (ev, evr) in enumerate(
        zip(pca_2d.explained_variance_, pca_2d.explained_variance_ratio_)):
    pc = pca_2d.components_[i]
    print(f"  PC{i+1}: direction=({pc[0]:+.3f}, {pc[1]:+.3f}), "
          f"eigenvalue={ev:.3f}, explains {evr*100:.1f}% of variance")
print()
print(f"  PC1 captures {pca_2d.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"  (The main spread direction of the data)")
print()

print("WHAT DOES THIS MEAN?")
print("   PC1 is the axis along which the data varies the MOST.")
print("   PC2 is perpendicular to PC1 — the remaining variation.")
print()
print("If we keep only PC1 (reduce to 1D), we lose only "
      f"{pca_2d.explained_variance_ratio_[1]*100:.1f}% of variance.")
print()

# Project to 1D and reconstruct
pca_1d = PCA(n_components=1)
X_1d = pca_1d.fit_transform(X_2d)
X_reconstructed_2d = pca_1d.inverse_transform(X_1d)

recon_err = np.mean((X_2d - X_reconstructed_2d) ** 2)
print(f"Reconstruction error (1 component): MSE = {recon_err:.4f}")
print("(This is the information we lost by compressing to 1D.)")
print()

# ============================================================================
# SECTION 5: IRIS DATASET — REDUCE 4D TO 2D
# ============================================================================

print("=" * 80)
print("SECTION 5: Iris Dataset — Reducing 4 Features to 2")
print("=" * 80)
print()

try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"Iris dataset: {X_iris.shape[0]} samples × {X_iris.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Classes: {list(target_names)}")
    print()
except ImportError:
    print("sklearn not available. Generating simplified iris-like data.")
    rng = np.random.RandomState(0)
    X_iris = np.vstack([rng.randn(50, 4) + [5, 3.4, 1.5, 0.3],
                        rng.randn(50, 4) + [5.9, 2.8, 4.2, 1.3],
                        rng.randn(50, 4) + [6.6, 3.0, 5.5, 2.0]])
    y_iris = np.repeat([0, 1, 2], 50)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target_names = np.array(['setosa', 'versicolor', 'virginica'])

print("IMPORTANT POINT ABOUT SCALING:")
print("   PCA is sensitive to feature scale (just like K-Means).")
print("   Features with larger ranges dominate the covariance matrix.")
print("   Always StandardScale before PCA unless features are already comparable.")
print()

# Manual standardization
X_mean = X_iris.mean(axis=0)
X_std = X_iris.std(axis=0)
X_iris_scaled = (X_iris - X_mean) / X_std

print("Applied standardization: subtract mean, divide by std.")
print(f"   Before: means = {X_iris.mean(axis=0).round(2)}, "
      f"stds = {X_iris.std(axis=0).round(2)}")
print(f"   After:  means = {X_iris_scaled.mean(axis=0).round(2)}, "
      f"stds = {X_iris_scaled.std(axis=0).round(2)}")
print()

# Fit PCA keeping ALL components first (to build scree plot)
pca_full = PCA(n_components=None)
pca_full.fit(X_iris_scaled)

print("PCA on scaled Iris — all 4 components:")
print("-" * 70)
cumulative = 0.0
print(f"{'PC':<6} {'Eigenvalue':<15} {'Var Explained':<18} {'Cumulative':<18} {'Direction (loadings)'}")
print("-" * 70)
for i in range(4):
    ev = pca_full.explained_variance_[i]
    evr = pca_full.explained_variance_ratio_[i]
    cumulative += evr
    pc = pca_full.components_[i]
    loading_str = "  ".join([f"{v:+.3f}" for v in pc])
    print(f"PC{i+1:<5} {ev:<15.4f} {evr*100:<18.1f}% {cumulative*100:<18.1f}% {loading_str}")
print()

print("KEY INSIGHT:")
print(f"   PC1 + PC2 together explain "
      f"{(pca_full.explained_variance_ratio_[:2].sum()*100):.1f}% of variance.")
print("   We can go from 4D → 2D with very little information loss!")
print("   This is why PCA is so powerful for visualization.")
print()

# Reduce to 2D
pca_2comp = PCA(n_components=2)
X_iris_2d = pca_2comp.fit_transform(X_iris_scaled)

print(f"Reduced Iris from {X_iris_scaled.shape[1]}D → {X_iris_2d.shape[1]}D")
print(f"   Original shape : {X_iris_scaled.shape}")
print(f"   Reduced shape  : {X_iris_2d.shape}")
print()

# Show how PC loadings relate to original features
print("PC LOADINGS (how much each original feature contributes to each PC):")
print("-" * 70)
print(f"{'Feature':<25}", end="")
for i in range(2):
    print(f"{'PC'+str(i+1):<20}", end="")
print()
print("-" * 70)
for j, fname in enumerate(feature_names):
    print(f"{fname:<25}", end="")
    for i in range(2):
        loading = pca_2comp.components_[i, j]
        bar = '#' * int(abs(loading) * 10)
        sign = '+' if loading > 0 else '-'
        print(f"{sign}{abs(loading):.3f} {bar:<12}", end="")
    print()
print()

print("Interpretation of PC1:")
print("   Positive loadings on petal length & petal width = 'flower size' axis")
print("   PC1 roughly separates small flowers (setosa) from large ones")
print()

# ============================================================================
# SECTION 6: DIGITS DATASET — RECONSTRUCTION DEMO
# ============================================================================

print("=" * 80)
print("SECTION 6: Digits Dataset — Compression and Reconstruction")
print("=" * 80)
print()

print("The digits dataset has 8x8 pixel images of handwritten numbers 0-9.")
print("Each image = 64 dimensions. We will compress and reconstruct with PCA.")
print()

try:
    from sklearn.datasets import load_digits
    digits = load_digits()
    X_digits = digits.data.astype(float)
    y_digits = digits.target
    print(f"Digits dataset: {X_digits.shape[0]} images × {X_digits.shape[1]} pixels")
    digits_loaded = True
except ImportError:
    print("sklearn not available — skipping digits demo.")
    digits_loaded = False

if digits_loaded:
    # Fit PCA with varying number of components
    components_to_try = [1, 4, 10, 20, 40, 64]
    print("Compression at different numbers of components:")
    print("-" * 65)
    print(f"{'Components':<15} {'Var Explained':<20} {'Recon MSE':<20} {'Compression'}")
    print("-" * 65)

    pca_records = {}
    for n_comp in components_to_try:
        pca_d = PCA(n_components=n_comp)
        X_reduced = pca_d.fit_transform(X_digits)
        X_recon = pca_d.inverse_transform(X_reduced)
        mse = np.mean((X_digits - X_recon) ** 2)
        var_explained = pca_d.explained_variance_ratio_.sum()
        comp_ratio = n_comp / X_digits.shape[1]
        pca_records[n_comp] = (pca_d, X_reduced, X_recon, mse, var_explained)
        print(f"{n_comp:<15} {var_explained*100:<20.1f}% {mse:<20.2f} "
              f"{comp_ratio*100:.0f}% of original")
    print()

    print("Observation:")
    print("   With only 20 components (31% of 64), we explain ~83% of variance.")
    print("   Images are recognizable with much less data!")
    print()

# ============================================================================
# SECTION 7: SCIKIT-LEARN PCA
# ============================================================================

print("=" * 80)
print("SECTION 7: Scikit-Learn PCA — The Production Version")
print("=" * 80)
print()

print("Our PCA taught us the internals. Scikit-learn's version is more")
print("numerically robust (uses SVD instead of explicit eigendecomposition)")
print("and handles edge cases better.")
print()

try:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.preprocessing import StandardScaler

    print("sklearn PCA key parameters:")
    print("   n_components : int, float, or 'mle'")
    print("                  int   → keep exactly that many components")
    print("                  float → keep enough to explain that fraction of variance")
    print("                          e.g., n_components=0.95 keeps 95% of variance")
    print("                  'mle' → automatically select using MLE (Minka's method)")
    print("   whiten       : bool — normalize component variance to 1")
    print("   random_state : seed for reproducibility")
    print()

    # Standardize iris
    scaler = StandardScaler()
    X_iris_std = scaler.fit_transform(X_iris)

    # Auto-select components for 95% variance
    pca_auto = SklearnPCA(n_components=0.95, random_state=42)
    X_iris_auto = pca_auto.fit_transform(X_iris_std)
    print(f"sklearn PCA with n_components=0.95 on Iris:")
    print(f"   Automatically selected {pca_auto.n_components_} components "
          f"to explain 95%+ of variance")
    print(f"   Explained variance ratios: "
          f"{[f'{r*100:.1f}%' for r in pca_auto.explained_variance_ratio_]}")
    print()

    # Exact 2 components
    pca_sk = SklearnPCA(n_components=2, random_state=42)
    X_iris_sk = pca_sk.fit_transform(X_iris_std)
    print(f"sklearn PCA (n_components=2):")
    print(f"   Total variance explained: "
          f"{pca_sk.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"   PC1: {pca_sk.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2: {pca_sk.explained_variance_ratio_[1]*100:.1f}%")
    print()

    # Compare our scratch vs sklearn
    print("Comparing scratch PCA vs sklearn PCA on iris:")
    our_var = pca_2comp.explained_variance_ratio_.sum()
    sk_var = pca_sk.explained_variance_ratio_.sum()
    print(f"   Our total variance explained : {our_var*100:.2f}%")
    print(f"   sklearn variance explained   : {sk_var*100:.2f}%")
    diff = abs(our_var - sk_var)
    print(f"   Difference                  : {diff*100:.4f}%  (essentially identical)")
    print()

    sklearn_pca_ok = True

except ImportError:
    print("sklearn not installed. Run: pip install scikit-learn")
    sklearn_pca_ok = False

# ============================================================================
# SECTION 8: WHEN DOES PCA HELP AND WHEN DOES IT NOT?
# ============================================================================

print("=" * 80)
print("SECTION 8: When PCA Helps — and When It Does Not")
print("=" * 80)
print()

print("PCA IS MOST USEFUL WHEN:")
print()
print("1. HIGH MULTICOLLINEARITY")
print("   - Many correlated features (e.g., stock prices in the same sector)")
print("   - PCA decorrelates them and finds independent factors")
print()
print("2. VISUALIZATION")
print("   - Any high-D dataset can be projected to 2D/3D with PCA")
print("   - Gives you a 'map' to spot clusters, outliers, and structure")
print()
print("3. PREPROCESSING FOR ML MODELS")
print("   - Reduce noise before KNN, SVM, or logistic regression")
print("   - Smaller input = faster training")
print()
print("4. IMAGE/AUDIO COMPRESSION")
print("   - Keep the most informative directions, discard noise")
print("   - 'Eigenfaces' in face recognition (famous application)")
print()
print("5. REMOVING NOISE")
print("   - Low-variance components often capture sensor noise")
print("   - Dropping them and reconstructing = denoising")
print()

print("PCA DOES NOT HELP (OR CAN HURT) WHEN:")
print()
print("1. NON-LINEAR STRUCTURE")
print("   - PCA only finds LINEAR directions")
print("   - Circular clusters, spirals, or manifolds require non-linear methods")
print("   - Alternatives: t-SNE, UMAP, Kernel PCA")
print()
print("2. FEATURES ARE ALREADY INDEPENDENT")
print("   - If features have zero correlation, PCA finds nothing useful")
print("   - Each feature is already its own principal component")
print()
print("3. WHEN INTERPRETABILITY MATTERS")
print("   - PCA components are combinations of ALL original features")
print("   - PC1 might be '0.6×sepal_len + 0.5×petal_len - 0.3×sepal_wid ...'")
print("   - Hard to explain to non-technical stakeholders")
print()
print("4. SPARSE DATA (text/NLP)")
print("   - PCA is dense; sparse data explodes in memory when densified")
print("   - Use TruncatedSVD (LSA — Latent Semantic Analysis) instead")
print()
print("5. CATEGORICAL FEATURES")
print("   - PCA assumes continuous features")
print("   - Use MCA (Multiple Correspondence Analysis) for categoricals")
print()

# ============================================================================
# SECTION 9: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 9: Generating Visualizations")
print("=" * 80)
print()

COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#9B59B6', '#F39C12']

# ---- Visualization 1: 2D PCA — data with PC arrows ----
print("Generating Visualization 1: 2D PCA projection with PC arrows...")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('PCA on 2D Data — Finding Principal Directions', fontsize=14,
             fontweight='bold')

# Panel 1: Raw (centered) data with PC arrows
ax = axes[0]
X_2d_c = X_2d - pca_2d.mean_
ax.scatter(X_2d_c[:, 0], X_2d_c[:, 1], s=25, alpha=0.5, color='#3498DB', edgecolors='none')
ax.scatter(0, 0, s=200, color='black', marker='+', linewidths=3, zorder=5, label='Mean (origin)')

scale1 = np.sqrt(pca_2d.explained_variance_[0]) * 2.5
scale2 = np.sqrt(pca_2d.explained_variance_[1]) * 2.5

pc1 = pca_2d.components_[0]
pc2 = pca_2d.components_[1]

ax.annotate('', xy=(pc1[0]*scale1, pc1[1]*scale1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3))
ax.annotate('', xy=(pc2[0]*scale2, pc2[1]*scale2), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#2ECC71', lw=3))

ax.text(pc1[0]*scale1*1.1, pc1[1]*scale1*1.1, 'PC1\n(max var)', color='#E74C3C',
        fontsize=10, fontweight='bold', ha='center')
ax.text(pc2[0]*scale2*1.15, pc2[1]*scale2*1.15, 'PC2\n(perpendicular)', color='#2ECC71',
        fontsize=10, fontweight='bold', ha='center')

ax.set_title('Step 1: Find Principal Directions\n(PC arrows scaled by std)', fontsize=11)
ax.set_xlabel('Feature 1 (centered)')
ax.set_ylabel('Feature 2 (centered)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 2: Data projected onto PC1 (1D reduction)
ax = axes[1]
proj_pc1 = (X_2d_c @ pc1[:, np.newaxis]) * pc1[np.newaxis, :]  # projection onto PC1
# Original points
ax.scatter(X_2d_c[:, 0], X_2d_c[:, 1], s=20, alpha=0.3, color='#95A5A6', edgecolors='none',
           label='Original (2D)')
# Projected (reconstructed) points
ax.scatter(proj_pc1[:, 0], proj_pc1[:, 1], s=20, alpha=0.7, color='#E74C3C',
           edgecolors='none', label='Projected onto PC1')
# Lines from original to projection
for i in range(0, len(X_2d_c), 5):  # every 5th point for clarity
    ax.plot([X_2d_c[i, 0], proj_pc1[i, 0]],
            [X_2d_c[i, 1], proj_pc1[i, 1]], 'k-', alpha=0.15, linewidth=0.8)

ax.annotate('', xy=(pc1[0]*scale1, pc1[1]*scale1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=3))

ax.set_title(f'Step 2: Project onto PC1\n({pca_2d.explained_variance_ratio_[0]*100:.0f}% var kept)',
             fontsize=11)
ax.set_xlabel('Feature 1 (centered)')
ax.set_ylabel('Feature 2 (centered)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Panel 3: After projection — new PC coordinate system
ax = axes[2]
X_2d_proj = pca_2d.transform(X_2d)
ax.scatter(X_2d_proj[:, 0], X_2d_proj[:, 1], s=25, alpha=0.6, color='#9B59B6',
           edgecolors='none')
ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
ax.set_title('Step 3: Data in PC coordinate system\n(rotated, decorrelated)', fontsize=11)
ax.set_xlabel('PC1 score (explains most variance)')
ax.set_ylabel('PC2 score')
ax.grid(True, alpha=0.3)

corr_after = np.corrcoef(X_2d_proj[:, 0], X_2d_proj[:, 1])[0, 1]
ax.text(0.02, 0.98, f'Original corr: {np.corrcoef(X_2d[:,0],X_2d[:,1])[0,1]:.3f}\n'
                    f'After PCA corr: {corr_after:.3f}',
        transform=ax.transAxes, va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '01_pca_2d_projection.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

# ---- Visualization 2: Scree plot ----
print("Generating Visualization 2: Scree Plot (explained variance)...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Scree Plot — How Many Components to Keep?', fontsize=14, fontweight='bold')

# Panel 1: Individual explained variance per component (Iris)
ax = axes[0]
n_comps_iris = len(pca_full.explained_variance_ratio_)
comp_labels = [f'PC{i+1}' for i in range(n_comps_iris)]
bars = ax.bar(comp_labels, pca_full.explained_variance_ratio_ * 100,
              color=['#3498DB', '#2ECC71', '#E74C3C', '#F39C12'][:n_comps_iris],
              edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, pca_full.explained_variance_ratio_):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_title('Iris — Individual Variance\n(4 features → top 2 explain 95.8%)', fontsize=11)
ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: Cumulative explained variance (Iris)
ax = axes[1]
cumulative_var = np.cumsum(pca_full.explained_variance_ratio_) * 100

ax.plot(comp_labels, cumulative_var, 'o-', color='#8E44AD', linewidth=2.5,
        markersize=10, markerfacecolor='white', markeredgewidth=2.5)

for i, (lbl, val) in enumerate(zip(comp_labels, cumulative_var)):
    ax.annotate(f'{val:.1f}%', (lbl, val), textcoords='offset points',
                xytext=(10, 5), fontsize=10, color='#8E44AD', fontweight='bold')

ax.axhline(y=95, color='#E74C3C', linestyle='--', linewidth=2,
           label='95% threshold')
ax.axhline(y=99, color='#E67E22', linestyle=':', linewidth=2,
           label='99% threshold')
ax.fill_between(comp_labels, 0, cumulative_var, alpha=0.15, color='#8E44AD')

ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
ax.set_title('Iris — Cumulative Variance\n(Choose k where curve levels off)', fontsize=11)
ax.set_ylim(0, 110)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '02_scree_plot.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

# ---- Visualization 3: Iris 2D projection colored by true class ----
print("Generating Visualization 3: Iris PCA 2D projection colored by species...")

CLASS_COLORS = ['#E74C3C', '#3498DB', '#2ECC71']
MARKERS = ['o', 's', '^']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA on Iris: 4D → 2D Visualization', fontsize=14, fontweight='bold')

# Panel 1: Our scratch PCA
ax = axes[0]
X_iris_sc = pca_2comp.fit_transform(X_iris_scaled)

for cls in range(3):
    mask = y_iris == cls
    ax.scatter(X_iris_sc[mask, 0], X_iris_sc[mask, 1],
               c=CLASS_COLORS[cls], marker=MARKERS[cls], s=70,
               edgecolors='black', linewidths=0.7, alpha=0.85,
               label=target_names[cls])

ax.set_xlabel(f"PC1 ({pca_2comp.explained_variance_ratio_[0]*100:.1f}% variance)",
              fontsize=12, fontweight='bold')
ax.set_ylabel(f"PC2 ({pca_2comp.explained_variance_ratio_[1]*100:.1f}% variance)",
              fontsize=12, fontweight='bold')
ax.set_title('Scratch PCA — 4D Iris compressed to 2D\n'
             '(Class labels used ONLY for coloring, not for PCA)',
             fontsize=11)
ax.legend(title='Species', fontsize=10, title_fontsize=10)
ax.grid(True, alpha=0.3)

total_var = pca_2comp.explained_variance_ratio_.sum()
ax.text(0.02, 0.02, f'Total variance captured: {total_var*100:.1f}%',
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

# Panel 2: Reconstruction quality on digits (if available)
ax = axes[1]
if digits_loaded:
    # Plot reconstruction MSE vs n_components
    comp_list = sorted(pca_records.keys())
    mse_list = [pca_records[c][3] for c in comp_list]
    var_list = [pca_records[c][4] * 100 for c in comp_list]

    color1 = '#E74C3C'
    color2 = '#3498DB'

    line1, = ax.plot(comp_list, mse_list, 'o-', color=color1, linewidth=2.5,
                     markersize=8, label='Reconstruction MSE')
    ax.set_ylabel('Reconstruction MSE', fontsize=12, fontweight='bold', color=color1)
    ax.tick_params(axis='y', labelcolor=color1)

    ax2 = ax.twinx()
    line2, = ax2.plot(comp_list, var_list, 's--', color=color2, linewidth=2.5,
                      markersize=8, label='Variance Explained (%)')
    ax2.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax.set_xlabel('Number of PCA Components', fontsize=12, fontweight='bold')
    ax.set_title('Digits Dataset: Compression Quality\n(64 pixel features → k components)',
                 fontsize=11)
    ax.set_xticks(comp_list)
    ax.grid(True, alpha=0.3)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=10, loc='center right')
else:
    # Fallback: show a simple covariance matrix of iris features
    cov = np.corrcoef(X_iris.T)
    im = ax.imshow(cov, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    short_names = ['SepalL', 'SepalW', 'PetalL', 'PetalW']
    ax.set_xticklabels(short_names, rotation=30, ha='right')
    ax.set_yticklabels(short_names)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cov[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Iris Feature Correlation Matrix\n(PCA removes these correlations)',
                 fontsize=11)

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '03_iris_projection.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

print()

# ============================================================================
# SECTION 10: SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY: What You Learned")
print("=" * 80)
print()
print("THE PROBLEM:")
print("   High-dimensional data is hard to visualize, model, and store.")
print("   Many features are correlated, adding redundancy but not information.")
print()
print("PCA ALGORITHM:")
print("   1. Center the data (subtract per-feature mean)")
print("   2. Compute covariance matrix (n_features × n_features)")
print("   3. Eigendecomposition: C × v = λ × v")
print("   4. Sort eigenvectors by eigenvalue (largest = most variance)")
print("   5. Project: X_reduced = X_centered × W  where W = top k eigenvectors")
print()
print("KEY CONCEPTS:")
print("   - Principal Components: orthogonal directions of maximum variance")
print("   - Explained Variance Ratio: fraction of information each PC captures")
print("   - Scree Plot: plot of variance per component — look for 'elbow'")
print("   - Reconstruction: X ≈ X_reduced × W.T + mean (approximate)")
print()
print("IMPLEMENTATION:")
print("   from scratch : NumPy eigendecomposition — great for understanding")
print("   production   : sklearn PCA — uses SVD, more numerically stable")
print()
print("WHEN TO USE PCA:")
print("   - Many correlated features")
print("   - Visualizing high-D data in 2D/3D")
print("   - Preprocessing to speed up downstream models")
print("   - Denoising (drop low-variance components)")
print()
print("WHEN NOT TO USE PCA:")
print("   - Non-linear structure → use t-SNE or UMAP")
print("   - When feature names must be preserved for interpretation")
print("   - Sparse data (text) → use TruncatedSVD instead")
print()
print("NEXT: Hierarchical clustering and DBSCAN — more flexible clustering!")
print()
print("=" * 80)
print("Module Complete! Visualizations saved to:")
print(f"   {os.path.abspath(VISUAL_DIR)}")
print("=" * 80)
