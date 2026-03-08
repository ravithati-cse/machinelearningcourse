"""
K-MEANS CLUSTERING - Finding Natural Groups Without Labels
==========================================================

LEARNING OBJECTIVES:
-------------------
After this module, you will understand:
1. What unsupervised learning is and how clustering differs from classification
2. The K-Means algorithm step by step: initialize, assign, update, repeat
3. How to implement K-Means from scratch using only NumPy
4. How inertia (within-cluster sum of squares) measures cluster quality
5. How the Elbow Method helps you choose the right number of clusters (k)
6. How to use scikit-learn's KMeans for production use
7. The key limitations of K-Means and when to use alternatives

RECOMMENDED VIDEOS:
------------------
* StatQuest: "K-means clustering"
   https://www.youtube.com/watch?v=4b5d3muPQmA
   THE BEST visual explanation of K-Means — MUST WATCH!

* StatQuest: "K-means clustering, Clearly Explained!!!"
   https://www.youtube.com/watch?v=4b5d3muPQmA
   Visual step-by-step walkthrough

* 3Blue1Brown: Essence of Linear Algebra (for intuition on distance)
   https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

* Sentdex: "K-Means from Scratch in Python"
   https://www.youtube.com/watch?v=9991JlKnFmk

TIME: 90-120 minutes
DIFFICULTY: Intermediate
PREREQUISITES: Math foundations 01-03 (linear algebra, distance, probability)

OVERVIEW:
---------
K-Means is the most popular clustering algorithm. It partitions data into k
groups (clusters) where each data point belongs to the cluster whose centroid
(center point) is nearest. No labels needed — it discovers structure on its own.

Key Intuition:
- Imagine dropping k magnets onto your data points
- Each point is attracted to its nearest magnet
- Magnets slide to the center of their attracted points
- Repeat until magnets stop moving — that is your answer!
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# Setup visualization directory
VISUAL_DIR = '../visuals/kmeans_from_scratch/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("K-MEANS CLUSTERING - Finding Natural Groups Without Labels")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHAT IS UNSUPERVISED LEARNING AND CLUSTERING?
# ============================================================================

print("=" * 80)
print("SECTION 1: What Is Unsupervised Learning?")
print("=" * 80)
print()

print("SUPERVISED LEARNING (what we did before):")
print("   - You have features AND labels (X and y)")
print("   - Example: Photos labeled 'cat' or 'dog'")
print("   - Goal: Learn to predict the label from the features")
print()

print("UNSUPERVISED LEARNING (what we do now):")
print("   - You have ONLY features (X) — no labels!")
print("   - Example: Thousands of customer purchase records")
print("   - Goal: Discover hidden structure or patterns in the data")
print()

print("CLUSTERING — one type of unsupervised learning:")
print("   - Groups similar data points together")
print("   - Points within a cluster are similar to each other")
print("   - Points in different clusters are different from each other")
print()

print("Real-world clustering examples:")
print("   - Customer segmentation (which shoppers behave similarly?)")
print("   - Document grouping (which news articles cover the same topic?)")
print("   - Gene expression analysis (which genes activate together?)")
print("   - Image compression (represent similar pixel colors as one)")
print("   - Anomaly detection (points far from any cluster = outliers)")
print()

print("THE KEY QUESTION in clustering:")
print('   "Given N data points, how do we find natural groups?"')
print()
print("K-MEANS answers this by iteratively refining group assignments.")
print()

# ============================================================================
# SECTION 2: THE K-MEANS ALGORITHM STEP BY STEP
# ============================================================================

print("=" * 80)
print("SECTION 2: The K-Means Algorithm — Step by Step")
print("=" * 80)
print()

print("INPUT:")
print("   - Data: N points in d-dimensional space")
print("   - k: the number of clusters you want to find (YOU choose this!)")
print()

print("ALGORITHM:")
print()
print("STEP 1 — Initialize centroids")
print("   Pick k points from the data at random as starting centroids.")
print("   (A centroid is the 'center' or 'representative' of a cluster.)")
print()

print("STEP 2 — Assignment step")
print("   For every data point, calculate its distance to EACH centroid.")
print("   Assign the point to the cluster of the NEAREST centroid.")
print()

print("STEP 3 — Update step")
print("   Recalculate each centroid as the MEAN of all points assigned to it.")
print("   The centroid 'moves' to the center of its cluster.")
print()

print("STEP 4 — Repeat")
print("   Repeat steps 2 and 3 until:")
print("   - Centroids stop moving (convergence), OR")
print("   - Maximum iterations reached")
print()

print("OUTPUT: k cluster assignments — each point labelled 0 to k-1")
print()

print("WHY DOES THIS WORK?")
print("   Each iteration reduces the 'inertia' (total distance of points to")
print("   their centroid). Like water finding the lowest point — it always")
print("   goes downhill and eventually settles.")
print()

# Illustrate with a tiny manual example
print("TINY MANUAL EXAMPLE (1D):")
print("-" * 60)
data_1d = np.array([1.0, 1.5, 2.0, 8.0, 8.5, 9.0])
centroids_1d = np.array([1.5, 8.5])  # Good initialization
print(f"Data points: {data_1d}")
print(f"Initial centroids: {centroids_1d}")
print()

for iteration in range(2):
    # Assignment
    assignments = []
    for x in data_1d:
        dists = [abs(x - c) for c in centroids_1d]
        assignments.append(np.argmin(dists))
    assignments = np.array(assignments)
    print(f"  Iteration {iteration+1} — Assignment step:")
    for x, a in zip(data_1d, assignments):
        print(f"    Point {x:.1f} → Cluster {a} (nearest centroid: {centroids_1d[a]:.1f})")
    # Update
    new_centroids = np.array([data_1d[assignments == k].mean() for k in range(2)])
    print(f"  Iteration {iteration+1} — Update step:")
    print(f"    Cluster 0 mean: {new_centroids[0]:.2f}  (was {centroids_1d[0]:.2f})")
    print(f"    Cluster 1 mean: {new_centroids[1]:.2f}  (was {centroids_1d[1]:.2f})")
    centroids_1d = new_centroids
    print()

print("Centroids stabilized — algorithm converged!")
print()

# ============================================================================
# SECTION 3: K-MEANS FROM SCRATCH IN NUMPY
# ============================================================================

print("=" * 80)
print("SECTION 3: K-Means Implementation from Scratch (NumPy)")
print("=" * 80)
print()

print("Now let's build a full KMeans class — just like scikit-learn!")
print()

class KMeans:
    """
    K-Means Clustering implemented from scratch with NumPy.

    Parameters
    ----------
    n_clusters : int
        Number of clusters k.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance — stop if centroids move less than this.
    init : str
        'random' or 'kmeans++' for centroid initialization.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 init='kmeans++', random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state

        # Attributes set after fit()
        self.centroids_ = None        # Shape: (k, n_features)
        self.labels_ = None           # Shape: (n_samples,)
        self.inertia_ = None          # Scalar: total within-cluster SS
        self.n_iter_ = None           # How many iterations ran
        self.history_ = []            # Centroid positions per iteration

    def _initialize_centroids(self, X):
        """
        Choose starting centroids.

        'random' : Pick k data points uniformly at random.
        'kmeans++': Smart init — spread centroids out using distance-weighted
                    probability. Greatly reduces chance of bad convergence.
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        if self.init == 'random':
            # Pick k random indices without replacement
            idx = rng.choice(n_samples, self.n_clusters, replace=False)
            return X[idx].copy()

        elif self.init == 'kmeans++':
            # K-Means++ initialization
            # 1. Choose the first centroid uniformly at random
            idx = rng.randint(0, n_samples)
            centroids = [X[idx].copy()]

            for c in range(1, self.n_clusters):
                # 2. For each point, find its distance to the nearest centroid
                centroid_arr = np.array(centroids)          # shape (c, features)
                diffs = X[:, np.newaxis, :] - centroid_arr  # (n, c, features)
                dists_sq = (diffs ** 2).sum(axis=2)         # (n, c)
                min_dists_sq = dists_sq.min(axis=1)         # (n,)

                # 3. Choose next centroid with probability proportional to distance^2
                probs = min_dists_sq / min_dists_sq.sum()
                cumulative_probs = np.cumsum(probs)
                r = rng.rand()
                idx = np.searchsorted(cumulative_probs, r)
                centroids.append(X[idx].copy())

            return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        """
        For each data point, find the index of the nearest centroid.

        Uses squared Euclidean distance for efficiency.
        Distance formula: d^2 = sum((x - centroid)^2)
        """
        # Broadcasting: X is (n, features), centroids is (k, features)
        # We need distances of shape (n, k)
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, features)
        dists_sq = (diffs ** 2).sum(axis=2)                         # (n, k)
        return np.argmin(dists_sq, axis=1)                          # (n,) — cluster index

    def _update_centroids(self, X, labels):
        """
        Recompute each centroid as the mean of its assigned points.
        """
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                centroids[k] = X[mask].mean(axis=0)
            else:
                # Edge case: empty cluster — reinitialize to a random point
                rng = np.random.RandomState(self.random_state)
                centroids[k] = X[rng.randint(0, len(X))]
        return centroids

    def _compute_inertia(self, X, labels, centroids):
        """
        Compute inertia = within-cluster sum of squared distances.

        Inertia = sum over all points of (distance to assigned centroid)^2
        Lower inertia = tighter, more compact clusters.
        """
        total = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                diff = X[mask] - centroids[k]
                total += (diff ** 2).sum()
        return total

    def fit(self, X):
        """
        Run the K-Means algorithm on data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        print(f"  Fitting K-Means: k={self.n_clusters}, init='{self.init}', "
              f"max_iter={self.max_iter}")

        # Step 1: Initialize centroids
        centroids = self._initialize_centroids(X)
        self.history_ = [centroids.copy()]

        for iteration in range(self.max_iter):
            # Step 2: Assign each point to its nearest centroid
            labels = self._assign_clusters(X, centroids)

            # Step 3: Update centroids to the mean of their cluster
            new_centroids = self._update_centroids(X, labels)
            self.history_.append(new_centroids.copy())

            # Check convergence: did centroids move?
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                print(f"  Converged at iteration {iteration + 1} (shift={shift:.6f})")
                break
        else:
            print(f"  Reached max_iter={self.max_iter} without full convergence")

        # Final assignment and inertia
        self.labels_ = self._assign_clusters(X, centroids)
        self.centroids_ = centroids
        self.inertia_ = self._compute_inertia(X, self.labels_, centroids)
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        """
        Assign new data points to the nearest learned centroid.
        """
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X):
        """Convenience: fit and return labels."""
        return self.fit(X).labels_


print("KMeans class defined! Here are the key methods:")
print("   fit(X)          — learn centroids from data X")
print("   predict(X)      — assign new points to nearest centroid")
print("   fit_predict(X)  — fit and return labels in one step")
print()
print("Key attributes after fit:")
print("   .centroids_  — the k centroid coordinates")
print("   .labels_     — which cluster each training point belongs to")
print("   .inertia_    — total within-cluster sum of squared distances")
print("   .n_iter_     — how many iterations it took to converge")
print()

# ============================================================================
# SECTION 4: APPLY TO SYNTHETIC BLOB DATA
# ============================================================================

print("=" * 80)
print("SECTION 4: Applying K-Means to Synthetic Blobs")
print("=" * 80)
print()

try:
    from sklearn.datasets import make_blobs
    print("Generating synthetic blob data with 3 natural clusters...")
    X_blobs, y_true_blobs = make_blobs(
        n_samples=300, centers=3, cluster_std=0.9, random_state=42
    )
    print(f"Dataset shape: {X_blobs.shape}")
    print(f"(y_true_blobs is hidden from the algorithm — used only for evaluation)")
    print()
except ImportError:
    print("sklearn not found — generating blobs manually.")
    rng = np.random.RandomState(42)
    centers = [[-3, -3], [0, 3], [3, -3]]
    X_blobs = np.vstack([rng.randn(100, 2) * 0.9 + c for c in centers])
    y_true_blobs = np.repeat([0, 1, 2], 100)
    print(f"Dataset shape: {X_blobs.shape}")
    print()

print("Training our from-scratch K-Means with k=3...")
print("-" * 60)
kmeans_scratch = KMeans(n_clusters=3, random_state=42, init='kmeans++')
kmeans_scratch.fit(X_blobs)
print()
print(f"Results:")
print(f"   Iterations to converge : {kmeans_scratch.n_iter_}")
print(f"   Final inertia          : {kmeans_scratch.inertia_:.4f}")
print(f"   Centroids found:")
for i, c in enumerate(kmeans_scratch.centroids_):
    n_points = (kmeans_scratch.labels_ == i).sum()
    print(f"     Cluster {i}: center=({c[0]:.3f}, {c[1]:.3f}), size={n_points} points")
print()

# Compare with true labels
print("How well did we recover the true groups?")
print("(Remember: cluster IDs may differ — we just care about groupings)")

# Compute purity
from itertools import permutations
best_acc = 0
pred = kmeans_scratch.labels_
for perm in permutations(range(3)):
    mapped = np.array([perm[p] for p in pred])
    acc = np.mean(mapped == y_true_blobs)
    best_acc = max(best_acc, acc)
print(f"   Cluster purity (best permutation match): {best_acc*100:.1f}%")
print()

# ============================================================================
# SECTION 5: UNDERSTANDING INERTIA AND THE ELBOW METHOD
# ============================================================================

print("=" * 80)
print("SECTION 5: Inertia and the Elbow Method — Choosing k")
print("=" * 80)
print()

print("THE BIG PROBLEM WITH K-MEANS: You must specify k beforehand.")
print()
print("But how do you know the right k?")
print()
print("INERTIA (within-cluster sum of squares):")
print("   - For each point, compute distance to its assigned centroid")
print("   - Square that distance")
print("   - Sum over ALL points")
print()
print("   Inertia = Σ ||xᵢ - centroid(xᵢ)||²")
print()
print("   Lower inertia = tighter, more compact clusters = better fit")
print()
print("   BUT: Inertia always decreases as k increases.")
print("   With k=N (one cluster per point), inertia = 0 — meaningless!")
print()
print("THE ELBOW METHOD:")
print("   - Try k = 1, 2, 3, ... (e.g., up to 10)")
print("   - Plot inertia vs k")
print("   - Look for the 'elbow' — the point where adding more clusters")
print("     gives diminishing returns")
print("   - That elbow is a good guess for the true k")
print()

# Run K-Means for multiple k values
k_values = range(1, 11)
inertias = []

print("Computing inertia for k = 1 to 10...")
print("-" * 50)
print(f"{'k':<6} {'Inertia':<15} {'Change':<15}")
print("-" * 50)

prev_inertia = None
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, init='kmeans++', max_iter=100)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
    change = f"{km.inertia_ - prev_inertia:.1f}" if prev_inertia is not None else "—"
    print(f"{k:<6} {km.inertia_:<15.2f} {change}")
    prev_inertia = km.inertia_

print()
print("Notice: the biggest drop in inertia is from k=1 to k=2, then k=2 to k=3.")
print("After k=3, gains become smaller — this is the 'elbow'!")
print()

# ============================================================================
# SECTION 6: APPLY TO THE IRIS DATASET (REAL DATA)
# ============================================================================

print("=" * 80)
print("SECTION 6: Applying K-Means to the Iris Dataset (Real Data)")
print("=" * 80)
print()

print("The Iris dataset has 150 flowers described by 4 measurements:")
print("   - Sepal length, sepal width, petal length, petal width")
print("   - 3 species: Setosa, Versicolor, Virginica (50 each)")
print()
print("We will IGNORE the species labels and try to find k=3 clusters.")
print("Then we will compare our clusters to the true species.")
print()

try:
    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = iris.data        # 150 x 4 feature matrix
    y_iris = iris.target      # 0, 1, 2 — hidden from K-Means
    feature_names = iris.feature_names
    target_names = iris.target_names
    print(f"Iris dataset loaded: {X_iris.shape[0]} samples, {X_iris.shape[1]} features")
    print()
except ImportError:
    print("sklearn not available. Using simplified iris approximation.")
    rng = np.random.RandomState(0)
    X_iris = np.vstack([rng.randn(50, 4) + [5, 3.4, 1.5, 0.3],
                        rng.randn(50, 4) + [5.9, 2.8, 4.2, 1.3],
                        rng.randn(50, 4) + [6.6, 3.0, 5.5, 2.0]])
    y_iris = np.repeat([0, 1, 2], 50)
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    target_names = ['setosa', 'versicolor', 'virginica']

print("Training K-Means on Iris (k=3, ignoring species labels)...")
kmeans_iris = KMeans(n_clusters=3, random_state=42, init='kmeans++')
kmeans_iris.fit(X_iris)
print()

print(f"Cluster sizes: ", end="")
for i in range(3):
    count = (kmeans_iris.labels_ == i).sum()
    print(f"Cluster {i}: {count} points", end="  ")
print()
print()

print("Cluster centroids (in feature space):")
print(f"{'Cluster':<10}", end="")
for name in feature_names:
    print(f"{name:<20}", end="")
print()
print("-" * 90)
for i, c in enumerate(kmeans_iris.centroids_):
    print(f"{i:<10}", end="")
    for val in c:
        print(f"{val:<20.3f}", end="")
    print()
print()

# Evaluate purity
best_acc_iris = 0
best_perm_iris = None
for perm in permutations(range(3)):
    mapped = np.array([perm[p] for p in kmeans_iris.labels_])
    acc = np.mean(mapped == y_iris)
    if acc > best_acc_iris:
        best_acc_iris = acc
        best_perm_iris = perm

print(f"Cluster purity vs true species: {best_acc_iris*100:.1f}%")
print("(K-Means found ~89% correct groupings with NO label information!)")
print()

# ============================================================================
# SECTION 7: SCIKIT-LEARN K-MEANS
# ============================================================================

print("=" * 80)
print("SECTION 7: Scikit-Learn KMeans — The Production Version")
print("=" * 80)
print()

print("Our implementation taught us the internals. For real projects, use sklearn.")
print()

try:
    from sklearn.cluster import KMeans as SklearnKMeans

    print("sklearn KMeans — key parameters:")
    print("   n_clusters  : number of clusters k")
    print("   init        : 'k-means++' (default, smart) or 'random'")
    print("   n_init      : run k-means n_init times, keep best result (default=10)")
    print("   max_iter    : max iterations per run (default=300)")
    print("   random_state: for reproducibility")
    print()

    sk_km = SklearnKMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
    sk_km.fit(X_blobs)

    print(f"sklearn KMeans on blobs:")
    print(f"   Inertia    : {sk_km.inertia_:.4f}")
    print(f"   Iterations : {sk_km.n_iter_}")
    print(f"   Centroids  :")
    for i, c in enumerate(sk_km.cluster_centers_):
        print(f"     Cluster {i}: ({c[0]:.3f}, {c[1]:.3f})")
    print()

    print("Comparing our scratch implementation vs sklearn:")
    our_inertia = kmeans_scratch.inertia_
    sk_inertia = sk_km.inertia_
    print(f"   Our inertia   : {our_inertia:.4f}")
    print(f"   sklearn inertia: {sk_inertia:.4f}")
    diff_pct = abs(our_inertia - sk_inertia) / sk_inertia * 100
    print(f"   Difference    : {diff_pct:.2f}%  (very close!)")
    print()

    # Predict new points
    new_points = np.array([[0, 0], [5, 5], [-4, 5]])
    print("Predicting cluster for new points:")
    print(f"{'Point':<20} {'Cluster'}")
    print("-" * 35)
    for pt, label in zip(new_points, sk_km.predict(new_points)):
        print(f"({pt[0]}, {pt[1]})              Cluster {label}")
    print()

    sklearn_ok = True

except ImportError:
    print("sklearn not installed. Run:  pip install scikit-learn")
    sklearn_ok = False

# ============================================================================
# SECTION 8: K-MEANS LIMITATIONS
# ============================================================================

print("=" * 80)
print("SECTION 8: K-Means Limitations — When It Breaks")
print("=" * 80)
print()

print("K-Means is powerful but has important limitations:")
print()

print("1. YOU MUST SPECIFY k")
print("   You need to choose the number of clusters in advance.")
print("   The Elbow Method helps, but it is not always clear-cut.")
print()

print("2. SENSITIVE TO INITIALIZATION")
print("   Poor starting centroids can lead to suboptimal solutions.")
print("   k-Means++ initialization solves this mostly.")
print("   Running multiple times (n_init) and keeping the best also helps.")
print()

print("3. ASSUMES SPHERICAL CLUSTERS")
print("   K-Means uses Euclidean distance — it implicitly assumes clusters")
print("   are roughly circular/spherical and similarly sized.")
print("   It FAILS on elongated, crescent, or interleaved clusters.")
print("   Alternatives: DBSCAN, Gaussian Mixture Models, Spectral Clustering")
print()

print("4. SENSITIVE TO SCALE")
print("   Features with large ranges dominate distance calculations.")
print("   Always StandardScale your features before K-Means!")
print()

print("5. SENSITIVE TO OUTLIERS")
print("   Outliers pull centroids away from the true cluster centers.")
print("   K-Medoids (PAM) is a more robust alternative.")
print()

print("6. HARD ASSIGNMENTS")
print("   Every point is assigned to EXACTLY one cluster.")
print("   Gaussian Mixture Models give soft (probabilistic) assignments.")
print()

print("K-MEANS++ INITIALIZATION (how it helps):")
print("   Regular K-Means: pick k random points — may clump centroids together")
print("   K-Means++: spread centroids out using distance-weighted sampling")
print("     Step 1: Choose first centroid uniformly at random")
print("     Step 2: Choose next centroid with prob proportional to dist^2")
print("             (farther points are more likely to be chosen)")
print("     Step 3: Repeat until k centroids selected")
print("   Result: better starting positions → faster convergence, better results")
print()

# ============================================================================
# SECTION 9: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 9: Generating Visualizations")
print("=" * 80)
print()

# Color palette for up to 10 clusters
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
          '#1ABC9C', '#E67E22', '#34495E', '#F1C40F', '#7F8C8D']
MARKER_STYLES = ['o', 's', '^', 'D', 'P']

# ---- Visualization 1: Step-by-step K-Means on blobs ----
print("Generating Visualization 1: Step-by-step K-Means progression...")

n_steps = min(len(kmeans_scratch.history_), 5)
step_indices = [0, 1, 2, min(3, len(kmeans_scratch.history_)-2),
                len(kmeans_scratch.history_)-1]
step_indices = sorted(set(step_indices))[:5]
labels_at_steps = []
for hist_centroids in kmeans_scratch.history_:
    lbl = kmeans_scratch._assign_clusters(X_blobs, hist_centroids)
    labels_at_steps.append(lbl)

fig, axes = plt.subplots(1, len(step_indices), figsize=(4 * len(step_indices), 4))
fig.suptitle('K-Means: Step-by-Step Convergence', fontsize=14, fontweight='bold', y=1.02)

for ax_idx, step in enumerate(step_indices):
    ax = axes[ax_idx]
    centroids_step = kmeans_scratch.history_[step]
    labels_step = labels_at_steps[step]

    for k in range(3):
        mask = labels_step == k
        ax.scatter(X_blobs[mask, 0], X_blobs[mask, 1],
                   c=COLORS[k], s=30, alpha=0.6, edgecolors='none')
        ax.scatter(centroids_step[k, 0], centroids_step[k, 1],
                   c=COLORS[k], s=250, marker='X',
                   edgecolors='black', linewidths=1.5, zorder=5)

    title = "Initial" if step == 0 else (
        "Final" if step == len(kmeans_scratch.history_) - 1
        else f"Iteration {step}")
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '01_kmeans_steps.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

# ---- Visualization 2: Elbow Method ----
print("Generating Visualization 2: Elbow Method (Inertia vs k)...")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(list(k_values), inertias, 'o-', color='#3498DB', linewidth=2.5,
        markersize=8, markerfacecolor='white', markeredgewidth=2.5)

# Annotate the elbow
ax.axvline(x=3, color='#E74C3C', linestyle='--', linewidth=2, label='Elbow at k=3')
ax.annotate('Elbow!\nTrue k=3', xy=(3, inertias[2]),
            xytext=(4.5, inertias[2] + (inertias[0] - inertias[-1]) * 0.15),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=2),
            fontsize=11, color='#E74C3C', fontweight='bold')

for i, (k, iner) in enumerate(zip(k_values, inertias)):
    ax.annotate(f'{iner:.0f}', (k, iner), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, color='#2C3E50')

ax.set_xlabel('Number of Clusters (k)', fontsize=13, fontweight='bold')
ax.set_ylabel('Inertia (Within-Cluster SS)', fontsize=13, fontweight='bold')
ax.set_title('The Elbow Method — Choosing the Right k', fontsize=14, fontweight='bold')
ax.set_xticks(list(k_values))
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax.text(0.98, 0.95,
        'Inertia always\ndecreases with k.\nLook for the "elbow"\nwhere gains diminish.',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '02_elbow_method.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

# ---- Visualization 3: Final clusters with centroids ----
print("Generating Visualization 3: Final clustering result with centroids...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('K-Means Final Clustering Result', fontsize=14, fontweight='bold')

# Left: our scratch result on blobs
ax = axes[0]
for k in range(3):
    mask = kmeans_scratch.labels_ == k
    ax.scatter(X_blobs[mask, 0], X_blobs[mask, 1],
               c=COLORS[k], s=50, alpha=0.7,
               edgecolors='none', label=f'Cluster {k}')

for k, c in enumerate(kmeans_scratch.centroids_):
    ax.scatter(c[0], c[1], c=COLORS[k], s=400, marker='X',
               edgecolors='black', linewidths=2.5, zorder=10)
    ax.annotate(f'C{k}', (c[0], c[1]), textcoords='offset points',
                xytext=(8, 8), fontsize=12, fontweight='bold')

ax.set_title('From-Scratch K-Means (k=3)\nX marks = centroids',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1', fontsize=11)
ax.set_ylabel('Feature 2', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Right: iris clustering result (first 2 PCA-like features for display)
ax = axes[1]
petal_len_idx = 2   # petal length
petal_wid_idx = 3   # petal width
for k in range(3):
    mask = kmeans_iris.labels_ == k
    ax.scatter(X_iris[mask, petal_len_idx], X_iris[mask, petal_wid_idx],
               c=COLORS[k], s=60, alpha=0.7, edgecolors='none',
               label=f'Cluster {k} (n={mask.sum()})')

for k, c in enumerate(kmeans_iris.centroids_):
    ax.scatter(c[petal_len_idx], c[petal_wid_idx], c=COLORS[k], s=400,
               marker='X', edgecolors='black', linewidths=2.5, zorder=10)

ax.set_title('Iris Dataset — K-Means k=3\n(Petal length vs Petal width)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Petal Length (cm)', fontsize=11)
ax.set_ylabel('Petal Width (cm)', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(VISUAL_DIR, '03_final_clusters.png')
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
print("UNSUPERVISED LEARNING:")
print("   - No labels — discover hidden structure from features alone")
print("   - Clustering groups similar points together")
print()
print("K-MEANS ALGORITHM:")
print("   1. Initialize k centroids (random or k-means++)")
print("   2. Assign each point to its nearest centroid")
print("   3. Update centroids to the mean of assigned points")
print("   4. Repeat until convergence (centroids stop moving)")
print()
print("INERTIA (within-cluster sum of squares):")
print("   - Measures cluster compactness — lower is better")
print("   - Used in the Elbow Method to choose k")
print()
print("ELBOW METHOD:")
print("   - Plot inertia vs k")
print("   - Look for the 'elbow' — diminishing returns")
print("   - That knee point is a good estimate of the true k")
print()
print("K-MEANS++ INITIALIZATION:")
print("   - Spreads starting centroids apart using distance-weighted sampling")
print("   - Avoids bad initializations that lead to poor solutions")
print()
print("LIMITATIONS:")
print("   - Must specify k in advance")
print("   - Assumes spherical, equal-sized clusters")
print("   - Sensitive to outliers and feature scale")
print("   - Only finds local optima (run multiple times with n_init)")
print()
print("NEXT: Principal Component Analysis (PCA) — reducing dimensions!")
print()
print("=" * 80)
print("Module Complete! Visualizations saved to:")
print(f"   {os.path.abspath(VISUAL_DIR)}")
print("=" * 80)
