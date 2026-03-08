"""
🔍 DBSCAN CLUSTERING - Density-Based Spatial Clustering of Applications with Noise
===================================================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. The core idea: clusters are DENSE regions of points separated by SPARSE regions
2. The three point types: core points, border points, and noise points
3. The two parameters that control DBSCAN: eps (radius) and min_samples (density)
4. How to implement DBSCAN from scratch using NumPy (region_query + expand_cluster)
5. Why DBSCAN beats K-Means on non-spherical, arbitrarily-shaped clusters
6. How to use sklearn's DBSCAN and interpret its output (-1 = noise)
7. How to select eps using a k-distance graph, and DBSCAN's limitations

RECOMMENDED VIDEOS:
------------------
* StatQuest: "DBSCAN, Clearly Explained"
  https://www.youtube.com/watch?v=RDZUdRSDOok
  Best visual walkthrough — MUST WATCH!

* Computerphile: "DBSCAN - Computerphile"
  https://www.youtube.com/watch?v=C3r7tGRe2eI
  Excellent visual intuition for density

* sklearn Docs Cluster Comparison:
  https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
  Side-by-side comparison of all clustering algorithms

TIME: 80-90 minutes
DIFFICULTY: Intermediate
PREREQUISITES: K-Means clustering, distance metrics, hierarchical clustering

OVERVIEW:
---------
K-Means and even hierarchical clustering (with Ward/complete linkage) struggle
with clusters that are:
  - Non-spherical (moons, rings, spirals)
  - Of very different sizes
  - Mixed with noise/outlier points

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) solves
all three problems by a single elegant idea:

  A CLUSTER is a maximal set of points where every point has at least
  min_samples neighbors within distance eps.

Points that are not reachable from any dense region are labeled NOISE (-1).

No need to specify k! The algorithm discovers k automatically.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

VISUAL_DIR = Path('../visuals/dbscan_clustering')
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  DBSCAN CLUSTERING - Find Clusters by Density, Not by Distance to Center")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: INTUITION — DENSITY AS THE KEY
# ============================================================================

print("=" * 80)
print("SECTION 1: The Big Idea — Density Defines Clusters")
print("=" * 80)
print()

print("K-Means ASSUMPTION: clusters are round blobs roughly equal in size.")
print("  → Fails on crescents, rings, spirals, and clusters with outliers.")
print()
print("DBSCAN ASSUMPTION: clusters are DENSE regions.")
print("  → Works on ANY shape, automatically ignores noise/outliers.")
print()
print("THE TWO PARAMETERS:")
print()
print("  eps (ε):          radius of the neighborhood around each point")
print("  min_samples (m):  minimum # of points needed to form a dense region")
print()
print("THE THREE POINT TYPES:")
print()
print("  ┌──────────────────────────────────────────────────────────────────┐")
print("  │                                                                  │")
print("  │   CORE POINT    ●   Has ≥ min_samples neighbors within eps      │")
print("  │                     (including itself)                          │")
print("  │                     → Part of a cluster interior                │")
print("  │                                                                  │")
print("  │   BORDER POINT  ◉   Fewer than min_samples neighbors in eps,    │")
print("  │                     but is within eps of a core point           │")
print("  │                     → On the edge of a cluster                  │")
print("  │                                                                  │")
print("  │   NOISE POINT   ○   Not a core point AND not within eps of any  │")
print("  │                     core point                                  │")
print("  │                     → Outlier; labeled -1 by sklearn            │")
print("  │                                                                  │")
print("  └──────────────────────────────────────────────────────────────────┘")
print()
print("VISUAL EXAMPLE (eps = 1.5, min_samples = 3):")
print()
print("  Points: A(0,0), B(0.5,0), C(0.3,0.4), D(5,5), E(5.2,5), F(10,10)")
print()
print("  Neighborhood of A: {A, B, C} → 3 neighbors ≥ min_samples=3 → CORE ●")
print("  Neighborhood of B: {A, B, C} → 3 neighbors ≥ min_samples=3 → CORE ●")
print("  Neighborhood of C: {A, B, C} → 3 neighbors ≥ min_samples=3 → CORE ●")
print("  Neighborhood of D: {D, E}    → 2 neighbors < min_samples=3")
print("      But D is within eps of E, and... only 2 nearby. → BORDER ◉")
print("  Neighborhood of F: {F}       → 1 neighbor  < min_samples=3")
print("      F is not within eps of any core point → NOISE ○")
print()
print("  Result: Cluster 1 = {A, B, C}    Cluster 2 = {D, E}    Noise = {F}")
print()

# ============================================================================
# SECTION 2: GENERATE DATASETS
# ============================================================================

print("=" * 80)
print("SECTION 2: Creating Datasets to Show DBSCAN's Strengths")
print("=" * 80)
print()

from sklearn.datasets import make_moons, make_blobs, make_circles

np.random.seed(42)

# Dataset 1: Moons — the classic non-spherical test
X_moons, y_moons = make_moons(n_samples=200, noise=0.06, random_state=42)
print(f"Dataset 1 (Moons):   {X_moons.shape[0]} points — crescent shapes, K-Means fails here")

# Dataset 2: Circles — concentric rings, another K-Means failure case
X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)
print(f"Dataset 2 (Circles): {X_circles.shape[0]} points — concentric rings, K-Means completely fails")

# Dataset 3: Blobs with noise — shows DBSCAN's outlier detection
X_blobs_raw, y_blobs_raw = make_blobs(
    n_samples=120, centers=[[0, 0], [6, 0], [3, 5]], cluster_std=0.5, random_state=42
)
# Add 15 uniformly spread noise points
rng = np.random.RandomState(7)
X_noise = rng.uniform(low=-3, high=9, size=(15, 2))
X_noisy = np.vstack([X_blobs_raw, X_noise])
y_noisy = np.hstack([y_blobs_raw, -1 * np.ones(15, dtype=int)])
print(f"Dataset 3 (Noisy):   {X_noisy.shape[0]} points — 3 clean clusters + 15 noise points")
print()

# ============================================================================
# SECTION 3: FROM-SCRATCH IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 3: DBSCAN from Scratch — region_query + expand_cluster")
print("=" * 80)
print()

print("The algorithm has two core helper functions:")
print()
print("  region_query(X, point_idx, eps)")
print("    → returns list of indices of all points within eps of point_idx")
print()
print("  expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples)")
print("    → recursively expands the cluster from a core point")
print()


UNVISITED = -2   # Internal: not yet processed
NOISE     = -1   # Outlier label (matches sklearn convention)


def region_query(X, point_idx, eps):
    """
    Return indices of all points within Euclidean distance eps of X[point_idx].
    This is the O(n) brute-force approach. A real implementation would use a
    k-d tree (O(log n) per query), which is what sklearn uses internally.
    """
    distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
    return list(np.where(distances <= eps)[0])


def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples):
    """
    Assign cluster_id to point_idx and recursively absorb density-reachable points.

    neighbors: initial neighborhood of point_idx (already confirmed as core point)
    """
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == UNVISITED:
            # Visit this neighbor for the first time
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, eps)
            if len(new_neighbors) >= min_samples:
                # neighbor_idx is also a core point; absorb its neighborhood
                neighbors.extend(new_neighbors)

        elif labels[neighbor_idx] == NOISE:
            # Previously labeled noise, but reachable from this cluster → border point
            labels[neighbor_idx] = cluster_id

        i += 1


def dbscan_from_scratch(X, eps=0.3, min_samples=5):
    """
    Full DBSCAN algorithm.

    Returns
    -------
    labels : (n_samples,) array
        Cluster labels. -1 = noise, 0,1,2,... = cluster IDs.
    point_types : (n_samples,) array
        'core', 'border', or 'noise' for each point.
    """
    n = len(X)
    labels = np.full(n, UNVISITED, dtype=int)
    cluster_id = 0

    for point_idx in range(n):
        if labels[point_idx] != UNVISITED:
            continue  # Already processed

        neighbors = region_query(X, point_idx, eps)

        if len(neighbors) < min_samples:
            labels[point_idx] = NOISE   # Tentative noise; may become border later
        else:
            # point_idx is a CORE POINT — start a new cluster
            expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_samples)
            cluster_id += 1

    # Classify point types (for visualization)
    point_types = np.full(n, 'noise', dtype=object)
    for idx in range(n):
        if labels[idx] == NOISE:
            point_types[idx] = 'noise'
        else:
            neighbors = region_query(X, idx, eps)
            if len(neighbors) >= min_samples:
                point_types[idx] = 'core'
            else:
                point_types[idx] = 'border'

    return labels, point_types


print("Running DBSCAN from scratch on Moons dataset...")
print()
eps_scratch = 0.25
min_s_scratch = 5

scratch_labels, scratch_types = dbscan_from_scratch(X_moons, eps=eps_scratch, min_samples=min_s_scratch)

n_clusters_scratch = len(set(scratch_labels)) - (1 if NOISE in scratch_labels else 0)
n_noise_scratch = np.sum(scratch_labels == NOISE)
n_core   = np.sum(scratch_types == 'core')
n_border = np.sum(scratch_types == 'border')

print(f"  eps = {eps_scratch}, min_samples = {min_s_scratch}")
print(f"  Clusters found: {n_clusters_scratch}")
print(f"  Core points:    {n_core}")
print(f"  Border points:  {n_border}")
print(f"  Noise points:   {n_noise_scratch}")
print()

for cid in sorted(set(scratch_labels)):
    if cid == NOISE:
        print(f"  Label -1 (noise): {np.sum(scratch_labels == NOISE)} points")
    else:
        print(f"  Cluster {cid}:       {np.sum(scratch_labels == cid)} points")
print()

# ============================================================================
# SECTION 4: SKLEARN IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 4: Using sklearn DBSCAN (The Professional Way)")
print("=" * 80)
print()

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import adjusted_rand_score

    print("sklearn.cluster.DBSCAN is O(n log n) using a k-d tree for neighbor queries.")
    print("It uses the same algorithm but runs much faster on large datasets.")
    print()

    # ── Moons: compare K-Means vs DBSCAN ─────────────────────────────────────
    print("TEST 1: Moon-shaped data")
    print("-" * 50)

    km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels_moons = km_moons.fit_predict(X_moons)
    km_ari_moons = adjusted_rand_score(y_moons, km_labels_moons)

    db_moons = DBSCAN(eps=0.25, min_samples=5)
    db_labels_moons = db_moons.fit_predict(X_moons)
    db_ari_moons = adjusted_rand_score(y_moons, db_labels_moons)
    n_noise_moons = np.sum(db_labels_moons == -1)
    n_clust_moons = len(set(db_labels_moons)) - (1 if -1 in db_labels_moons else 0)

    print(f"  K-Means (k=2):         ARI = {km_ari_moons:.3f}  (expected: low)")
    print(f"  DBSCAN (eps=0.25, m=5): ARI = {db_ari_moons:.3f}  "
          f"clusters={n_clust_moons}, noise={n_noise_moons}")
    print()

    # ── Circles: compare K-Means vs DBSCAN ───────────────────────────────────
    print("TEST 2: Concentric circles")
    print("-" * 50)

    km_circles = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels_circles = km_circles.fit_predict(X_circles)
    km_ari_circles = adjusted_rand_score(y_circles, km_labels_circles)

    db_circles = DBSCAN(eps=0.2, min_samples=5)
    db_labels_circles = db_circles.fit_predict(X_circles)
    db_ari_circles = adjusted_rand_score(y_circles, db_labels_circles)
    n_noise_circles = np.sum(db_labels_circles == -1)
    n_clust_circles = len(set(db_labels_circles)) - (1 if -1 in db_labels_circles else 0)

    print(f"  K-Means (k=2):         ARI = {km_ari_circles:.3f}  (expected: ~0)")
    print(f"  DBSCAN (eps=0.2, m=5):  ARI = {db_ari_circles:.3f}  "
          f"clusters={n_clust_circles}, noise={n_noise_circles}")
    print()

    # ── Noisy blobs: show outlier detection ───────────────────────────────────
    print("TEST 3: Blobs + noise — outlier detection")
    print("-" * 50)

    db_noisy = DBSCAN(eps=0.8, min_samples=4)
    db_labels_noisy = db_noisy.fit_predict(X_noisy)
    n_noise_noisy = np.sum(db_labels_noisy == -1)
    n_clust_noisy = len(set(db_labels_noisy)) - (1 if -1 in db_labels_noisy else 0)
    true_noise = np.sum(y_noisy == -1)

    print(f"  True noise points injected: {true_noise}")
    print(f"  DBSCAN detected noise:      {n_noise_noisy}")
    print(f"  DBSCAN clusters found:      {n_clust_noisy}")
    print()
    print("  K-Means CANNOT label noise — it assigns every point to a cluster.")
    print("  DBSCAN AUTOMATICALLY identifies outliers as noise (label = -1).")
    print()

    sklearn_available = True

except ImportError as e:
    print(f"sklearn not installed: {e}")
    print("Install with: pip install scikit-learn")
    sklearn_available = False

# ============================================================================
# SECTION 5: PARAMETER SELECTION — THE k-DISTANCE GRAPH
# ============================================================================

print("=" * 80)
print("SECTION 5: Choosing eps — The k-Distance Graph")
print("=" * 80)
print()

print("The biggest challenge with DBSCAN: choosing eps.")
print()
print("METHOD: k-Distance Graph")
print()
print("  1. For every point, compute its distance to its k-th nearest neighbor")
print("     (use k = min_samples - 1)")
print("  2. Sort these distances in descending order and plot them")
print("  3. Look for the 'elbow' in the curve")
print("  4. The eps value at the elbow is a good starting point")
print()
print("WHY THE ELBOW WORKS:")
print("  • Below the elbow: distances are small → points inside dense clusters")
print("  • Above the elbow: distances jump up → points are outliers/noise")
print("  • The elbow is where clusters end and noise begins")
print()
print("EXAMPLE READING:")
print()
print("  k-distance (sorted, desc)")
print("  │")
print("  │\\")
print("  │  \\")
print("  │   \\___________")
print("  │                \\___________")
print("  └─────────────────────────────→ Points (sorted)")
print("         ↑")
print("      Elbow here → set eps to this distance value")
print()

print("LIMITATIONS OF DBSCAN:")
print()
print("  1. VARIABLE DENSITY CLUSTERS")
print("     DBSCAN uses a SINGLE eps for all clusters.")
print("     If clusters have very different densities, one eps can't fit all.")
print("     → HDBSCAN (hierarchical DBSCAN) solves this.")
print()
print("  2. HIGH DIMENSIONS")
print("     In high dimensions, all points become roughly equidistant")
print("     (the 'curse of dimensionality') — eps becomes meaningless.")
print("     → Reduce dimensions first (PCA/UMAP) before applying DBSCAN.")
print()
print("  3. TWO PARAMETERS TO TUNE")
print("     Both eps and min_samples interact, making parameter search 2D.")
print("     → Use the k-distance graph + domain knowledge.")
print()
print("  4. SENSITIVITY TO eps")
print("     Small change in eps can split one cluster into many, or merge all.")
print("     → Visualize for several eps values (see Visualization 3).")
print()

# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 6: Creating Visualizations")
print("=" * 80)
print()

if sklearn_available:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.neighbors import NearestNeighbors

    cluster_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00', '#a65628']
    noise_color    = '#aaaaaa'

    # ── Visualization 1: Core / Border / Noise on Moons ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('DBSCAN Point Types: Core, Border, and Noise',
                 fontsize=15, fontweight='bold')

    for ax_idx, (X_data, title, eps_val, min_s) in enumerate([
        (X_moons,   'Moon-shaped Data', 0.25, 5),
        (X_noisy,   'Blobs + Injected Noise', 0.8, 4),
    ]):
        ax = axes[ax_idx]
        db = DBSCAN(eps=eps_val, min_samples=min_s)
        labels = db.fit_predict(X_data)

        # Determine point types
        nbrs = NearestNeighbors(radius=eps_val).fit(X_data)
        neighbor_counts = np.array([
            len(nbrs.radius_neighbors([X_data[i]], radius=eps_val)[1][0])
            for i in range(len(X_data))
        ])
        is_core   = (labels != -1) & (neighbor_counts >= min_s)
        is_border = (labels != -1) & ~is_core
        is_noise  = labels == -1

        # Draw eps circles around a few core points (to illustrate the radius)
        core_indices = np.where(is_core)[0]
        for ci in core_indices[:5]:
            circle = plt.Circle(
                (X_data[ci, 0], X_data[ci, 1]),
                radius=eps_val,
                color='gold', fill=True, alpha=0.08, linewidth=0
            )
            ax.add_patch(circle)

        ax.scatter(X_data[is_core,   0], X_data[is_core,   1],
                   c='#1a9641', s=90, alpha=0.9, edgecolors='black', lw=0.6,
                   zorder=3, label=f'Core ({is_core.sum()})')
        ax.scatter(X_data[is_border, 0], X_data[is_border, 1],
                   c='#2b83ba', s=60, alpha=0.9, edgecolors='black', lw=0.6,
                   zorder=3, label=f'Border ({is_border.sum()})')
        ax.scatter(X_data[is_noise,  0], X_data[is_noise,  1],
                   c='#d7191c', s=80, marker='x', lw=2,
                   zorder=3, label=f'Noise ({is_noise.sum()})')

        # Gold annotation for eps
        ax.text(0.02, 0.97,
                f'eps = {eps_val},  min_samples = {min_s}\n'
                f'Clusters = {len(set(labels)) - (1 if -1 in labels else 0)}',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '01_core_border_noise.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/01_core_border_noise.png")
    plt.close()

    # ── Visualization 2: K-Means fails, DBSCAN succeeds ──────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('K-Means vs DBSCAN on Non-Spherical Datasets',
                 fontsize=15, fontweight='bold')

    datasets_v2 = [
        (X_moons,   y_moons,   2, 'Moons',   0.25, 5),
        (X_circles, y_circles, 2, 'Circles', 0.20, 5),
        (X_noisy,   y_noisy,   3, 'Blobs+Noise', 0.80, 4),
    ]

    for col, (X_data, y_true, k, name, eps_val, min_s) in enumerate(datasets_v2):
        # K-Means (top row)
        ax_km = axes[0][col]
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_lbl = km.fit_predict(X_data)
        unique_km = sorted(set(km_lbl))
        for i, lbl in enumerate(unique_km):
            mask = km_lbl == lbl
            ax_km.scatter(X_data[mask, 0], X_data[mask, 1],
                          c=cluster_colors[i % len(cluster_colors)],
                          s=50, alpha=0.8, edgecolors='black', lw=0.4)
        # Plot centroids
        ax_km.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                      c='black', marker='*', s=250, zorder=5, label='Centroids')
        valid_mask = y_true != -1
        ari_km = adjusted_rand_score(y_true[valid_mask], km_lbl[valid_mask])
        ax_km.set_title(f'K-Means on {name}\nARI = {ari_km:.3f}',
                        fontsize=11, fontweight='bold',
                        color='darkred' if ari_km < 0.7 else 'darkgreen')
        ax_km.set_xlabel('Feature 1')
        ax_km.set_ylabel('Feature 2')
        ax_km.legend(fontsize=8)
        ax_km.grid(True, alpha=0.25)

        # DBSCAN (bottom row)
        ax_db = axes[1][col]
        db = DBSCAN(eps=eps_val, min_samples=min_s)
        db_lbl = db.fit_predict(X_data)
        unique_db = sorted(set(db_lbl))
        for lbl in unique_db:
            mask = db_lbl == lbl
            if lbl == -1:
                ax_db.scatter(X_data[mask, 0], X_data[mask, 1],
                              c=noise_color, s=80, marker='x', lw=1.5, label='Noise')
            else:
                ax_db.scatter(X_data[mask, 0], X_data[mask, 1],
                              c=cluster_colors[lbl % len(cluster_colors)],
                              s=50, alpha=0.8, edgecolors='black', lw=0.4,
                              label=f'Cluster {lbl}')

        valid_mask = y_true != -1
        ari_db = adjusted_rand_score(y_true[valid_mask], db_lbl[valid_mask])
        n_clust = len(unique_db) - (1 if -1 in unique_db else 0)
        n_ns    = np.sum(db_lbl == -1)
        ax_db.set_title(f'DBSCAN on {name}\nARI = {ari_db:.3f}  '
                        f'(k={n_clust}, noise={n_ns})',
                        fontsize=11, fontweight='bold',
                        color='darkred' if ari_db < 0.7 else 'darkgreen')
        ax_db.set_xlabel('Feature 1')
        ax_db.set_ylabel('Feature 2')
        ax_db.legend(fontsize=8)
        ax_db.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '02_kmeans_vs_dbscan.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/02_kmeans_vs_dbscan.png")
    plt.close()

    # ── Visualization 3: eps sensitivity + k-distance graph ──────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('DBSCAN Sensitivity to eps (top) and k-Distance Graph (bottom)',
                 fontsize=14, fontweight='bold')

    eps_values = [0.05, 0.15, 0.25, 0.50]
    for col, eps_val in enumerate(eps_values):
        ax = axes[0][col]
        db = DBSCAN(eps=eps_val, min_samples=5)
        labels = db.fit_predict(X_moons)
        unique = sorted(set(labels))
        n_clust = len(unique) - (1 if -1 in unique else 0)
        n_ns    = np.sum(labels == -1)

        for lbl in unique:
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(X_moons[mask, 0], X_moons[mask, 1],
                           c=noise_color, s=60, marker='x', lw=1.5)
            else:
                ax.scatter(X_moons[mask, 0], X_moons[mask, 1],
                           c=cluster_colors[lbl % len(cluster_colors)],
                           s=50, alpha=0.8, edgecolors='black', lw=0.4)

        title_color = 'darkgreen' if n_clust == 2 and n_ns < 20 else 'darkorange'
        ax.set_title(f'eps = {eps_val}\nclusters={n_clust}, noise={n_ns}',
                     fontsize=11, fontweight='bold', color=title_color)
        ax.set_xlabel('Feature 1', fontsize=9)
        ax.set_ylabel('Feature 2', fontsize=9)
        ax.grid(True, alpha=0.25)

    # Bottom row: k-distance graph for each eps context
    for col in range(4):
        ax = axes[1][col]
        k_nn = 4  # min_samples - 1
        nbrs = NearestNeighbors(n_neighbors=k_nn + 1).fit(X_moons)
        distances, _ = nbrs.kneighbors(X_moons)
        k_distances = distances[:, k_nn]
        k_distances_sorted = np.sort(k_distances)[::-1]

        ax.plot(range(len(k_distances_sorted)), k_distances_sorted,
                color='#2b83ba', linewidth=2)
        ax.axhline(y=eps_values[col], color='red', linestyle='--', linewidth=2,
                   label=f'eps = {eps_values[col]}')

        # Shade region above and below eps
        ax.fill_between(range(len(k_distances_sorted)), 0, k_distances_sorted,
                        where=(k_distances_sorted <= eps_values[col]),
                        alpha=0.2, color='green', label='Dense (cluster)')
        ax.fill_between(range(len(k_distances_sorted)), 0, k_distances_sorted,
                        where=(k_distances_sorted > eps_values[col]),
                        alpha=0.2, color='red', label='Sparse (noise)')

        ax.set_title(f'k-Distance Graph (k={k_nn})\neps = {eps_values[col]}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Points (sorted by distance)', fontsize=9)
        ax.set_ylabel(f'{k_nn}-th Nearest Neighbor Distance', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '03_eps_sensitivity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/03_eps_sensitivity.png")
    plt.close()

else:
    print("Skipping visualizations — sklearn not available.")
    print("Install with: pip install scikit-learn matplotlib")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("  SUMMARY: What You Learned")
print("=" * 80)
print()
print("DBSCAN finds clusters as DENSE REGIONS separated by sparse space.")
print()
print("THREE POINT TYPES:")
print("  Core point   → ≥ min_samples neighbors within eps")
print("  Border point → fewer than min_samples neighbors, but near a core point")
print("  Noise point  → not reachable from any core point → labeled -1")
print()
print("ALGORITHM (two steps):")
print("  1. region_query(p, eps) — find all neighbors of p within eps")
print("  2. expand_cluster(...)  — recursively grow cluster from core points")
print()
print("TWO PARAMETERS:")
print("  eps         → neighborhood radius (use k-distance graph to choose)")
print("  min_samples → density threshold (try: 2 × n_features as a start)")
print()
print("DBSCAN vs K-MEANS:")
print("  DBSCAN  → arbitrary shapes, automatic k, handles outliers")
print("  K-Means → spherical clusters, fast, scalable to millions of points")
print()
print("DBSCAN LIMITATIONS:")
print("  Variable-density clusters → use HDBSCAN")
print("  High dimensions           → reduce first (PCA/UMAP)")
print("  Two parameters to tune    → use k-distance graph + grid search")
print()
print("=" * 80)
print("  Module Complete! Visualizations saved to:")
print(f"  {VISUAL_DIR.resolve()}/")
print("    01_core_border_noise.png")
print("    02_kmeans_vs_dbscan.png")
print("    03_eps_sensitivity.png")
print("=" * 80)
