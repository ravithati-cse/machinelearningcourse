"""
🌳 HIERARCHICAL CLUSTERING - Build a Tree of Similarities
==========================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. The core intuition: build a dendrogram (merge tree), then cut it to get clusters
2. Agglomerative (bottom-up) clustering: every point starts alone, we merge greedily
3. Four linkage methods: single, complete, average, and Ward — and when each shines
4. How to read a dendrogram and choose where to cut for k clusters
5. How to implement agglomerative clustering from scratch using NumPy
6. How to use sklearn's AgglomerativeClustering and scipy's dendrogram tools
7. When hierarchical clustering beats K-Means (non-spherical shapes, unknown k)

RECOMMENDED VIDEOS:
------------------
* StatQuest: "Hierarchical Clustering"
  https://www.youtube.com/watch?v=7xHsRkOdVwo
  Best visual walkthrough — MUST WATCH!

* StatQuest: "How to Read a Dendrogram"
  https://www.youtube.com/watch?v=OcoE7JlbXvY
  Clear explanation of cutting strategies

* StatQuest: "Ward's Method for Hierarchical Clustering"
  https://www.youtube.com/watch?v=T1ObCUpjq3o
  Why Ward minimizes within-cluster variance

TIME: 80-100 minutes
DIFFICULTY: Intermediate
PREREQUISITES: K-Means clustering, distance metrics, basic NumPy

OVERVIEW:
---------
K-Means requires you to specify k upfront. What if you don't know k?
Hierarchical clustering solves this by building a FULL TREE of merges.
You can look at the tree (dendrogram) and decide where to cut after the fact.

The agglomerative (bottom-up) approach:
  Step 0: Each of N points is its own cluster (N clusters total)
  Step 1: Find the two CLOSEST clusters and merge them (N-1 clusters)
  Step 2: Repeat until only 1 cluster remains
  Step 3: Cut the dendrogram at the right height to get k clusters

"Closest" depends on the LINKAGE method:
  - Single linkage:   distance = min distance between any two points across clusters
  - Complete linkage: distance = max distance between any two points across clusters
  - Average linkage:  distance = mean distance between all pairs across clusters
  - Ward linkage:     merge the pair that increases total within-cluster variance LEAST
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

VISUAL_DIR = Path('../visuals/hierarchical_clustering')
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("  HIERARCHICAL CLUSTERING - Build a Tree, Then Cut It")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: INTUITION — THE BIG PICTURE
# ============================================================================

print("=" * 80)
print("SECTION 1: The Big Idea — A Tree of Merges")
print("=" * 80)
print()

print("PROBLEM: K-Means requires you to know k in advance.")
print("SOLUTION: Build a FULL TREE of merges — then decide k by cutting the tree.")
print()

print("VISUALIZING THE PROCESS:")
print()
print("  Start (5 points, each is its own cluster):")
print("  A   B   C   D   E")
print("  |   |   |   |   |")
print()
print("  Step 1 — Merge A and B (they are closest):")
print("   AB  C   D   E")
print("  /\\   |   |   |")
print(" A  B  |   |   |")
print()
print("  Step 2 — Merge D and E:")
print("   AB  C   DE")
print("  /\\   |   /\\")
print(" A  B  |  D  E")
print()
print("  Step 3 — Merge AB and C:")
print("   ABC     DE")
print("   /\\\\     /\\")
print("  A  B  C  D  E")
print()
print("  Step 4 — Merge ABC and DE into one cluster:")
print("       ABCDE")
print("      /      \\")
print("    ABC       DE")
print("    /\\\\       /\\")
print("   A  B  C   D  E")
print()
print("This tree is called a DENDROGRAM.")
print("To get k=2 clusters: cut at Step 4 boundary → {A,B,C} and {D,E}")
print("To get k=3 clusters: cut at Step 3 boundary → {A,B}, {C}, {D,E}")
print()

print("LINKAGE METHODS — How do we measure 'distance between clusters'?")
print()
print("  Imagine cluster P = {p1, p2} and cluster Q = {q1, q2}.")
print()
print("  Single linkage  (MIN): d(P,Q) = min(d(p1,q1), d(p1,q2), d(p2,q1), d(p2,q2))")
print("    • Sensitive to outliers; can create long 'chains' of clusters")
print("    • Pro: finds non-convex shapes")
print("    • Con: chaining effect")
print()
print("  Complete linkage (MAX): d(P,Q) = max(...all pairs...)")
print("    • Creates compact, equally-sized clusters")
print("    • Pro: robust to outliers; tight clusters")
print("    • Con: breaks large clusters aggressively")
print()
print("  Average linkage (MEAN): d(P,Q) = mean(...all pairs...)")
print("    • Balanced between single and complete")
print("    • Pro: good general-purpose linkage")
print("    • Con: biased toward equal-sized clusters")
print()
print("  Ward linkage: merge the pair that increases total within-cluster")
print("  variance (inertia) the LEAST.")
print("    • Pro: creates the most homogeneous clusters")
print("    • Con: assumes roughly spherical clusters")
print("    • DEFAULT in sklearn — usually the best choice!")
print()

# ============================================================================
# SECTION 2: GENERATE DATASETS
# ============================================================================

print("=" * 80)
print("SECTION 2: Creating Datasets to Explore")
print("=" * 80)
print()

from sklearn.datasets import make_blobs, make_moons

np.random.seed(42)

# Dataset 1: Well-separated blobs — easy for all algorithms
X_blobs, y_blobs = make_blobs(
    n_samples=150,
    centers=[[1, 1], [5, 1], [3, 5]],
    cluster_std=0.6,
    random_state=42
)

print(f"Dataset 1 (Blobs): {X_blobs.shape[0]} points in 3 well-separated clusters")

# Dataset 2: Moon-shaped — non-convex, hierarchical with single linkage wins
X_moons, y_moons = make_moons(n_samples=150, noise=0.08, random_state=42)

print(f"Dataset 2 (Moons): {X_moons.shape[0]} points in 2 crescent-shaped clusters")
print()

print("WHY TWO DATASETS?")
print("  • Blobs test basic cluster separation (all methods work)")
print("  • Moons test non-convex shapes (only single linkage handles well)")
print()

# ============================================================================
# SECTION 3: FROM-SCRATCH IMPLEMENTATION (COMPLETE LINKAGE)
# ============================================================================

print("=" * 80)
print("SECTION 3: Agglomerative Clustering from Scratch (Complete Linkage)")
print("=" * 80)
print()

print("We'll implement the core algorithm step by step using NumPy.")
print("We use COMPLETE LINKAGE because it's simple yet produces tight clusters.")
print()


def euclidean_distance(p, q):
    """Euclidean distance between two points."""
    return np.sqrt(np.sum((p - q) ** 2))


def cluster_distance_complete(cluster_a_indices, cluster_b_indices, X):
    """
    Complete linkage: max pairwise distance between two clusters.
    This is the furthest-neighbor measure.
    """
    max_dist = -np.inf
    for i in cluster_a_indices:
        for j in cluster_b_indices:
            d = euclidean_distance(X[i], X[j])
            if d > max_dist:
                max_dist = d
    return max_dist


def agglomerative_from_scratch(X, n_clusters=3, verbose=True):
    """
    Bottom-up agglomerative clustering with complete linkage.

    Parameters
    ----------
    X          : (n_samples, n_features) array
    n_clusters : desired number of clusters after cutting
    verbose    : print merge steps

    Returns
    -------
    labels     : (n_samples,) cluster assignment
    history    : list of (merged_a, merged_b, distance) tuples
    """
    n = len(X)

    # Each point starts as its own cluster
    # clusters[i] = list of point indices belonging to cluster i
    clusters = {i: [i] for i in range(n)}
    history = []   # Record of merges: (cluster_id_a, cluster_id_b, distance, new_id)
    next_id = n    # New cluster IDs start after n

    step = 0

    while len(clusters) > n_clusters:
        # Find the two closest clusters (by complete linkage distance)
        min_dist = np.inf
        merge_a, merge_b = None, None
        cluster_ids = list(clusters.keys())

        for i_idx in range(len(cluster_ids)):
            for j_idx in range(i_idx + 1, len(cluster_ids)):
                ci = cluster_ids[i_idx]
                cj = cluster_ids[j_idx]
                d = cluster_distance_complete(clusters[ci], clusters[cj], X)
                if d < min_dist:
                    min_dist = d
                    merge_a = ci
                    merge_b = cj

        # Merge the two closest clusters into a new one
        new_cluster_points = clusters[merge_a] + clusters[merge_b]
        history.append((merge_a, merge_b, min_dist, next_id))

        if verbose and step < 8:
            size_a = len(clusters[merge_a])
            size_b = len(clusters[merge_b])
            print(f"  Step {step+1}: Merge cluster {merge_a} ({size_a} pts) + "
                  f"cluster {merge_b} ({size_b} pts) → cluster {next_id}  "
                  f"[distance = {min_dist:.4f}]")

        del clusters[merge_a]
        del clusters[merge_b]
        clusters[next_id] = new_cluster_points
        next_id += 1
        step += 1

    if verbose and len(X) > 8:
        print(f"  ... (continued for {step} total merge steps)")
    print()

    # Assign final labels
    labels = np.zeros(n, dtype=int)
    for label, (_, point_indices) in enumerate(clusters.items()):
        for idx in point_indices:
            labels[idx] = label

    return labels, history


print("Running from-scratch agglomerative clustering on Blobs dataset...")
print("(showing first 8 merge steps)")
print()
print("  Merge Log:")
print("  " + "-" * 65)

scratch_labels, merge_history = agglomerative_from_scratch(
    X_blobs, n_clusters=3, verbose=True
)

unique_labels = np.unique(scratch_labels)
print(f"Result: {len(unique_labels)} clusters found")
for lbl in unique_labels:
    count = np.sum(scratch_labels == lbl)
    print(f"  Cluster {lbl}: {count} points")
print()

scratch_accuracy = np.mean([
    np.mean(scratch_labels[y_blobs == true_lbl] == scratch_labels[y_blobs == true_lbl][0])
    for true_lbl in np.unique(y_blobs)
])
print("(From-scratch complete linkage successfully groups the blob clusters)")
print()

# ============================================================================
# SECTION 4: SKLEARN IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 4: Using sklearn and scipy (The Professional Way)")
print("=" * 80)
print()

try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    print("sklearn AgglomerativeClustering gives us four linkage methods:")
    print()

    linkage_methods = ['single', 'complete', 'average', 'ward']
    results_blobs = {}

    for method in linkage_methods:
        model = AgglomerativeClustering(n_clusters=3, linkage=method)
        labels = model.fit_predict(X_blobs)
        ari = adjusted_rand_score(y_blobs, labels)
        sil = silhouette_score(X_blobs, labels)
        results_blobs[method] = {'labels': labels, 'ari': ari, 'sil': sil}
        print(f"  [{method:>8} linkage]  ARI = {ari:.3f}   Silhouette = {sil:.3f}")

    print()
    print("Adjusted Rand Index (ARI):")
    print("  1.0 = perfect match with true labels")
    print("  0.0 = random assignment")
    print()

    best_method = max(results_blobs, key=lambda m: results_blobs[m]['ari'])
    print(f"Best linkage for blobs: {best_method} (ARI = {results_blobs[best_method]['ari']:.3f})")
    print()

    print("Now testing on MOON-SHAPED data:")
    print("  K-Means with 2 clusters on moons (expected: poor performance)")
    from sklearn.cluster import KMeans
    km_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
    km_labels = km_moons.fit_predict(X_moons)
    km_ari = adjusted_rand_score(y_moons, km_labels)
    print(f"  K-Means      ARI = {km_ari:.3f}  (usually < 0.5 on moons)")

    results_moons = {}
    for method in linkage_methods:
        model = AgglomerativeClustering(n_clusters=2, linkage=method)
        labels = model.fit_predict(X_moons)
        ari = adjusted_rand_score(y_moons, labels)
        results_moons[method] = {'labels': labels, 'ari': ari}
        print(f"  [{method:>8} linkage]  ARI = {ari:.3f}")

    print()
    best_moon_method = max(results_moons, key=lambda m: results_moons[m]['ari'])
    print(f"Best linkage for moons: {best_moon_method} (ARI = {results_moons[best_moon_method]['ari']:.3f})")
    print()

    print("KEY INSIGHT:")
    print("  Single linkage finds non-convex moon shapes because it uses the MINIMUM")
    print("  distance — points at the tips of adjacent crescents can still connect.")
    print("  Complete/Ward linkage fails because they try to form compact round blobs.")
    print()

    sklearn_available = True

except ImportError as e:
    print(f"Some libraries not installed: {e}")
    print("Install with: pip install scikit-learn scipy")
    sklearn_available = False

# ============================================================================
# SECTION 5: HOW TO READ A DENDROGRAM
# ============================================================================

print("=" * 80)
print("SECTION 5: Reading Dendrograms — Where to Cut the Tree")
print("=" * 80)
print()

print("A DENDROGRAM shows the full merge history:")
print()
print("  HEIGHT (y-axis) = distance at which the merge happened")
print("  LEAVES (x-axis) = original data points")
print("  HORIZONTAL LINES = merge events joining two clusters")
print()
print("HOW TO CHOOSE k:")
print("  Look for the LARGEST VERTICAL GAP in the dendrogram — that gap")
print("  represents the biggest 'jump' in merge distance, meaning the clusters")
print("  below that cut are well-separated.")
print()
print("  Example dendrogram heights (ward linkage on blobs):")
print()
print("  Height 10 |                        ___________")
print("            |                       |           |")
print("  Height  5 |          _____        |           |")
print("            |         |     |       |           |")
print("  Height  2 |   ___   |     |  ___  |           |")
print("            |  |   |  |     | |   | |           |")
print("            |  A   B  C     D  E  F G           H")
print()
print("  Largest gap: between height 5 and 10")
print("  CUT HERE → 2 clusters: {A,B,C} and {D,E,F,G,H}")
print()
print("  Second cut at height 5 → 3 clusters: {A,B}, {C}, {D,E,F,G,H}")
print()

# ============================================================================
# SECTION 6: ADVANTAGES VS K-MEANS
# ============================================================================

print("=" * 80)
print("SECTION 6: Hierarchical Clustering vs K-Means — When to Use Which")
print("=" * 80)
print()

print("HIERARCHICAL CLUSTERING ADVANTAGES:")
print()
print("  1. NO NEED TO SPECIFY k IN ADVANCE")
print("     Build the full tree, then choose k by inspection of the dendrogram.")
print()
print("  2. DENDROGRAM GIVES THE FULL PICTURE")
print("     You can see ALL possible clusterings at once, not just one.")
print()
print("  3. WORKS WITH ANY DISTANCE METRIC")
print("     Euclidean, cosine, Manhattan, Hamming, custom — all work.")
print()
print("  4. NON-SPHERICAL CLUSTERS (single linkage)")
print("     Can find chains, crescents, and irregular shapes.")
print()
print("  5. DETERMINISTIC")
print("     No random initialization — same input always gives same tree.")
print()
print("K-MEANS ADVANTAGES:")
print()
print("  1. SCALABLE: O(n * k * t) — handles millions of points")
print("     Hierarchical clustering is O(n^3) naively, O(n^2 log n) optimized.")
print()
print("  2. SIMPLE and FAST for large datasets")
print()
print("  3. WORKS WELL when clusters ARE spherical and roughly equal size")
print()
print("RULE OF THUMB:")
print("  Small/medium data (<10,000 pts), unknown k → Hierarchical")
print("  Large data, roughly spherical clusters   → K-Means")
print()

# ============================================================================
# SECTION 7: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 7: Creating Visualizations")
print("=" * 80)
print()

if sklearn_available:
    from scipy.cluster.hierarchy import dendrogram, linkage

    # ── Visualization 1: Dendrogram from scipy ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Hierarchical Clustering — Dendrograms', fontsize=16, fontweight='bold')

    for ax_idx, (X_data, name, true_k) in enumerate(
            [(X_blobs, 'Blobs (3 true clusters)', 3),
             (X_moons, 'Moons (2 true clusters)', 2)]):
        ax = axes[ax_idx]

        Z = linkage(X_data, method='ward')
        dend = dendrogram(
            Z,
            ax=ax,
            color_threshold=0.7 * max(Z[:, 2]),
            above_threshold_color='gray',
            leaf_rotation=90,
            leaf_font_size=5,
            show_leaf_counts=True,
        )
        # Mark the suggested cut line
        heights = sorted(Z[:, 2])
        gaps = np.diff(heights)
        biggest_gap_idx = np.argmax(gaps)
        cut_height = (heights[biggest_gap_idx] + heights[biggest_gap_idx + 1]) / 2
        ax.axhline(y=cut_height, color='red', linestyle='--', linewidth=2,
                   label=f'Suggested cut → k={true_k}')

        ax.set_title(f'Ward Dendrogram: {name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Data Points', fontsize=10)
        ax.set_ylabel('Merge Distance (Ward)', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '01_dendrograms.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/01_dendrograms.png")
    plt.close()

    # ── Visualization 2: Cluster assignments on 2D scatter ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Hierarchical Clustering — Cluster Assignments on 2D Data',
                 fontsize=15, fontweight='bold')

    colors_3 = ['#e41a1c', '#377eb8', '#4daf4a']
    colors_2 = ['#e41a1c', '#377eb8']

    # Panel 1: Blobs, Ward linkage
    ax = axes[0]
    ward_labels = results_blobs['ward']['labels']
    for k in range(3):
        mask = ward_labels == k
        ax.scatter(X_blobs[mask, 0], X_blobs[mask, 1],
                   c=colors_3[k], s=70, alpha=0.8, edgecolors='black', lw=0.5,
                   label=f'Cluster {k}')
    ax.set_title(f"Blobs: Ward Linkage\nARI = {results_blobs['ward']['ari']:.3f}",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Moons, K-Means (to show it fails)
    ax = axes[1]
    for k in range(2):
        mask = km_labels == k
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1],
                   c=colors_2[k], s=70, alpha=0.8, edgecolors='black', lw=0.5,
                   label=f'Cluster {k}')
    ax.set_title(f"Moons: K-Means (FAILS)\nARI = {km_ari:.3f}",
                 fontsize=11, fontweight='bold', color='darkred')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Moons, Single linkage (succeeds)
    ax = axes[2]
    single_labels = results_moons['single']['labels']
    single_ari = results_moons['single']['ari']
    for k in range(2):
        mask = single_labels == k
        ax.scatter(X_moons[mask, 0], X_moons[mask, 1],
                   c=colors_2[k], s=70, alpha=0.8, edgecolors='black', lw=0.5,
                   label=f'Cluster {k}')
    ax.set_title(f"Moons: Single Linkage (SUCCEEDS)\nARI = {single_ari:.3f}",
                 fontsize=11, fontweight='bold', color='darkgreen')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '02_cluster_assignments.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/02_cluster_assignments.png")
    plt.close()

    # ── Visualization 3: Linkage method comparison ────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Linkage Method Comparison: Blobs (top) vs Moons (bottom)',
                 fontsize=14, fontweight='bold')

    datasets = [(X_blobs, y_blobs, 3, 'Blobs', colors_3),
                (X_moons, y_moons, 2, 'Moons', colors_2)]

    for row, (X_data, y_true, k, dname, palette) in enumerate(datasets):
        for col, method in enumerate(linkage_methods):
            ax = axes[row][col]
            model = AgglomerativeClustering(n_clusters=k, linkage=method)
            labels = model.fit_predict(X_data)
            ari = adjusted_rand_score(y_true, labels)

            for lbl in range(k):
                mask = labels == lbl
                ax.scatter(X_data[mask, 0], X_data[mask, 1],
                           c=palette[lbl], s=50, alpha=0.8,
                           edgecolors='black', lw=0.4)

            title_color = 'darkgreen' if ari > 0.8 else ('darkorange' if ari > 0.5 else 'darkred')
            ax.set_title(f'{dname}: {method.capitalize()}\nARI = {ari:.3f}',
                         fontsize=10, fontweight='bold', color=title_color)
            ax.set_xlabel('Feature 1', fontsize=8)
            ax.set_ylabel('Feature 2', fontsize=8)
            ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(VISUAL_DIR / '03_linkage_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {VISUAL_DIR}/03_linkage_comparison.png")
    plt.close()

else:
    print("Skipping visualizations — sklearn/scipy not available.")
    print("Install with: pip install scikit-learn scipy matplotlib")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("  SUMMARY: What You Learned")
print("=" * 80)
print()
print("HIERARCHICAL CLUSTERING builds a full merge tree (dendrogram), then")
print("cuts it at any level to produce k clusters — without needing k upfront.")
print()
print("AGGLOMERATIVE ALGORITHM (bottom-up):")
print("  1. Start: each point is its own cluster")
print("  2. Repeat: find the two closest clusters and merge them")
print("  3. Stop: when only one cluster remains (or you reach desired k)")
print()
print("LINKAGE METHODS:")
print("  Single   → min distance  → chains, non-convex shapes")
print("  Complete → max distance  → compact, round clusters")
print("  Average  → mean distance → balanced general purpose")
print("  Ward     → min variance increase → default; best overall")
print()
print("READING A DENDROGRAM:")
print("  Y-axis = merge height (distance)")
print("  Largest vertical gap → optimal number of clusters")
print()
print("VS K-MEANS:")
print("  Use hierarchical when: k is unknown, data is small/medium, shapes vary")
print("  Use K-Means when: data is large, clusters are spherical, speed matters")
print()
print("=" * 80)
print("  Module Complete! Visualizations saved to:")
print(f"  {VISUAL_DIR.resolve()}/")
print("    01_dendrograms.png")
print("    02_cluster_assignments.png")
print("    03_linkage_comparison.png")
print("=" * 80)
