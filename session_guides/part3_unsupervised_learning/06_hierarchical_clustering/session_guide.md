# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 06: Hierarchical Clustering
### Two-Session Teaching Script

> **No need to specify k. Build a tree, cut where you want.**
> Hierarchical clustering gives you a full picture of how your data
> nests and relates — not just one partition, but all possible partitions at once.

---

# SESSION 1 (~90 min)
## "Dendrograms — the tree of all possible clusterings"

## Before They Arrive
- Terminal open in `unsupervised_learning/algorithms/`
- Draw a family tree on the board (great analogy)

---

## OPENING (10 min)

> *"K-Means had a fundamental problem: you must choose k before running.*
> *What if you don't know how many clusters there are?*
>
> *Hierarchical clustering: don't choose. Build ALL clusterings.*
> *Start with n clusters (each point alone).*
> *Merge the two closest. Now n-1 clusters.*
> *Merge again. n-2.*
> *Keep merging until one big cluster.*
>
> *Then look at the full tree — the dendrogram.*
> *CUT the tree wherever makes sense for your use case.*
> *Need 3 clusters? Cut there. Need 10? Cut there.*
> *One algorithm run, every possible k."*

Write on board:
```
START: [A] [B] [C] [D] [E]  ← 5 clusters

Merge closest pair (B,C):
       [A] [BC] [D] [E]     ← 4 clusters

Merge closest pair (D,E):
       [A] [BC] [DE]        ← 3 clusters

Merge closest pair (BC, DE):
       [A] [BCDE]           ← 2 clusters

Merge all:
       [ABCDE]              ← 1 cluster

DENDROGRAM = the record of all these merges, with distances
```

---

## SECTION 1: Reading a Dendrogram (25 min)

```bash
python3 hierarchical_clustering.py
```

Look at the output dendrogram:

> *"The y-axis is the distance at which the merge happened.*
> *Low merges: two similar points joining.*
> *High merges: two distant groups forced together.*
>
> *Where to cut: draw a horizontal line across the dendrogram.*
> *Count how many vertical lines you cross = number of clusters.*
>
> *The best cut: find the LARGEST GAP in merge distances.*
> *A big jump means: we're merging very different things.*
> *The point just before that jump is the natural number of clusters."*

Draw on board:
```
Height
│
│ ─── gap ────────────────────────
│
│
│─── cut here ─────────────────────  ← crosses 3 lines = 3 clusters
│
│     │       │     │
│    / \     / \   / \
│   A   B   C   D E   F
```

---

## SECTION 2: Linkage Methods (20 min)

> *"When merging two clusters, 'distance between clusters' is ambiguous.*
> *The distance between two POINTS is clear (Euclidean, Manhattan...).*
> *But distance between two GROUPS? Several options."*

Write:
```
LINKAGE METHODS:

Single linkage:
  distance(A, B) = min distance between any point in A and any in B
  "Nearest neighbor"
  Problem: creates long chains (chaining effect)

Complete linkage:
  distance(A, B) = max distance between any point in A and any in B
  "Furthest neighbor"
  Creates compact, equal-sized clusters

Average linkage:
  distance(A, B) = average of all pairwise distances A→B
  Compromise between single and complete

Ward linkage (most common):
  Merge clusters that minimize TOTAL within-cluster variance
  Usually produces the most natural clusters
  sklearn default: linkage='ward'
```

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Hierarchical clustering: merge bottom-up, one step at a time
  Dendrogram: tree of all merges with distances
  Cut: draw horizontal line → count clusters
  Best cut: find largest gap in merge heights
  Linkage: Ward is usually best (minimizes variance)
```

---

# SESSION 2 (~90 min)
## "When to use hierarchical clustering vs K-Means"

## OPENING (5 min)

> *"Session 1: how it works.*
> *Today: when to use it, implement it from scratch, and compare to K-Means."*

---

## SECTION 1: From Scratch (Agglomerative) (25 min)

```python
def agglomerative_clustering(X, n_clusters):
    n = len(X)
    # Start: each point is its own cluster
    clusters = {i: [i] for i in range(n)}

    # Distance matrix
    dist = np.array([[np.linalg.norm(X[i] - X[j]) for j in range(n)]
                     for i in range(n)])
    np.fill_diagonal(dist, np.inf)  # ignore self-distance

    while len(clusters) > n_clusters:
        # Find two closest clusters (complete linkage)
        min_dist = np.inf
        merge_i, merge_j = -1, -1
        for i in clusters:
            for j in clusters:
                if i >= j: continue
                # Complete linkage: max distance between cluster members
                d = max(dist[a, b]
                        for a in clusters[i]
                        for b in clusters[j])
                if d < min_dist:
                    min_dist, merge_i, merge_j = d, i, j

        # Merge cluster j into cluster i
        clusters[merge_i].extend(clusters[merge_j])
        del clusters[merge_j]

    # Assign labels
    labels = np.zeros(n, dtype=int)
    for label, (_, members) in enumerate(clusters.items()):
        for m in members:
            labels[m] = label
    return labels
```

> *"This is O(n³) — slow for large datasets.*
> *sklearn's implementation is much faster: O(n² log n) with optimizations.*
> *But this shows the algorithm clearly."*

---

## SECTION 2: sklearn in 3 Lines (10 min)

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc

# Dendrogram (scipy)
linked = shc.linkage(X, method='ward')
dendrogram(linked)

# Clustering (sklearn)
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = hc.fit_predict(X)
```

---

## SECTION 3: K-Means vs Hierarchical — When to Use Which (20 min)

```
USE K-MEANS WHEN:
  ✓ Large dataset (10K+ points) — hierarchical is slow
  ✓ You roughly know k
  ✓ Clusters are roughly spherical and similar size
  ✓ Speed is important

USE HIERARCHICAL WHEN:
  ✓ Small to medium dataset (< 10K points)
  ✓ You don't know k (use dendrogram to decide)
  ✓ You want to explore the cluster hierarchy
  ✓ Nested cluster structure matters (species → genus → family)
  ✓ Non-spherical clusters (use complete or average linkage)
  ✓ You need reproducible results (no random init)
```

> *"The dendrogram is hierarchical's superpower.*
> *In biology: clustering gene expression data, species classification.*
> *In market research: market segments nest within each other.*
> *In document analysis: topics have subtopics.*
> *K-Means loses that structure. Hierarchical preserves it."*

---

## CLOSING SESSION 2 (10 min)

```
HIERARCHICAL CLUSTERING — COMPLETE:
  Agglomerative: bottom-up merging (most common)
  Dendrogram: tree of all clusterings
  Ward linkage: minimize variance on merge (usually best)
  No need to pre-specify k
  But slower than K-Means

  Next: DBSCAN — what if clusters aren't blobs at all?
```

---

## INSTRUCTOR TIPS

**"What's divisive vs agglomerative?"**
> *"Agglomerative: bottom-up (start with n clusters, merge).*
> *Divisive: top-down (start with 1 cluster, split).*
> *Agglomerative is almost always used — cheaper and more stable.*
> *Divisive requires deciding 'which cluster to split and how' at each step — much harder."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — the k problem          10 min
├── Dendrogram reading               25 min
├── Linkage methods                  20 min
└── Close                             5 min  (+ 30 min buffer)

SESSION 2  (90 min)
├── Opening                           5 min
├── From scratch implementation      25 min
├── sklearn in 3 lines               10 min
├── K-Means vs hierarchical          20 min
└── Close                            10 min  (+ 20 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 06*
