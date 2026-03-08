# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 07: DBSCAN Clustering
### Two-Session Teaching Script

> **The algorithm that handles what K-Means can't.**
> Arbitrary shapes. Noise. No need to specify k.
> DBSCAN finds clusters by density — the way humans actually see groups.

---

# SESSION 1 (~90 min)
## "Density-based clustering — finding clusters of any shape"

## Before They Arrive
- Terminal open in `unsupervised_learning/algorithms/`
- Draw two crescent shapes on the board (K-Means fails here)

---

## OPENING (10 min)

> *"What if your clusters look like crescents? Like rings?*
> *Like two interleaved spirals?*
>
> *K-Means completely fails. It assumes blobs.*
> *Hierarchical clustering with Ward linkage also struggles.*
>
> *Human brains don't think in spheres.*
> *We think: 'these points are all part of the same dense region.'*
>
> *DBSCAN: Density-Based Spatial Clustering of Applications with Noise.*
> *It clusters by density.*
> *Points in dense regions: cluster members.*
> *Points in sparse regions: noise (outliers).*
>
> *And critically: you never specify k. It finds k automatically.*
> *The trade-off: you specify density parameters instead."*

Draw on board then show the reveal:
```
K-MEANS sees:                DBSCAN sees:
  two blobs (wrong)            two crescents (correct!)

K-Means splits by distance to centroid.
DBSCAN asks: "Is this point in a dense neighborhood?"
```

---

## SECTION 1: The Three Types of Points (25 min)

Write on board:
```
DBSCAN PARAMETERS:
  eps         = neighborhood radius (how far counts as "nearby")
  min_samples = minimum points to form a dense region

THREE TYPES OF POINTS:

1. CORE POINT:
   Has ≥ min_samples neighbors within eps distance
   → The backbone of a cluster

2. BORDER POINT:
   Has < min_samples neighbors within eps
   But is within eps of a core point
   → Part of the cluster, but not the core

3. NOISE POINT (outlier):
   Not within eps of any core point
   → Label = -1 (not in any cluster)

Visual:
  ● ● ● ● ●   ← core points (dense)
   ● ● ●      ← border points (near a core)
                    *      ← noise point (alone)
```

---

## SECTION 2: The Algorithm (20 min)

```
DBSCAN ALGORITHM:

For each unvisited point p:
  1. Find all points within eps of p (the neighborhood)
  2. If neighborhood has < min_samples points:
       label p as noise (for now)
  3. If neighborhood has ≥ min_samples points:
       p is a CORE POINT → start a new cluster
       Add all neighborhood points to this cluster
       For each of THOSE points:
         If they're also core points → expand their neighborhoods too
         (this is the "density reachability" propagation)
  4. Continue until no unvisited points remain

Result: clusters grow by density-connection
        Noise points remain unclustered (label = -1)
```

```bash
python3 dbscan_clustering.py
```

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Core point: ≥ min_samples neighbors within eps
  Border point: near a core, but sparse itself
  Noise: not near any core
  Algorithm: expand clusters by density-reachability
  Key: clusters can be ANY shape
```

---

# SESSION 2 (~90 min)
## "Choosing eps and min_samples — and when to use DBSCAN"

## OPENING (5 min)

> *"The hardest part of DBSCAN: choosing good parameters.*
> *Today: a principled method for picking eps.*
> *And the full comparison: K-Means vs hierarchical vs DBSCAN."*

---

## SECTION 1: From Scratch DBSCAN (25 min)

```python
def dbscan(X, eps, min_samples):
    n = len(X)
    labels = np.full(n, -1)  # -1 = noise initially
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    def region_query(p_idx):
        neighbors = []
        for i in range(n):
            if np.linalg.norm(X[p_idx] - X[i]) <= eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(p_idx, neighbors, cluster_id):
        labels[p_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            q = neighbors[i]
            if not visited[q]:
                visited[q] = True
                new_neighbors = region_query(q)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)
            if labels[q] == -1:  # was noise, now border
                labels[q] = cluster_id
            i += 1

    for p_idx in range(n):
        if visited[p_idx]:
            continue
        visited[p_idx] = True
        neighbors = region_query(p_idx)
        if len(neighbors) < min_samples:
            pass  # stays as noise (-1)
        else:
            expand_cluster(p_idx, neighbors, cluster_id)
            cluster_id += 1

    return labels
```

---

## SECTION 2: Choosing eps — The k-Distance Graph (20 min)

> *"The principled way to choose eps:"*

Write on board:
```
K-DISTANCE GRAPH METHOD:
  1. For each point, compute distance to its k-th nearest neighbor
     (use min_samples as k)
  2. Sort these distances in ascending order
  3. Plot the sorted distances
  4. Look for the "elbow" — sharp increase in distance
  5. The elbow = good eps value

Intuition:
  Below the elbow: points are tightly packed (within clusters)
  Above the elbow: points are sparse (noise / between clusters)
  The elbow is where "nearby" ends and "distant" begins
```

```python
from sklearn.neighbors import NearestNeighbors

nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X)
distances, _ = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, -1])  # k-th neighbor distance
# Plot k_distances, look for elbow
```

---

## SECTION 3: Algorithm Comparison (20 min)

Show all three on the moon/crescent dataset:

```
COMPARISON ON NON-SPHERICAL DATA:

K-Means (k=2):          DBSCAN:              Hierarchical:
  ┌──────────┐            ┌──────────┐         ┌──────────┐
  │ A  │  B  │            │ ╭──╮ ╭──╮│         │   mixed  │
  │ ╭──│──╮  │            │(  )(  ) │          │  result  │
  │(   │   ) │            │ ╰──╯ ╰──╯│         │          │
  │ ╰──│──╯  │            │          │         │          │
  └────│─────┘            └──────────┘         └──────────┘
  Wrong: splits           Correct!              Wrong (Ward)
  moons by vertical       Finds crescents       better with
  boundary                and noise             single linkage

RULE: K-Means for blobs, DBSCAN for arbitrary shapes + noise detection
```

---

## CLOSING SESSION 2 (10 min)

```
DBSCAN — COMPLETE:
  Parameters: eps (radius) and min_samples (density threshold)
  No need to specify k!
  Handles any cluster shape
  Identifies noise automatically (label = -1)
  Choose eps: k-distance graph elbow method

  Use DBSCAN when:
  → Non-spherical clusters
  → Unknown number of clusters
  → Outlier/noise detection matters
  → Geographic data (dense city centers vs sparse rural areas)
```

---

## INSTRUCTOR TIPS

**"What if DBSCAN gives me one giant cluster?"**
> *"eps is too large. Every point is within eps of another. Reduce eps.*
> *Use the k-distance graph — the elbow tells you the right eps."*

**"What if DBSCAN gives me all noise?"**
> *"eps is too small OR min_samples too high.*
> *Try increasing eps slightly or reducing min_samples to 3.*
> *Rule: min_samples = 2 × n_features is a good starting point."*

**"What about HDBSCAN?"**
> *"Hierarchical DBSCAN — handles varying density clusters.*
> *Standard DBSCAN struggles when clusters have different densities.*
> *HDBSCAN adapts eps automatically per region.*
> *pip install hdbscan — worth knowing for advanced work."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — shape problem          10 min
├── Three types of points            25 min
├── Algorithm walkthrough            20 min
└── Close                             5 min  (+ 30 min buffer)

SESSION 2  (90 min)
├── Opening                           5 min
├── From scratch DBSCAN             25 min
├── eps selection (k-distance)      20 min
├── Algorithm comparison            20 min
└── Close                           10 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 07*
