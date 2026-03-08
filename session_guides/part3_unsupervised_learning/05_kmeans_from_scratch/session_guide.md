# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 05: K-Means from Scratch
### Two-Session Teaching Script

> **The most famous unsupervised algorithm.**
> Simple enough to implement in 30 lines. Powerful enough to run inside Google Photos.
> Build it from scratch, then watch it discover structure no one told it about.

---

# SESSION 1 (~90 min)
## "K-Means — teaching data to sort itself"

## Before They Arrive
- Terminal open in `unsupervised_learning/algorithms/`
- Draw 3 blobs of dots on the board, unlabeled

---

## OPENING (10 min)

> *"Look at this scatter plot I drew.*
> *No labels. No colors. Just dots.*
>
> *How many groups do you see?*
> *(Everyone will say 3.)*
>
> *How did you know? You looked for dense regions.*
> *Points close together — that's a cluster.*
>
> *K-Means does exactly this — algorithmically.*
> *'Find k groups where each group's points are close to each other.'*
>
> *It's used in customer segmentation, image compression,*
> *anomaly detection, document clustering, gene expression analysis.*
> *One of the most widely deployed algorithms ever built."*

Write on board:
```
K-MEANS GOAL:
  Partition n points into k clusters
  Minimize: sum of (distance from each point to its cluster center)²

INPUT:  data points (no labels), k (number of clusters)
OUTPUT: cluster assignment for each point + k centroids
```

---

## SECTION 1: The Algorithm — 4 Steps (25 min)

Write step by step, then animate it on the board:

```
K-MEANS ALGORITHM:

Step 1: INITIALIZE
  Randomly pick k points as initial centroids

Step 2: ASSIGN
  For each point: find nearest centroid
  Assign point to that cluster

Step 3: UPDATE
  For each cluster: compute new centroid
  (centroid = mean of all assigned points)

Step 4: REPEAT
  Go back to Step 2
  Stop when centroids don't move (convergence)
```

Act it out with volunteers if energy allows:
> *"Three people come to the front — you're the centroids.*
> *Everyone else: walk toward the centroid closest to you.*
> *Now centroids: walk to the average position of your group.*
> *Repeat twice.*
> *That's K-Means."*

```bash
python3 kmeans_from_scratch.py
```

---

## SECTION 2: Building It from Scratch (30 min)

Live code together:

```python
import numpy as np

def kmeans(X, k, max_iters=100, random_state=42):
    np.random.seed(random_state)

    # Step 1: Random initialization
    idx = np.random.choice(len(X), k, replace=False)
    centroids = X[idx].copy()

    for iteration in range(max_iters):
        # Step 2: Assign each point to nearest centroid
        distances = np.array([[np.linalg.norm(x - c) for c in centroids]
                              for x in X])
        labels = np.argmin(distances, axis=1)

        # Step 3: Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0)
                                  for i in range(k)])

        # Step 4: Check convergence
        if np.allclose(centroids, new_centroids):
            print(f"Converged at iteration {iteration+1}")
            break
        centroids = new_centroids

    return labels, centroids
```

> *"This is the complete K-Means algorithm.*
> *The real sklearn version adds random restarts and K-Means++*
> *but the core logic is identical to what you just wrote."*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  K-Means: initialize → assign → update → repeat
  Objective: minimize within-cluster sum of squares (inertia)
  Convergence: when centroids stop moving
  Key parameter: k (number of clusters)
```

**Homework:** *"Think about: what happens if you choose k=1? k=n?*
*What are the failure modes of random initialization?"*

---

# SESSION 2 (~90 min)
## "Choosing k, limitations, and the sklearn version"

## OPENING (5 min)

> *"Session 1: we built K-Means.*
> *Today: the hardest practical question — how do we choose k?*
> *And: when does K-Means fail, and what do we do about it?"*

---

## SECTION 1: The Elbow Method (20 min)

Write on board:
```
INERTIA = sum of squared distances from each point to its centroid

k=1: one big cluster, high inertia
k=2: two clusters, lower inertia
k=3: three clusters, even lower
k=n: each point is its own cluster, inertia=0

But k=n is useless — we've just memorized the data.
We want: "the elbow" — where adding more clusters
         gives diminishing returns.

    Inertia
    │ *
    │   *
    │     *
    │       * ← ELBOW (optimal k)
    │         * * * *
    └─────────────────── k
```

> *"The elbow is where the curve bends.*
> *Before the elbow: each new cluster significantly reduces inertia.*
> *After the elbow: we're just overfitting to noise.*
> *Choose k at the elbow."*

---

## SECTION 2: K-Means++ Initialization (15 min)

> *"Standard K-Means picks initial centroids randomly.*
> *Problem: if two initial centroids land in the same cluster,*
> *the algorithm might never recover.*
>
> *K-Means++ fixes this with smarter initialization:*
> *1. Pick first centroid randomly.*
> *2. Pick next centroid proportional to distance² from nearest existing centroid.*
> *3. Repeat until k centroids chosen.*
>
> *Result: initial centroids spread across the data.*
> *Typically converges faster and to better solutions.*
> *sklearn uses K-Means++ by default: KMeans(init='k-means++')"*

---

## SECTION 3: K-Means Limitations (20 min)

Write:
```
K-MEANS ASSUMPTIONS (and when they break):

1. MUST SPECIFY K
   Problem: real data doesn't come with "use k=3"
   Fix: elbow + silhouette, domain knowledge

2. SPHERICAL CLUSTERS
   K-Means finds circular/spherical clusters
   Fails on: crescents, rings, elongated shapes
   Fix: DBSCAN (Module 07), Gaussian Mixture Models

3. EQUAL CLUSTER SIZE
   K-Means prefers equally sized clusters
   Fix: weighted K-Means, different algorithm

4. SENSITIVE TO OUTLIERS
   One outlier can pull a centroid far off
   Fix: K-Medoids (uses median instead of mean)

5. SENSITIVE TO SCALE
   Always normalize features first!
   Fix: StandardScaler before KMeans
```

Show side by side: blobs (K-Means wins) vs moons (K-Means fails).

---

## SECTION 4: sklearn K-Means (10 min)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Always normalize first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with K-Means++ init and 10 random restarts
km = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)

print("Inertia:", km.inertia_)
print("Iterations:", km.n_iter_)
print("Centroids:", km.cluster_centers_)
```

---

## CLOSING SESSION 2 (10 min)

```
K-MEANS — COMPLETE:
  Algorithm: 4-step iterative assign-update loop
  Choose k: elbow method + silhouette score (use both)
  Initialization: K-Means++ (sklearn default)
  Limitations: needs k, assumes spherical, sensitive to scale

  Always: StandardScaler before KMeans
  Always: try multiple k values and compare
  Always: multiple random restarts (n_init=10)
```

---

## INSTRUCTOR TIPS

**"Can K-Means cluster text?"**
> *"Yes — TF-IDF vectors + cosine distance K-Means.*
> *sklearn's KMeans uses Euclidean by default.*
> *For text: use SphericalKMeans (spherical K-Means) or normalize vectors to unit length first.*
> *Cosine similarity becomes Euclidean on unit-length vectors."*

**"What's a Gaussian Mixture Model?"**
> *"K-Means: hard assignment — each point belongs to exactly one cluster.*
> *GMM: soft assignment — each point has a PROBABILITY of belonging to each cluster.*
> *Also handles non-spherical clusters (ellipsoidal).*
> *sklearn: GaussianMixture(n_components=3).*
> *More powerful but harder to interpret."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — the visual              10 min
├── Algorithm — 4 steps              25 min
├── Live coding from scratch         30 min
└── Close                             5 min  (+ 20 min buffer)

SESSION 2  (90 min)
├── Opening                           5 min
├── Elbow method                     20 min
├── K-Means++ initialization         15 min
├── Limitations                      20 min
├── sklearn version                  10 min
└── Close                            10 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 05*
