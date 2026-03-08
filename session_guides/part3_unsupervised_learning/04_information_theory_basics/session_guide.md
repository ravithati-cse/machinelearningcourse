# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 04: Information Theory Basics
### Two-Session Teaching Script

> **Entropy: how do you measure "surprise" mathematically?**
> Information theory gives us tools to evaluate clustering quality,
> measure how informative a feature is, and understand model uncertainty.

---

# SESSION 1 (~90 min)
## "Entropy — measuring uncertainty and surprise"

## Before They Arrive
- Terminal open in `unsupervised_learning/math_foundations/`
- Think of a coin flip vs a loaded die as openers

---

## OPENING (10 min)

> *"Flip a fair coin. Heads or tails?*
> *You're maximally uncertain — 50/50.*
>
> *Now flip a weighted coin: 99% heads.*
> *You're barely uncertain — you know it'll be heads.*
>
> *When it comes up tails, that's SURPRISING.*
> *When heads: expected, not surprising.*
>
> *Information theory quantifies surprise mathematically.*
> *Claude Shannon invented this in 1948 — before modern computers.*
> *It underlies compression (ZIP files), communication, and machine learning.*
>
> *For us: entropy helps evaluate how 'spread out' clusters are,*
> *how informative a feature is, and whether a model is overconfident."*

Write on board:
```
ENTROPY = average surprise of a distribution

High entropy = high uncertainty = uniform distribution
Low entropy  = low uncertainty  = peaked distribution

Fair coin:      H = 1 bit (maximum uncertainty)
Loaded coin:    H < 1 bit (less uncertain)
Certain outcome: H = 0 bits (no uncertainty)
```

---

## SECTION 1: The Formula (20 min)

Write on board:
```
Shannon Entropy:
  H(X) = -Σ p(x) × log₂(p(x))

For fair coin (p=0.5 for each):
  H = -(0.5 × log₂(0.5) + 0.5 × log₂(0.5))
  H = -(0.5 × (-1) + 0.5 × (-1))
  H = 1.0 bit

For loaded coin (p=0.9, p=0.1):
  H = -(0.9 × log₂(0.9) + 0.1 × log₂(0.1))
  H = -(0.9 × (-0.15) + 0.1 × (-3.32))
  H = -(−0.135 + −0.332) = 0.47 bits

For certain outcome (p=1.0, p=0.0):
  H = -(1.0 × log₂(1) + 0 × log₂(0))
  H = -(0 + 0) = 0 bits
  (convention: 0 × log(0) = 0)
```

> *"The -log₂(p(x)) term is the 'surprise' of event x.*
> *Rare events (small p) → large surprise.*
> *Common events (large p) → small surprise.*
> *Entropy = the AVERAGE surprise across all possible events."*

```bash
python3 04_information_theory_basics.py
```

---

## SECTION 2: Entropy in Clustering (20 min)

> *"For clustering: entropy measures how 'pure' a cluster is.*
>
> *A cluster with all class A: entropy = 0 (perfectly pure)*
> *A cluster with 50% A, 50% B: entropy = 1 (maximum impurity)*
>
> *Good clustering → low entropy within clusters."*

Write:
```
CLUSTER EVALUATION WITH ENTROPY:

Cluster 1: [A, A, A, A, A] → p(A)=1.0 → H=0 (pure ✓)
Cluster 2: [B, B, B, B, B] → p(B)=1.0 → H=0 (pure ✓)

vs

Cluster 1: [A, B, A, B, A] → p(A)=0.6, p(B)=0.4 → H=0.97 (mixed ✗)
Cluster 2: [B, A, B, A, B] → p(A)=0.4, p(B)=0.6 → H=0.97 (mixed ✗)

Lower entropy within clusters = better clustering
(when you have labels to check against)
```

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Entropy: average surprise of a probability distribution
  H = -Σ p(x) log₂ p(x)
  High entropy → high uncertainty (uniform)
  Low entropy  → low uncertainty (peaked)
  For clustering: low within-cluster entropy = purer clusters
```

---

# SESSION 2 (~90 min)
## "Silhouette score — evaluating clusters without labels"

## OPENING (5 min)

> *"Session 1: entropy with labels.*
> *But in true unsupervised learning, we DON'T have labels.*
> *How do you evaluate clustering quality when you can't check?*
> *That's the silhouette score — today's main topic."*

---

## SECTION 1: The Silhouette Score (30 min)

Write on board:
```
SILHOUETTE SCORE — evaluating clustering without labels

For each point i:
  a(i) = mean distance to other points in SAME cluster
         (how tight is the cluster around i?)

  b(i) = mean distance to points in NEAREST OTHER cluster
         (how far is the next cluster?)

  s(i) = (b(i) - a(i)) / max(a(i), b(i))

s(i) ranges from -1 to +1:
  +1: well inside its own cluster, far from others ✓ perfect
   0: on the boundary between two clusters
  -1: probably in the wrong cluster ✗

Mean silhouette score over all points = cluster quality
```

> *"Think of it as: 'Is this point closer to its own cluster or the neighboring one?'*
> *A good clustering: every point is deep inside its own cluster (s ≈ +1).*
> *A bad clustering: points hover between clusters (s ≈ 0 or negative)."*

---

## SECTION 2: Using Silhouette to Choose K (20 min)

> *"K-Means needs you to choose k — the number of clusters.*
> *Elbow method: look at inertia (within-cluster sum of squares).*
> *Silhouette method: look at silhouette score.*
>
> *Use BOTH. They often agree. When they disagree,*
> *prefer the silhouette — it's more principled."*

```python
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

scores = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)
    print(f"k={k}: silhouette={score:.3f}")

best_k = np.argmax(scores) + 2
print(f"Best k: {best_k}")
```

---

## SECTION 3: Other Evaluation Metrics (15 min)

```
CLUSTERING EVALUATION TOOLKIT:

WITHOUT LABELS (unsupervised):
  Silhouette score:    -1 to +1, higher = better
  Inertia/WCSS:        lower = tighter clusters
  Calinski-Harabasz:   higher = better separated clusters
  Davies-Bouldin:      lower = better (ratio of within/between scatter)

WITH LABELS (supervised check):
  Adjusted Rand Index:      0=random, 1=perfect
  Normalized Mutual Info:   0=no info, 1=perfect
  Homogeneity/Completeness: precision/recall for clusters
```

> *"When you have ground truth labels even for just a sample,*
> *use Adjusted Rand Index to validate.*
> *Most of the time you don't — use silhouette."*

---

## CLOSING SESSION 2 (10 min)

```
INFORMATION THEORY BASICS — COMPLETE:
  Entropy: H = -Σ p log p (uncertainty measurement)
  Silhouette score: (b-a)/max(a,b) (cluster quality without labels)
  Use silhouette to choose k in K-Means

  Next: K-Means from scratch — you have all the tools you need.
```

---

## INSTRUCTOR TIPS

**"Why use log base 2 instead of natural log?"**
> *"Base 2 → units are 'bits' (information theory origin in binary transmission).*
> *Base e → units are 'nats'.*
> *For ML purposes: either works, the choice just scales the result.*
> *sklearn uses natural log (nats) internally.*
> *Information theory textbooks use log₂ (bits)."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — surprise and uncertainty   10 min
├── Entropy formula + worked examples    20 min
├── Entropy in clustering               20 min
└── Close                                5 min  (+ 35 min buffer)

SESSION 2  (90 min)
├── Opening                              5 min
├── Silhouette score                    30 min
├── Using silhouette to choose k        20 min
├── Other evaluation metrics            15 min
└── Close                              10 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 04*
