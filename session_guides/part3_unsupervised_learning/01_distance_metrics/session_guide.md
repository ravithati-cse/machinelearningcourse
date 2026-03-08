# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 01: Distance Metrics
### Two-Session Teaching Script

> **The first question in unsupervised learning: "How similar are these two things?"**
> Before you can cluster, you need to measure closeness.
> Distance metrics are the rulers of machine learning.

---

# SESSION 1 (~90 min)
## "How do you measure similarity? — The geometry of ML"

## Before They Arrive
- Terminal open in `unsupervised_learning/math_foundations/`
- Draw a 2D grid on the board with 3-4 labeled points

---

## OPENING (10 min)

> *"Welcome to Part 3 — Unsupervised Learning.*
>
> *Parts 1 and 2 were supervised: you had labels.*
> *Someone told you: this email is spam, this house costs $300K.*
>
> *Unsupervised learning asks: what if you have NO labels?*
> *Can the data organize itself? Can you find hidden structure?*
>
> *The answer is yes — and the entire foundation is one question:*
> 'How far apart are these two points?'*
>
> *That's today. Distance metrics.*
> *Every clustering algorithm, every similarity search, every recommendation system*
> *starts right here."*

Write on board:
```
SUPERVISED:   Data + Labels → Learn mapping
UNSUPERVISED: Data only     → Find structure

The key question:
  "Are these two things similar or different?"
  → You need a way to MEASURE similarity/distance
```

---

## SECTION 1: Euclidean Distance — The Straight Line (20 min)

> *"The distance you learned in middle school."*

Write on board:
```
Two points: A = (1, 2),  B = (4, 6)

Euclidean distance:
  d = √((4-1)² + (6-2)²)
  d = √(9 + 16)
  d = √25 = 5

"As the crow flies" — straight line through space
```

Work through it manually. Then generalize:

```
N dimensions: d = √(Σᵢ (aᵢ - bᵢ)²)

Two emails with 1000-word TF-IDF features:
  d = √((email1[word1] - email2[word1])² + ... + 1000 terms)

Works in any number of dimensions.
```

> *"Euclidean distance assumes all dimensions are equally important.*
> *And it assumes dimensions are measured in the same units.*
> *If one feature is 'age' (0-100) and another is 'income' (0-100000),*
> *income dominates. This is why we normalize."*

```bash
python3 01_distance_metrics.py
```

---

## SECTION 2: Manhattan Distance — City Blocks (15 min)

> *"Imagine you're in Manhattan. You can't walk diagonally through buildings.*
> *You walk along the grid: right, right, up, up.*
> *That's Manhattan distance."*

Write on board:
```
Manhattan (L1) distance:
  d = Σᵢ |aᵢ - bᵢ|

A = (1, 2),  B = (4, 6):
  d = |4-1| + |6-2| = 3 + 4 = 7

vs Euclidean = 5

When to use Manhattan:
  → Features are independent categories
  → You want to downweight large individual differences
  → High-dimensional data (less affected by the "curse")
```

---

## SECTION 3: Cosine Similarity — Direction, Not Distance (20 min)

> *"This one surprises people. We don't care how FAR apart two vectors are.*
> *We care about the ANGLE between them."*

Write on board:
```
cosine_similarity = (A · B) / (|A| × |B|)

Example — two documents:
  Doc A: "machine learning is great"   → [3, 1, 1, 1, 0, ...]
  Doc B: "machine learning is amazing" → [3, 1, 1, 0, 1, ...]

They're about the same topic. Cosine similarity ≈ 0.95

Short vs Long version of same document:
  Doc C: "machine learning" → [1, 1, 0, ...]
  Doc D: "machine learning machine learning machine" → [3, 2, 0, ...]

C and D have different magnitudes but same DIRECTION.
Cosine similarity = 1.0 (identical topic)
Euclidean distance would say they're different!
```

> *"For text: cosine similarity. Always.*
> *We don't care if one doc is 500 words and one is 5000 words.*
> *We care if they're about the same topic.*
> *Direction = topic. Magnitude = length."*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Euclidean: straight-line distance (geometry)
  Manhattan: sum of absolute differences (city blocks)
  Cosine: angle between vectors (text similarity)

  The right metric depends on your data and what "similar" means.
```

**Homework:** *"Think of one real-world problem where each metric makes sense.*
*Euclidean, Manhattan, Cosine — one use case each."*

---

# SESSION 2 (~90 min)
## "Choosing the right metric — and visualizing similarity"

## OPENING (5 min)

> *"Today we see what different distance metrics do to real data.*
> *Same dataset, different ruler = different clusters.*
> *That's how much the metric matters."*

---

## SECTION 1: Pairwise Distance Heatmaps (20 min)

> *"A pairwise distance matrix shows the distance between EVERY pair of points.*
> *It's a window into the structure of your data."*

Watch the heatmap visualization:
- Dark = close / similar
- Light = far / different

> *"Look at the block structure. Points within a natural cluster*
> *show dark regions. That's what clustering algorithms exploit.*
> *You can almost SEE the clusters just from the distance matrix."*

---

## SECTION 2: The Curse of Dimensionality (20 min)

> *"Here's a problem nobody warns you about early enough."*

Write on board:
```
THE CURSE OF DIMENSIONALITY

In 2D: distances range from 0 to √2 (normalized)
In 100D: distances CONCENTRATE — everything is "far"

Intuition:
  In 1D: random points scattered 0 to 1, some are close
  In 100D: random points in a 100-dimensional box
            Almost all pairs have similar distance!
            The "signal" of closeness disappears.

Result:
  Euclidean distance becomes meaningless in high dimensions
  K-Means struggles. Cosine similarity holds up better.

Solution:
  → Dimensionality reduction FIRST (PCA, Module 08)
  → Then cluster in lower-dimensional space
```

---

## SECTION 3: When to Use What (20 min)

Write the guide:
```
DISTANCE METRIC DECISION GUIDE:

EUCLIDEAN:
  ✓ Spatial data (GPS coordinates, pixel positions)
  ✓ Continuous features measured in same units
  ✓ After normalization
  ✗ Text data   ✗ High dimensions   ✗ Mixed units

MANHATTAN (L1):
  ✓ Features measured independently
  ✓ Sparse data (many zeros)
  ✓ When outliers should have less influence
  ✗ When diagonal distance has meaning

COSINE:
  ✓ Text/document similarity (ALWAYS)
  ✓ High-dimensional sparse vectors
  ✓ When magnitude shouldn't matter
  ✗ When zero vectors are possible
  ✗ When actual magnitude is meaningful

OTHERS:
  Jaccard:   Set similarity (used in recommendation systems)
  Hamming:   Binary strings, DNA sequences
  Minkowski: Generalization of Euclidean and Manhattan (p=2 and p=1)
```

---

## CLOSING SESSION 2 (10 min)

> *"Every algorithm we build this part depends on distance.*
> *K-Means: distance to centroids.*
> *DBSCAN: distance to neighbors.*
> *Hierarchical: distance between clusters.*
>
> *You now have the foundation.*
> *Module 02: we go deeper on the statistics that power PCA."*

---

## INSTRUCTOR TIPS

**"Does the distance metric matter that much?"**
> *"Run the same dataset through K-Means with Euclidean vs Cosine.*
> *On text data: wildly different clusters.*
> *On normalized continuous data: often similar.*
> *Always worth experimenting. The metric is a hyperparameter."*

**"What about Mahalanobis distance?"**
> *"It accounts for correlations between features — essentially normalizes by*
> *the covariance matrix. Very powerful for anomaly detection.*
> *We'll see a version of it in Module 11 (Anomaly Detection).*
> *For now: Euclidean → Manhattan → Cosine is the decision tree."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — supervised vs unsupervised   10 min
├── Euclidean distance                     20 min
├── Manhattan distance                     15 min
├── Cosine similarity                      20 min
└── Close + homework                        5 min  (+ 20 min buffer)

SESSION 2  (90 min)
├── Opening                                 5 min
├── Pairwise heatmaps                      20 min
├── Curse of dimensionality                20 min
├── Decision guide                         20 min
└── Close                                  10 min  (+ 15 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 01*
