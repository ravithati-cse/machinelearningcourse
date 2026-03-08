# MLForBeginners — Instructor Guide
## Part 2 · Module 07: K-Nearest Neighbors Classifier
### Two-Session Teaching Script

> **Prerequisites:** Part 2 Modules 01–06 complete. They know sigmoid, log-loss,
> confusion matrix, decision boundaries, and logistic regression from scratch.
> **Payoff today:** They learn an algorithm that needs NO training phase —
> and understand exactly when this is powerful and when it falls apart.

---

# SESSION 1 (~90 min)
## "No training, just memory — the lazy learner that just works"

## Before They Arrive
- Terminal open in `classification_algorithms/algorithms/`
- Whiteboard ready, draw a 2D scatter plot with two clear clusters (circles and squares)
- Have 5 colored dots ready to mark "the new point" and its neighbors

---

## OPENING (10 min)

> *"Last module, logistic regression spent hundreds of iterations adjusting
> weights with gradient descent. All that work just to learn beta coefficients.*
>
> *Today's algorithm does zero training. Literally none.*
>
> *I'll give it 10,000 training examples, and it won't compute a single weight,
> coefficient, or parameter.*
>
> *Instead, when you give it a new point to classify, it just asks:
> who are your nearest neighbors? And it takes a majority vote.*
>
> *It's called K-Nearest Neighbors. It's deceptively simple — and it works
> surprisingly well in many real-world situations."*

Draw on board:
```
NEW PATIENT: age=45, tumor size=2.3cm
             Is this benign or malignant?

KNN SAYS:
  "Find the 5 most similar patients in our records.
   4 of them had benign tumors.
   1 had malignant.
   Vote: BENIGN (4 to 1)"
```

> *"That's it. That's K-Nearest Neighbors.
> You are the average of your 5 closest neighbors."*

---

## SECTION 1: The Intuition — Similarity in Feature Space (20 min)

> *"Here's what 'nearest' means in machine learning."*

Draw on board:
```
FEATURE SPACE (2D example):
  Feature 1 = Age
  Feature 2 = Tumor Size

         Tumor Size (cm)
              │
          5   │    ×        ×
              │  ×    ×
          3   │    ×   ?    ○    ○
              │  ×      ○    ○
          1   │         ○  ○
              └──────────────────── Age
                  30   45   60

  × = malignant  ○ = benign  ? = new patient

  The 5 nearest neighbors to ? are mostly ○ → predict BENIGN
```

> *"'Nearest' means closest by Euclidean distance.*
>
> *The same distance formula you learned in geometry:*"

Write on board:
```
EUCLIDEAN DISTANCE:

d(A, B) = sqrt( (x₁_A - x₁_B)² + (x₂_A - x₂_B)² )

EXAMPLE:
  New patient:      age=45, tumor=2.3cm
  Existing patient: age=42, tumor=2.5cm

  d = sqrt( (45-42)² + (2.3-2.5)² )
    = sqrt( 9 + 0.04 )
    = sqrt( 9.04 )
    ≈ 3.01

  For 10,000 training patients — compute ALL distances,
  sort them, take the K smallest.
```

> *"This is why KNN is called a 'lazy learner' — it defers ALL computation
> to prediction time. Training? Just store the data.*
>
> *Prediction? Compute distances on the fly.*
>
> *The cost is reversed from logistic regression: training is instant,
> but prediction is expensive on large datasets."*

**Ask the room:** *"If KNN takes distance to EVERY training point at prediction time,
and we have 10 million patients — how many distance calculations does each prediction need?"*

Answer: 10 million. That's slow. We'll come back to this.

---

## SECTION 2: The K Parameter — How Many Neighbors? (20 min)

> *"The K in K-Nearest Neighbors is your main tuning knob.*
>
> *Let's see what happens at the extremes."*

Draw on board:
```
K = 1  ("Nearest neighbor"):
  Always labels as whatever the single closest point is.
  Very sensitive to noise.
  Perfect on training data (every point is its own neighbor).
  → OVERFITTING

K = N  (all training points):
  Always predicts the majority class.
  Ignores the test point completely.
  → UNDERFITTING

The sweet spot:
  Small K → flexible, complex boundary → overfits
  Large K → smooth, simple boundary  → underfits
  Optimal K → found by cross-validation
```

Illustrate with the boundary visualization:
```
K=1:   wiggly, irregular boundary — memorizes noise
K=5:   smoother boundary — generalizes better
K=20:  very smooth — may miss real structure
K=100: nearly a straight line — too simple
```

> *"This is the bias-variance tradeoff again — just like choosing model
> complexity in regression.*
>
> *Small K = low bias, high variance (overfits)*
> *Large K = high bias, low variance (underfits)*
>
> *In practice: start with K=5 or K=7. Use odd numbers to avoid ties.
> Then try K=1,3,5,7,9,15,20 and pick by cross-validation accuracy."*

**Ask the room:** *"Why use odd K values?"*

Answer: To avoid ties in the majority vote with binary classification.

---

## SECTION 3: Distance Metrics and Feature Scaling (15 min)

> *"One practical issue that trips people up: feature scaling.*
>
> *If one feature is measured in thousands (income: $50,000)
> and another in small numbers (age: 35), which feature dominates
> the Euclidean distance?"*

Work it out:
```
Person A: income=$50,000  age=30
Person B: income=$50,100  age=60

d = sqrt( (50000-50100)² + (30-60)² )
  = sqrt( 10000 + 900 )
  = sqrt( 10900 )
  ≈ 104.4

Income difference: $100  (30 year age gap... irrelevant!)
```

> *"Age is completely drowned out by income.*
>
> *KNN is blind to what the features represent — it only sees numbers.*
>
> *Solution: standardize. Make every feature have mean=0, std=1.*"

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Now income and age contribute equally to distance
```

> *"This is critical for KNN. Logistic regression is more robust to this.*
> *But for any distance-based algorithm — KNN, SVM, clustering —
> always scale your features first."*

---

## SECTION 4: The Curse of Dimensionality — A Warning (10 min)

> *"Here's the most important limitation of KNN.*
>
> *In 2 dimensions, 'nearest neighbors' makes intuitive sense.*
> *In 100 dimensions, everything is far from everything else."*

Draw on board:
```
CURSE OF DIMENSIONALITY:

1D: Need ~10 neighbors, they're nearby
2D: Need ~100 neighbors to cover same fraction of space
3D: Need ~1000
...
100D: Need 10^100 neighbors — more than atoms in the universe!

Result in high dimensions:
  - All points become roughly equidistant
  - "Nearest" neighbor is almost as far as farthest neighbor
  - KNN loses its meaning
```

> *"This is why KNN shines on low-dimensional data (2-20 features)
> but struggles when you have hundreds of features.*
>
> *Feature selection or PCA (dimensionality reduction) can help.*
> *But KNN is generally not your first choice for text data with 10,000 features."*

---

## CLOSING SESSION 1 (15 min)

Board summary:
```
KNN KEY FACTS:
  Training:    O(1) — just store the data
  Prediction:  O(n) — compute distance to every training point
  K too small: overfits
  K too large: underfits
  Requires:    feature scaling (StandardScaler)
  Struggles:   high dimensions, large datasets, imbalanced data

THE ALGORITHM:
  1. Compute distances from new point to ALL training points
  2. Sort by distance
  3. Take K closest
  4. Majority vote → prediction
```

**Homework:** Draw a 2D feature space with 10 training points (5 circles, 5 squares).
Mark a new point. Manually compute which point is nearest (by Euclidean distance).
What would K=1 and K=5 predict?

---

# SESSION 2 (~90 min)
## "From scratch code, sklearn, and knowing when to use KNN"

## OPENING (10 min)

> *"Last session we built the full conceptual picture.*
>
> *Today we write KNN from scratch — and you'll see it's actually
> the most readable ML algorithm we've implemented yet.*
>
> *Then we benchmark it against logistic regression from Module 06
> and discuss when you'd choose one over the other."*

---

## SECTION 1: Implement KNN From Scratch (25 min)

Code together:

```python
import numpy as np
from collections import Counter

class KNNClassifierScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # "Training" — just store the data
        self.X_train = X
        self.y_train = y
        print(f"KNN 'trained' — stored {len(X)} examples.")

    def _euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    def _predict_one(self, x):
        # Compute distance to all training points
        distances = [self._euclidean_distance(x, x_train)
                     for x_train in self.X_train]

        # Sort and get K closest indices
        k_indices = np.argsort(distances)[:self.k]

        # Get their labels
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority vote
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])
```

> *"Look at how clean this is.*
>
> *fit(): three lines. Just store data.*
> *predict_one(): four steps. Distance, sort, slice, vote.*
>
> *No gradient descent. No loss function. No epochs.*
>
> *Run it — and notice how slow it gets with large datasets.
> That slowness is real and it's the main production limitation."*

---

## SECTION 2: Test and Visualize (15 min)

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate data
X, y = make_classification(n_samples=500, n_features=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

# Scale! (critical for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# Try different K values
for k in [1, 3, 5, 7, 15, 25]:
    knn = KNNClassifierScratch(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"K={k:2d}: accuracy = {acc:.1%}")
```

> *"Run it. Watch accuracy peak at some K and then drop.*
> *That peak is your bias-variance sweet spot for this dataset."*

---

## SECTION 3: sklearn KNN in Production (10 min)

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import time

# Much faster than our scratch version
model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start

start = time.time()
y_pred = model.predict(X_test)
predict_time = time.time() - start

print(f"Train time:   {train_time:.4f}s")
print(f"Predict time: {predict_time:.4f}s")
print(classification_report(y_test, y_pred))
```

> *"sklearn uses KD-trees or ball trees to speed up the search.*
> *Instead of checking all N training points, it only checks nearby ones.*
> *This makes prediction O(log N) instead of O(N) in many cases."*

---

## SECTION 4: KNN vs Logistic Regression — When to Use What (20 min)

Write on board — work through this together:
```
                    KNN                 Logistic Regression
─────────────────────────────────────────────────────────────
Training time       Instant             Minutes to hours
Prediction time     Slow (O(n))         Fast (O(features))
Works with n=100    Yes                 Barely (needs data)
Works with n=1M     No (too slow)       Yes
High dimensions     No                  Yes
Non-linear boundary Yes (naturally)     Only with eng features
Interpretable       Somewhat            Very (coefficients)
Needs scaling       YES (critical)      Helpful but not critical
Handles missing val No                  Some imputation possible
```

> *"Here's the practitioner's heuristic:*
>
> *Use KNN when: you have fewer than ~10,000 training examples,
> 2-30 features, and you want a quick-to-implement baseline.*
>
> *Use Logistic Regression when: you need interpretable predictions,
> large datasets, speed at prediction time, or regulated decisions
> where you must explain the output.*
>
> *In industry, KNN is often a useful baseline — a quick sanity check.
> But it's rarely the final production model for large systems."*

**Ask the room:** *"You're building a recommendation system for a streaming service
with 50 million users and 500 features. KNN or logistic regression?"*

Answer: Neither in this raw form. But logistic regression scales; KNN would be
catastrophically slow. This motivates why we'll later learn tree-based methods.

---

## SECTION 5: Real Application — Anomaly Detection (10 min)

> *"One place KNN genuinely shines: anomaly detection.*
>
> *If a new point's K nearest neighbors are all very far away —
> that point is probably anomalous. It doesn't belong to any cluster.*
>
> *Credit card fraud detection, network intrusion detection —
> KNN-based outlier scoring works really well for these."*

Quick demo:
```python
# KNN-based anomaly score = distance to K-th neighbor
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)
anomaly_score = distances[:, -1]  # Distance to 5th nearest neighbor

# High score = potentially anomalous
threshold = np.percentile(anomaly_score, 95)
anomalies = X_scaled[anomaly_score > threshold]
print(f"Found {len(anomalies)} potential anomalies (top 5%)")
```

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
KNN SUMMARY:
  "Find K most similar training examples. Take majority vote."

  Strength:  Simple, no assumptions, naturally non-linear
  Weakness:  Slow at prediction, breaks in high dimensions

  ALWAYS:
    Scale features with StandardScaler
    Try multiple K values
    Compare against logistic regression baseline

WHAT'S NEXT:
  Module 08: Decision Trees — a completely different approach
  that splits feature space with if-then rules.
  Much faster at prediction. Still non-linear. Very interpretable.
```

**Homework:**
```python
# Using sklearn's KNeighborsClassifier on the Iris dataset:
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# 1. Scale the features
# 2. Try K = 1, 3, 5, 7, 10, 15, 20 with cross-validation (cv=5)
# 3. Plot K vs validation accuracy
# 4. At what K does accuracy peak?
# 5. Compare to a LogisticRegression on the same data.
#    Which wins? By how much?
```

---

## INSTRUCTOR TIPS

**"Why does K=1 always get 100% training accuracy?"**
> *"Because every point is its own nearest neighbor!*
> *The model memorizes training data perfectly — classic overfitting.*
> *On new data it's much worse because it overfit to noise."*

**"Can KNN output probabilities?"**
> *"Yes! Instead of majority vote, count fraction: if 3 out of 5
> neighbors are spam, output P(spam) = 0.6.*
> *sklearn's predict_proba() does exactly this.*
> *These probabilities are less calibrated than logistic regression's,
> but still useful."*

**"What about Manhattan distance?"**
> *"Manhattan (L1) = sum of absolute differences instead of Euclidean (L2).*
> *More robust to outliers. Works better for high-dimensional sparse data.*
> *sklearn supports many metrics: 'euclidean', 'manhattan', 'minkowski', etc.*
> *Try them — they can make a real difference."*

**"My KNN is really slow. What do I do?"**
> *"First: use sklearn's built-in KD-tree or ball tree (algorithm='kd_tree').*
> *Second: reduce training set size with random sampling.*
> *Third: reduce dimensions with PCA.*
> *Fourth: honestly consider a different algorithm (decision trees, random forests)."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                   10 min
├── Intuition and feature space    20 min
├── The K parameter                20 min
├── Feature scaling + why          15 min
├── Curse of dimensionality        10 min
└── Close + homework               15 min

SESSION 2  (90 min)
├── Opening bridge                 10 min
├── Implement from scratch         25 min
├── Test and visualize             15 min
├── sklearn production version     10 min
├── KNN vs Logistic Regression     20 min
├── Anomaly detection application  10 min
└── Close + homework               10 min
```

---
*MLForBeginners · Part 2: Classification · Module 07*
