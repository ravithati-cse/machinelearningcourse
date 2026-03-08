# MLForBeginners — Instructor Guide
## Part 2 · Module 08: Decision Trees
### Two-Session Teaching Script

> **Prerequisites:** Part 2 Modules 01–07 complete. They know sigmoid, log-loss,
> confusion matrix, decision boundaries, logistic regression, and KNN.
> **Payoff today:** They build an algorithm that makes decisions like a human —
> a series of yes/no questions — and understand the math behind which questions to ask.

---

# SESSION 1 (~90 min)
## "Playing 20 Questions with data — Gini impurity and tree growth"

## Before They Arrive
- Terminal open in `classification_algorithms/algorithms/`
- Whiteboard ready; draw a simple tree diagram: boxes connected by branches with yes/no labels
- Optionally: print a small table of 10 rows of patient data (age, smoker, tumor_size, malignant)

---

## OPENING (10 min)

> *"Let me ask you something. Imagine you're a doctor.*
>
> *A patient comes in. You need to decide: is this tumor malignant or benign?*
>
> *You don't do logistic regression in your head. You ask questions.*
>
> *'Is the tumor larger than 3cm?' Yes? 'Has the patient smoked for 10+ years?'*
>
> *That chain of yes/no questions — that's a decision tree.*
>
> *And today we're going to answer the hardest part: how does the algorithm
> decide which question to ask first?"*

Draw on board:
```
              Is tumor > 3cm?
             /               \
           YES                NO
           /                   \
   Smoked > 10 years?        Older than 60?
    /          \               /          \
  YES           NO           YES           NO
  |             |             |             |
MALIGNANT    BENIGN       MALIGNANT      BENIGN
```

> *"This tree makes predictions in milliseconds.*
> *It's also completely interpretable — you can trace every decision.*
> *A doctor can audit it. A judge can challenge it.*
>
> *Today we learn to grow trees like this automatically from data."*

---

## SECTION 1: Gini Impurity — Measuring Messiness (25 min)

> *"The core question: at each split, which feature should we use?*
>
> *We need a way to measure how 'pure' a group is.*
> *If all examples in a group are the same class — that's pure.*
> *If it's a mix of classes — that's impure.*
>
> *The measure we use is called Gini Impurity."*

Write on board:
```
GINI IMPURITY:
  Gini = 1 - Σ p_i²

  where p_i = fraction of class i in the group

EXAMPLE 1: Pure group — all malignant
  10 malignant, 0 benign
  p_malignant = 1.0,  p_benign = 0.0
  Gini = 1 - (1.0² + 0.0²) = 1 - 1 = 0   ← PERFECT PURITY

EXAMPLE 2: Worst case — perfectly mixed
  5 malignant, 5 benign
  p_malignant = 0.5,  p_benign = 0.5
  Gini = 1 - (0.5² + 0.5²) = 1 - 0.5 = 0.5  ← MAXIMUM IMPURITY

EXAMPLE 3: Mostly pure
  8 malignant, 2 benign
  p_malignant = 0.8,  p_benign = 0.2
  Gini = 1 - (0.64 + 0.04) = 1 - 0.68 = 0.32
```

> *"The lower the Gini, the purer the group.*
> *0 = perfectly pure. 0.5 = completely mixed (for binary classification)."*

**Ask the room:** *"What is the Gini impurity of a group with 9 cats and 1 dog?"*

Calculate: p_cat = 0.9, p_dog = 0.1
Gini = 1 - (0.81 + 0.01) = 1 - 0.82 = 0.18

> *"Pretty pure — only one outlier. That's a leaf we'd be happy to stop at."*

---

## SECTION 2: Information Gain — The Splitting Criterion (25 min)

> *"Gini measures a single group.*
> *But when we split, we create TWO groups: left branch and right branch.*
>
> *We need to measure: how much did this split IMPROVE the purity?*
> *That's called Information Gain."*

Write on board:
```
INFORMATION GAIN:
  IG = Gini(parent) - [weighted average of children's Gini]

  IG = Gini(parent) - [ (n_left/n_total) × Gini(left)
                       + (n_right/n_total) × Gini(right) ]

EXAMPLE: Splitting 10 patients on "tumor > 3cm"

Before split (parent):  6 malignant, 4 benign
  Gini_parent = 1 - (0.6² + 0.4²) = 1 - 0.52 = 0.48

After split:
  Left (tumor > 3cm):  5 malignant, 1 benign  → n=6
    Gini_left = 1 - ((5/6)² + (1/6)²) = 1 - (0.694 + 0.028) = 0.278

  Right (tumor ≤ 3cm): 1 malignant, 3 benign  → n=4
    Gini_right = 1 - ((1/4)² + (3/4)²) = 1 - (0.0625 + 0.5625) = 0.375

  Weighted children:
    (6/10) × 0.278 + (4/10) × 0.375 = 0.167 + 0.150 = 0.317

  IG = 0.48 - 0.317 = 0.163
```

> *"So this split gained 0.163 in purity.*
>
> *The tree algorithm tries EVERY possible feature and EVERY possible threshold.*
> *It picks the split with the HIGHEST information gain.*
>
> *Then it recurses into each child and does the same thing again.*
> *That's the entire tree-building algorithm."*

Draw the algorithm:
```
TREE GROWING ALGORITHM (ID3 / CART):
  function grow_tree(data):
    if data is pure enough → return leaf node
    if max_depth reached → return leaf node

    best_feature, best_threshold = find_best_split(data)
    left_data  = data[feature ≤ threshold]
    right_data = data[feature > threshold]

    return Node(
      left  = grow_tree(left_data),
      right = grow_tree(right_data),
      split = (best_feature, best_threshold)
    )
```

> *"It's recursive. The tree builds itself by asking:
> 'what's the best question here?' — and then doing the same for each branch.*
>
> *This is called a greedy algorithm — it doesn't plan ahead.
> At each step it makes locally the best split.*
> *This means we might not get the globally optimal tree —
> but in practice, greedy works very well."*

---

## SECTION 3: Overfitting and Pruning (15 min)

> *"What happens if we let the tree grow without stopping?"*

Draw on board:
```
DEEP TREE (no limit):
  The tree can always perfectly classify training data.
  Just grow until every leaf has exactly one sample.

  Training accuracy:   100%
  Test accuracy:       Much worse

  WHY: The tree memorized noise and outliers in training data.
  On new data, those outliers don't appear — so predictions fail.
```

> *"This is overfitting again. We've seen it before — remember KNN with K=1?*
> *Same problem: the model memorized training data instead of learning patterns.*
>
> *The solution: stop the tree from growing too deep.*
> *This is called pre-pruning (or stopping criteria)."*

Write on board:
```
STOPPING CRITERIA (pre-pruning):
  max_depth        — don't go deeper than D levels
  min_samples_split — don't split if fewer than N samples
  min_samples_leaf  — each leaf must have at least N samples
  min_impurity_decrease — only split if IG > threshold

Post-pruning (less common):
  Grow full tree, then cut back branches that hurt validation accuracy
```

> *"In sklearn, max_depth=3 or max_depth=5 is often a good start.*
> *We tune this with cross-validation, just like K in KNN."*

---

## CLOSING SESSION 1 (15 min)

Board summary:
```
KEY FORMULAS:
  Gini = 1 - Σ p_i²         (impurity of a group)
  IG = Gini(parent) - Σ weighted Gini(children)  (gain from split)

TREE GROWING:
  1. Find feature + threshold that maximizes IG
  2. Split data
  3. Recurse on each child
  4. Stop when: pure enough, max_depth reached, or too few samples

OVERFITTING:
  Deep tree → memorizes training data → fails on test
  Solution: max_depth, min_samples_leaf
```

**Homework:** Draw a decision tree by hand for this dataset:

| Weather | Windy | Play?  |
|---------|-------|--------|
| Sunny   | No    | Yes    |
| Sunny   | Yes   | No     |
| Rain    | No    | Yes    |
| Rain    | Yes   | No     |
| Cloudy  | No    | Yes    |
| Cloudy  | Yes   | Yes    |

Which feature should be the root? Calculate Gini impurity for both options.

---

# SESSION 2 (~90 min)
## "Code the tree, visualize it, and know when it beats other algorithms"

## OPENING (10 min)

> *"Last session: the math. Today: the code and the craft.*
>
> *We're going to implement a decision tree from scratch,
> then use sklearn to visualize a real tree.*
>
> *The visual is important — decision trees are one of the few ML algorithms
> you can literally print out and hand to a non-technical colleague."*

---

## SECTION 1: Implement Decision Tree From Scratch (25 min)

Code together:

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None,
                 left=None, right=None, value=None):
        self.feature = feature      # Which feature to split on
        self.threshold = threshold  # Split threshold
        self.left = left            # Left subtree
        self.right = right          # Right subtree
        self.value = value          # Leaf prediction (if leaf)

class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y, depth=0)

    def _gini(self, y):
        n = len(y)
        counts = Counter(y)
        return 1 - sum((count/n)**2 for count in counts.values())

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        parent_gini = self._gini(y)

        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                if left_mask.sum() == 0 or (~left_mask).sum() == 0:
                    continue

                left_gini  = self._gini(y[left_mask])
                right_gini = self._gini(y[~left_mask])
                n_left, n_right = left_mask.sum(), (~left_mask).sum()
                n = len(y)

                gain = parent_gini - (n_left/n)*left_gini - (n_right/n)*right_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        # Stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):
            # Return leaf: most common class
            most_common = Counter(y).most_common(1)[0][0]
            return Node(value=most_common)

        # Find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            return Node(value=Counter(y).most_common(1)[0][0])

        # Split and recurse
        left_mask = X[:, feature] <= threshold
        left  = self._grow_tree(X[left_mask],  y[left_mask],  depth+1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth+1)

        return Node(feature=feature, threshold=threshold,
                    left=left, right=right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
```

> *"The recursive structure matches the algorithm perfectly.*
>
> *_grow_tree calls itself on left and right subsets.*
> *_predict_one navigates the tree from root to leaf.*
>
> *This is actually how all decision tree libraries work at their core —
> sklearn's C implementation is faster but the logic is identical."*

---

## SECTION 2: sklearn Decision Tree + Visualization (20 min)

```python
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load real data
X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Try different depths
for depth in [1, 2, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc  = accuracy_score(y_test,  dt.predict(X_test))
    print(f"Depth={str(depth):4}: train={train_acc:.1%}  test={test_acc:.1%}")

# Best model — visualize
best_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
best_tree.fit(X_train, y_train)

# Print as text
print("\nTree structure:")
print(export_text(best_tree, feature_names=list(feature_names)))

# Plot the tree
plt.figure(figsize=(20, 8))
plot_tree(best_tree, feature_names=feature_names,
          class_names=['malignant', 'benign'],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree (max_depth=3)")
plt.savefig("decision_tree_visualization.png", dpi=300, bbox_inches='tight')
```

> *"Look at this output. The depth=1 tree uses one question for everything.*
> *As depth increases, training accuracy climbs — but at depth 10 (or None),
> test accuracy starts dropping.*
>
> *Find the sweet spot. In this dataset it's usually around depth 3-5.*
>
> *Now look at the visualization. You can trace every path.*
> *Each box shows: the question, the Gini impurity, the sample counts.*
> *This is a complete audit trail of every decision."*

---

## SECTION 3: Feature Importances (10 min)

```python
import pandas as pd

importances = pd.Series(
    best_tree.feature_importances_,
    index=feature_names
).sort_values(ascending=False)

print("Top 10 most important features:")
print(importances.head(10))
```

> *"Feature importance in decision trees = sum of information gain
> across all splits that used this feature, weighted by samples.*
>
> *Features that appear near the root and handle many samples
> get high importance scores.*
>
> *This is free insight. You ran the model to classify — and it
> tells you which features actually matter. In medicine, this
> can guide future data collection."*

---

## SECTION 4: When Decision Trees Win (15 min)

Draw on board:
```
DECISION TREES SHINE WHEN:
  You need interpretable decisions (medicine, law, finance)
  Mixed feature types (numerical + categorical)
  Non-linear relationships (XOR, checkerboard patterns)
  Feature importance is needed
  Quick training and fast prediction

DECISION TREES STRUGGLE WHEN:
  Data is small (overfit easily)
  Very high accuracy is needed (logistic regression, ensembles beat them)
  Decision boundary is smooth/linear (logistic regression is better)
  Slight data change → very different tree (high variance)
```

> *"That last point is crucial. Decision trees are unstable.*
> *Remove 5 training examples and the entire tree structure might change.*
>
> *This instability is actually the seed of next module's idea:
> Random Forests. What if instead of one tree, you averaged many?
> The instability cancels out.*
>
> *But for now: decision trees are your go-to when interpretability matters
> more than maximum accuracy."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
DECISION TREE COMPLETE PICTURE:
  Grows by: maximizing information gain at each split
  Gini impurity: measures class mixture (0 = pure, 0.5 = mixed)
  Overfits when: no depth limit → use max_depth, min_samples_leaf
  Strengths: interpretable, fast, no scaling needed, handles mixed types
  Weakness: unstable, high variance, not the most accurate alone

TUNE WITH CROSS-VALIDATION:
  max_depth: [1, 2, 3, 5, 7, 10]
  min_samples_leaf: [1, 5, 10, 20]
```

**Homework:**
```python
# Using the Titanic-style synthetic dataset:
# 1. Fit a DecisionTreeClassifier with max_depth=None
#    Report train and test accuracy.
# 2. Tune max_depth from 1 to 15. Plot depth vs test accuracy.
# 3. Use export_text() to print your best tree.
#    Write in plain English what the 3 most important questions are.
# 4. Compare accuracy to your logistic regression from Module 06.
#    Which wins on this dataset? Why do you think?
```

---

## INSTRUCTOR TIPS

**"Why do we use Gini instead of information gain (entropy)?"**
> *"Both measure impurity. Entropy = -Σ p_i × log(p_i)*
> *Gini is slightly faster to compute (no logarithm).*
> *In practice they produce almost identical trees.*
> *sklearn defaults to Gini; you can try criterion='entropy' to compare."*

**"Can I use decision trees for regression?"**
> *"Yes! DecisionTreeRegressor exists.*
> *Instead of Gini, it minimizes variance in each leaf.*
> *The leaf prediction is the mean of all samples in it.*
> *The same overfitting problem applies — you still need max_depth."*

**"My tree gives 100% training accuracy. Is that good?"**
> *"No — that means max_depth=None and the tree memorized training data.*
> *Check your test accuracy. Almost certainly much lower.*
> *Set max_depth and retrain."*

**"How do I know if I need a deeper tree?"**
> *"Plot training vs validation accuracy vs depth.*
> *If validation accuracy is still rising, go deeper.*
> *If it plateaus or drops, you've found your sweet spot.*
> *Use cross-validation for a more reliable estimate."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                   10 min
├── Gini impurity                  25 min
├── Information gain               25 min
├── Overfitting and pruning        15 min
└── Close + homework               15 min

SESSION 2  (90 min)
├── Opening bridge                 10 min
├── Implement from scratch         25 min
├── sklearn + visualization        20 min
├── Feature importances            10 min
├── When trees win                 15 min
└── Close + homework               10 min
```

---
*MLForBeginners · Part 2: Classification · Module 08*
