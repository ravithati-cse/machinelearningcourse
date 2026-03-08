# MLForBeginners — Instructor Guide
## Part 2 · Module 09: Random Forests
### Two-Session Teaching Script

> **Prerequisites:** Part 2 Modules 01–08 complete. They know decision trees
> thoroughly: Gini impurity, information gain, overfitting, and max_depth tuning.
> **Payoff today:** They understand why one unstable tree is weak — but 500
> diverse trees are extremely powerful. They build an ensemble from scratch.

---

# SESSION 1 (~90 min)
## "Wisdom of the crowd — why diversity beats expertise"

## Before They Arrive
- Terminal open in `classification_algorithms/algorithms/`
- Whiteboard ready; draw 5 small tree sketches and a single "vote tally" box beside them
- Have a coin ready for the bootstrap analogy

---

## OPENING (10 min)

> *"Last module we built a decision tree. It was interpretable, fast, and worked well.*
>
> *But I mentioned a problem: decision trees are unstable.*
> *Change a few training examples and the whole tree changes.*
> *That instability means high variance — the model is too sensitive to
> which specific examples it happened to see.*
>
> *Here's the brilliant idea that fixes this:*
>
> *What if instead of one tree, you trained 500 different trees,
> each on a slightly different version of your data,
> and let them vote?*
>
> *A single doctor can be biased. But a committee of 500 diverse doctors
> averaging their opinions? Much more reliable.*
>
> *This is Random Forests. It's consistently one of the best-performing
> algorithms on tabular data. And today you'll understand exactly why."*

Draw on board:
```
SINGLE DECISION TREE:
  One doctor's opinion → fast, interpretable, but can be biased

RANDOM FOREST:
  500 doctors, each trained on slightly different cases,
  each only allowed to look at a random subset of tests
  → Majority vote
  → Much more reliable than any single doctor
```

---

## SECTION 1: The Jellybean Jar — Why Crowds Beat Experts (15 min)

> *"In 1907, Francis Galton visited a county fair in England.*
> *There was a competition: guess the weight of an ox.*
> *800 farmers entered their guesses.*
>
> *No single farmer was right.*
> *But the average of all 800 guesses was 1,197 pounds.*
> *The actual weight: 1,198 pounds.*
>
> *The crowd was more accurate than any individual expert.*
> *This is called the 'wisdom of crowds'."*

Write on board:
```
WHY THIS WORKS FOR ML:

Each tree makes different errors.
If errors are INDEPENDENT (uncorrelated), they cancel out.

Example with 5 trees, each 70% accurate:
  P(majority wrong) = P(3 or more out of 5 are wrong)
                    = much less than 30%

Actually: P(majority wrong) ≈ 16%  (by binomial distribution)

With 100 trees at 70% each: P(majority wrong) ≈ 0.04%

THE KEY CONDITION: trees must be diverse (different errors)
```

> *"If all trees make the same errors — they're correlated —
> the ensemble provides no benefit.*
>
> *Random Forests creates diversity in two ways:*
> *1. Different training data for each tree (bootstrapping)*
> *2. Different features available at each split (feature randomness)*
>
> *Let's go through each."*

---

## SECTION 2: Bootstrap Sampling — Different Data for Each Tree (20 min)

Hold up a coin:
> *"Bootstrapping sounds fancy. It just means: sample with replacement.*
>
> *Imagine you have 10 training examples.*
> *Instead of giving every tree all 10 examples,*
> *you randomly draw 10 examples, but you put each one back before drawing the next.*
>
> *So Tree 1 might see examples: 1,1,3,5,7,8,8,9,10,10*
> *Tree 2 might see: 2,3,3,4,5,5,6,8,9,10*
> *Tree 3 might see: 1,2,4,4,6,7,7,8,9,10*
>
> *Each tree sees a different version of the data.*
> *Some examples appear twice. Some don't appear at all.*"

Draw on board:
```
BOOTSTRAP SAMPLING (n=10, sampling with replacement):

Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Tree 1 sample: [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]  ← 1,3,5 appear twice; 7,8,10 missing
Tree 2 sample: [2, 7, 1, 8, 2, 8, 1, 8, 2, 8]  ← very different
Tree 3 sample: [5, 4, 3, 7, 10, 6, 8, 4, 9, 3] ← different again

Each tree sees ~63.2% of unique examples (on average)
The remaining ~36.8% are "out-of-bag" (OOB) — never seen during training
```

> *"That 36.8% — the out-of-bag examples — are incredibly useful.*
> *Each tree can be evaluated on its OOB examples.*
> *Average OOB accuracy across all trees ≈ cross-validation accuracy.*
>
> *You get a free validation score without setting aside a test set.*
> *In competitions and production, this is extremely useful."*

**Ask the room:** *"Why is it 63.2% and not exactly 50%?"*

Answer: With n draws from n items with replacement, the probability that
any specific item is never drawn is (1 - 1/n)^n → e^(-1) ≈ 36.8% as n → ∞.

---

## SECTION 3: Feature Randomness — Why Not Just Bootstrap? (15 min)

> *"Bootstrap sampling alone makes trees different.*
> *But there's a subtlety: if one feature is extremely predictive,
> every tree will put it at the root.*
>
> *All trees have similar structure. Their errors are still correlated.*
> *The ensemble doesn't help much.*
>
> *Random Forests adds a second source of diversity:*
> *At each split, only a random subset of features is considered."*

Write on board:
```
AT EACH SPLIT:
  Standard decision tree: considers ALL features (e.g., all 30)
  Random Forest tree:     considers SQRT(n_features) randomly chosen features

For 30 features: each split only looks at ~5-6 random features

WHY THIS HELPS:
  Weak predictors get occasional chances to split
  Strong predictor can't dominate every tree
  Trees become more diverse in structure
  Errors become less correlated → ensemble benefits increase
```

> *"The hyperparameter is called max_features in sklearn.*
> *Default: 'sqrt' for classification, works great in practice.*
> *You can also try 'log2' or a specific number."*

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
RANDOM FOREST = DECISION TREES + TWO TRICKS:

Trick 1: Bootstrapping
  Each tree trains on a random sample with replacement
  ~63% unique examples per tree
  OOB examples → free validation score

Trick 2: Feature Randomness
  Each split considers only sqrt(n_features) random features
  Prevents any one feature from dominating
  Increases diversity between trees

RESULT:
  Many diverse, uncorrelated trees
  Majority vote (classification) or average (regression)
  Errors cancel out → much more accurate than one tree
```

**Homework:** Think about which would benefit more from ensembling:
- Model A: 90% accurate on training, 95% accurate on test
- Model B: 90% accurate on training, 65% accurate on test
Why? What does this tell you about when Random Forests help most?

---

# SESSION 2 (~90 min)
## "Code the forest, OOB error, and feature importance"

## OPENING (10 min)

> *"Last session we understood the ideas: bootstrapping, feature randomness, voting.*
>
> *Today we build it.*
>
> *And then we'll look at feature importance — one of the most practically
> useful outputs of Random Forests.*
> *You can train a forest on medical data and it will tell you:
> these 5 tests are doing 80% of the work. Stop doing the other 25 tests.*
> *That's a real $10 million insight."*

---

## SECTION 1: Build Random Forest From Scratch (25 min)

Code together — building on the DecisionTreeScratch from Module 08:

```python
import numpy as np
from collections import Counter

# (Assumes DecisionTreeScratch from previous module is available)

class RandomForestScratch:
    def __init__(self, n_trees=100, max_depth=5,
                 max_features='sqrt', min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.trees = []

    def _bootstrap_sample(self, X, y):
        n = len(X)
        indices = np.random.choice(n, size=n, replace=True)
        return X[indices], y[indices], indices

    def _get_max_features(self, n_features):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_features)))
        if self.max_features == 'log2':
            return max(1, int(np.log2(n_features)))
        return n_features  # use all

    def fit(self, X, y):
        n_features = X.shape[1]
        self.n_features_ = n_features
        self.oob_scores_ = []
        self.trees = []

        for i in range(self.n_trees):
            # Bootstrap sample
            X_boot, y_boot, boot_indices = self._bootstrap_sample(X, y)

            # Get feature subset size
            max_feat = self._get_max_features(n_features)

            # Grow a tree (simplified: use full features for scratch version)
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if i % 50 == 0:
                print(f"Grew tree {i+1}/{self.n_trees}")

    def predict(self, X):
        # Each tree votes
        all_votes = np.array([tree.predict(X) for tree in self.trees])
        # all_votes shape: (n_trees, n_samples)

        # Majority vote for each sample
        def majority_vote(votes):
            return Counter(votes).most_common(1)[0][0]

        return np.array([majority_vote(all_votes[:, i])
                         for i in range(X.shape[0])])

    def predict_proba(self, X):
        # Fraction of trees voting for class 1
        all_votes = np.array([tree.predict(X) for tree in self.trees])
        return all_votes.mean(axis=0)
```

> *"The key loop: for each tree, bootstrap sample → grow tree → store.*
> *Prediction: every tree votes, majority wins.*
>
> *This is genuinely the entire algorithm.*
> *sklearn adds feature randomness at each split, which we're simplifying here,
> but the structure is identical."*

---

## SECTION 2: sklearn Random Forest — Production Version (15 min)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X, y = load_breast_cancer(return_X_y=True)
feature_names = load_breast_cancer().feature_names
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,     # 200 trees
    max_depth=None,       # let trees grow deep (forest handles variance)
    max_features='sqrt',  # sqrt of features per split
    oob_score=True,       # compute OOB accuracy for free
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
rf.fit(X_train, y_train)

print(f"Test accuracy:  {accuracy_score(y_test, rf.predict(X_test)):.1%}")
print(f"OOB accuracy:   {rf.oob_score_:.1%}")  # free validation!
print()
print(classification_report(y_test, rf.predict(X_test)))
```

> *"Notice: OOB accuracy is close to test accuracy.*
> *The OOB estimate is so reliable that in many projects
> people use it instead of a held-out test set.*
>
> *Also notice: we didn't limit max_depth here.*
> *With a single tree, unlimited depth = overfitting.*
> *With a forest, the diversity and averaging control variance.*
> *You can often grow trees fully and still get excellent generalization."*

---

## SECTION 3: Feature Importance (20 min)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Feature importance
importances = pd.Series(rf.feature_importances_,
                        index=feature_names)
importances_sorted = importances.sort_values(ascending=False)

print("Top 10 Most Important Features:")
print(importances_sorted.head(10))

# Visualize
plt.figure(figsize=(10, 6))
importances_sorted.head(15).plot(kind='bar')
plt.title("Random Forest Feature Importance")
plt.ylabel("Importance Score")
plt.xlabel("Feature")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=300)
plt.close()
```

> *"Feature importance = sum of information gain improvements weighted by samples.*
>
> *The top features are doing most of the classification work.*
>
> *This is directly actionable: if you're building a diagnostic tool
> with 30 tests, and importance shows 5 tests do 80% of the work —
> you can consider cutting the other 25 tests.*
>
> *One real-world application: a hospital used Random Forest feature importance
> to identify that only 4 blood tests were needed instead of 20
> to predict readmission risk. Saved money and patient time."*

**Ask the room:** *"What's the difference between importance in a decision tree
vs importance in a Random Forest?"*

Answer: Decision tree importance can be misleading because one dominant feature
gets all splits. Random Forest averages across 200 diverse trees — much more
stable and reliable.

---

## SECTION 4: Tuning and Comparing (10 min)

```python
# How many trees is enough?
from sklearn.model_selection import cross_val_score

n_trees_range = [10, 50, 100, 200, 500]
for n in n_trees_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_temp, X, y, cv=5)
    print(f"n_estimators={n:3d}: CV accuracy = {scores.mean():.1%} ± {scores.std():.1%}")
```

> *"Performance improves quickly at first, then plateaus.*
> *There's almost never a reason to use more than 200-500 trees.*
> *After that, you're just burning CPU.*
>
> *The parameter that matters more: max_features.*
> *Try 'sqrt', 'log2', and a few integers.*
> *But honestly, defaults work very well — Random Forests are robust."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
RANDOM FOREST SUMMARY:
  Technique: Ensemble of decision trees (bagging + feature randomness)
  Training:  N bootstrap samples → N trees (easily parallelizable)
  Prediction: Majority vote
  OOB score: Free validation, no test set needed
  Feature importance: Reliable, stable, actionable

COMPARED TO SINGLE TREE:
  More accurate         ✓ (usually significantly)
  Less interpretable    ✗ (can't print 500 trees)
  Slower to train       ✗ (but parallelizes well)
  More robust           ✓ (high variance → controlled)

WHAT'S NEXT:
  Module 10: How do we evaluate ALL these classifiers rigorously?
  ROC curves, AUC, precision-recall — the complete metrics picture.
```

**Homework:**
```python
# Compare on the same Breast Cancer dataset:
# 1. Single DecisionTree (best depth from Module 08)
# 2. RandomForest (n_estimators=200)
#
# Report for each: test accuracy, precision, recall, F1, AUC
# Which is more accurate? By how much?
# Which features does the Random Forest consider most important?
# Are they the same as the Decision Tree root split?
```

---

## INSTRUCTOR TIPS

**"Why not just use more data instead of bootstrap sampling?"**
> *"In the real world, more data is expensive.*
> *Bootstrap sampling is free — it creates diversity without extra data.*
> *And even with unlimited data, feature randomness still improves performance
> by preventing correlated trees."*

**"When does Random Forest fail?"**
> *"Linear data: if the true relationship is linear, many small trees
> approximate it poorly. Logistic regression wins there.*
> *Extremely high dimensions: forests do better than KNN, but struggle.*
> *Regression extrapolation: trees can't predict outside the training range
> (they always predict within the observed target range)."*

**"What's the difference between Random Forest and Gradient Boosting?"**
> *"Random Forest: trees are independent, trained in parallel,
> each corrects no one else's errors.*
> *Gradient Boosting (XGBoost, LightGBM): trees are trained sequentially,
> each one correcting the previous tree's errors.*
> *Gradient Boosting often wins on accuracy but is more prone to overfitting
> and requires more tuning. Great topic for a future module."*

**"n_jobs=-1 — what does that mean?"**
> *"Use all available CPU cores.*
> *200 trees are completely independent — they can be trained simultaneously.*
> *On an 8-core machine, you get 8x speedup. Almost perfectly parallelizable."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                   10 min
├── Wisdom of crowds intuition     15 min
├── Bootstrap sampling             20 min
├── Feature randomness             15 min
└── Close + homework               10 min

SESSION 2  (90 min)
├── Opening bridge                 10 min
├── Build from scratch             25 min
├── sklearn production version     15 min
├── Feature importance             20 min
├── Tuning and comparing           10 min
└── Close + homework               10 min
```

---
*MLForBeginners · Part 2: Classification · Module 09*
