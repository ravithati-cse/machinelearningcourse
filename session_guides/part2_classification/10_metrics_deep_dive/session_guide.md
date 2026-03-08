# MLForBeginners — Instructor Guide
## Part 2 · Module 10: Metrics Deep Dive
### Two-Session Teaching Script

> **Prerequisites:** Part 2 Modules 01–09 complete. They know confusion matrix,
> precision, recall, F1, and all four classifiers. They are ready for the advanced
> evaluation toolkit that separates professionals from beginners.
> **Payoff today:** They will never again evaluate a model with only accuracy.
> They'll understand ROC curves, AUC, precision-recall tradeoffs, and most
> importantly — how to choose the right metric for a real business problem.

---

# SESSION 1 (~90 min)
## "The ROC curve — visualizing every possible threshold at once"

## Before They Arrive
- Terminal open in `classification_algorithms/algorithms/`
- Whiteboard ready; draw a confusion matrix reminder and a blank ROC axes
- Optional: print a small table with 10 patients, their true labels and model scores

---

## OPENING (10 min)

> *"You've been evaluating models with accuracy, precision, recall, and F1.*
> *You know your confusion matrix.*
>
> *But here's a question you haven't fully answered yet:*
>
> *You have a logistic regression model that outputs probabilities.*
> *P(cancer) = 0.73. Do you say 'cancer'? What if P = 0.51?*
> *What if you set the threshold to 0.3? To 0.8?*
>
> *Every threshold gives a different model — different precision, recall, confusion matrix.*
>
> *Today you'll learn how to evaluate ALL possible thresholds at once,
> in a single beautiful curve.*
>
> *This is the ROC curve. It's how cancer screening systems,
> fraud detection engines, and hiring algorithms are evaluated in the real world."*

Draw on board:
```
THE THRESHOLD QUESTION:

Model outputs: P(spam) = 0.85 → obviously spam
               P(spam) = 0.55 → probably spam?
               P(spam) = 0.45 → probably not spam?
               P(spam) = 0.12 → probably ham

Threshold = 0.5  (default): classify as spam if P > 0.5
Threshold = 0.3  (aggressive): catch more spam, but more false positives
Threshold = 0.8  (conservative): very certain before marking spam

QUESTION: Which threshold should I use?
ROC CURVE ANSWER: Look at ALL thresholds and pick the one that fits your needs.
```

---

## SECTION 1: Quick Review — Confusion Matrix Vocabulary (10 min)

> *"Before we build the ROC curve, let's cement the vocabulary.*
> *You've seen this before — but these exact terms are what ROC is built from."*

Write on board:
```
                    PREDICTED
                  Spam    Ham
              ┌────────┬────────┐
  ACTUAL  Spam│  TP    │  FN    │  ← Real spam. Did we catch it?
  (P)         │        │        │
              ├────────┼────────┤
  ACTUAL  Ham │  FP    │  TN    │  ← Real ham. Did we falsely alarm?
  (N)         │        │        │
              └────────┴────────┘

True Positive Rate (TPR) = TP / (TP + FN)  ← "Recall" / "Sensitivity"
  Of all actual spam, what fraction did we catch?

False Positive Rate (FPR) = FP / (FP + TN) ← "1 - Specificity"
  Of all actual ham, what fraction did we wrongly flag as spam?

IDEAL: TPR = 1 (catch everything), FPR = 0 (no false alarms)
REALITY: You can't have both. Raising TPR usually raises FPR too.
```

**Ask the room:** *"If I set threshold = 0 (flag everything as spam),
what are TPR and FPR?"*

Answer: TPR = 1.0 (catch all spam), FPR = 1.0 (flag all ham too). Useless.

**Ask the room:** *"If I set threshold = 1.0 (never flag anything as spam),
what are TPR and FPR?"*

Answer: TPR = 0.0, FPR = 0.0. Also useless — we never catch anything.

> *"Every threshold between 0 and 1 gives a different (FPR, TPR) pair.*
> *Plot all of them — that's the ROC curve."*

---

## SECTION 2: Building the ROC Curve Step by Step (25 min)

> *"Let's build a ROC curve by hand with 10 patients.*
> *This is the only time you'll do this manually — to understand the mechanics."*

Write on board:
```
10 PATIENTS: Sorted by model score (high to low)
Score  True Label   TP  FP  TPR         FPR
─────────────────────────────────────────────
0.95   Positive     1   0   1/5=0.20    0/5=0.00
0.90   Positive     2   0   2/5=0.40    0/5=0.00
0.80   Negative     2   1   2/5=0.40    1/5=0.20
0.70   Positive     3   1   3/5=0.60    1/5=0.20
0.65   Positive     4   1   4/5=0.80    1/5=0.20
0.60   Negative     4   2   4/5=0.80    2/5=0.40
0.50   Positive     5   2   5/5=1.00    2/5=0.40
0.40   Negative     5   3   5/5=1.00    3/5=0.60
0.30   Negative     5   4   5/5=1.00    4/5=0.80
0.20   Negative     5   5   5/5=1.00    5/5=1.00

(5 positive, 5 negative patients)
```

> *"Plot these (FPR, TPR) points on axes.*
> *Connect them with a line.*
> *That's the ROC curve."*

Draw on the whiteboard:
```
     TPR
  1.0 │            *────────────
      │         *
  0.8 │      *
      │      *
  0.6 │   *
      │
  0.4 │   *
      │*
  0.2 │*
      │
  0.0 └──────────────────────── FPR
      0.0  0.2  0.4  0.6  0.8  1.0

Diagonal dashed line = random classifier (AUC = 0.5)
Our curve = above diagonal (better than random)
```

> *"The diagonal line represents random guessing.*
> *A random classifier that just flips a coin gives a diagonal ROC curve.*
> *AUC = 0.5.*
>
> *A perfect classifier: goes straight up to TPR=1, then right.*
> *AUC = 1.0.*
>
> *Our curve: somewhere in between. The closer to the top-left corner,
> the better."*

---

## SECTION 3: AUC — Collapsing the Curve to One Number (20 min)

> *"ROC curves are great for visualization. But how do you compare two models?*
>
> *Model A has higher TPR at low FPR.*
> *Model B has higher TPR at high FPR.*
> *Which is better overall?*
>
> *AUC answers this: Area Under the Curve.*"

Write on board:
```
AUC = Area Under the ROC Curve

AUC = 1.0  → Perfect classifier
AUC = 0.9  → Excellent
AUC = 0.8  → Good
AUC = 0.7  → Fair
AUC = 0.6  → Poor
AUC = 0.5  → No better than random

INTERPRETATION:
  AUC = probability that the model ranks a random positive
  example higher than a random negative example.

  AUC = 0.8 means: if you pick one spam and one ham,
  the model gives the spam a higher score 80% of the time.
```

> *"AUC is threshold-independent. It measures the model's discriminating power
> across ALL possible thresholds.*
>
> *This is why it's the standard metric in medical machine learning.*
> *'Our cancer detector has AUC = 0.93' — clean, interpretable, universal."*

**Ask the room:** *"If I shuffle my model's predictions randomly,
what AUC would I expect?"*

Answer: 0.5 (random ordering = diagonal ROC = AUC 0.5).

---

## CLOSING SESSION 1 (25 min)

Live demo and board summary:

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Run module first:
# python3 metrics_deep_dive.py

# Then demonstrate in interactive Python:
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_scores)
auc = roc_auc_score(y_test, y_scores)

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_demo.png', dpi=300)
print(f"AUC = {auc:.3f}")
```

**Homework:** Given two classifiers on the same dataset:
- Classifier A: AUC = 0.88, F1 = 0.82
- Classifier B: AUC = 0.80, F1 = 0.87

Which would you choose for a cancer screening tool? Which for an email spam filter?
Think about what each metric emphasizes before next session.

---

# SESSION 2 (~90 min)
## "Precision-recall, imbalanced classes, and choosing the right metric"

## OPENING (10 min)

> *"Last session: ROC curve, AUC.*
> *This session: when AUC lies to you.*
>
> *Wait — AUC can lie?*
>
> *Sort of. AUC is great when your classes are balanced.*
> *But real-world data often isn't.*
>
> *99% of credit card transactions are legitimate. 1% are fraud.*
> *A model that labels everything 'legitimate' gets 99% accuracy.*
> *And has a decent-looking ROC curve.*
> *But it catches ZERO fraud.*
>
> *For imbalanced problems, we need the Precision-Recall curve."*

---

## SECTION 1: Precision-Recall — When Class Balance Matters (20 min)

> *"Precision and recall you know from Module 04.*
> *Precision-Recall curves work like ROC curves but use different axes."*

Write on board:
```
PRECISION-RECALL CURVE:
  X-axis: Recall    = TP / (TP + FN)   "Of all positives, what fraction caught?"
  Y-axis: Precision = TP / (TP + FP)   "Of all flagged, what fraction is real?"

WHY BETTER FOR IMBALANCED CLASSES:
  FPR (used in ROC) = FP / (FP + TN)
  When TN is huge (99% legitimate), even many FP look like small FPR.
  ROC curve looks great even if the model is awful at finding positives.

  Precision-recall IGNORES true negatives entirely.
  It only cares about: did we find the positives? Were our alarms real?

PERFECT PRECISION-RECALL:
  P = 1.0, R = 1.0 (top right corner)
  Area Under PR curve (AUPRC) near 1.0 = excellent
  Random baseline AUPRC ≈ fraction of positive class (e.g., 0.01 for 1% fraud)
```

Draw on board:
```
      Precision
  1.0 │*
      │ *
  0.8 │  *
      │   *
  0.6 │     *
      │       *
  0.4 │          *
      │               *
  0.2 │                     *
      │                            *
  0.0 └──────────────────────────────── Recall
      0.0  0.2  0.4  0.6  0.8  1.0

The curve always descends: as you increase recall (catch more),
precision drops (more false positives).
```

> *"This is the fundamental tradeoff — you cannot maximize both.*
> *You must choose based on what's more important for your problem."*

---

## SECTION 2: The Threshold Decision — A Doctor's Dilemma (20 min)

> *"Here's the real-world question: once you have the ROC or PR curve,
> where do you set the threshold?*
>
> *There's no mathematical answer. It depends on costs.*"

Draw on board:
```
SCENARIO: Cancer Screening
  False Negative (FN): Missed cancer → patient goes untreated → potentially dies
  False Positive (FP): False alarm → patient gets extra tests → anxiety, cost

  Which is worse? Almost everyone says FN.
  → Set threshold LOW to maximize recall (catch everything)
  → Accept more false positives

SCENARIO: Spam Filter
  False Negative (FN): Spam gets through → minor annoyance
  False Positive (FP): Real email flagged → critical business email lost

  Which is worse? Most say FP.
  → Set threshold HIGH to maximize precision
  → Accept more spam getting through

SCENARIO: Hiring Algorithm
  High stakes for both → need balanced threshold
  → Use F1 score or equal-cost assumption
```

> *"This is a business decision, not a math decision.*
> *The data scientist's job: present the ROC/PR curve to stakeholders*
> *and let them choose the operating point.*
>
> *Never silently pick threshold = 0.5. Always discuss with the team."*

Code the threshold analysis:
```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Find threshold that maximizes F1 score
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores[:-1])  # last element has no threshold
best_threshold = thresholds[best_idx]
best_f1 = f1_scores[best_idx]

print(f"Best threshold for F1: {best_threshold:.3f}")
print(f"At this threshold: F1={best_f1:.3f}, "
      f"Precision={precision[best_idx]:.3f}, "
      f"Recall={recall[best_idx]:.3f}")
```

---

## SECTION 3: Which Metric to Use When (20 min)

> *"Let's make this practical. Here's a decision framework."*

Write on board:
```
METRIC SELECTION GUIDE:

Problem Type                  Recommended Metric
─────────────────────────────────────────────────
Classes balanced, general     Accuracy or F1
Classes imbalanced            F1, Precision-Recall AUC
Need to explain one number    AUC-ROC (threshold-independent)
Medical diagnosis (FN costly) Recall (sensitivity)
Spam filter (FP costly)       Precision
Fraud detection (FN costly,   F1 with cost weighting,
  imbalanced)                 or Precision-Recall AUC
Information retrieval         Precision@K, Average Precision
Multiple classes              Macro F1 or Weighted F1

NEVER use accuracy alone when:
  Classes are imbalanced (even slightly)
  The cost of FP and FN are different
  You need to set a specific operating threshold
```

> *"Here's a mental checklist:*
> *1. Are my classes balanced? If no → use F1 and PR-AUC*
> *2. Is one error type more costly? Yes → optimize for recall or precision*
> *3. Am I comparing models across thresholds? Yes → use ROC-AUC*
> *4. Am I deploying at a fixed threshold? Yes → compute F1/precision/recall at that threshold*"

**Ask the room:** *"You're building a fraud detection system.*
*0.1% of transactions are fraudulent. The bank loses $1,000 on missed fraud,*
*but investigating a false alarm costs $10 in staff time.*
*What metric should guide your threshold choice?"*

Answer: Weighted cost analysis. Missing fraud costs 100x more than false alarm.
Set a very low threshold (maximize recall), even if precision is low.

---

## SECTION 4: Imbalanced Classes — Practical Solutions (10 min)

```python
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
import numpy as np

# Option 1: class_weight='balanced' in sklearn
model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X_train, y_train)

# Option 2: Oversample minority class (basic version)
# Find minority class samples
minority_mask = (y_train == 1)
n_majority = (~minority_mask).sum()
n_minority = minority_mask.sum()
n_to_add = n_majority - n_minority

# Random oversample
minority_indices = np.where(minority_mask)[0]
extra_indices = np.random.choice(minority_indices, size=n_to_add, replace=True)
X_balanced = np.vstack([X_train, X_train[extra_indices]])
y_balanced = np.concatenate([y_train, y_train[extra_indices]])

print(f"Original: {n_majority} majority, {n_minority} minority")
print(f"Balanced: {len(y_balanced)//2} each")
```

> *"In production, the most commonly used library for this is imbalanced-learn.*
> *It provides SMOTE (Synthetic Minority Oversampling Technique).*
> *We'll use it in the churn project — Module 12.*
>
> *For now, class_weight='balanced' is a quick, effective fix.*
> *Try it on every imbalanced problem as your first step."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
METRICS COMPLETE PICTURE:
  ROC Curve:    TPR vs FPR at all thresholds — balanced classes
  AUC-ROC:      Single number, threshold-independent, 0.5→random, 1.0→perfect
  PR Curve:     Precision vs Recall — imbalanced classes
  AUPRC:        Area under PR curve — better for rare events

  Threshold:    Business decision, not a math decision
                Depends on relative cost of FP vs FN

SUMMARY TABLE:
  High recall needed?     → Low threshold
  High precision needed?  → High threshold
  Classes imbalanced?     → Use class_weight='balanced'
  Report a single number? → AUC-ROC (balanced) or PR-AUC (imbalanced)
```

**Homework:**
```python
# Using RandomForest from Module 09 on breast cancer data:
# 1. Generate ROC curve and compute AUC
# 2. Generate Precision-Recall curve and compute AUPRC
# 3. Find the threshold that maximizes F1 score
# 4. Find the threshold that achieves recall ≥ 0.95
#    (high-recall scenario: don't miss any cancer)
#    At this threshold, what is the precision?
# 5. Write 2 sentences: when would you use threshold from #3 vs #4?
```

---

## INSTRUCTOR TIPS

**"ROC AUC vs accuracy — which is better?"**
> *"AUC is almost always better for model selection.*
> *Accuracy depends on threshold and class balance.*
> *AUC measures intrinsic discriminative power regardless of threshold.*
> *Use accuracy for final reporting at a chosen threshold.*
> *Use AUC for comparing models during development."*

**"What does AUC = 0.5 actually mean?"**
> *"The model is useless — its score for positives and negatives are*
> *drawn from the same distribution. Random ordering.*
> *If you see AUC = 0.5 or below, something is probably wrong:*
> *maybe you accidentally inverted your labels."*

**"What if my PR curve is all over the place?"**
> *"PR curves are 'noisy' with small datasets.*
> *Use interpolated PR curves (scipy or sklearn's average_precision_score)*
> *for a cleaner single-number summary.*
> *Also, more data → smoother PR curves."*

**"Should I always use class_weight='balanced' for imbalanced data?"**
> *"Not always. class_weight='balanced' can overcorrect if classes are very unequal.*
> *Start with it as a baseline.*
> *Then try adjusting the ratio manually or using SMOTE.*
> *Always evaluate with PR-AUC on imbalanced data, not accuracy."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                   10 min
├── Confusion matrix review        10 min
├── Building ROC curve by hand     25 min
├── AUC interpretation             20 min
└── Live demo + homework           25 min

SESSION 2  (90 min)
├── Opening bridge                 10 min
├── Precision-Recall curves        20 min
├── Threshold decision framework   20 min
├── Which metric when              20 min
├── Imbalanced class solutions     10 min
└── Close + homework               10 min
```

---
*MLForBeginners · Part 2: Classification · Module 10*
