# MLForBeginners — Instructor Guide
## Module 4 (Part 2): Confusion Matrix  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who understand sigmoid and probability.
> **Tone:** Casual — this module is very concrete, very relatable, almost fun.
> **Goal:** Everyone can build a confusion matrix from scratch, knows TP/FP/TN/FN by heart,
> and understands which metric to use when. This is the most interview-critical module.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "The 2x2 table that explains everything about your classifier"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal ready in `MLForBeginners/classification_algorithms/math_foundations/`
- Visuals folder `visuals/04_confusion_matrix/` open
- Whiteboard with a large 2x2 grid drawn before they arrive
- The module is called "MOST IMPORTANT" in the source — tell them that

**Room vibe:** Draw the 2x2 grid big. You'll write on it a lot.

---

## OPENING  (10 min)

### Hook — The 95% accurate model that's useless

> *"Quick scenario. I just built a cancer detection model.*
> *It's 95% accurate. Should I deploy it?*
>
> *You say: probably! 95% is good.*
>
> *I tell you: the dataset has 95% healthy patients and 5% cancer patients.*
> *My model predicts 'healthy' for every single person, no matter what.*
> *Accuracy: 95%. Cancer caught: zero.*
>
> *This is the accuracy trap. And it's why you need the confusion matrix.*
> *One table. Tells you everything a single accuracy number hides."*

**Write on board:**

```
The most important question is NOT "how often am I right overall?"
The real questions are:
  - "Of people who have cancer, how many did I catch?"
  - "Of people I said have cancer, how many actually do?"

ACCURACY can't answer these. The confusion matrix can.
```

---

## SECTION 1: Building the Confusion Matrix  (25 min)

> *"Two dimensions: what actually happened, and what we predicted.*
> *Four possible combinations."*

**Draw the 2x2 grid large on the board:**

```
                        PREDICTED
                     Positive    Negative
                  ┌───────────┬───────────┐
         Positive │    TP     │    FN     │
ACTUAL            │           │           │
         Negative │    FP     │    TN     │
                  └───────────┴───────────┘
```

> *"Four boxes. Each gets a name. Let's learn them with a medical example.*
>
> *TEST RESULT: Does the patient have Disease X?*"

**Fill in the boxes with explanations:**

```
TRUE POSITIVE (TP):
  Actually sick. Predicted sick. CORRECT!
  "We caught a real case." → Great, this is what we want.

FALSE NEGATIVE (FN):
  Actually sick. Predicted healthy. WRONG!
  "We missed a real case." → In cancer: potentially fatal.
  Also called: "miss," "Type II error"

FALSE POSITIVE (FP):
  Actually healthy. Predicted sick. WRONG!
  "We raised a false alarm." → In cancer: scary but survivable.
  Also called: "false alarm," "Type I error"

TRUE NEGATIVE (TN):
  Actually healthy. Predicted healthy. CORRECT!
  "Correctly cleared." → Good.
```

**Ask the room after each box:**
> *"In a medical test, which of these four is the most dangerous to have?"*

Let them think. Answer: FN. Missing a real sick person. They never get treatment.

---

## SECTION 2: A Worked Example  (20 min)

> *"Let's put real numbers in."*

**Write on board:**

```
SCENARIO: Spam detection. You classified 100 emails.
  50 were actually spam.  50 were actually not spam.

RESULTS:
  Of the 50 spam:    model caught 45, missed 5
  Of the 50 not spam: model falsely flagged 10, cleared 40

BUILD THE MATRIX:
                        PREDICTED
                     Spam       Not Spam
                  ┌──────────┬──────────┐
         Spam     │   45     │    5     │  ← 45 TP, 5 FN
ACTUAL            │          │          │
         Not Spam │   10     │   40     │  ← 10 FP, 40 TN
                  └──────────┴──────────┘

TP=45, FN=5, FP=10, TN=40
```

> *"Now let's calculate accuracy the old way:*
> *Correct predictions = TP + TN = 45 + 40 = 85*
> *Accuracy = 85/100 = 85%*
>
> *But what does this hide?*
> *5 spam emails got through (FN).*
> *10 legit emails were blocked (FP).*
>
> *Those are very different problems.*
> *The confusion matrix shows them separately.*"

**Have them help calculate from the matrix:**

> *"These numbers come straight from the four boxes.
> No formulas yet — just counts."*

---

## SECTION 3: The Four Derived Metrics  (20 min)

> *"From TP/FP/TN/FN we can derive every metric you'll ever use."*

**Write each formula and calculate with the spam numbers:**

```
1. ACCURACY  = (TP + TN) / Total
   = (45 + 40) / 100 = 85%
   "Overall, how often am I right?"

2. PRECISION = TP / (TP + FP)
   = 45 / (45 + 10) = 45/55 ≈ 82%
   "Of emails I called spam, how many actually were?"
   (Quality of positive predictions)

3. RECALL    = TP / (TP + FN)
   = 45 / (45 + 5) = 45/50 = 90%
   Also called: SENSITIVITY, TRUE POSITIVE RATE
   "Of actual spam, how much did I catch?"
   (Coverage of actual positives)

4. F1 SCORE  = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.82 × 0.90) / (0.82 + 0.90)
   = 2 × 0.738 / 1.72 ≈ 0.858
   "Balanced average of precision and recall"
```

**Write the simple mnemonic on the board:**

```
PRECISION:  "When I say yes, am I right?"     → TP / (TP+FP)
RECALL:     "Do I say yes when I should?"     → TP / (TP+FN)

Precision penalizes false alarms (FP).
Recall penalizes misses (FN).
F1 balances both.
```

---

## SECTION 4: Live Demo  (5 min)

```bash
python3 04_confusion_matrix.py
```

Walk through the output:
> *"See it building the matrix from scratch. See the metrics printing.*
> *Then it generates a heatmap — the visual version of our 2x2 table.*"

Open the confusion matrix heatmap visualization.

---

## CLOSING SESSION 1  (10 min)

### Recap board

```
CONFUSION MATRIX — SESSION 1
────────────────────────────────────────────
THE 4 BOXES:
  TP: right about YES   FN: missed YES (Type II)
  FP: wrong YES alarm   TN: right about NO

THE 4 METRICS (from those 4 boxes):
  Accuracy  = (TP+TN) / Total
  Precision = TP / (TP+FP)   "Quality of yes calls"
  Recall    = TP / (TP+FN)   "Coverage of actual yes"
  F1        = harmonic mean(Precision, Recall)
```

### Homework

From `04_confusion_matrix_lab.md` — Quick Win and Calculate ALL Metrics:

```
Medical test: 10 patients
Actual:    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
Predicted: [1, 1, 0, 0, 0, 0, 0, 1, 0, 0]

Task: Count TP, TN, FP, FN by hand.
Then calculate accuracy, precision, recall, F1.
Then verify with sklearn.
```

> *"Do the counting first. No code.*
> *Then verify with Python.*
> *The numbers should match."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "When does which metric matter — and the accuracy trap"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Homework debrief

> *"What did you get for the medical test?*
> *TP=2, TN=5, FP=1, FN=2.*
> *Accuracy = 70%. Precision = 66.7%. Recall = 50%.*
>
> *Now the important question: this is a medical test.*
> *Is 50% recall acceptable?"*

Let them react. (No — you miss half the sick patients.)

> *"Today we answer: when do you optimize for which metric?*
> *And we'll see specificity, the ROC curve preview, and imbalanced data.*"

---

## SECTION 1: The Precision-Recall Tradeoff  (20 min)

> *"Precision and recall are in tension. You can't max both simultaneously.*
> *Improving one usually hurts the other.*
> *Here's why."*

**Write on board:**

```
SPAM FILTER EXAMPLE:

Conservative model (high threshold):
  Only flags emails it's VERY sure about (99%+ probability)
  TP=40, FP=2, FN=10, TN=48

  Precision = 40/(40+2) = 95%    ← great!
  Recall    = 40/(40+10) = 80%  ← some spam gets through

Aggressive model (low threshold):
  Flags anything that might be spam (>30% probability)
  TP=49, FP=20, FN=1, TN=30

  Precision = 49/(49+20) = 71%  ← more false alarms
  Recall    = 49/(49+1) = 98%   ← catches almost all spam!
```

> *"See the tradeoff? You literally cannot have both.*
> *Lower threshold → more recall, less precision.*
> *Higher threshold → more precision, less recall.*
>
> *Which is right? Depends on your use case."*

**Draw decision guide on board:**

```
USE PRECISION WHEN:
  False alarms are expensive.
  Example: Spam filter (blocking real emails is costly)
  Example: News recommendation (recommending junk annoys users)

USE RECALL WHEN:
  Misses are expensive.
  Example: Cancer detection (missing cancer is fatal)
  Example: Fraud detection (missing fraud = money lost)
  Example: Security screening (missing threats = disaster)

USE F1 WHEN:
  You need to balance both.
  Example: General-purpose classifier
  Example: Imbalanced data evaluation
```

**Ask the room:**
> *"Imagine a model that predicts whether a social media post is misinformation.*
> *Would you optimize for precision or recall?"*

Debate is the point. There's a legitimate case for both.

---

## SECTION 2: Specificity and the Full Grid  (15 min)

> *"There's a fifth metric you'll see: specificity."*

**Write:**

```
SPECIFICITY = TN / (TN + FP)
              "Of actual negatives, how many did we correctly call negative?"
              Also called: TRUE NEGATIVE RATE

With our spam numbers (TP=45, FP=10, TN=40, FN=5):
  Specificity = 40 / (40+10) = 80%

  "Of the 50 legit emails, we correctly identified 80% as legit."
  "We falsely flagged 20% of legit emails as spam."
```

> *"Recall = 'catching sick people' (True Positive Rate)*
> *Specificity = 'clearing healthy people' (True Negative Rate)*
>
> *These two are the axes of the ROC curve.*
> *We'll cover that in the metrics deep dive module.*
> *For now: know specificity exists and what it measures."*

---

## SECTION 3: The Accuracy Trap — Imbalanced Data  (20 min)

> *"Remember the opening: 95% accurate cancer model that catches zero cancers.*
> *Let's see this in code and understand why it happens."*

**Write on board:**

```
IMBALANCED DATASET:
  95 healthy patients
  5 cancer patients

MODEL: Predict "healthy" for everyone.

Confusion Matrix:
              PREDICTED
           Healthy   Cancer
Actual  H  |  95  |    0   |   ← All 95 healthy correctly cleared
        C  |   5  |    0   |   ← ALL 5 cancer missed!

TP=0, TN=95, FP=0, FN=5

Accuracy = (0+95)/100 = 95%  ← LOOKS AMAZING
Recall   = 0/(0+5) = 0%      ← CATCHES ZERO CANCER
```

> *"This is why you NEVER report just accuracy on imbalanced data.*
> *A model that does nothing gets 95% accuracy.*
> *A model that actually tries might only get 92% accuracy but catches 80% of cancers.*
>
> *The second model is infinitely better. Accuracy hides this.*"

**From the lab file — run the accuracy trap exercise together:**

```python
# Lab exercise — The Accuracy Trap
actual = [0]*95 + [1]*5

# Model A: Always predicts healthy
pred_a = [0]*100

# Model B: Actually tries to find cancer
pred_b = [0]*92 + [1]*8  # Gets some right, some false alarms

from sklearn.metrics import accuracy_score, recall_score
print("Model A:", accuracy_score(actual, pred_a), recall_score(actual, pred_a))
print("Model B:", accuracy_score(actual, pred_b), recall_score(actual, pred_b))
```

---

## SECTION 4: Sklearn Demo + Heatmap  (15 min)

```bash
python3 04_confusion_matrix.py
```

Walk through SECTION 2 and beyond — the full metrics comparison.

Open the heatmap visualization:
> *"Color = magnitude. Bright diagonal = good.*
> *Dark off-diagonal = errors.*
> *You can scan this visually in 2 seconds to see where a model fails."*

---

## CLOSING SESSION 2  (10 min)

### Full recap board

```
CONFUSION MATRIX — FULL PICTURE
─────────────────────────────────────────────────────
TP = caught real yes   FN = missed real yes (Type II)
FP = false alarm yes   TN = correct no      (Type I)

ACCURACY  = (TP+TN)/Total      ← misleading with imbalance
PRECISION = TP/(TP+FP)         ← quality of yes calls
RECALL    = TP/(TP+FN)         ← coverage of real yes
F1        = harmonic(P,R)      ← balance of both
SPECIFICITY = TN/(TN+FP)       ← quality of no calls

CHOOSE BY PROBLEM:
  Cancer/fraud/security → Recall (don't miss)
  Spam/recommendations → Precision (don't alarm)
  General → F1 or both

IMBALANCED DATA → Never use accuracy alone!
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"I keep mixing up FP and FN"**
> *"FP: you CALLED it positive (yes), but you were FALSE (wrong).*
> *FN: you CALLED it negative (no), but you were FALSE (wrong).*
> *The first word describes the outcome. The second describes whether it was correct."*

**"Why is F1 harmonic mean, not regular mean?"**
> *"Harmonic mean punishes imbalance more.*
> *If Precision=1.0 and Recall=0.01, regular mean = 50.5% (sounds OK!).*
> *F1 = 2%, which reflects that the model is terrible.*
> *Harmonic mean requires BOTH to be good."*

**"Is accuracy ever useful?"**
> *"Yes — when classes are balanced and errors cost the same.*
> *Balanced dataset, two equally bad error types → accuracy is fine.*
> *Imbalanced data or asymmetric error costs → use recall, precision, or F1."*

## Energy Management

- This module is FUN because examples are relatable. Let debates run.
- **30-min mark:** Natural break. Sketch the heatmap together.
- **If ahead of schedule:** Have them calculate specificity for the homework dataset.

## The Golden Rule

> Every metric needs a story: *"This is important because if you get this wrong,
> [specific bad thing] happens."* Numbers without consequences don't stick.

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook (accuracy trap)   10 min
├── Building the 2x2 matrix        25 min
├── Worked spam example            20 min
├── Four metrics (formulas)        20 min
├── Live demo                       5 min
└── Close + homework               10 min

SESSION 2  (90 min)
├── Homework debrief               10 min
├── Precision-recall tradeoff      20 min
├── Specificity                    15 min
├── Accuracy trap (imbalanced)     20 min
├── Sklearn + heatmap demo         15 min
└── Close + full recap             10 min
```

---

*Generated for MLForBeginners — Module 04 · Part 2: Classification*
