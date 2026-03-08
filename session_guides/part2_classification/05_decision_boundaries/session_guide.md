# MLForBeginners — Instructor Guide
## Module 5 (Part 2): Decision Boundaries  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who know sigmoid, probability, and the confusion matrix.
> **Tone:** Very visual — this is the most fun module to draw. Get them drawing.
> **Goal:** Everyone understands what a decision boundary is, can sketch linear vs.
> non-linear boundaries, and sees why overfitting/underfitting shows up visually.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Drawing the line between spam and not-spam"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal ready in `MLForBeginners/classification_algorithms/math_foundations/`
- Visuals folder `visuals/05_decision_boundaries/` open
- Large blank whiteboard — you'll do a lot of 2D sketching today
- Different colored markers for different classes

**Note:** This module is the most visual. Spend more time at the whiteboard, less at the terminal.

---

## OPENING  (10 min)

### Hook — Drawing a line

> *"I'm going to describe some fruit. You tell me what kind it is.*
>
> *Weight: 200g. Color: red. It's round.*
> *(Pause) What fruit?*
>
> *Apple or pomegranate — probably apple.*
>
> *Weight: 500g. Color: yellow.*
> *Banana? No, too heavy. Mango? Maybe. Grapefruit?*
>
> *You're running an algorithm in your head.*
> *You're asking: which side of some boundary does this fall on?*
>
> *Today we make those boundaries explicit.*
> *We'll draw them, understand them, and see how algorithms learn them."*

**Draw two clusters on the board:**

```
                   FEATURE 2 (Color score)
                   ↑
                   |     × × ×
              ×    |   × × × ×      × = Spam
              ×    |                o = Not Spam
                   |
              o    |     o o o
              o o  |   o o o
                   └──────────────→ FEATURE 1 (Exclamation marks)
```

> *"Somewhere in this space there's a line that separates the ×'s from the o's.*
> *Finding that line — that's what a classifier learns.*
> *We call it the decision boundary."*

---

## SECTION 1: What Is a Decision Boundary?  (20 min)

**Write the definition:**

```
DECISION BOUNDARY:
  The line (or surface) that separates different classes in feature space.

On one side:  Model predicts Class 1 (spam, sick, fraud)
On the other: Model predicts Class 0 (not spam, healthy, legit)

Every classifier draws some kind of decision boundary.
Different algorithms → different shaped boundaries.
```

> *"In 2 features: the boundary is a line or curve.*
> *In 3 features: the boundary is a plane.*
> *In N features: an (N-1)-dimensional surface (a hyperplane).*
>
> *We'll work in 2D because we can see it.*
> *But everything we learn applies to 100 dimensions."*

**Draw the spam example properly:**

```
Feature 1 (x₁): Number of exclamation marks
Feature 2 (x₂): Number of suspicious links

Linear boundary: x₁ + x₂ = 5 (a line)

If x₁ + x₂ > 5  → SPAM (one side of the line)
If x₁ + x₂ ≤ 5  → NOT SPAM (other side)
```

**Draw on board:**

```
x₂
 |         SPAM zone
10|     × × × × ×
 |   × × × ×  /
 5| × /──────/ ← Decision boundary: x₁+x₂=5
 | / o o o
 |/o o o o o NOT SPAM zone
 └──────────────→ x₁
           5    10
```

> *"Notice: every point on the line has equal probability of being spam or not spam.*
> *Probability = 0.5 right on the boundary.*
> *The further you go into spam territory, the higher the probability.*"

---

## SECTION 2: Linear Decision Boundaries  (20 min)

> *"The simplest boundary: a straight line.*
> *Logistic regression draws exactly this — one straight line."*

**Write:**

```
LOGISTIC REGRESSION BOUNDARY:

The model computes: z = β₀ + β₁x₁ + β₂x₂
Decision: spam if sigmoid(z) > 0.5
         → this happens when z > 0
         → which happens when β₀ + β₁x₁ + β₂x₂ > 0

That equation describes a LINE! (in 2D)
                   a PLANE! (in 3D)
             a HYPERPLANE! (in N dimensions)

Logistic regression = linear decision boundary.
```

> *"So if you have a problem where classes are linearly separable
> — where you can draw a straight line between them —
> logistic regression is perfect.*
>
> *But what if the data looks like this?"*

**Draw on board:**

```
NON-LINEARLY SEPARABLE DATA:

× × ×   o o   × ×
  × × o o o o × ×
  o o o o × × × ×
      × × × o o

No single straight line separates × and o.
You need a CURVED boundary.
```

> *"That's when you need non-linear classifiers.*
> *KNN, decision trees, random forests — all can learn curved boundaries.*"

---

## SECTION 3: Non-Linear Boundaries  (15 min)

**Draw three boundary shapes:**

```
LINEAR (Logistic Regression):
  Classes separated by a straight line
  ×|o  ×|o  ×|o  ×|o

POLYNOMIAL (degree 2):
  Curved line (parabola, ellipse, etc.)
         o o o
       ×       ×
      ×    o    ×
       ×       ×
         × × ×

HIGHLY NON-LINEAR (KNN, Trees, Neural Nets):
  Any shape — can be very wiggly
  Can follow every twist in the data

        × × × × ×
    ×   ×   ×   ×   ×
    ×   o   o   o   ×
    ×   o   o   o   ×
    ×   ×   ×   ×   ×
```

> *"More complex boundary = more flexible model.*
> *But more flexibility has a cost: overfitting.*
> *We'll see this in a moment."*

---

## SECTION 4: Live Demo  (15 min)

```bash
python3 05_decision_boundaries.py
```

> *"Watch the output — it trains different classifiers and plots their boundaries.*"

**Open the generated visualizations:**
- Point at the linear boundary (logistic regression)
- Point at the wiggly KNN boundary
- Point at the tree-like rectangular boundary (decision tree)

> *"See how each algorithm draws a different shape?*
> *Logistic regression: always a straight line.*
> *KNN: follows the local data.*
> *Decision tree: axis-aligned rectangles.*
>
> *None is universally better. It depends on your data's true shape."*

---

## CLOSING SESSION 1  (10 min)

### Recap board

```
DECISION BOUNDARIES — SESSION 1
────────────────────────────────────────────
DEFINITION:
  The line/surface separating predicted classes.

LINEAR BOUNDARY:
  Straight line. Used by logistic regression.
  Works when data is linearly separable.

NON-LINEAR BOUNDARY:
  Curved or complex. KNN, trees, neural nets.
  Needed when classes are interleaved.

ON THE BOUNDARY:
  Probability = 0.5 (maximum uncertainty)
```

### Homework
> *"Sketch three different 2D datasets on paper:*
> *1) One that's linearly separable (can draw a straight line between classes)*
> *2) One that needs a circle boundary (one class surrounds the other)*
> *3) One that's a total mess (no clean boundary possible)*
>
> *Bring them next session — we'll discuss what each implies for algorithm choice."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Overfitting, underfitting, and complexity — through the lens of boundaries"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Homework debrief

> *"Show me your sketches. Which datasets did you draw?*
> *What algorithm would you use for each?"*

Look at their drawings. Key questions to ask:
- "Can a straight line separate those? Then logistic regression!"
- "Is one class inside the other? Then you need a circle — polynomial features."
- "Complete mess? More data or more features might help."

> *"Today: what happens when the boundary is TOO simple or TOO complex.*
> *Underfitting and overfitting — explained through decision boundary shapes."*

---

## SECTION 1: Overfitting Through Decision Boundaries  (25 min)

> *"The most visual way to understand overfitting:*
> *A boundary that's TOO wiggly memorizes the training data.*"

**Draw on board:**

```
TRAINING DATA — 10 points each class:

× × ×  o o
  ×  o  ×  o

UNDERFITTING (linear model):
  ───────────── (misclassifies many points, too simple)

GOOD FIT (moderate complexity):
  ~~~~~~~~~  (curves reasonably, captures the pattern)

OVERFITTING (KNN with K=1):
  All the × training points are correctly classified
  But the boundary looks like this:
  ×──×  o─o  ×─×
  It snakes around every single training point
  → Will fail badly on new data
```

> *"With K=1 in KNN, you always perfectly classify training data.*
> *For each training point, its nearest neighbor IS ITSELF.*
> *But the boundary is horrifying.*
>
> *Test a new point near an outlier — it gets classified wrong.*"

**The bias-variance analogy:**

```
UNDERFITTING = HIGH BIAS
  Model too simple. Ignores real patterns.
  Same wrong answer even with more data.

OVERFITTING = HIGH VARIANCE
  Model too complex. Memorizes noise.
  Small change in training data → completely different model.

GOLDILOCKS ZONE:
  Just right. Captures the real pattern, ignores noise.
```

**Ask the room:**
> *"If you trained a decision tree with unlimited depth on 100 training points,
> what would the training accuracy be?"*

Answer: 100%. It just memorizes every point. Training accuracy is useless as a solo metric.

---

## SECTION 2: Complexity Control  (20 min)

> *"Every algorithm has a way to control complexity.*
> *That's the dial between underfitting and overfitting."*

**Write on board:**

```
LOGISTIC REGRESSION:
  Complexity dial: regularization (C parameter)
  C → small: smoother, simpler boundary
  C → large: fits training data harder

K-NEAREST NEIGHBORS:
  Complexity dial: K (number of neighbors)
  K=1: extremely complex, jagged boundary
  K=large: smoother, more stable boundary

DECISION TREES:
  Complexity dial: max_depth
  depth=1: single split (underfitting)
  depth=unlimited: memorizes training data (overfitting)

GENERAL RULE:
  More complexity = lower training error, higher test error (at extremes)
  Less complexity = higher training error, but often better test error
```

**Draw the validation curve:**

```
Error
  |
  |  ←underfitting→  ←good zone→  ←overfitting→
  |
  |____
  |    \___                            _______  ← Test error
  |        \___                   ___/
  |             \_______________/
  |
  |                                            ← Training error (always goes down)
  └────────────────────────────────────────→ Complexity
     simple                          complex
```

> *"The test error curve is a U-shape.*
> *Too simple → both errors high.*
> *Too complex → training error low, test error high.*
> *Sweet spot in the middle: best generalization."*

---

## SECTION 3: How to Detect Overfitting  (15 min)

> *"You can't see the boundary in high-dimensional data.*
> *But you can detect overfitting numerically."*

**Write:**

```
SIGNS OF OVERFITTING:
  Training accuracy: 99%
  Test accuracy:     70%
  Gap > 10-15%: likely overfitting

SIGNS OF UNDERFITTING:
  Training accuracy: 70%
  Test accuracy:     68%
  Both low, small gap: model too simple

HOW TO FIX OVERFITTING:
  → More training data
  → Reduce model complexity (decrease depth, increase K, add regularization)
  → Use cross-validation

HOW TO FIX UNDERFITTING:
  → More features
  → More complex model
  → Less regularization
```

---

## SECTION 4: Running the Full Module  (10 min)

```bash
python3 05_decision_boundaries.py
```

Focus on the visualizations — boundary shape for each algorithm:

> *"This is the best visualization in the foundations section.*
> *You can see exactly what each algorithm learns.*
> *Remember these shapes — you'll see them again when we run the algorithms."*

---

## CLOSING SESSION 2  (10 min)

### Full recap board

```
DECISION BOUNDARIES — FULL PICTURE
─────────────────────────────────────────────────────
WHAT IT IS:
  Line/surface separating class predictions.

LINEAR VS NON-LINEAR:
  Linear: logistic regression (straight line)
  Non-linear: KNN, trees, neural nets (any shape)

COMPLEXITY CONTROL:
  LogReg: C (regularization)
  KNN: K value
  Trees: max_depth

OVER/UNDERFITTING:
  Overfit: boundary too wiggly, memorizes training noise
  Underfit: boundary too simple, misses real pattern

DETECTION:
  Train accuracy >> Test accuracy → overfitting
  Both low → underfitting
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"Why does K=1 overfit?"**
> *"When K=1, the only neighbor of a training point is itself.*
> *The boundary wraps around every individual point.*
> *It's like memorizing the answer key instead of understanding the material."*

**"Can you always find a perfect boundary?"**
> *"Not always. If classes genuinely overlap (noisy data, mislabeled examples),
> no classifier can get 100% accuracy.*
> *That's Bayes error rate — the irreducible minimum error.*
> *Accept it and optimize what you can control."*

**"Should I always use the most complex model?"**
> *"No! Complex models need more data to train reliably.*
> *If you have 100 training examples, a simple model often beats a complex one.*
> *Complex models shine with thousands or millions of examples."*

## Energy Management

- **This session is fun.** More drawing, less formulas.
- **If energy is high:** Have everyone sketch their own boundary for a dataset you draw.
- **30-min mark:** Natural break. Open the visualization files.
- **Best quote to end on:** "The goal isn't the most complex boundary. It's the simplest boundary that works."

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook                  10 min
├── What is a decision boundary   20 min
├── Linear boundaries             20 min
├── Non-linear boundaries         15 min
├── Live demo + visuals           15 min
└── Close + homework              10 min

SESSION 2  (90 min)
├── Homework debrief              10 min
├── Overfitting through boundaries 25 min
├── Complexity control dials      20 min
├── How to detect overfitting     15 min
├── Full module run               10 min
└── Close + recap                 10 min
```

---

*Generated for MLForBeginners — Module 05 · Part 2: Classification*
