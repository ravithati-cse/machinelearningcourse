# MLForBeginners — Instructor Guide
## Part 2 · Module 06: Logistic Regression Introduction
### Two-Session Teaching Script

> **Prerequisites:** Part 1 complete + Part 2 Modules 01–05. They know y=mx+b,
> MSE, gradient descent, sigmoid function, probability, log-loss, confusion matrix,
> and decision boundaries.
> **Payoff today:** They will connect every math concept from Part 2 into one
> complete algorithm — and build the most widely-used classifier from scratch.

---

# SESSION 1 (~90 min)
## "From straight lines to probabilities — the logistic pipeline"

## Before They Arrive
- Terminal open in `classification_algorithms/algorithms/`
- Whiteboard ready, draw two columns: "Linear Regression" vs "Logistic Regression"
- Have a simple scatter plot drawn: two clusters of points, one labelled "spam", one "ham"

---

## OPENING (10 min)

> *"You've spent the last month building intuition for classification.
> You know sigmoid. You know log-loss. You know confusion matrices.*
>
> *Today everything clicks together into one algorithm that powers
> spam filters, fraud detectors, disease screeners, and credit scoring.*
>
> *Logistic regression is probably the most deployed ML algorithm on Earth.
> And you're going to build it from scratch today."*

Draw on board:
```
THE JOURNEY SO FAR:
  y = mx + b              (Part 1 — predicts any number)
  sigmoid(z) → (0, 1)    (Module 01 — squashes to probability)
  log-loss                (Module 03 — measures how wrong we are)
  confusion matrix        (Module 04 — counts our mistakes)
  decision boundary       (Module 05 — the line we're learning)

TODAY: Combine ALL of these into one algorithm.
```

> *"Every piece of math you've learned was a Lego brick.
> Today we snap them together."*

---

## SECTION 1: Why Not Just Use Linear Regression? (15 min)

Write on board:
```
LINEAR REGRESSION OUTPUT:
  z = β₀ + β₁x₁ + β₂x₂ + ...
  Range: -∞ to +∞

PROBLEM: We need probabilities. 0 to 1. Not 500. Not -17.
```

> *"Imagine trying to classify an email as spam.*
>
> *Linear regression might output: 2.7, -0.3, 15.4*
>
> *What does 15.4 mean as a probability of spam?
> Nothing. You can't interpret that.*
>
> *We need a number between 0 and 1.
> We need: 0.95 → very probably spam. 0.08 → almost certainly not spam."*

Draw the problem visually:
```
SPAM DATA:   x = number of exclamation marks

Linear:  z = 0.1 + 0.8x
         When x=10: z = 8.1    ← meaningless as probability!

We need: P(spam) = ??? such that 0 ≤ P ≤ 1
```

**Ask the room:** *"You already know the function that squashes any number into (0,1).
What is it?"*

They should say: sigmoid.

> *"Exactly. That's the entire idea of logistic regression in one sentence:
> run linear regression, then push the result through sigmoid."*

---

## SECTION 2: The Full Pipeline (20 min)

Write on board — this is the core diagram they should memorize:
```
INPUT FEATURES                  LOGISTIC REGRESSION PIPELINE
x₁ = num exclamation marks  →  z = β₀ + β₁x₁ + β₂x₂     (linear model)
x₂ = contains "free"        →                               ↓
x₃ = email length           →  log-odds = z                (unbounded)
                                                             ↓
                                P(spam) = sigmoid(z)        (0 to 1)
                                        = 1 / (1 + e^(-z))
                                                             ↓
                                Decision: P > 0.5 → SPAM   (binary)
                                          P ≤ 0.5 → HAM
```

> *"Let's walk through this step by step with real numbers.*
>
> *Say we have an email with 3 exclamation marks, no 'free', and length 200.*
>
> *Our (made-up) model has: β₀ = -2, β₁ = 0.8, β₂ = 1.5, β₃ = 0.001*"

Calculate together:
```
z = -2 + (0.8 × 3) + (1.5 × 0) + (0.001 × 200)
  = -2 + 2.4 + 0 + 0.2
  = 0.6

P(spam) = sigmoid(0.6)
        = 1 / (1 + e^(-0.6))
        = 1 / (1 + 0.549)
        = 1 / 1.549
        ≈ 0.645

Decision: 0.645 > 0.5 → SPAM
```

> *"That's the whole prediction pipeline. Now how do we find the right betas?
> Same as always — gradient descent. But now minimizing log-loss instead of MSE."*

Briefly remind them:
```
LOG-LOSS (from Module 03):
  L = -[y·log(P) + (1-y)·log(1-P)]

  Actual spam    (y=1), predicted P=0.95 → small loss (good)
  Actual spam    (y=1), predicted P=0.10 → large loss (bad)
  Actual not spam(y=0), predicted P=0.90 → large loss (bad)
```

> *"We minimize the average log-loss across all emails.
> Gradient descent adjusts β₀, β₁, β₂... to make our predictions
> as close to the truth as possible."*

---

## SECTION 3: Log-Odds — What the Betas Really Mean (20 min)

> *"Before we code, I want to explain what the betas actually mean.
> This is the part most courses skip — and it trips people up later."*

Write on board:
```
z = β₀ + β₁x₁ + ...     ← this is called the LOG-ODDS

WHY "LOG-ODDS"?

Odds = P(event) / P(not event)
     = P / (1 - P)

     If P = 0.75 (75% chance of spam):
       Odds = 0.75 / 0.25 = 3  ("3 to 1 odds of spam")

Log-odds = log(Odds) = log(P / (1-P))

It turns out: sigmoid(log-odds) = probability
And:          log(P / (1-P)) = β₀ + β₁x₁ + ...
```

> *"So each beta tells you: for one unit increase in that feature,
> the log-odds of the outcome change by that beta.*
>
> *β₁ = 0.8 for exclamation marks means: each extra '!'
> increases the log-odds of spam by 0.8.*
>
> *In practice you'll exponentiate: e^0.8 ≈ 2.2 means each '!'
> multiplies the odds of spam by 2.2. That's an interpretable insight."*

**Ask the room:** *"If a feature has β = 0, what does that mean for prediction?"*

Answer: That feature has no effect. The model ignores it.

**Ask the room:** *"If β is negative, what does that mean?"*

Answer: Higher values of that feature make the positive class less likely.

---

## SECTION 4: Live Demo — Watch It Learn (20 min)

```bash
python3 logistic_regression_intro.py
```

Watch the output together. Point at:
- The loss decreasing per training epoch
- The betas converging to stable values
- The final probability predictions on test emails
- The decision boundary visualization

Open the visuals:
> *"Look at the decision boundary plot. That straight line is where P=0.5.*
>
> *Everything to the right is classified as spam.
> Everything to the left as ham.*
>
> *The line's position and angle are determined entirely by β₀, β₁, β₂.*
> *Gradient descent found those betas — we just watched it happen."*

---

## CLOSING SESSION 1 (5 min)

```
TODAY'S PIPELINE:
  z = β₀ + β₁x₁ + β₂x₂ + ...   (linear model = log-odds)
  P = sigmoid(z)                   (convert to probability)
  ŷ = 1 if P > 0.5, else 0        (decision)
  Loss = -[y·log(P) + (1-y)·log(1-P)]   (log-loss)
  β := β - α × gradient           (gradient descent)
```

**Homework:** Given β₀ = -3, β₁ = 1.2 (feature = years of credit history):
What is P(default) for someone with 0, 2, 5, and 10 years of credit history?
What does the sign of β₁ tell you about the relationship?

---

# SESSION 2 (~90 min)
## "From scratch code + sklearn + interpreting what the model learned"

## OPENING (10 min)

> *"Last session we understood the pipeline conceptually.*
> *Today we write every line of code.*
>
> *By the end of this session, you'll have a working logistic regression
> implementation in pure numpy AND in sklearn.*
>
> *More importantly, you'll know how to read what the model learned —
> what features matter, and why."*

---

## SECTION 1: Implement From Scratch (25 min)

Code together line by line:

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.n_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            predictions = self.sigmoid(z)

            # Gradients (from calculus — derived from log-loss)
            dw = (1/n_samples) * X.T @ (predictions - y)
            db = (1/n_samples) * np.sum(predictions - y)

            # Update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            if epoch % 100 == 0:
                loss = self._log_loss(y, predictions)
                print(f"Epoch {epoch}: loss = {loss:.4f}")

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _log_loss(self, y, p):
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
```

> *"Look at the gradient formulas: dw = X.T @ (predictions - y).*
>
> *The error is (predictions - y) — exactly like linear regression!*
>
> *The math works out elegantly because of how log-loss and sigmoid interact.*
>
> *The clip on sigmoid prevents numerical overflow — a practical detail
> you'll see in production code."*

**Ask the room:** *"What does `X @ self.weights` do in numpy terms?"*

Answer: Matrix multiplication — computes the dot product for every sample at once.

---

## SECTION 2: Test Your Implementation (10 min)

```python
# Breast cancer style data (2 features for visualization)
np.random.seed(42)
n = 200

# Class 0: benign tumors
X0 = np.random.randn(n//2, 2) + np.array([-2, -2])
# Class 1: malignant tumors
X1 = np.random.randn(n//2, 2) + np.array([2, 2])

X = np.vstack([X0, X1])
y = np.array([0]*(n//2) + [1]*(n//2))

# Shuffle
idx = np.random.permutation(n)
X, y = X[idx], y[idx]

# Train
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=500)
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"\nAccuracy: {accuracy:.1%}")
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias:.4f}")
```

> *"Run it. Watch the loss drop each 100 epochs.*
>
> *The final weights tell you: which feature matters more for this prediction.*
>
> *Larger absolute weight = more influential feature."*

---

## SECTION 3: sklearn in 5 Lines (10 min)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model_sk = LogisticRegression()
model_sk.fit(X_train, y_train)
y_pred = model_sk.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.1%}")
print(classification_report(y_test, y_pred))
print(f"Coefficients: {model_sk.coef_}")
print(f"Intercept: {model_sk.intercept_}")
```

> *"Same algorithm, different API. The coefficients should be close to what
> our scratch version found.*
>
> *sklearn uses a more sophisticated optimizer — L-BFGS by default —
> but the math is the same underlying idea."*

---

## SECTION 4: Interpreting What the Model Learned (20 min)

> *"Here's the part that separates good practitioners from great ones:
> reading the model's output.*
>
> *Let's use a real-world example — predicting loan default.*"

Write on board:
```
IMAGINARY LOAN MODEL:
  Feature               Beta     e^Beta (odds multiplier)
  ─────────────────────────────────────────────────────────
  Years employed        -0.4     0.67   (more years = safer)
  Debt-to-income ratio  +1.2     3.32   (more debt = riskier)
  Has savings account   -0.9     0.41   (savings = much safer)
  Number of late pays   +2.1     8.17   (huge red flag!)
  Intercept (β₀)        -1.5     ---
```

> *"Let's read this table:*
>
> *Beta = -0.4 for years employed: each extra year of employment
> multiplies your odds of default by 0.67 — your risk drops by 33%.*
>
> *Beta = +2.1 for late payments: each late payment multiplies your
> default odds by 8.17. That's the strongest signal in the model.*
>
> *This is why banks use logistic regression — it's not a black box.
> You can explain every decision to a regulator."*

**Ask the room:** *"Which feature has the biggest impact in this model?
How do you know?"*

---

## SECTION 5: The Lab (15 min)

Point to `logistic_regression_intro_lab.md` — they have a working lab file.

Quick preview of Task 1:
```python
# The lab starts with this quick win:
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Spam score model
def predict_spam(exclamation_marks):
    z = 2 * exclamation_marks - 3
    probability = sigmoid(z)
    return probability

# Their job: test it for 0, 1, 2, 3, 5 exclamation marks
# and interpret: at what point does it tip to SPAM?
```

> *"Work through the lab tasks. The key goal is Task 3:
> implement your own LogisticRegression from scratch on a new dataset.
> Don't copy the class we wrote — retype it, understand each line."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
LOGISTIC REGRESSION SUMMARY:
  Linear model:   z = β₀ + β₁x₁ + β₂x₂ + ...  (log-odds)
  Probability:    P = sigmoid(z) = 1/(1+e^(-z))
  Decision:       ŷ = 1 if P > threshold
  Training:       minimize log-loss via gradient descent
  Interpretation: e^βᵢ = odds multiplier for feature i

WHEN TO USE IT:
  Binary classification problem
  Want a probability, not just a label
  Need to explain decisions (regulated industries)
  Large dataset, need speed
  Good baseline before trying complex models
```

**Homework — from `logistic_regression_intro_lab.md`:**
```python
# Complete the remaining lab tasks:
# Task 2: Train on the diabetes dataset, report accuracy + confusion matrix
# Task 3: Implement LogisticRegression from scratch, train, compare to sklearn
# Task 4: Interpret coefficients — which feature is most important?
# Task 5: Try thresholds 0.3 and 0.7. How do precision and recall change?
```

---

## INSTRUCTOR TIPS

**"Why log-loss and not MSE for classification?"**
> *"MSE is bowl-shaped for linear regression — gradient descent works perfectly.*
> *For classification with sigmoid, MSE creates a non-convex surface with local
> minima. Log-loss stays convex, so gradient descent always finds the global minimum."*

**"What's the difference between beta and the odds ratio?"**
> *"Beta is the log-odds change. e^beta is the odds multiplier.*
> *Practitioners usually report e^beta because 'risk multiplies by 2.2'
> is more intuitive than 'log-odds increases by 0.8'."*

**"Can logistic regression handle non-linear boundaries?"**
> *"Not directly. The decision boundary is always a straight line (or hyperplane).*
> *But you can engineer polynomial features: add x₁² as a feature, and suddenly
> the boundary can curve. This is called feature engineering."*

**"What if my data is not balanced — 95% spam, 5% ham?"**
> *"sklearn has `class_weight='balanced'` to handle this.*
> *Or use `sample_weight` to up-weight rare classes.*
> *We'll cover this in detail in Module 10 and the churn project."*

**"How does regularization work in LogisticRegression?"**
> *"sklearn's `C` parameter controls regularization (opposite convention to Ridge).*
> *Small C = strong regularization = simpler model.*
> *Large C = weak regularization = fits training data more closely.*
> *Default C=1 is usually a good starting point."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                   10 min
├── Why not linear regression      15 min
├── The full pipeline              20 min
├── Log-odds and interpretation    20 min
├── Live demo                      20 min
└── Close + homework                5 min

SESSION 2  (90 min)
├── Opening bridge                 10 min
├── Implement from scratch         25 min
├── Test implementation            10 min
├── sklearn in 5 lines             10 min
├── Interpreting coefficients      20 min
├── Lab preview                    15 min
└── Close + homework               10 min
```

---
*MLForBeginners · Part 2: Classification · Module 06*
