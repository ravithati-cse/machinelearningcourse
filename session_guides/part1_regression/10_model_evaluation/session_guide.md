# MLForBeginners — Instructor Guide
## Part 1 · Module 10: Model Evaluation
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–09 complete. They can fit models, interpret coefficients,
> and have a working intuition for R². They've seen RMSE mentioned but never derived it.
> **Payoff today:** They learn to measure model quality rigorously — the difference between
> a model that looks good and one that actually IS good.

---

# SESSION 1 (~90 min)
## "Measuring how wrong we are — MAE, MSE, and RMSE"

## Before They Arrive
- Terminal open in `regression_algorithms/examples/`
- `model_evaluation.py` ready to run
- Whiteboard ready — draw the "error cartoon" below
- Calculator on your phone (for the by-hand calculation in Section 2)

---

## OPENING (10 min)

> *"You've been asking R² this whole time. Is 0.85 good? Is 0.70 good?*
>
> *But R² is just one way to measure a model's quality.*
> *There are three others — MAE, MSE, RMSE — and each answers a slightly different question.*
>
> *Today we're going to derive all four from scratch.*
> *With actual numbers. By hand. No sklearn allowed for the first 40 minutes.*
>
> *Because if you can calculate them by hand, you actually understand them.
> And if you understand them, you know WHICH one to use and WHY."*

Draw on board:
```
THE FUNDAMENTAL QUESTION:
  We predicted ŷ.
  The actual answer was y.
  The error = y - ŷ

  error =  0   → perfect prediction
  error = +50  → we underestimated by 50
  error = -50  → we overestimated by 50

  How do we summarize MANY errors into ONE number?
  That's what MAE, MSE, RMSE, and R² each do differently.
```

---

## SECTION 1: MAE — Mean Absolute Error (20 min)

> *"MAE is the simplest and most intuitive metric.*
>
> *Calculate each error. Take the absolute value. Average them.*
> *That's the MAE: the average distance between prediction and reality.*"

Write on board:
```
MAE = (1/n) × Σ |yᵢ - ŷᵢ|

  y = actual:    [200, 250, 300, 180, 220]
  ŷ = predicted: [210, 240, 310, 170, 235]
  errors:        [-10, +10,  -10, +10, -15]
  |errors|:      [ 10,  10,   10,  10,  15]
  MAE = (10+10+10+10+15)/5 = 55/5 = 11
```

> *"MAE = 11 means our predictions are off by 11 units on average.*
>
> *If these are prices in thousands: we're off by $11,000 on average.*
> *If these are temperatures in Celsius: we're off by 11 degrees.*
>
> *MAE is in the SAME UNITS as your target. That's its superpower.*
> *You can hand this number to a non-technical person and they understand it."*

**Ask the room:** *"If your target is 'number of hospital patients per day'
and MAE = 8.3, what does that mean in plain English?"*

Code together (type it, no copy-paste):
```python
import numpy as np

y_actual    = np.array([200, 250, 300, 180, 220])
y_predicted = np.array([210, 240, 310, 170, 235])

# MAE from scratch
errors = y_actual - y_predicted
mae = np.mean(np.abs(errors))
print(f"Errors:    {errors}")
print(f"|Errors|:  {np.abs(errors)}")
print(f"MAE:       {mae:.2f}")

# Verify with sklearn
from sklearn.metrics import mean_absolute_error
mae_sklearn = mean_absolute_error(y_actual, y_predicted)
print(f"Sklearn MAE: {mae_sklearn:.2f}  (should match)")
```

---

## SECTION 2: MSE — Mean Squared Error (20 min)

> *"MAE treats a $10 error and a $100 error differently — $100 is just 10× worse.*
>
> *But what if a $100 error is MUCH more than 10× problematic?
> In medical dosing, a 10× error is catastrophic.*
> *In house prices, a $100K error ruins a deal.*
>
> *MSE punishes large errors more severely — by squaring them."*

Write on board:
```
MSE = (1/n) × Σ (yᵢ - ŷᵢ)²

  Using same data:
  errors:   [-10, +10, -10, +10, -15]
  squared:  [100,  100, 100, 100, 225]
  MSE = (100+100+100+100+225)/5 = 625/5 = 125

  Notice: error of -15 contributed 225 to MSE
          error of -10 contributed 100 to MSE
  15/10 = 1.5x bigger error → 225/100 = 2.25x bigger contribution
  → Large errors penalized more than small ones
```

> *"Why MSE instead of MAE?*
>
> *MSE is differentiable everywhere — that makes it perfect for gradient descent.*
> *MAE has a kink at 0 where the derivative is undefined.*
> *In practice, MSE is the default cost function for regression for this reason.*
>
> *Downside: MSE is in squared units. If price is in dollars, MSE is in dollars².*
> *That's not interpretable. So we take the square root."*

Code together:
```python
# MSE from scratch
mse = np.mean(errors ** 2)
print(f"Squared errors: {errors ** 2}")
print(f"MSE:            {mse:.2f}")

from sklearn.metrics import mean_squared_error
mse_sklearn = mean_squared_error(y_actual, y_predicted)
print(f"Sklearn MSE:    {mse_sklearn:.2f}")
```

**Ask the room:** *"If we have one prediction that's off by 100,
and nine predictions that are off by 1,
what does MAE give us? What does MSE give us?
Which would a doctor prefer for a drug dosing model?"*

Work through together on board:
```
MAE: (100 + 9×1) / 10 = 109/10 = 10.9
MSE: (10000 + 9×1) / 10 = 10009/10 = 1000.9
RMSE: √1000.9 ≈ 31.6

MAE says "10.9 average error" — sounds manageable
RMSE says "31.6 typical error" — screams that something is very wrong
```

---

## SECTION 3: RMSE — Root Mean Squared Error (15 min)

> *"RMSE fixes MSE's unit problem. Same formula, just square rooted at the end.*
>
> *RMSE is back in original units, like MAE.*
> *But it still punishes large errors more than MAE does.*
>
> *This is the most commonly reported metric in regression papers.*"

Write on board:
```
RMSE = √MSE = √[(1/n) × Σ (yᵢ - ŷᵢ)²]

From our example:
  MSE = 125
  RMSE = √125 = 11.18

Compare to MAE = 11.0

They're close here because our errors are similar sizes.
When there are large outlier errors, RMSE >> MAE.
The gap between RMSE and MAE tells you about your outlier errors.
```

Code together:
```python
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"Gap:  {rmse - mae:.2f}")
print()
print("Interpretation: if gap is large, outlier errors are present")
```

**Common confusion box:**

> **"When do I use MAE vs RMSE?"**
> *"Use MAE when:*
> *— All errors are equally bad regardless of size*
> *— You want an easy-to-explain metric*
> *— Robustness to outliers matters*
>
> *Use RMSE when:*
> *— Large errors are especially costly*
> *— You want consistency with your MSE training loss*
> *— Gradient descent optimization is involved*"

---

## SECTION 4: Run the Module (15 min)

```bash
python3 model_evaluation.py
```

Walk through the output together:
- Section 1: Sample data generated — 100 points, true relationship `y = 3 + 2x + noise`
- Section 2: MAE, MSE, RMSE printed
- The generated visualizations in `visuals/model_evaluation/`

Open the residual plot visualization:
> *"This is new — a residual plot. The x-axis is predicted values, the y-axis is error.*
>
> *What should this look like for a good model?*
> *Random scatter around zero. No patterns.*
>
> *If you see a curve, your relationship isn't actually linear.*
> *If you see a funnel shape, your errors are bigger for bigger predictions.*
> *If you see a clear slope, you're missing a feature."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
METRIC CHEAT SHEET:
  MAE:  (1/n)Σ|y-ŷ|     — average error, easy to explain
  MSE:  (1/n)Σ(y-ŷ)²    — punishes large errors, used in training
  RMSE: √MSE             — like MAE but penalizes outliers more
  All three: lower is better, 0 = perfect

  Large gap between RMSE and MAE → outlier errors present
```

**Homework:** From `model_evaluation_lab.md` Challenge 1 — calculate MAE, MSE, RMSE
by hand for the given 5-point dataset. Verify with sklearn. Write out each step.

---

# SESSION 2 (~90 min)
## "R² score, residual diagnostics, and cross-validation"

## OPENING (10 min)

> *"Last session we covered three error metrics: MAE, MSE, RMSE.*
> *All three measure absolute error — how many units off are we?*
>
> *Today we cover R² — which measures relative quality.*
> *Not 'how many dollars off are we?' but 'how much better are we than doing nothing?'*
>
> *And then cross-validation — the answer to 'but does it GENERALIZE?'"*

---

## SECTION 1: R² — Coefficient of Determination (20 min)

> *"Here's the comparison R² makes.*
>
> *Baseline: the dumbest possible model is to predict the mean of y for everyone.*
> *How much better is our regression model than that dumb baseline?*
> *R² answers: 'what fraction of the variance does our model explain?'"*

Write on board:
```
SS_total   = Σ(y - ȳ)²      ← total variance in y
SS_residual = Σ(y - ŷ)²     ← variance our model DIDN'T explain

R² = 1 - (SS_residual / SS_total)

  R² = 1.0  → SS_residual = 0  → perfect, no unexplained variance
  R² = 0.0  → SS_residual = SS_total → model = baseline (just predict mean)
  R² < 0    → model is WORSE than the dumb baseline (disaster)

If SS_residual < SS_total → R² > 0 → we're beating the baseline
```

Code together — build R² from scratch:
```python
import numpy as np

y_actual    = np.array([200, 250, 300, 180, 220, 260, 290, 210])
y_predicted = np.array([210, 240, 310, 185, 225, 255, 285, 215])

y_mean = np.mean(y_actual)

ss_total    = np.sum((y_actual - y_mean) ** 2)
ss_residual = np.sum((y_actual - y_predicted) ** 2)

r2 = 1 - (ss_residual / ss_total)

print(f"Mean of y:     {y_mean:.1f}")
print(f"SS_total:      {ss_total:.1f}")
print(f"SS_residual:   {ss_residual:.1f}")
print(f"R²:            {r2:.4f}")
print(f"Interpretation: model explains {r2*100:.1f}% of variance in y")

from sklearn.metrics import r2_score
print(f"Sklearn R²:    {r2_score(y_actual, y_predicted):.4f}")
```

**Ask the room:** *"If R² = 0.82, what fraction of the variance is NOT explained by the model?
Where does that unexplained variance come from?"*

Desired answer: 18% unexplained — due to missing features, measurement error, inherent randomness.

---

## SECTION 2: Residual Plots — Diagnosing Your Model (20 min)

> *"Numbers like R² tell you HOW GOOD the model is.*
> *Residual plots tell you WHAT'S WRONG with it.*"*

Draw on board:
```
READING RESIDUAL PLOTS:

  GOOD — random scatter:     BAD — curved pattern:
   ●  ●                           ●
 ●   ●  ●    → linear fit OK    ●    ●   → relationship isn't linear
   ●    ●                      ●      ●    need polynomial features
  ────────────                ─────────────

  BAD — funnel shape:        BAD — trend:
  ●                            ●
   ●●                         ●●  → systematic bias
    ●●●●  → heteroscedasticity ●●●   model is missing
     ●●●●●  error variance      ●●●● a key feature
  ───────────                ──────────────
```

Code together:
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)
y_true = 3 + 2 * X.ravel() + np.random.normal(0, 2, 100)

model = LinearRegression()
model.fit(X, y_true)
y_pred = model.predict(X)

residuals = y_true - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Residuals vs predicted
axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Predicted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Predicted (should be random)')

# Residual histogram
axes[1].hist(residuals, bins=20, edgecolor='black', color='steelblue')
axes[1].set_xlabel('Residual')
axes[1].set_title('Residual Distribution (should be bell-shaped)')

plt.tight_layout()
plt.savefig('../visuals/residual_analysis.png', dpi=300)
print("Saved to visuals/residual_analysis.png")
```

> *"Two things to check:*
> *1. Residuals vs Predicted: should look like TV static — random, centered at zero.*
> *2. Residual histogram: should be roughly bell-shaped — symmetric around zero.*
>
> *If either looks structured, your model has a systematic problem.*"

---

## SECTION 3: Cross-Validation (20 min)

> *"Here's the problem with a single train/test split.*
>
> *You split your data. Maybe you got lucky and the test set was easy.*
> *Maybe you got unlucky and it was hard.*
> *Either way, your R² is a guess based on one random split.*
>
> *Cross-validation fixes this: split multiple times, average the results."*

Draw on board:
```
5-FOLD CROSS-VALIDATION:

  Full dataset: [==========|==========|==========|==========|==========]
                  Fold 1    Fold 2    Fold 3    Fold 4    Fold 5

  Round 1: TRAIN on [2,3,4,5], TEST on [1]  → R² = 0.82
  Round 2: TRAIN on [1,3,4,5], TEST on [2]  → R² = 0.79
  Round 3: TRAIN on [1,2,4,5], TEST on [3]  → R² = 0.84
  Round 4: TRAIN on [1,2,3,5], TEST on [4]  → R² = 0.81
  Round 5: TRAIN on [1,2,3,4], TEST on [5]  → R² = 0.83

  Mean R²: 0.818 ± 0.018
           ↑ more reliable   ↑ tells you how stable it is
```

Code together:
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

np.random.seed(42)
X = np.random.uniform(0, 10, 150).reshape(-1, 1)
y = 3 + 2 * X.ravel() + np.random.normal(0, 2, 150)

model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("5-Fold Cross-Validation R² scores:")
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean R²: {scores.mean():.4f}")
print(f"Std  R²: {scores.std():.4f}")
print(f"\nConclusion: model explains {scores.mean()*100:.1f}% of variance")
print(f"(± {scores.std()*100:.1f}% across folds)")
```

> *"The standard deviation of scores tells you stability.*
> *Std of 0.02 → model is reliable across different data slices.*
> *Std of 0.15 → model is sensitive to which data it sees — a warning sign.*
>
> *Always report mean AND std from cross-validation.
> One number without the other is incomplete."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
COMPLETE EVALUATION TOOLKIT:
  MAE          → average error, easy to explain
  RMSE         → penalizes outliers, standard in papers
  R²           → fraction of variance explained (0=baseline, 1=perfect)
  Residual plot → WHAT is wrong (not just HOW wrong)
  Cross-val    → HOW STABLE is the score (not just what is it)

THE EVALUATION LOOP:
  1. Fit model
  2. Calculate MAE and RMSE (how many units off?)
  3. Calculate R² (how much better than baseline?)
  4. Plot residuals (is there a systematic pattern?)
  5. Cross-validate (is R² consistent across folds?)
```

**Homework — from `model_evaluation_lab.md`:**
```python
# Boss Challenge: Cross-Validation
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Generate data: y = 3x + 5 + noise
# Run 5-fold cross-validation
# Answer:
#   1. What is mean R²?
#   2. What is std of R²? Is the model stable?
#   3. When would you prefer CV over a single split?
```

---

## INSTRUCTOR TIPS

**"Is R² = 0.70 good or bad?"**
> *"Depends on the problem. For predicting stock prices? Phenomenal.*
> *For predicting whether a bolt will fail? Dangerous.*
>
> *The right question is: what R² does the baseline get?
> (Always 0.) How much better are you than that?
> And what does the error mean in the real-world context?"*

**"Why doesn't sklearn have RMSE as a built-in scorer?"**
> *"It does now in newer versions — `neg_root_mean_squared_error`.*
> *But traditionally RMSE was just `np.sqrt(mean_squared_error(...))`.*
> *This is why knowing the formula matters — API gaps happen,
> and you need to be able to roll your own."*

**"What's the difference between training R² and test R²?"**
> *"Training R²: how well the model fits the data it was trained on.*
> *Always optimistic — the model has seen this data.*
>
> *Test R²: how well it generalizes to new data.*
> *This is the number that actually matters.*
>
> *If training R² = 0.99 and test R² = 0.55, your model memorized training data.*
> *That's overfitting — coming up in classification modules."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                    10 min
├── MAE — derivation + code         20 min
├── MSE — squaring errors           20 min
├── RMSE — back to original units   15 min
├── Run model_evaluation.py         15 min
└── Close + homework                10 min

SESSION 2  (90 min)
├── Opening bridge                  10 min
├── R² from scratch                 20 min
├── Residual plots                  20 min
├── Cross-validation                20 min
└── Close + homework                20 min
```

---
*MLForBeginners · Part 1: Regression · Module 10*
