# 🎓 MLForBeginners — Instructor Guide
## Part 1 · Module 06: Linear Regression Introduction
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–05 complete. They know y=mx+b, stats, a bit of
> calculus (derivatives as slope), and basic NumPy.
> **Payoff today:** They will code the most important ML algorithm from scratch.

---

# SESSION 1 (~90 min)
## "Making the math learn — cost functions and gradient descent"

## Before They Arrive
- Terminal open in `regression_algorithms/algorithms/`
- Whiteboard ready
- Have a scatter plot of house data drawn loosely on the board

---

## OPENING (10 min)

> *"Two months ago, if I asked you 'can you predict house prices with math?'
> you probably would have said no. Today you're going to do exactly that.
> And you're going to write every single line of the code yourself.*
>
> *Everything we've done — y = mx + b, derivatives, statistics —
> was leading here. This is the moment it all clicks."*

Draw on board:
```
DATA:                    GOAL:
Size(sqft)  Price        Draw the BEST line through this data
  1000     $200K         so we can predict any size
  1500     $270K
  2000     $340K             Price
  2500     $400K               │    /
                               │   /  ← this line
                               │  /
                               │ /
                               └───────── Size
```

> *"The question is: what makes a line 'best'?
> How do we measure 'how wrong' our line is?
> That's what today is about."*

---

## SECTION 1: The Regression Equation (15 min)

Write on board:
```
ŷ = β₀ + β₁x

ŷ  = predicted value (y-hat)
β₀ = intercept (same as b)
β₁ = slope (same as m)
x  = input feature
```

> *"You've seen this before — it's y = mx + b in disguise.
> We use β (beta) notation in ML because we'll soon have
> β₀, β₁, β₂, β₃... for many features.*
>
> *The hat on the ŷ means 'predicted'. We use it to distinguish
> from y — the actual real value."*

**Interactive:** Given β₀=50,000 and β₁=150, what do we predict for 1500 sqft?
```
ŷ = 50,000 + 150 × 1500 = 50,000 + 225,000 = $275,000
```
> *"Is that right? Who knows. That depends on whether we chose
> good β₀ and β₁. Finding the BEST ones — that's the algorithm."*

---

## SECTION 2: Mean Squared Error — The Measuring Stick (20 min)

> *"To find the best line, we need to measure how bad each guess is.*
>
> *For every data point, we have:
> — Actual price (y)
> — Predicted price (ŷ)
> — Error = y − ŷ (how far off we were)*
>
> *We want to minimize total error across ALL data points."*

Draw on board:
```
      •  ← actual
      |  ← error (residual)
      ×  ← predicted (on the line)

MSE = (1/n) × Σ(y - ŷ)²

Why SQUARED?
1. Negative errors don't cancel positive ones
2. Large errors are punished more (50² = 2500, but 10² = 100)
```

**Calculate together on board** (3 data points):
```
y=[200, 250, 300]   ŷ=[210, 240, 290]
Errors: [-10, +10, +10]
Squared: [100, 100, 100]
MSE = 300/3 = 100
RMSE = √100 = $10K off on average
```

> *"The goal of linear regression:
> Find β₀ and β₁ that minimize MSE.*
>
> *This is an optimization problem. We're searching for the lowest point
> in a bowl-shaped surface of all possible (β₀, β₁) combinations."*

Draw the bowl:
```
     MSE
      │     loss surface
      │  ╲          ╱
      │    ╲      ╱
      │      ╲  ╱
      │       ╲╱  ← minimum (best β₀, β₁)
      └──────────── β₁
```

---

## SECTION 3: Gradient Descent (25 min)

> *"Here's the brilliant idea. Instead of trying every possible β,
> we start somewhere random and walk downhill.*
>
> *Remember derivatives? The derivative tells us the slope of the surface
> at any point. If the slope is positive, we go left.
> If it's negative, we go right. Always toward the minimum.*
>
> *That's gradient descent."*

Write the update rule:
```
β₁ := β₁ - α × (∂MSE/∂β₁)
β₀ := β₀ - α × (∂MSE/∂β₀)

α = learning rate (how big a step to take)
∂MSE/∂β₁ = -(2/n) × Σ xᵢ(yᵢ - ŷᵢ)
∂MSE/∂β₀ = -(2/n) × Σ (yᵢ - ŷᵢ)
```

> *"Don't panic at the formula. In code it's 5 lines.*
>
> *The learning rate α is critical:
> Too big → overshoot, bounce around, never settle
> Too small → takes forever
> Just right → converges smoothly*"

**Ask the room:** *"What happens if α = 0? What happens if α = 10,000?"*

---

## SECTION 4: Live Demo — Watch it Learn (15 min)

```bash
python3 linear_regression_intro.py
```

Watch the output together. Point at:
- The loss decreasing per epoch
- The β₀ and β₁ converging
- The final predictions

Open the visuals:
> *"Look at this — the line finding its way through the data.
> Each frame is one gradient descent step.*
>
> *This is what 'training a model' means.
> The model is adjusting β₀ and β₁ to minimize its own error."*

---

## CLOSING SESSION 1 (5 min)

```
TODAY:
  ŷ = β₀ + β₁x          (regression equation)
  MSE = (1/n)Σ(y-ŷ)²    (cost function)
  β := β - α × gradient  (gradient descent update)
```

**Homework:** What is ŷ when β₀=10, β₁=0.5, and the following x values: 20, 50, 100?

---

# SESSION 2 (~90 min)
## "The normal equation + sklearn — two ways to the same answer"

## OPENING (10 min)

> *"Last time we found the best line by walking downhill slowly —
> gradient descent, thousands of steps.*
>
> *Today I'll show you the shortcut. There's a formula that goes
> straight to the answer in one shot. It's called the normal equation.*
>
> *Then we'll let sklearn do both, and verify they all match."*

---

## SECTION 1: The Normal Equation (20 min)

Write on board:
```
β = (XᵀX)⁻¹Xᵀy

X = feature matrix (all our input data, with a column of 1s)
y = target vector
β = optimal weights — SOLVED DIRECTLY
```

> *"Remember linear algebra? Matrix multiplication, inverse?
> This formula gives us the exact optimal β in one calculation.*
>
> *Why don't we always use this?
> If X has 10 million rows, (XᵀX)⁻¹ takes forever to compute.
> Gradient descent is slower per step but scales to huge datasets.*
>
> *Small data → normal equation
> Big data → gradient descent"*

Code together:
```python
import numpy as np

X = np.array([[1, 1000], [1, 1500], [1, 2000], [1, 2500]])  # 1s for β₀
y = np.array([200000, 270000, 340000, 400000])

# Normal equation
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"β₀ = {beta[0]:.0f}, β₁ = {beta[1]:.1f}")

# Predict 1800 sqft
pred = beta[0] + beta[1] * 1800
print(f"1800 sqft → ${pred:,.0f}")
```

---

## SECTION 2: Interpreting Coefficients (15 min)

> *"Now here's what people miss: understanding what the numbers MEAN.*
>
> *β₁ = 150 means: for every extra square foot, price goes up $150.
> β₀ = 50,000 means: a 0-sqft house costs $50,000 (doesn't make real sense,
> it's just the mathematical baseline).*
>
> *In real projects, interpreting coefficients is as important
> as the prediction accuracy. A business needs to know WHY.*"

**Ask the room:** *"If β₁ for 'years of experience' in a salary model is 5000,
what does that tell you?"*

---

## SECTION 3: sklearn in 4 Lines (15 min)

```python
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1000], [1500], [2000], [2500]])
y = np.array([200000, 270000, 340000, 400000])

model = LinearRegression()
model.fit(X, y)

print(f"Slope: {model.coef_[0]:.1f}")
print(f"Intercept: {model.intercept_:.0f}")
print(f"Predict 1800: ${model.predict([[1800]])[0]:,.0f}")
```

> *"Four lines. That's it. The same algorithm, just abstracted.*
>
> *sklearn is what you'll use in production. But now you know
> exactly what it's doing inside — gradient descent or normal equation.*
>
> *Never use a tool you don't understand."*

---

## SECTION 4: R² Score — How Good Is Our Line? (15 min)

```
R² = 1 - (SS_residual / SS_total)

SS_residual = Σ(y - ŷ)²     (how wrong our model is)
SS_total    = Σ(y - ȳ)²     (how wrong "just predict the mean" is)

R² = 1.0  → perfect fit
R² = 0.0  → our model is no better than predicting the mean
R² < 0    → our model is WORSE than the mean (bad!)
```

> *"R² answers: 'What fraction of the variance in y does our model explain?'
> R² = 0.85 means our model explains 85% of why prices vary.*
>
> *Anything above 0.7 is generally useful. Above 0.9 is great.
> But beware — high R² on training data doesn't mean it generalizes!"*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
THREE WAYS TO FIT A LINE:
  1. Gradient descent    → iterative, scales to big data
  2. Normal equation     → direct formula, fast for small data
  3. sklearn             → production API, same math inside

MEASURE YOUR MODEL:
  MSE / RMSE            → error in original units
  R²                    → fraction of variance explained (0–1)
```

**Homework — from `linear_regression_intro_lab.md`:**
```python
# Given this data about study hours vs test scores:
hours  = [1, 2, 3, 4, 5, 6, 7, 8]
scores = [45, 55, 60, 65, 70, 78, 82, 90]

# 1. Plot a scatter plot
# 2. Fit a LinearRegression
# 3. What score does the model predict for 5.5 hours?
# 4. What is the R² score?
# 5. What does the slope coefficient mean in plain English?
```

---

## INSTRUCTOR TIPS

**"I don't get gradient descent"**
> *"Think of it as being blindfolded on a hilly landscape.
> You can only feel the slope under your feet.
> You always step in the downhill direction.
> Eventually you reach the valley. That's the minimum."*

**"Why not just try all possible β values?"**
> *"If β can be any real number from −∞ to +∞,
> there are infinite possibilities. We can't try them all.
> Gradient descent is the smart shortcut."*

**"When does gradient descent fail?"**
> *"If the learning rate is too big, it overshoots.
> If the function has multiple valleys (local minima),
> it might get stuck. Linear regression has only ONE minimum —
> the bowl shape guarantees it. Neural networks are harder."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                10 min
├── Regression equation         15 min
├── MSE cost function           20 min
├── Gradient descent            25 min
├── Live demo                   15 min
└── Close + homework             5 min

SESSION 2  (90 min)
├── Opening bridge              10 min
├── Normal equation             20 min
├── Interpreting coefficients   15 min
├── sklearn demo                15 min
├── R² score                    15 min
└── Close + homework            15 min
```

---
*MLForBeginners · Part 1: Regression · Module 06*
