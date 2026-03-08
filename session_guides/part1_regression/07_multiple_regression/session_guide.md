# MLForBeginners — Instructor Guide
## Part 1 · Module 07: Multiple Regression
### Two-Session Teaching Script

> **Prerequisites:** Module 06 complete. They can write `ŷ = β₀ + β₁x`, understand gradient descent,
> and have used `LinearRegression` from sklearn. They know y = mx + b inside and out.
> **Payoff today:** They graduate from one-input models to the real world — where every prediction
> uses many features at once.

---

# SESSION 1 (~90 min)
## "From one feature to many — the feature matrix"

## Before They Arrive
- Terminal open in `regression_algorithms/algorithms/`
- Whiteboard ready — draw the two-column vs multi-column table shown below
- Have `multiple_regression.py` ready to scroll through
- Think of a local real-estate example with at least 3 obvious features

---

## OPENING (10 min)

> *"Last module we predicted house prices from size alone.*
>
> *But when you actually buy a house, what do you look at?
> Size. Number of bedrooms. Age. Neighborhood. Distance to schools.*
> *Dozens of things. One feature is never the whole story.*
>
> *Today we extend everything you know — y = mx + b — to handle
> as many features as we want. Same math. Same gradient descent.
> Just more columns."*

Draw on board:
```
SIMPLE regression:           MULTIPLE regression:

   x     →   y                 x₁  x₂  x₃  →   y
 (size)    (price)           (size)(bed)(age)  (price)

 1 column of input           3 columns of input
     ↓                             ↓
  ŷ = β₀ + β₁x              ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃
```

> *"Notice what changed: we added more x's, and more β's.
> β₁ is no longer 'the slope' — it's the slope for feature 1,
> holding everything else fixed. That phrase — 'holding everything else fixed' —
> is the key idea of the entire session."*

---

## SECTION 1: Why One Feature Is Never Enough (15 min)

Write on board:
```
SIMPLE MODEL:
  Price = β₀ + β₁ × size
  Prediction: $280,000

REALITY:
  House A: 1,800 sqft, 2 bedrooms, built 1950 → $240K
  House B: 1,800 sqft, 4 bedrooms, built 2020 → $420K

Same size. Very different prices.
The model that only knows SIZE is confused.
```

> *"Both houses are 1,800 sqft. Our simple model predicts the same price for both.
> But we know that's wrong — the age and bedroom count matter.*
>
> *That's the fundamental limitation of simple regression:
> any variation that ISN'T captured in your one feature
> becomes unexplained error. More features → less unexplained error."*

**Ask the room:** *"What other features would help predict house prices?
Name three that aren't on the board already."*

Take answers, write them up. Common good ones: garage, school rating, lot size.

> *"Every one of those becomes another column in our feature matrix.
> Multiple regression handles all of them at once."*

---

## SECTION 2: The Feature Matrix X (20 min)

> *"Here is the core data structure for all of machine learning.
> It's called the feature matrix, written as capital X.*
>
> *Every row is one sample. Every column is one feature."*

Draw on board:
```
         β₀    β₁      β₂       β₃
         │     │        │        │
        [1,  1500,     3,       10]   ← house 1: 1500sqft, 3bed, 10yr
   X  = [1,  2000,     4,        5]   ← house 2: 2000sqft, 4bed,  5yr
        [1,  1200,     2,       20]   ← house 3: 1200sqft, 2bed, 20yr
        [1,  2200,     5,        2]   ← house 4: 2200sqft, 5bed,  2yr
         ↑
  column of 1s for β₀ (the intercept)

   y  = [300000, 450000, 200000, 520000]
```

> *"That first column of 1s is the trick that absorbs the intercept β₀ into the matrix math.
> When we multiply X by our weights β, the 1 in the first column ensures β₀
> always gets added in. Elegant.*
>
> *Now our prediction for any house is just a dot product:*"

Write on board:
```
ŷ = X · β

ŷ = [1, 1500, 3, 10] · [β₀, β₁, β₂, β₃]
  = β₀×1 + β₁×1500 + β₂×3 + β₃×10
  = intercept + size_contrib + bedroom_contrib + age_contrib
```

> *"This works for 3 features, 30 features, or 300 features.
> The same matrix multiplication handles all of them.
> This is why linear algebra is in your prerequisites."*

Code together (live, from scratch):
```python
import numpy as np

X = np.array([
    [1, 1500, 3, 10],
    [1, 2000, 4,  5],
    [1, 1200, 2, 20],
    [1, 2200, 5,  2],
])
y = np.array([300000, 450000, 200000, 520000])

# Guess some weights
beta = np.array([50000, 100, 40000, -5000])

# Predict house 1
pred_house1 = X[0] @ beta   # dot product
print(f"House 1 prediction: ${pred_house1:,.0f}")
print(f"House 1 actual:     ${y[0]:,.0f}")
print(f"Error:              ${y[0] - pred_house1:,.0f}")
```

**Ask the room:** *"How would you change beta to reduce that error?"*

---

## SECTION 3: The Normal Equation for Multiple Features (20 min)

> *"Remember the normal equation from last module?
> β = (XᵀX)⁻¹Xᵀy — find the optimal weights in one shot.*
>
> *The beautiful thing: this formula works EXACTLY the same way
> for 1 feature or 1,000 features. X just gets more columns.*"

Code together:
```python
import numpy as np

X = np.array([
    [1, 1500, 3, 10],
    [1, 2000, 4,  5],
    [1, 1200, 2, 20],
    [1, 2200, 5,  2],
    [1, 1800, 3,  8],
])
y = np.array([300000, 450000, 200000, 520000, 350000])

# Normal equation: β = (XᵀX)⁻¹Xᵀy
beta_optimal = np.linalg.inv(X.T @ X) @ X.T @ y

print("Optimal coefficients:")
print(f"  β₀ (intercept):  ${beta_optimal[0]:>10,.0f}")
print(f"  β₁ (per sqft):   ${beta_optimal[1]:>10,.2f}")
print(f"  β₂ (per bedroom):${beta_optimal[2]:>10,.0f}")
print(f"  β₃ (per year):   ${beta_optimal[3]:>10,.0f}")

# Predict a new house: 1700sqft, 3bed, 15yr
new_house = np.array([1, 1700, 3, 15])
prediction = new_house @ beta_optimal
print(f"\nNew house prediction: ${prediction:,.0f}")
```

> *"Look at β₃ — the coefficient for age.
> It should be negative: older houses are worth less, all else equal.
> Does the number make intuitive sense?"*

**Common confusion box:**

> **"Why is β₂ (bedrooms) sometimes negative or weird?"**
> *"Multicollinearity — bedrooms and size are highly correlated.
> Add more bedrooms → house tends to be bigger too.
> The model gets confused about which feature deserves credit.*
>
> *This is normal. We'll explore it in Session 2.*
> *For now: when features are correlated, individual coefficients
> can behave oddly even when predictions are accurate."*

---

## SECTION 4: Run the Module (15 min)

```bash
python3 multiple_regression.py
```

Walk through the output together. Point at each section:
- Section 1: The limitation of simple regression printed clearly
- Section 2: The feature matrix being constructed
- Section 3: The normal equation solution
- The generated visualizations in `visuals/regression/`

Open the 3D scatter plot visualization:
> *"This is what multiple regression looks like geometrically.*
>
> *With one feature, we fit a LINE through 2D data.*
> *With two features, we fit a PLANE through 3D data.*
> *With three features, we fit a hyperplane through 4D space —
> impossible to visualize, but the math is identical."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
MULTIPLE REGRESSION:
  ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

MATRIX FORM:
  ŷ = X · β

SOLVE WITH NORMAL EQUATION:
  β = (XᵀX)⁻¹Xᵀy

GEOMETRY:
  1 feature  → best LINE (2D)
  2 features → best PLANE (3D)
  n features → best HYPERPLANE (n+1 D)
```

**Homework:** On paper — if β₀=50000, β₁=100, β₂=20000, β₃=-3000,
what does the model predict for a 1,600 sqft, 3-bedroom, 12-year-old house?

---

# SESSION 2 (~90 min)
## "Feature importance, multicollinearity, and sklearn"

## OPENING (10 min)

> *"Last session we got the math working — we can predict prices from multiple features.*
>
> *Today's question is harder: WHICH features actually matter?
> Is bedroom count as important as size? Is age as important as location?*
>
> *And there's a trap called multicollinearity that trips up every data scientist at some point.
> We're going to step right into it, on purpose, so you know what it looks like."*

---

## SECTION 1: Feature Importance and Standardization (20 min)

> *"Here's the trap. Our coefficients are:
> β₁ (sqft) = $80/sqft, β₂ (bedrooms) = $30,000/bedroom.*
>
> *Which feature matters more?*
>
> *You might say bedrooms — the number is bigger. But size varies by hundreds of sqft,
> while bedrooms vary by just 1 or 2. The SCALE is different.*
>
> *To compare feature importance fairly, we need to standardize.*"

Draw on board:
```
BEFORE SCALING:              AFTER STANDARDIZING (z-score):
  size: 1000 to 3000 sqft      size: -1.5 to +1.5
  bedrooms: 1 to 6             bedrooms: -1.5 to +1.5
  age: 1 to 40 years           age: -1.5 to +1.5

  Now all features are on the SAME SCALE.
  The coefficient magnitude = the actual importance.
```

Code together:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = np.array([
    [1500, 3, 10],
    [2000, 4,  5],
    [1200, 2, 20],
    [1800, 3,  8],
    [2200, 5,  2],
])
y = np.array([300000, 450000, 200000, 350000, 520000])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

features = ['Size (sqft)', 'Bedrooms', 'Age (years)']
print("Feature Importance (standardized coefficients):")
print("-" * 50)
for name, coef in sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True):
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {name:<15}: ${abs(coef):>8,.0f} impact ({direction} price)")
```

> *"Now we can actually compare them.
> The feature with the biggest standardized coefficient IS the most important,
> in the sense that a one-standard-deviation change has the most impact on price."*

**Ask the room:** *"If size has a bigger coefficient than bedrooms after scaling,
what does that tell a real estate agent?"*

---

## SECTION 2: Multicollinearity — The Hidden Problem (20 min)

> *"Here's the trap I promised. What happens if two features are nearly identical?*
>
> *Imagine predicting salary from 'years of experience' AND 'years since college graduation'.
> For most people, those are almost the same number.*
>
> *The model tries to split the credit between them — and fails."*

Draw on board:
```
MULTICOLLINEARITY:
  If x₁ ≈ x₂ (highly correlated):
    - β₁ and β₂ become unstable
    - Small changes in data → huge swings in coefficients
    - Individual coefficients are uninterpretable
    - But PREDICTIONS can still be accurate!

  Detect with: correlation matrix
  Fix with:
    1. Remove one of the correlated features
    2. Combine them (e.g., average)
    3. Use Ridge/Lasso regression (Part 2 preview)
```

Code together (show the problem):
```python
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

# Correlated features: total_rooms ≈ total_bedrooms
np.random.seed(42)
n = 100
size = np.random.uniform(1000, 3000, n)
bedrooms = size / 400 + np.random.normal(0, 0.3, n)  # nearly = size/400
price = 50000 + 80 * size + 25000 * bedrooms + np.random.normal(0, 10000, n)

X_correlated = np.column_stack([size, bedrooms])
model = LinearRegression().fit(X_correlated, price)
print("Correlated features:")
print(f"  β_size:     {model.coef_[0]:,.2f}")
print(f"  β_bedrooms: {model.coef_[1]:,.2f}")
print(f"  (True β_size should be ~80, β_bedrooms should be ~25000)")
print(f"  Correlation between size and bedrooms: {np.corrcoef(size, bedrooms)[0,1]:.3f}")
```

> *"The individual coefficients are wrong — they don't reflect the true relationships.
> Yet the overall predictions might still be decent.*
>
> *This is why 'the model works' and 'the coefficients make sense' are two different things.
> Always check your correlation matrix before interpreting coefficients."*

---

## SECTION 3: sklearn — The Production API (15 min)

> *"Now let's use multiple regression the way you'll actually use it in the real world."*

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# Build a realistic dataset
np.random.seed(42)
n = 200
size     = np.random.uniform(800, 3500, n)
bedrooms = np.random.randint(1, 7, n)
age      = np.random.uniform(0, 50, n)
price    = 30000 + 100*size + 20000*bedrooms - 2000*age + np.random.normal(0, 20000, n)

X = np.column_stack([size, bedrooms, age])
y = price

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"Test RMSE:  ${rmse:,.0f}")
print(f"Test R²:    {r2:.4f}")
print(f"\nCoefficients:")
print(f"  Per sqft:    ${model.coef_[0]:,.2f}")
print(f"  Per bedroom: ${model.coef_[1]:,.0f}")
print(f"  Per year:    ${model.coef_[2]:,.0f}")
```

> *"Compare these recovered coefficients to the 'true' ones we set:
> 100 per sqft, 20000 per bedroom, -2000 per year.
> How close are they? That's how well multiple regression recovers reality."*

---

## SECTION 4: Lab Walkthrough (10 min)

Open `regression_algorithms/algorithms/multiple_regression_lab.md`.

Walk through the "Feature Importance" task together:
> *"This is the same pattern you just saw — fit on scaled data, read coefficients by size.*
>
> *The 'Interpretation Challenge' is the one I want everyone to actually write.*
> Write two sentences that a non-technical person could read.
> That skill — translating math to plain English — is what separates
> good data scientists from great ones."*

Assign the Boss Challenge (add `bedrooms²`) as homework.

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
MULTIPLE REGRESSION TOOLKIT:
  Model:           ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
  Matrix form:     ŷ = X · β
  Best β:          Normal equation or gradient descent

  Feature importance:  Standardize first, then compare |βᵢ|
  Multicollinearity:   When features correlate, coefficients get unstable
                       Check: correlation matrix
                       Fix: drop one feature, or use regularization

  sklearn API:         model.fit(X, y)
                       model.coef_     ← one per feature
                       model.intercept_
```

**Homework — from `multiple_regression_lab.md`:**
```python
# Boss Challenge: Add a new engineered feature
# Original features: [size, bedrooms, age]
# Add bedrooms² as feature 4
bedrooms_squared = X_original[:, 1] ** 2
X_enhanced = np.column_stack([X_original, bedrooms_squared])
# Compare R² of original vs enhanced model
```

---

## INSTRUCTOR TIPS

**"Why do we need to scale before comparing coefficients?"**
> *"Imagine you're comparing 'height in meters' vs 'height in centimeters'.
> The number for centimeters is 100× bigger but it's measuring the same thing.
> Features measured in different units have this problem.
> Standardizing puts every feature on the same 'number of standard deviations' scale."*

**"My coefficients look wrong after adding more features"**
> *"That's multicollinearity. Which features are correlated with each other?
> Print `np.corrcoef(X.T)` — any pair above 0.8 is suspicious.
> Remove one, refit, and see if the coefficients stabilize."*

**"When should I use multiple regression vs adding more features?"**
> *"You almost always want more relevant features — they reduce unexplained variance.
> The danger is adding irrelevant or redundant features.
> A good rule: only add a feature if you can explain WHY it should affect the target."*

**"Can β be negative?"**
> *"Absolutely. Age has a negative coefficient: older house → lower price.
> Distance to city center often has a negative coefficient too.
> Negative doesn't mean wrong — it means that feature works against the prediction."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                    10 min
├── Why one feature fails           15 min
├── Feature matrix X                20 min
├── Normal equation — multiple      20 min
├── Run multiple_regression.py      15 min
└── Close + homework                 5 min

SESSION 2  (90 min)
├── Opening bridge                  10 min
├── Standardization + importance    20 min
├── Multicollinearity               20 min
├── sklearn production API          15 min
├── Lab walkthrough                 15 min
└── Close + homework                10 min
```

---
*MLForBeginners · Part 1: Regression · Module 07*
