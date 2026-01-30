# 🧪 Practice Lab: Model Evaluation

**Is your model good?** Learn to measure and compare performance!

---

## 🎯 Challenge 1: Calculate Metrics by Hand (10 min)

```python
import numpy as np

# Your model's predictions vs actual values
actual =    np.array([100, 200, 150, 180, 220])
predicted = np.array([110, 185, 155, 175, 230])

# Step 1: Calculate errors
errors = actual - predicted
print("Errors:", errors)

# Step 2: MAE (Mean Absolute Error)
# Average of |errors|
# YOUR HAND CALCULATION: (10 + 15 + 5 + 5 + 10) / 5 = ???
mae = np.mean(np.abs(errors))
print(f"MAE: {mae}")

# Step 3: MSE (Mean Squared Error)
# Average of errors²
# YOUR HAND CALCULATION: (100 + 225 + 25 + 25 + 100) / 5 = ???
mse = np.mean(errors ** 2)
print(f"MSE: {mse}")

# Step 4: RMSE (Root Mean Squared Error)
# √MSE
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# 🤔 Interpretation:
# MAE = Average error of $__ (in same units as data)
# RMSE = Typical error of $__ (penalizes big errors more)
```

---

## 🎯 Challenge 2: R² Score - How Much is Explained? (10 min)

```python
import numpy as np

actual = np.array([100, 200, 150, 180, 220])
predicted = np.array([110, 185, 155, 175, 230])

# R² = 1 - (SS_res / SS_tot)
# SS_res = Σ(actual - predicted)²
# SS_tot = Σ(actual - mean)²

ss_res = np.sum((actual - predicted) ** 2)
ss_tot = np.sum((actual - np.mean(actual)) ** 2)

r2 = 1 - (ss_res / ss_tot)

print(f"SS_res (unexplained): {ss_res}")
print(f"SS_tot (total variance): {ss_tot}")
print(f"R² Score: {r2:.4f}")
print(f"\n📊 Your model explains {r2*100:.1f}% of the variance!")

# Verify with sklearn
from sklearn.metrics import r2_score
print(f"Sklearn R²: {r2_score(actual, predicted):.4f}")
```

---

## 📊 Challenge 3: Compare Two Models (15 min)

Which model is better?

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Generate data
np.random.seed(42)
X = np.random.uniform(0, 10, 50).reshape(-1, 1)
y = 3 * X.ravel() + 5 + np.random.normal(0, 2, 50)

# Split manually
X_train, X_test = X[:40], X[40:]
y_train, y_test = y[:40], y[40:]

# Model A: Linear Regression
model_a = LinearRegression()
model_a.fit(X_train, y_train)
pred_a = model_a.predict(X_test)

# Model B: Always predict the mean (baseline)
pred_b = np.full_like(y_test, np.mean(y_train))

# Compare!
print("📊 MODEL COMPARISON")
print("=" * 50)
print(f"{'Metric':<15} {'Model A (LR)':<15} {'Model B (Mean)':<15}")
print("-" * 50)

for name, metric_fn in [('MAE', mean_absolute_error),
                         ('RMSE', lambda a, p: np.sqrt(mean_squared_error(a, p))),
                         ('R²', r2_score)]:
    score_a = metric_fn(y_test, pred_a)
    score_b = metric_fn(y_test, pred_b)
    print(f"{name:<15} {score_a:<15.3f} {score_b:<15.3f}")

# 🤔 Questions:
# 1. Which model wins? ___
# 2. What's the R² of always predicting the mean? ___
# 3. Why is Model B useful as a baseline? ___
```

---

## 🏆 Boss Challenge: Cross-Validation (15 min)

Don't trust a single train/test split!

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Generate data
np.random.seed(42)
X = np.random.uniform(0, 10, 100).reshape(-1, 1)
y = 3 * X.ravel() + 5 + np.random.normal(0, 2, 100)

# 5-fold cross-validation
model = LinearRegression()
scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("🔄 5-Fold Cross-Validation Results:")
print("-" * 40)
for i, score in enumerate(scores, 1):
    print(f"  Fold {i}: R² = {score:.4f}")

print(f"\n📊 Mean R²: {scores.mean():.4f} (±{scores.std():.4f})")

# 🤔 Think about:
# 1. Why are the scores different for each fold?
# 2. What does the std tell you about stability?
# 3. When would you use CV vs a simple train/test split?
```

---

## 🎯 When to Use Which Metric?

```
| Metric | Use When... |
|--------|-------------|
| MAE    | You want easy interpretation, outliers aren't critical |
| RMSE   | Big errors are especially bad (penalizes outliers) |
| R²     | You want to explain "% of variance explained" |
| CV     | You want robust estimates, small dataset |
```

---

## ✅ Ready for Projects When...

- [ ] You can calculate MAE, MSE, RMSE by hand
- [ ] You understand what R² means (0 = bad, 1 = perfect)
- [ ] You know why cross-validation gives more reliable estimates
- [ ] You can compare two models and pick the better one

**Congratulations!** 🎉 You've completed all the foundations! Time for real projects!

**Next up:** Capstone Projects! 🚀
