# 🧪 Practice Lab: Multiple Regression

**More features = more predictive power!** Let's harness it.

---

## 🎯 Quick Win: Feature Importance (10 min)

Which feature matters most for house prices?

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# House data: [size_sqft, bedrooms, age_years]
X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 20],
    [1800, 3, 8],
    [2200, 5, 2],
])
y = np.array([300000, 450000, 200000, 350000, 520000])

# Scale features for fair comparison
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_scaled, y)

# Reveal importance!
features = ['Size (sqft)', 'Bedrooms', 'Age (years)']
print("🏆 Feature Importance Ranking:")
print("-" * 40)

for name, coef in sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True):
    direction = "↑" if coef > 0 else "↓"
    print(f"  {name}: {direction} ${abs(coef):,.0f} impact")

# 🤔 Questions:
# 1. Which feature has the BIGGEST impact?
# 2. Does age INCREASE or DECREASE price?
# 3. Is this what you expected?
```

---

## 🔮 Prediction Challenge (10 min)

Predict prices for new houses!

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X_train = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 20],
    [1800, 3, 8],
    [2200, 5, 2],
])
y_train = np.array([300000, 450000, 200000, 350000, 520000])

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# New houses to predict
new_houses = np.array([
    [1700, 3, 15],  # House A
    [2500, 5, 3],   # House B
    [1000, 1, 30],  # House C (small, old)
])

# Predict!
predictions = model.predict(new_houses)

print("🔮 Price Predictions:")
print("-" * 40)
labels = ['A (1700sqft, 3bed, 15yr)', 'B (2500sqft, 5bed, 3yr)', 'C (1000sqft, 1bed, 30yr)']
for label, pred in zip(labels, predictions):
    print(f"  House {label}: ${pred:,.0f}")

# 🤔 Does House C's prediction make sense?
# (Might be negative or very low - why?)
```

---

## 📝 Interpretation Challenge (10 min)

Explain the model to a non-technical person!

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 20],
    [1800, 3, 8],
    [2200, 5, 2],
])
y = np.array([300000, 450000, 200000, 350000, 520000])

model = LinearRegression()
model.fit(X, y)

print("📝 Model Interpretation:")
print(f"Base price (intercept): ${model.intercept_:,.0f}")
print(f"Per square foot: ${model.coef_[0]:,.0f}")
print(f"Per bedroom: ${model.coef_[1]:,.0f}")
print(f"Per year of age: ${model.coef_[2]:,.0f}")

# YOUR TURN: Write a 2-sentence business summary!
# Example: "Our model suggests that size has the biggest impact..."
#
# Your summary:
# _______________________________________________
# _______________________________________________
```

---

## 🏆 Boss Challenge: Add a New Feature (15 min)

What if we add `bedrooms²` as a feature? (This captures diminishing returns!)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Original data
X_original = np.array([
    [1500, 3, 10],
    [2000, 4, 5],
    [1200, 2, 20],
    [1800, 3, 8],
    [2200, 5, 2],
    [1600, 3, 12],
    [1900, 4, 7],
])
y = np.array([300000, 450000, 200000, 350000, 520000, 280000, 400000])

# Add bedrooms² as a new feature!
bedrooms_squared = X_original[:, 1] ** 2
X_enhanced = np.column_stack([X_original, bedrooms_squared])

# Compare models
model_original = LinearRegression().fit(X_original, y)
model_enhanced = LinearRegression().fit(X_enhanced, y)

r2_original = r2_score(y, model_original.predict(X_original))
r2_enhanced = r2_score(y, model_enhanced.predict(X_enhanced))

print("📊 Model Comparison:")
print(f"  Original (3 features): R² = {r2_original:.4f}")
print(f"  Enhanced (4 features): R² = {r2_enhanced:.4f}")
print(f"  Improvement: +{(r2_enhanced - r2_original)*100:.2f}%")

# 🤔 Did the extra feature help?
```

---

## ✅ You're Ready to Move On When...

- [ ] You can interpret feature coefficients
- [ ] You understand standardization for comparing importance
- [ ] You can engineer new features
- [ ] You can explain the model to non-technical people

**You're now doing real data science!** 📊

**Next up:** Data Exploration (the art of understanding your data)
