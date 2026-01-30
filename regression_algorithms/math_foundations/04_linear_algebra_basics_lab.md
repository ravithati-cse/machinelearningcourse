# 🧪 Practice Lab: Linear Algebra Basics

**Vectors and matrices are the language of ML!** Let's get fluent.

---

## 🎯 Quick Win Challenge (5 min)

Calculate these BY HAND, then verify:

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 1. Vector addition: a + b = ???
# Your answer: [?, ?, ?]

# 2. Scalar multiplication: 3 × a = ???
# Your answer: [?, ?, ?]

# 3. Dot product: a · b = 1×4 + 2×5 + 3×6 = ???
# Your answer: ???

# Verify!
print(f"a + b = {a + b}")
print(f"3 × a = {3 * a}")
print(f"a · b = {np.dot(a, b)}")
```

---

## 🏠 The Matrix Challenge (10 min)

You're predicting house prices with this data:

| House | Size (sqft) | Bedrooms |
|-------|-------------|----------|
| 1 | 1500 | 3 |
| 2 | 2000 | 4 |
| 3 | 1200 | 2 |

Price formula: `$100/sqft + $50,000/bedroom`

```python
import numpy as np

# Houses as a matrix: each row is a house, columns are features
houses = np.array([
    [1500, 3],
    [2000, 4],
    [1200, 2]
])

# Weights (price per unit of each feature)
weights = np.array([100, 50000])

# 🎯 YOUR TASK: Calculate all prices at once!
# Use matrix multiplication: prices = houses @ weights

prices = ???  # Your code here!

print("Predicted prices:")
for i, price in enumerate(prices):
    print(f"  House {i+1}: ${price:,}")

# Verify House 1 by hand:
# 1500 × $100 + 3 × $50,000 = $150,000 + $150,000 = $300,000
```

---

## 🔄 Transpose Challenge (10 min)

Understanding transpose is key for ML!

```python
import numpy as np

# Original matrix (3 houses × 2 features)
X = np.array([
    [1500, 3],
    [2000, 4],
    [1200, 2]
])

print("Original shape:", X.shape)
print(X)

print("\nTransposed shape:", X.T.shape)
print(X.T)

# 🤔 THINK ABOUT IT:
# 1. What was (3, 2) became (?, ?)
# 2. Rows became ___ and columns became ___
# 3. When would you need this in ML?
```

---

## 🏆 Boss Challenge: The Normal Equation (15 min)

Linear regression can be solved with ONE matrix equation!

`weights = (X^T × X)^(-1) × X^T × y`

```python
import numpy as np

# Training data: study hours → exam score
X = np.array([[1], [2], [3], [4], [5]])  # Features (with shape for matrix math)
y = np.array([50, 55, 65, 70, 80])       # Targets

# Add bias column (column of 1s)
X_bias = np.c_[np.ones(5), X]
print("X with bias:")
print(X_bias)

# The Normal Equation! 🎯
# weights = (X^T × X)^(-1) × X^T × y

XtX = X_bias.T @ X_bias           # X^T × X
XtX_inv = np.linalg.inv(XtX)      # Inverse
Xty = X_bias.T @ y                 # X^T × y
weights = XtX_inv @ Xty            # Final weights!

print(f"\nWeights: intercept={weights[0]:.2f}, slope={weights[1]:.2f}")
print(f"Model: score = {weights[1]:.2f} × hours + {weights[0]:.2f}")

# Verify with sklearn!
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
print(f"Sklearn: score = {model.coef_[0]:.2f} × hours + {model.intercept_:.2f}")

# Do they match? 🎯
```

---

## ✅ You're Ready to Move On When...

- [ ] You can do vector addition and dot products
- [ ] You understand matrix multiplication does multiple dot products at once
- [ ] You know transpose flips rows ↔ columns

**Mind Blown Moment:** Scikit-learn's LinearRegression does exactly what you just did!

**Next up:** Probability! 🎲
