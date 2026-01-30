# 🧪 Practice Lab: Linear Regression

**You've learned the most important algorithm in ML!** Time to build it yourself.

---

## 🎯 Quick Win: Implement from Scratch! (10 min)

NO scikit-learn allowed! Use only numpy.

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Calculate slope and intercept using the formulas!
x_mean = np.mean(X)
y_mean = np.mean(y)

# slope = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²
numerator = np.sum((X - x_mean) * (y - y_mean))
denominator = np.sum((X - x_mean) ** 2)
slope = numerator / denominator

# intercept = ȳ - slope × x̄
intercept = y_mean - slope * x_mean

print(f"YOUR model: y = {slope:.3f}x + {intercept:.3f}")

# Now verify with sklearn!
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
print(f"Sklearn:    y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}")

# 🎉 Do they match?
```

---

## 🔮 Prediction Challenge (10 min)

Using your model, make predictions!

```python
# Your slope and intercept from above
slope = ???     # Fill this in!
intercept = ??? # Fill this in!

def predict(x):
    return slope * x + intercept

# Predictions
print("🔮 Predictions:")
print(f"  x=6: y = {predict(6):.2f}")
print(f"  x=10: y = {predict(10):.2f}")
print(f"  x=0: y = {predict(0):.2f}")  # This is just the intercept!

# 🤔 THINK: Should you trust the x=10 prediction?
# Your training data only went up to x=5!
# This is called EXTRAPOLATION - be careful!
```

---

## 📊 Visualize Your Model (10 min)

```python
import matplotlib.pyplot as plt
import numpy as np

# Your data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Your model
slope = ???
intercept = ???

# Create the plot
plt.figure(figsize=(10, 6))

# Plot data points
plt.scatter(X, y, s=100, color='blue', zorder=5, label='Training Data')

# Plot regression line
x_line = np.linspace(0, 7, 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')

# Make it pretty
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('🎉 Your First Linear Regression Model!', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../visuals/my_linear_regression.png', dpi=150)
plt.show()
```

---

## 🏆 Boss Challenge: Gradient Descent Training (20 min)

Instead of the formula, train using gradient descent!

```python
import numpy as np

# Training data
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)

# Initialize random weights
np.random.seed(42)
slope = np.random.randn()
intercept = np.random.randn()
learning_rate = 0.01

print("Training with Gradient Descent...")
print("-" * 40)

for epoch in range(1000):
    # Forward pass: predictions
    predictions = slope * X + intercept

    # Calculate error (MSE)
    error = predictions - y
    mse = np.mean(error ** 2)

    # Calculate gradients
    grad_slope = 2 * np.mean(error * X)
    grad_intercept = 2 * np.mean(error)

    # Update weights
    slope = slope - learning_rate * grad_slope
    intercept = intercept - learning_rate * grad_intercept

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: MSE={mse:.4f}, slope={slope:.3f}, intercept={intercept:.3f}")

print(f"\n🎯 Final: y = {slope:.3f}x + {intercept:.3f}")
print("Compare to the formula solution above!")
```

---

## ✅ You're Ready to Move On When...

- [ ] You can implement linear regression with the formula
- [ ] You can implement it with gradient descent
- [ ] You can visualize your model
- [ ] You understand why extrapolation is risky

**You just built what powers billion-dollar predictions!** 🚀

**Next up:** Multiple Regression (more features = more power!)
