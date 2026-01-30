# 🧪 Practice Lab: Logistic Regression

**Build the most widely-used classification algorithm from scratch!**

---

## 🎯 Quick Win: Implement Sigmoid + Prediction (5 min)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simple model: spam_score = 2 × (num_exclamation_marks) - 3

def predict_spam(exclamation_marks):
    z = 2 * exclamation_marks - 3
    probability = sigmoid(z)
    return probability

# Test it!
print("Exclamation Marks → Spam Probability")
print("-" * 40)
for n in [0, 1, 2, 3, 5]:
    prob = predict_spam(n)
    label = "SPAM" if prob > 0.5 else "HAM"
    print(f"  {n} → {prob:.1%} → {label}")
```

---

## 💪 Build from Scratch! (20 min)

```python
import numpy as np

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Training data: [study_hours, sleep_hours] → pass (1) or fail (0)
X = np.array([
    [2, 4], [3, 5], [4, 6], [5, 7],  # Failed
    [6, 5], [7, 6], [8, 7], [9, 8]   # Passed
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Add bias column
X_bias = np.c_[np.ones(len(X)), X]

# Initialize weights randomly
np.random.seed(42)
weights = np.random.randn(3) * 0.1
learning_rate = 0.1

# Training loop
print("Training Logistic Regression...")
for epoch in range(1000):
    # Forward pass
    z = X_bias @ weights
    predictions = sigmoid(z)

    # Log loss
    loss = -np.mean(y * np.log(predictions + 1e-15) +
                    (1 - y) * np.log(1 - predictions + 1e-15))

    # Gradient
    gradient = X_bias.T @ (predictions - y) / len(y)

    # Update
    weights = weights - learning_rate * gradient

    if epoch % 200 == 0:
        print(f"  Epoch {epoch}: Loss = {loss:.4f}")

print(f"\n✅ Final weights: {weights}")

# Test predictions
print("\n🔮 Predictions:")
for i, (x, actual) in enumerate(zip(X, y)):
    prob = sigmoid(np.dot([1, x[0], x[1]], weights))
    pred = 1 if prob > 0.5 else 0
    status = "✓" if pred == actual else "✗"
    print(f"  [{x[0]}, {x[1]}] → {prob:.1%} → {pred} (actual: {actual}) {status}")
```

---

## 📊 Compare with Sklearn (5 min)

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

X = np.array([
    [2, 4], [3, 5], [4, 6], [5, 7],
    [6, 5], [7, 6], [8, 7], [9, 8]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

print("Sklearn Logistic Regression:")
print(f"  Intercept: {model.intercept_[0]:.3f}")
print(f"  Weights: {model.coef_[0]}")

# Predict new student: 5 hours study, 6 hours sleep
new_student = [[5, 6]]
prob = model.predict_proba(new_student)[0][1]
print(f"\n🔮 Student [5 hrs study, 6 hrs sleep]: {prob:.1%} chance of passing")
```

---

## 🏆 Boss Challenge: Visualize Decision Boundary

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Data
X = np.array([
    [2, 4], [3, 5], [4, 6], [5, 7],
    [6, 5], [7, 6], [8, 7], [9, 8]
])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train
model = LogisticRegression()
model.fit(X, y)

# Create grid
x1_range = np.linspace(0, 12, 100)
x2_range = np.linspace(2, 10, 100)
xx1, xx2 = np.meshgrid(x1_range, x2_range)
grid = np.c_[xx1.ravel(), xx2.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xx1.shape)

# Plot
plt.figure(figsize=(10, 8))
plt.contourf(xx1, xx2, probs, levels=20, cmap='RdYlBu', alpha=0.7)
plt.colorbar(label='P(Pass)')
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, label='Failed', edgecolors='black')
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, label='Passed', edgecolors='black')
plt.contour(xx1, xx2, probs, levels=[0.5], colors='black', linewidths=2)
plt.xlabel('Study Hours')
plt.ylabel('Sleep Hours')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.savefig('../visuals/my_logistic_boundary.png', dpi=150)
plt.show()
```

---

## ✅ You're Ready When...

- [ ] You can implement logistic regression from scratch
- [ ] You understand sigmoid converts score → probability
- [ ] You can visualize the decision boundary
- [ ] You know this powers spam filters, ad targeting, medical diagnosis!

**Congratulations!** You've mastered the most important classification algorithm! 🎉

**Next up:** KNN Classifier! 🗳️
