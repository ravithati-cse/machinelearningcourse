# 🧪 Practice Lab: Derivatives & Gradient Descent

**You've learned THE algorithm that powers all of ML!** Let's master it.

---

## 🎯 Quick Win Challenge (5 min)

For the function `f(x) = x²`, the derivative is `f'(x) = 2x`.

What's the slope at these points?

```python
def f(x):
    return x ** 2

def slope(x):
    return 2 * x

# Fill in your predictions FIRST, then run!
# At x = 0, slope = ???  (Hint: bottom of the valley)
# At x = 3, slope = ???
# At x = -2, slope = ???

for x in [0, 3, -2]:
    print(f"At x={x}: slope = {slope(x)}")
```

**Think:** Why is the slope 0 at x=0? What does this mean?

---

## 🎢 Gradient Descent by Hand (15 min)

Roll down the hill `f(x) = x²` starting at x = 4!

**Learning rate = 0.1**

Fill in this table BY HAND:

| Step | x | f(x) | slope | x_new = x - 0.1 × slope |
|------|---|------|-------|------------------------|
| 0 | 4.0 | 16.0 | 8.0 | 4 - 0.8 = 3.2 |
| 1 | 3.2 | ? | ? | ? |
| 2 | ? | ? | ? | ? |
| 3 | ? | ? | ? | ? |
| 4 | ? | ? | ? | ? |

```python
# Verify your hand calculations!
x = 4.0
learning_rate = 0.1

for step in range(5):
    fx = x ** 2
    grad = 2 * x
    x_new = x - learning_rate * grad
    print(f"Step {step}: x={x:.3f}, f(x)={fx:.3f}, slope={grad:.3f}, x_new={x_new:.3f}")
    x = x_new

# Are you approaching 0? 🎯
```

---

## 🔥 Experiment: Learning Rate Chaos (10 min)

What happens with different learning rates?

```python
import matplotlib.pyplot as plt

def gradient_descent(start, lr, steps):
    """Track x values during descent"""
    x = start
    history = [x]
    for _ in range(steps):
        x = x - lr * (2 * x)
        history.append(x)
    return history

# Try different learning rates!
plt.figure(figsize=(12, 4))

for i, lr in enumerate([0.1, 0.5, 1.1]):
    plt.subplot(1, 3, i+1)
    history = gradient_descent(start=4, lr=lr, steps=20)
    plt.plot(history, 'o-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'Learning Rate = {lr}')
    plt.xlabel('Step')
    plt.ylabel('x value')

plt.tight_layout()
plt.savefig('../visuals/learning_rate_experiment.png')
plt.show()

# 🤔 What do you notice?
# lr = 0.1: ???
# lr = 0.5: ???
# lr = 1.1: ???
```

---

## 🏆 Boss Challenge: 2D Gradient Descent (15 min)

Now try with TWO variables! Minimize `f(x, y) = x² + y²`

```python
import numpy as np

# Starting point
x, y = 4.0, 3.0
lr = 0.1

print("Finding the minimum of f(x,y) = x² + y²")
print("-" * 40)

for step in range(10):
    # Gradients: df/dx = 2x, df/dy = 2y
    grad_x = 2 * x
    grad_y = 2 * y

    # Update both variables
    x = x - lr * grad_x
    y = y - lr * grad_y

    f_val = x**2 + y**2
    print(f"Step {step}: x={x:.3f}, y={y:.3f}, f(x,y)={f_val:.4f}")

# The minimum is at (0, 0). How close did you get?
```

---

## ✅ You're Ready to Move On When...

- [ ] You can calculate a derivative for simple functions
- [ ] You understand gradient descent moves OPPOSITE to the slope
- [ ] You know why learning rate matters (too small = slow, too big = chaos)

**Aha Moment:** Neural networks do this same process with MILLIONS of parameters! 🤯

**Next up:** Linear Algebra! 🔢
