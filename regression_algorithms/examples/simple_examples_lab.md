# 🧪 Practice Lab: Simple Examples

**Practice makes permanent!** Apply regression to different scenarios.

---

## 🎯 Challenge 1: Temperature Prediction (10 min)

Morning temp = 55°F. Rises 3°F per hour until noon.

```python
import numpy as np

# Build the model
hours = np.array([0, 1, 2, 3, 4, 5, 6])  # 0 = 6am, 6 = noon
temps = 55 + 3 * hours  # Our model!

print("🌡️ Temperature Predictions:")
for h, t in zip(hours, temps):
    print(f"  {6+h}:00 → {t}°F")

# 🤔 YOUR TURN:
# 1. What's the temperature at 2pm (8 hours)? ___
# 2. When does it hit 70°F? ___
```

---

## 🎯 Challenge 2: Advertising ROI (10 min)

Every $1000 in ads generates $150 in sales (plus base sales of $5000).

```python
import numpy as np
import matplotlib.pyplot as plt

def sales(ad_spend):
    """ad_spend in thousands"""
    return 5000 + 150 * ad_spend

# Test different budgets
budgets = np.array([0, 10, 20, 30, 50])

print("💰 Advertising Analysis:")
print("-" * 40)
for budget in budgets:
    revenue = sales(budget)
    profit = revenue - (budget * 1000)  # Subtract ad cost
    print(f"  ${budget}k ads → ${revenue:,} sales → ${profit:,} profit")

# 🤔 At what ad spend do you START losing money?
# (Hint: When does ad cost exceed the extra sales it generates?)
```

---

## 🎯 Challenge 3: Study Hours vs Grade (15 min)

Build a model from scratch with real-looking data!

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Student data: [study_hours, grade]
data = [
    (1, 52), (2, 58), (3, 65), (4, 71),
    (5, 75), (6, 82), (7, 85), (8, 91)
]

hours = np.array([d[0] for d in data]).reshape(-1, 1)
grades = np.array([d[1] for d in data])

# Fit model
model = LinearRegression()
model.fit(hours, grades)

print(f"📚 Grade = {model.coef_[0]:.1f} × hours + {model.intercept_:.1f}")
print(f"\nInterpretation: Each study hour adds {model.coef_[0]:.1f} points!")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(hours, grades, s=100, zorder=5)
plt.plot(hours, model.predict(hours), 'r-', linewidth=2)
plt.xlabel('Study Hours')
plt.ylabel('Grade')
plt.title('Study Time vs Grade')
plt.grid(True, alpha=0.3)
plt.savefig('../visuals/study_grade.png')
plt.show()

# 🤔 YOUR TURN:
# 1. Predicted grade with 10 hours of study? ___
# 2. Hours needed for grade of 100? ___
# 3. Is that realistic? Why or why not?
```

---

## 🏆 Boss Challenge: Real-World Scenario

A gym membership costs $30/month. Each member costs $5/month in expenses.

```python
# Revenue = $30 × members
# Costs = Fixed costs + $5 × members
# Profit = Revenue - Costs

fixed_costs = 5000  # Rent, equipment, etc.

def profit(members):
    revenue = 30 * members
    costs = fixed_costs + 5 * members
    return revenue - costs

# Find the break-even point!
# Profit = 0 when: 30m - 5000 - 5m = 0
# 25m = 5000
# m = ???

break_even = ???
print(f"Break-even at {break_even} members")

# How much profit at 500 members?
profit_500 = profit(500)
print(f"Profit at 500 members: ${profit_500:,}")
```

---

## ✅ Ready for the Next Module When...

- [ ] You can model real scenarios with y = mx + b
- [ ] You can find break-even points
- [ ] You can interpret what slope and intercept MEAN

**Next up:** Data Exploration! 🔍
