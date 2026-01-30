# 🧪 Practice Lab: Statistics Fundamentals

**You now speak the language of data!** Let's practice finding patterns.

---

## 🎯 Quick Win Challenge (5 min)

Calculate the mean and standard deviation BY HAND first, then verify:

```python
import numpy as np

# House prices (in $1000s)
prices = [150, 200, 180, 220, 190]

# YOUR HAND CALCULATIONS:
# Mean = sum / count = ???
# Variance = average of (x - mean)² = ???
# Std Dev = √variance = ???

# Verify!
print(f"Mean: ${np.mean(prices)}k")
print(f"Std Dev: ${np.std(prices, ddof=1):.1f}k")

# Did your hand calculation match? 🎯
```

---

## 🔍 Detective Challenge (10 min)

Look at this data and GUESS the correlation before calculating:

| Study Hours | Exam Score |
|-------------|------------|
| 1 | 45 |
| 2 | 52 |
| 4 | 65 |
| 6 | 78 |
| 8 | 88 |

**Your guess:** Is correlation positive/negative? Strong (>0.8) or weak (<0.5)?

```python
import numpy as np

hours = [1, 2, 4, 6, 8]
scores = [45, 52, 65, 78, 88]

# Calculate correlation
correlation = np.corrcoef(hours, scores)[0, 1]
print(f"Actual correlation: {correlation:.3f}")

# How close was your guess?
```

---

## 🏆 Boss Challenge (15 min)

You have two investment options. Which is better?

```python
import numpy as np

# Monthly returns (%)
stock_A = [5, -3, 8, 2, -1, 6, 4, -2, 7, 3]
stock_B = [2, 1, 3, 2, 1, 2, 3, 1, 2, 2]

# Calculate for each:
# 1. Mean return (higher is better)
# 2. Standard deviation (lower = less risky)
# 3. Which would you choose and why?

print("Stock A:")
print(f"  Mean return: {np.mean(stock_A):.1f}%")
print(f"  Risk (std): {np.std(stock_A):.1f}%")

print("\nStock B:")
print(f"  Mean return: {np.mean(stock_B):.1f}%")
print(f"  Risk (std): {np.std(stock_B):.1f}%")

# Your decision: _______________
# Why: _______________
```

---

## 🎲 Bonus Quest: The Z-Score Mystery

A student scores 85 on a test. Class mean = 70, std = 10.

```python
# How unusual is this score?
score = 85
mean = 70
std = 10

z_score = (score - mean) / std
print(f"Z-score: {z_score}")

# Interpretation:
# z = 0: exactly average
# z = 1: one std above average (top ~16%)
# z = 2: two std above (top ~2.5%)
# z = 3: very rare (top ~0.1%)

# Is 85 impressive?
```

---

## ✅ You're Ready to Move On When...

- [ ] You can calculate mean, variance, std by hand
- [ ] You can interpret correlation (direction + strength)
- [ ] You understand z-scores tell you "how unusual"

**Next up:** Derivatives & Gradient Descent! 🎢
