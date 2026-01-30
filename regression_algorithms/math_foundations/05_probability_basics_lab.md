# 🧪 Practice Lab: Probability Basics

**Embrace uncertainty!** Probability is what makes ML predictions meaningful.

---

## 🎯 Quick Win: The 68-95-99.7 Rule (5 min)

House prices: Mean = $300,000, Std Dev = $50,000

Using ONLY the rule (no code!), answer:

1. What % of houses cost between $250k - $350k? → ___% (within 1 std)
2. What % cost between $200k - $400k? → ___% (within 2 std)
3. What % cost more than $400k? → ___% (beyond 2 std on one side)

```python
from scipy import stats

# Verify your mental math!
mean, std = 300000, 50000
dist = stats.norm(mean, std)

print("Your answers vs reality:")
print(f"1. Between $250k-$350k: {(dist.cdf(350000) - dist.cdf(250000))*100:.1f}%")
print(f"2. Between $200k-$400k: {(dist.cdf(400000) - dist.cdf(200000))*100:.1f}%")
print(f"3. Above $400k: {(1 - dist.cdf(400000))*100:.1f}%")
```

---

## 🏠 Real Estate Agent Challenge (10 min)

You're a real estate agent. A client asks:

"What price puts me in the top 10% of the market?"

```python
from scipy import stats

mean = 300000
std = 50000
dist = stats.norm(mean, std)

# Find the 90th percentile (top 10%)
price_90th = dist.ppf(0.90)
print(f"Top 10% starts at: ${price_90th:,.0f}")

# 🤔 YOUR TURN:
# 1. What price is the 75th percentile (top 25%)?
price_75th = ???
print(f"Top 25% starts at: ${price_75th:,.0f}")

# 2. If a house costs $425,000, what percentile is it?
percentile_425k = dist.cdf(425000) * 100
print(f"$425k is the {percentile_425k:.1f}th percentile")
```

---

## 🎲 Simulation Challenge (10 min)

Generate random house prices and see the bell curve!

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate 10,000 random house prices
np.random.seed(42)
prices = np.random.normal(loc=300000, scale=50000, size=10000)

# Visualize!
plt.figure(figsize=(10, 6))
plt.hist(prices, bins=50, density=True, alpha=0.7, color='steelblue')
plt.axvline(x=300000, color='red', linestyle='--', label='Mean ($300k)')
plt.axvline(x=250000, color='orange', linestyle=':', label='±1 std')
plt.axvline(x=350000, color='orange', linestyle=':')
plt.xlabel('House Price ($)')
plt.ylabel('Density')
plt.title('10,000 Random House Prices - The Bell Curve!')
plt.legend()
plt.savefig('../visuals/bell_curve_simulation.png')
plt.show()

# Count houses in each range
within_1std = np.sum((prices > 250000) & (prices < 350000)) / len(prices)
print(f"Within 1 std (250k-350k): {within_1std*100:.1f}%")  # Should be ~68%
```

---

## 🏆 Boss Challenge: Why This Matters for ML (15 min)

In regression, we assume errors are normally distributed!

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# True relationship: y = 2x + 5 + noise
np.random.seed(42)
X = np.random.uniform(0, 10, 100)
noise = np.random.normal(0, 2, 100)  # Normal noise!
y = 2 * X + 5 + noise

# Fit model
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
predictions = model.predict(X.reshape(-1, 1))

# Calculate errors (residuals)
errors = y - predictions

# Plot error distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, predictions, 'r-', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data + Model')

plt.subplot(1, 2, 2)
plt.hist(errors, bins=20, density=True, alpha=0.7)
plt.xlabel('Error')
plt.ylabel('Density')
plt.title('Distribution of Errors (Should be bell-shaped!)')

plt.tight_layout()
plt.savefig('../visuals/error_distribution.png')
plt.show()

# Check: Are errors normally distributed?
print(f"Error mean: {np.mean(errors):.3f} (should be ~0)")
print(f"Error std: {np.std(errors):.3f}")
```

**Think:** If the error histogram is NOT bell-shaped, what might be wrong with your model?

---

## ✅ You're Ready to Move On When...

- [ ] You can apply the 68-95-99.7 rule mentally
- [ ] You understand percentiles and can calculate them
- [ ] You know WHY regression assumes normal errors

**Congratulations!** 🎉 You've completed ALL Math Foundations! Time for the algorithms!

**Next up:** Linear Regression! 📈
