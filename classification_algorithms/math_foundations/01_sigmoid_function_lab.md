# 🧪 Practice Lab: The Sigmoid Function

**The magic transformation that makes classification possible!**

---

## 🎯 Quick Win: Sigmoid Explorer (5 min)

What does sigmoid do to different inputs?

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# PREDICT first, then verify!
test_values = [-10, -2, 0, 2, 10]

print("Input → Output")
print("-" * 25)
for x in test_values:
    print(f"  {x:>4} → {sigmoid(x):.4f}")

# 🤔 Notice:
# - Very negative → close to ???
# - Zero → exactly ???
# - Very positive → close to ???
```

---

## 🎨 Visualization Challenge (10 min)

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = 1 / (1 + np.exp(-x))

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision boundary (0.5)')
plt.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
plt.axhline(y=1, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Input (z)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.title('The Sigmoid Function', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../visuals/my_sigmoid.png', dpi=150)
plt.show()

# 🎯 Add these to your plot:
# 1. Mark where sigmoid(0) = 0.5
# 2. Shade the "Class 1" region (above 0.5)
# 3. Shade the "Class 0" region (below 0.5)
```

---

## 🏆 Boss Challenge: From Score to Class (10 min)

A model outputs these "raw scores" for emails. Convert to spam probabilities!

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Raw model scores (before sigmoid)
emails = {
    'Meeting tomorrow': -3.5,
    'FREE MONEY NOW!!!': 4.2,
    'Please review doc': -1.8,
    'You won a prize!': 2.1,
}

print("Email Classification:")
print("-" * 60)
for email, score in emails.items():
    prob = sigmoid(score)
    prediction = "SPAM 🚨" if prob > 0.5 else "HAM ✅"
    print(f"{email:<25} score={score:>5.1f} → prob={prob:.1%} → {prediction}")

# 🤔 Questions:
# 1. Why does a negative score give probability < 0.5?
# 2. What score gives exactly 50% probability?
# 3. What if you changed the threshold from 0.5 to 0.3?
```

---

## ✅ You're Ready When...

- [ ] You know sigmoid converts ANY number to 0-1
- [ ] You know sigmoid(0) = 0.5 (the decision point)
- [ ] You understand why this is perfect for classification

**Next up:** Probability for Classification! 🎲
