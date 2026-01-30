# 🧪 Practice Lab: Algebra Basics

**Congratulations on completing the Algebra module!** Now let's solidify your understanding with some hands-on challenges.

---

## 🎯 Quick Win Challenge (5 min)

A coffee shop finds: for every $1 price drop, they sell 50 more cups. At $5/cup, they sell 100 cups.

```python
# Can you build the equation? 🔍

# Step 1: What's the slope? (price DOWN → cups UP)
slope = ???  # Hint: It's negative!

# Step 2: Find the intercept using: 100 = slope × 5 + b
intercept = ???

def predict_sales(price):
    return slope * price + intercept

# Test it!
print(f"At $3/cup: {predict_sales(3)} cups")
print(f"At $0/cup: {predict_sales(0)} cups")  # Max sales!
```

<details>
<summary>💡 Click for solution</summary>

```python
slope = -50  # Negative because price down = cups up
intercept = 350  # 100 = -50 × 5 + b → b = 350

# At $3: -50 × 3 + 350 = 200 cups
# At $0: -50 × 0 + 350 = 350 cups (max possible)
```
</details>

---

## 🔥 Level Up Challenge (10 min)

You're analyzing temperature data. Morning temp = 60°F. It rises 5°F every hour.

1. Write the equation: `temp = m × hours + b`
2. What's the temperature at 3pm (6 hours later)?
3. When does it hit 100°F?

```python
# Your code here!
m = ???
b = ???

def get_temp(hours):
    return m * hours + b

# At 3pm (6 hours)
print(f"3pm temp: {get_temp(6)}°F")

# When does it hit 100°F? Solve: 100 = m × hours + b
hours_to_100 = ???
print(f"Hits 100°F at: {hours_to_100} hours")
```

---

## 🏆 Boss Challenge (15 min)

A company's revenue follows: `revenue = 1000 × employees - 50000`

1. How many employees to break even (revenue = 0)?
2. Revenue with 100 employees?
3. Graph this relationship!

```python
import matplotlib.pyplot as plt
import numpy as np

def revenue(employees):
    return 1000 * employees - 50000

# Break even point
break_even = ???
print(f"Break even at {break_even} employees")

# Plot it!
employees = np.arange(0, 150, 1)
plt.plot(employees, revenue(employees))
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Employees')
plt.ylabel('Revenue ($)')
plt.title('Your First Business Model!')
plt.savefig('../visuals/my_first_model.png')
plt.show()
```

---

## ✅ You're Ready to Move On When...

- [ ] You can write `y = mx + b` from a word problem
- [ ] You know slope = rate of change
- [ ] You can find where a line crosses zero

**Next up:** Statistics Fundamentals! 📊
