# 🧪 Practice Lab: Data Exploration

**Find the hidden stories in your data!** EDA is where insights begin.

---

## 🎯 Challenge 1: The EDA Checklist (10 min)

Every dataset needs this inspection!

```python
import pandas as pd
import numpy as np

# Messy dataset - find the problems!
data = {
    'price': [200000, 300000, None, 400000, 9999999, 250000],
    'size': [1200, 1500, 1800, -100, 2200, 1400],
    'bedrooms': [2, 3, 3, 4, 4, 100]
}
df = pd.DataFrame(data)

# 🔍 RUN THE CHECKLIST:

# 1. Shape
print("1️⃣ Shape:", df.shape)

# 2. Missing values
print("\n2️⃣ Missing values:")
print(df.isnull().sum())

# 3. Data types
print("\n3️⃣ Data types:")
print(df.dtypes)

# 4. Basic stats (look for weird min/max!)
print("\n4️⃣ Statistics:")
print(df.describe())

# 🎯 YOUR MISSION - Find:
# - Which column has missing data? ___
# - Which row has an impossible price? ___
# - Which row has negative size? ___
# - Which row has unrealistic bedrooms? ___
```

---

## 🔍 Challenge 2: Correlation Detective (10 min)

Which features are most correlated with price?

```python
import pandas as pd
import numpy as np

# Clean dataset
np.random.seed(42)
df = pd.DataFrame({
    'price': [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000],
    'size': [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600],
    'bedrooms': [2, 2, 3, 3, 4, 4, 5, 5],
    'age': [20, 15, 12, 10, 8, 5, 3, 1],
    'distance_to_city': [15, 12, 10, 8, 6, 5, 4, 2]
})

# Calculate correlations with price
correlations = df.corr()['price'].drop('price').sort_values(key=abs, ascending=False)

print("🔍 Correlation with Price:")
print("-" * 40)
for feature, corr in correlations.items():
    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
    direction = "positive" if corr > 0 else "negative"
    print(f"  {feature}: {corr:.3f} ({strength} {direction})")

# 🤔 Answer these:
# 1. Which feature has the STRONGEST relationship? ___
# 2. Does older age INCREASE or DECREASE price? ___
# 3. Which features should you include in your model? ___
```

---

## 📊 Challenge 3: Visualize Relationships (15 min)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({
    'price': np.random.normal(300000, 50000, 100),
    'size': np.random.normal(1800, 400, 100),
    'age': np.random.uniform(1, 30, 100)
})

# Make size and price correlated
df['price'] = df['size'] * 150 + np.random.normal(0, 20000, 100)

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Histogram of prices
axes[0].hist(df['price'], bins=20, color='steelblue', edgecolor='black')
axes[0].set_xlabel('Price ($)')
axes[0].set_title('Price Distribution')

# Plot 2: Scatter - Size vs Price
axes[1].scatter(df['size'], df['price'], alpha=0.5)
axes[1].set_xlabel('Size (sqft)')
axes[1].set_ylabel('Price ($)')
axes[1].set_title('Size vs Price')

# Plot 3: YOUR TURN - Age vs Price
# axes[2].scatter(???, ???, alpha=0.5)
# axes[2].set_xlabel('???')
# axes[2].set_ylabel('???')
# axes[2].set_title('???')

plt.tight_layout()
plt.savefig('../visuals/eda_visualizations.png', dpi=150)
plt.show()
```

---

## 🏆 Boss Challenge: Outlier Detection

Find and handle outliers using the IQR method!

```python
import pandas as pd
import numpy as np

# Data with outliers
prices = [200000, 220000, 250000, 280000, 300000,
          320000, 350000, 380000, 9000000, 50000]  # 9M and 50k are outliers!

df = pd.DataFrame({'price': prices})

# IQR Method
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1: ${Q1:,.0f}")
print(f"Q3: ${Q3:,.0f}")
print(f"IQR: ${IQR:,.0f}")
print(f"\nValid range: ${lower_bound:,.0f} to ${upper_bound:,.0f}")

# Find outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"\n🚨 Outliers detected: {len(outliers)}")
print(outliers)

# Remove outliers
df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print(f"\nClean dataset: {len(df_clean)} rows")
```

---

## ✅ Ready for Next Module When...

- [ ] You can run the EDA checklist on any dataset
- [ ] You can find and interpret correlations
- [ ] You can detect outliers using IQR
- [ ] You can visualize distributions and relationships

**Next up:** Model Evaluation! 📏
