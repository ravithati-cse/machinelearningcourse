# 📊 Datasets for Regression Learning

This directory contains information about datasets used in the regression learning modules.

---

## 🏠 California Housing Dataset

**Used in**:
- `projects/house_price_prediction.py`
- `projects/housing_analysis.py`

**Source**: Built into scikit-learn (`sklearn.datasets.fetch_california_housing`)

**Description**:
Real California housing data from the 1990 census with 20,640 samples.

**Features** (8 total):
1. **MedInc**: Median income in block group
2. **HouseAge**: Median house age in block group
3. **AveRooms**: Average number of rooms per household
4. **AveBedrms**: Average number of bedrooms per household
5. **Population**: Block group population
6. **AveOccup**: Average number of household members
7. **Latitude**: Block group latitude
8. **Longitude**: Block group longitude

**Target**:
- **MedHouseVal**: Median house value for California districts (in $100,000s)

**How to Load**:
```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing(as_frame=True)
X = housing.data  # Features
y = housing.target  # Target (prices)
```

**Characteristics**:
- No missing values
- All features are continuous
- Target values range from $0.15k to $5M (capped at $500k in 1990)
- Good for learning regression

---

## 📈 Synthetic Datasets

**Used in**:
- `math_foundations/*.py`
- `examples/simple_examples.py`

**Description**:
Programmatically generated data for learning specific concepts

**Examples**:

### 1. Perfect Linear Relationship
```python
import numpy as np
x = np.array([1, 2, 3, 4, 5])
y = 2 * x + 3  # Perfect line: y = 2x + 3
```

### 2. Linear with Noise
```python
x = np.linspace(0, 10, 100)
y = 2 * x + 3 + np.random.normal(0, 2, 100)  # Add noise
```

### 3. Multiple Features
```python
# House prices: price = 150*size + 10000*bedrooms + noise
sizes = np.random.normal(1500, 300, 100)
bedrooms = np.random.randint(1, 5, 100)
prices = 150*sizes + 10000*bedrooms + np.random.normal(0, 10000, 100)
```

---

## 🔢 Other Built-in Datasets (For Future Learning)

### Boston Housing (Deprecated, use California)
```python
# Don't use - has ethical concerns with LSTAT feature
# Use California Housing instead
```

### Diabetes Dataset
```python
from sklearn.datasets import load_diabetes
diabetes = load_diabetes(as_frame=True)
# Predict disease progression using 10 baseline features
```

### Make Regression (Synthetic)
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5, noise=10)
# Generate custom regression problems
```

---

## 🌐 External Datasets to Explore

### Beginner-Friendly:

1. **Car Prices**
   - Source: Kaggle "Car Price Prediction"
   - Features: Make, model, year, mileage, engine size
   - Target: Price

2. **Student Performance**
   - Source: UCI ML Repository
   - Features: Study time, absences, parent education
   - Target: Final grade

3. **Insurance Costs**
   - Source: Kaggle "Medical Cost Personal"
   - Features: Age, BMI, children, smoker status
   - Target: Insurance charges

### Intermediate:

4. **Ames Housing**
   - Source: Kaggle "House Prices - Advanced Regression"
   - Features: 79 features describing houses
   - Target: Sale price
   - Note: More complex than California Housing

5. **Bike Sharing**
   - Source: UCI ML Repository
   - Features: Weather, season, time
   - Target: Bike rental count

6. **Real Estate (Taiwan)**
   - Source: UCI ML Repository
   - Features: Location, age, distance to MRT
   - Target: House price per unit area

---

## 📥 How to Download External Datasets

### From Kaggle:
1. Create Kaggle account
2. Go to dataset page
3. Click "Download" button
4. Extract zip file
5. Place CSV in this `data/` folder

### From UCI ML Repository:
1. Visit https://archive.ics.uci.edu/ml/
2. Search for dataset
3. Download data files
4. Place in this folder

### Example Code to Load:
```python
import pandas as pd

# Load CSV
df = pd.read_csv('data/your_dataset.csv')

# Separate features and target
X = df.drop('target_column', axis=1)
y = df['target_column']
```

---

## 🧹 Data Preprocessing Tips

### Handling Missing Values:
```python
# Check for missing values
print(df.isnull().sum())

# Fill with mean
df['column'].fillna(df['column'].mean(), inplace=True)

# Or drop rows
df.dropna(inplace=True)
```

### Encoding Categorical Variables:
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Or use one-hot encoding
df = pd.get_dummies(df, columns=['category'])
```

### Feature Scaling:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# X_scaled has mean=0, std=1
```

---

## 📊 Dataset Selection Guidelines

### Choose a dataset with:
- ✅ Clear documentation
- ✅ Reasonable size (< 100k rows for learning)
- ✅ Mix of feature types (continuous, categorical)
- ✅ Clear target variable
- ✅ Real-world context you find interesting

### Avoid datasets with:
- ❌ Too many missing values (> 20%)
- ❌ Too many features for beginners (> 50)
- ❌ Privacy concerns
- ❌ Highly imbalanced targets
- ❌ Poor documentation

---

## 🎯 Practice Projects

After completing the main course, try these projects:

### Beginner Projects:
1. **Predict Car Prices**
   - Dataset: Kaggle Car Price Prediction
   - Goal: R² > 0.7
   - Focus: Feature engineering, handling categorical variables

2. **Student Grade Prediction**
   - Dataset: UCI Student Performance
   - Goal: MAE < 2 points
   - Focus: Understanding educational features

3. **Temperature Forecasting**
   - Dataset: Historical weather data
   - Goal: RMSE < 5°F
   - Focus: Time-based patterns

### Intermediate Projects:
4. **Advanced House Price Prediction**
   - Dataset: Ames Housing (Kaggle)
   - Goal: Top 50% on leaderboard
   - Focus: Complex feature engineering

5. **Demand Forecasting**
   - Dataset: Store sales data
   - Goal: Accurate weekly predictions
   - Focus: Time series, seasonality

---

## 📁 Directory Structure

```
data/
├── README.md (this file)
├── raw/                    # Original downloads
│   └── .gitkeep
├── processed/              # Cleaned datasets
│   └── .gitkeep
└── external/               # Downloaded external data
    └── .gitkeep
```

**Best Practice**:
- Keep raw data untouched
- Save processed data separately
- Document all preprocessing steps

---

## 🔍 Quick Dataset Checks

Before using any dataset, always:

```python
import pandas as pd

# Load data
df = pd.read_csv('dataset.csv')

# 1. Check shape
print(f"Shape: {df.shape}")  # (rows, columns)

# 2. Check data types
print(df.dtypes)

# 3. Check missing values
print(df.isnull().sum())

# 4. Check basic statistics
print(df.describe())

# 5. Check first few rows
print(df.head())

# 6. Check target distribution
print(df['target'].describe())
import matplotlib.pyplot as plt
df['target'].hist()
plt.show()
```

---

## 📚 Additional Resources

### Learning About Data:
- **Kaggle Learn**: Data Cleaning course
- **YouTube**: "Data Cleaning in Python" by Corey Schafer
- **Book**: "Python for Data Analysis" by Wes McKinney

### Finding Datasets:
- **Kaggle**: kaggle.com/datasets
- **UCI ML Repository**: archive.ics.uci.edu/ml/
- **Data.gov**: data.gov
- **Google Dataset Search**: datasetsearch.research.google.com

---

## ⚠️ Important Notes

### Data Ethics:
- Always check dataset licenses
- Respect privacy and sensitive information
- Be aware of potential biases in data
- Don't use data in ways that could harm others

### Performance Expectations:
- **California Housing**: R² ~ 0.6-0.7 is good for linear regression
- **Simple datasets**: R² > 0.8 possible
- **Complex datasets**: R² ~ 0.5-0.6 might be realistic
- **Don't chase perfect scores**: Understanding > accuracy

---

## 🎓 Dataset Analysis Checklist

Before building a model:

- [ ] Load and inspect data
- [ ] Check for missing values
- [ ] Identify feature types (continuous, categorical)
- [ ] Check target variable distribution
- [ ] Calculate basic statistics
- [ ] Visualize relationships (scatter plots)
- [ ] Check for outliers
- [ ] Understand domain context
- [ ] Split train/test sets
- [ ] Plan preprocessing steps

---

*Remember: In real-world ML, you'll spend 80% of time on data and 20% on models!*
