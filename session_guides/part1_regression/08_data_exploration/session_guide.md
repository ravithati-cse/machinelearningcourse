# MLForBeginners — Instructor Guide
## Part 1 · Module 08: Data Exploration
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–07 complete. They can fit a linear regression model and
> interpret coefficients. They have basic pandas familiarity from examples.
> **Payoff today:** They gain the skill that separates junior from senior practitioners —
> EDA is 80% of the job, and they'll feel that by the end.

---

# SESSION 1 (~90 min)
## "Know your data before you model it"

## Before They Arrive
- Terminal open in `regression_algorithms/examples/`
- `data_exploration.py` ready to run
- Whiteboard ready — draw the "messy data" table below
- Have a fake dataset printed or on screen with obvious problems

---

## OPENING (10 min)

> *"Here's a dirty secret about machine learning:
> most of the time you spend on a real project is NOT training models.*
>
> *It's this: staring at data, asking questions, finding problems.*
> *Exploratory Data Analysis — EDA. And I mean 70-80% of total project time.*
>
> *Today's module teaches you to never trust data you haven't examined.*
> *Because bad data in, bad model out. Always."*

Draw on board:
```
THE ML PIPELINE:
  RAW DATA → [EDA] → CLEAN DATA → MODEL → PREDICTIONS
               ↑
         We are here.
         This step takes longer than the model.
         And it determines whether the model is worth anything.
```

> *"I'm going to show you a dataset with hidden problems.
> Your job is to find them before we touch sklearn.
> Ready?"*

---

## SECTION 1: Load and Inspect — The First Five Commands (20 min)

> *"Every data exploration starts the same way. Five commands.
> Burn these into muscle memory."*

Write on board:
```
THE EDA STARTER PACK:
  df.shape          → rows, columns
  df.info()         → column names, types, non-null counts
  df.describe()     → min, max, mean, std, quartiles
  df.isnull().sum() → missing values per column
  df.head(10)       → first 10 rows, eyeball check
```

> *"Run these on every new dataset before doing anything else.
> Not because a textbook says so — because each one answers a specific question
> that will catch problems that would otherwise break your model silently."*

Code together (type from scratch, don't copy):
```python
import pandas as pd
import numpy as np

# Simulate house data with problems baked in
data = {
    'price':    [200000, 300000, None, 400000, 9999999, 250000, 280000, 315000],
    'size':     [1200,   1500,   1800, -100,   2200,    1400,   1650,   1750],
    'bedrooms': [2,      3,      3,    4,      4,       100,    3,      3],
    'age':      [10,     5,      20,   8,      2,       15,     12,     7]
}
df = pd.DataFrame(data)

print("--- shape ---")
print(df.shape)

print("\n--- info ---")
df.info()

print("\n--- describe ---")
print(df.describe())

print("\n--- missing values ---")
print(df.isnull().sum())

print("\n--- first rows ---")
print(df.head(8))
```

**Ask the room after running:** *"How many problems can you spot?
Look at describe() carefully — check the min and max of every column."*

Walk through together:
```
Problems in this dataset:
  price:    one None (missing), one 9,999,999 (probably an error)
  size:     -100 sqft (impossible)
  bedrooms: 100 bedrooms (impossible — outlier)
  age:      looks fine
```

> *"describe() is your best friend. Always check min and max.
> Minimum negative size? Flag it immediately.
> Maximum 100 bedrooms? Flag it immediately.*
>
> *These would be silent errors if you just fed the data straight into sklearn.
> Your model would train, R² would look fine, and your predictions would be garbage."*

---

## SECTION 2: Distributions — What Does the Data Actually Look Like? (20 min)

> *"After shape and missing values, we want to SEE each feature.*
>
> *A histogram tells you things that mean and std can hide.*
> *Is it symmetric? Skewed? Bimodal? Are there spikes at round numbers?*"

Draw on board:
```
COMMON DISTRIBUTION SHAPES:
              NORMAL            SKEWED RIGHT         BIMODAL
               ▓▓               ▓                   ▓      ▓
              ▓▓▓▓▓             ▓▓▓               ▓▓▓▓    ▓▓▓▓
             ▓▓▓▓▓▓▓▓          ▓▓▓▓▓▓            ▓▓▓▓▓▓  ▓▓▓▓▓
            ▓▓▓▓▓▓▓▓▓▓        ▓▓▓▓▓▓▓▓▓          ▓▓▓▓▓▓▓▓▓▓▓▓▓

  Ideal for linear    Income data,         Might be two groups
  regression         house prices         mixed together
```

Code together:
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Use a cleaner dataset for this demo
np.random.seed(42)
clean_df = pd.DataFrame({
    'price':    np.random.lognormal(12.5, 0.4, 500),
    'size':     np.random.normal(1800, 400, 500),
    'bedrooms': np.random.choice([1,2,3,4,5], 500, p=[0.05,0.2,0.4,0.25,0.1]),
    'age':      np.random.uniform(0, 50, 500)
})

for ax, col in zip(axes, clean_df.columns):
    ax.hist(clean_df[col], bins=30, edgecolor='black', color='steelblue')
    ax.set_title(col)
    ax.set_xlabel(col)

plt.tight_layout()
plt.savefig('../visuals/distributions.png', dpi=300)
print("Saved to visuals/distributions.png")
```

> *"Look at price — that lognormal shape is typical for real housing data.
> Prices are skewed right: lots of affordable houses, fewer luxury ones.*
>
> *Linear regression assumes errors are normally distributed.
> Skewed targets often benefit from a log transform: log(price) instead of price.*
> *That's a technique you'll use in the housing project coming up."*

---

## SECTION 3: Correlation — Who Predicts Who? (20 min)

> *"Now the question: which features actually relate to the target?*
>
> *Correlation is a number between -1 and +1.*
> *+1 means: as X goes up, Y goes up in lockstep.*
> *-1 means: as X goes up, Y goes down in lockstep.*
> *0 means: no linear relationship at all."*

Draw on board:
```
CORRELATION SCALE:
  |r| > 0.7   Strong — definitely include in model
  |r| > 0.4   Moderate — probably useful
  |r| > 0.2   Weak — might help with other features
  |r| < 0.1   Nearly none — think carefully before including

NOTE: Correlation measures LINEAR relationships only.
  If the relationship is curved, correlation can be 0
  even when X perfectly predicts Y.
```

Code together:
```python
# Correlation heatmap
import seaborn as sns

corr_matrix = clean_df.corr()
print("Correlation matrix:")
print(corr_matrix.round(3))

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('../visuals/correlation_heatmap.png', dpi=300)
print("Saved to visuals/correlation_heatmap.png")
```

> *"Read the 'price' row. Which features correlate most with price?*
>
> *Also check the feature-to-feature correlations. If two features
> strongly correlate with EACH OTHER, remember from last module:
> that's multicollinearity. You'll want to pick one or transform."*

**Ask the room:** *"If age has a -0.6 correlation with price, what does that tell you?
Would you include it in a model? Why?"*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
EDA SESSION 1:
  1. df.shape, .info(), .describe(), .isnull().sum(), .head()
     → Find shape, types, missing values, obvious outliers

  2. Histograms
     → Find distribution shape, skew, bimodal surprises

  3. Correlation heatmap
     → Find which features predict the target
     → Find which features are redundant with each other
```

**Homework:** Run `data_exploration.py` all the way through.
Write down: (a) the correlation between size and price, (b) any suspicious values in describe().

---

# SESSION 2 (~90 min)
## "Outliers, scatter plots, and cleaning for modeling"

## OPENING (10 min)

> *"Last session we got the bird's-eye view — shape, distributions, correlation matrix.*
>
> *Today we go deeper: what do outliers look like visually?
> How do we decide which ones to remove vs keep?
> And how do we turn EDA findings into actual cleaning decisions
> before we hand the data to a model?"*

---

## SECTION 1: Scatter Plots — Feature vs Target (20 min)

> *"A scatter plot of each feature against the target is the single most
> informative visualization you can make.*
>
> *It shows you:*
> *Is the relationship actually linear? (Good for linear regression)*
> *Are there outlier points dragging the line off course?*
> *Are there clusters or gaps?"*

Code together:
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300
df = pd.DataFrame({
    'size':     np.random.normal(1800, 400, n).clip(500, 4000),
    'bedrooms': np.random.choice([1,2,3,4,5], n),
    'age':      np.random.uniform(0, 50, n),
})
df['price'] = 40000 + 90*df['size'] + 15000*df['bedrooms'] - 1500*df['age']
df['price'] += np.random.normal(0, 25000, n)

# Inject outliers
df.loc[0, 'price'] = 5000000   # luxury mansion outlier
df.loc[1, 'price'] = -50000    # data entry error

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ['size', 'bedrooms', 'age']):
    ax.scatter(df[col], df['price'], alpha=0.4, s=20)
    ax.set_xlabel(col)
    ax.set_ylabel('price')
    ax.set_title(f'{col} vs price')

plt.tight_layout()
plt.savefig('../visuals/scatter_plots.png', dpi=300)
```

> *"See that lone point at $5 million? That's the outlier.*
> *Now, is it a data error or a real luxury house?*
>
> *That's the judgment call. You cannot answer it from the data alone.*
> *You have to ask: is this in my domain? A $5M house in rural Kansas?
> Probably an error. In Beverly Hills? Might be real."*

**Ask the room:** *"What would happen to our regression line if we kept that $5M point?
Would it pull the line up? Down? Which end?"*

---

## SECTION 2: Outlier Detection with IQR (20 min)

> *"The interquartile range method is the standard rule-of-thumb for outlier detection.*
>
> *Q1 is the 25th percentile, Q3 is the 75th percentile.*
> *IQR = Q3 - Q1 — the middle 50% of the data.*
>
> *Anything beyond 1.5 × IQR above Q3 or below Q1 is flagged."*

Draw on board:
```
IQR METHOD:
  |-------|-------------|-------------|-------|
  Q1-1.5IQR    Q1     median    Q3     Q3+1.5IQR
       ↑                                    ↑
   Lower fence                         Upper fence

  Points outside the fences → potential outliers
```

Code together:
```python
def find_outliers_iqr(series, label=""):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = series[(series < lower) | (series > upper)]
    pct = 100 * len(outliers) / len(series)

    print(f"{label}:")
    print(f"  Q1={Q1:,.0f}  Q3={Q3:,.0f}  IQR={IQR:,.0f}")
    print(f"  Valid range: [{lower:,.0f}, {upper:,.0f}]")
    print(f"  Outliers: {len(outliers)} ({pct:.1f}%)")
    return lower, upper

for col in ['price', 'size']:
    find_outliers_iqr(df[col], col)
    print()
```

> *"1.5 × IQR catches about 0.7% of normally distributed data — a reasonable threshold.*
> *You can use 3 × IQR to be more conservative and only catch extreme outliers.*
>
> *What do you do with them?*
> *Option 1: Remove them — only if they're clearly errors.*
> *Option 2: Cap them — clip to the fence value.*
> *Option 3: Keep them — if they're genuinely rare but real.*
>
> *Document every decision. Future you will thank you."*

---

## SECTION 3: Run the Full Module (15 min)

```bash
python3 data_exploration.py
```

Point out each section as it runs:
- Section 1: Load and inspect the simulated dataset
- Section 2: Statistical summaries via describe()
- Section 3: Distribution visualizations saved to `visuals/regression/`
- Section 4: Correlation heatmap

Open the generated visualizations from `visuals/regression/`:
> *"This is the deliverable of EDA: a set of charts that tell the story
> of your data before you've built a single model.*
>
> *In a real project, these go into your EDA notebook.
> Your teammates — or your future self — can read them and understand the data
> without re-running any code."*

---

## SECTION 4: EDA → Decisions (10 min)

> *"EDA is only useful if it drives decisions. Let's connect findings to actions."*

Write on board:
```
FINDING → DECISION

  Missing values found         → Impute with mean/median, or drop rows
  Negative size values         → Remove those rows (impossible)
  Price outlier at $9M         → Investigate; cap or remove
  Price is right-skewed        → Consider log(price) as target
  size-price correlation: 0.85 → Definitely include size
  bedrooms-size correlation: 0.7 → Watch for multicollinearity
  age-price correlation: -0.5   → Include age, expect negative β
```

> *"Every EDA chart should answer: does this change what I do next?
> If not, you don't need the chart.*
>
> *EDA is not decoration. It's the scientific process of data science."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
COMPLETE EDA CHECKLIST:
  Shape & types:      df.shape, df.info()
  Missing values:     df.isnull().sum()
  Summary stats:      df.describe() — check min/max carefully
  Distributions:      histograms per feature
  Relationships:      correlation heatmap + scatter plots
  Outliers:           IQR method; then decide keep/cap/remove
  Document decisions: comment every cleaning step with WHY
```

**Homework — from `data_exploration_lab.md`:**
```python
# Boss Challenge: Outlier Detection
prices = [200000, 220000, 250000, 280000, 300000,
          320000, 350000, 380000, 9000000, 50000]

# Use IQR method:
# 1. Calculate Q1, Q3, IQR
# 2. Find lower and upper bounds
# 3. Identify and print the outlier rows
# 4. Print the "clean" dataset after removing them
```

---

## INSTRUCTOR TIPS

**"Isn't EDA just looking at data? Seems simple."**
> *"That's what everyone thinks before they encounter real data.*
> *Real data has: mixed-type columns, dates stored as strings,
> negative values in columns that can't be negative,
> missing values encoded as 999 or -1 instead of NaN,
> duplicate rows, and columns with 90% missing.*
>
> *The 'simple' checklist is how you catch all of that systematically
> instead of discovering it at 11pm when your model behaves strangely."*

**"When do I stop doing EDA and start modeling?"**
> *"When you can answer: What's the shape of my target distribution?
> Which features correlate with the target? Are there any impossible values?
> Are there high correlations between features?*
>
> *Once you can answer all four from memory, you're ready to model."*

**"What if I find outliers but I'm not sure if they're real?"**
> *"Default rule: don't remove data you can't explain.*
> *Flag it — train with and without — and compare model performance.
> Report both results. That's honest science."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                    10 min
├── Five starter commands           20 min
├── Histograms + distributions      20 min
├── Correlation heatmap             20 min
└── Close + homework                 5 min  [run data_exploration.py]

SESSION 2  (90 min)
├── Opening bridge                  10 min
├── Scatter plots (feature vs target) 20 min
├── Outlier detection with IQR      20 min
├── Run data_exploration.py         15 min
├── EDA → decisions summary         10 min
└── Close + homework                15 min
```

---
*MLForBeginners · Part 1: Regression · Module 08*
