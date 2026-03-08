# MLForBeginners — Instructor Guide
## Part 1 · Module 11: Housing Analysis (Project)
### Single-Session Teaching Script

> **Prerequisites:** Modules 01–10 complete. They understand all regression metrics,
> can run EDA, and have used sklearn's LinearRegression. This is their first real-world dataset.
> **Payoff today:** They run a full EDA on the California Housing dataset from sklearn.
> No more simulated toy data — this is 20,640 actual records from the 1990 Census.

---

# SESSION (~120 min)
## "Real data, real EDA — the California Housing Dataset"

## Before They Arrive
- Terminal open in `regression_algorithms/projects/`
- `housing_analysis.py` ready to run (confirm sklearn installed: `python3 -c "from sklearn.datasets import fetch_california_housing"`)
- Whiteboard ready — draw the California map outline loosely
- Print or project a California district map for geographic context
- Have the script output already saved so you can reference specific numbers during discussion

---

## OPENING (15 min)

> *"Everything we've done so far has been on data we invented.*
> *We controlled the distributions, the correlations, the noise.*
>
> *Today that stops.*
>
> *The California Housing dataset is real census data from 1990.*
> *20,640 California districts. Real house prices. Real population counts.*
> *Real problems: skewed distributions, geographic patterns, outliers we didn't put there.*
>
> *This is the closest thing to a professional EDA session you can do
> in a classroom. Let's treat it like one."*

Draw on board:
```
CALIFORNIA HOUSING DATASET:
  20,640 rows (census districts)
  8 features:
    MedInc      — median household income (10,000s of dollars)
    HouseAge    — median age of houses in district
    AveRooms    — average number of rooms per household
    AveBedrms   — average number of bedrooms per household
    Population  — district population
    AveOccup    — average occupants per household
    Latitude    — geographic location
    Longitude   — geographic location

  Target: MedHouseVal — median house value (in $100,000s)
```

> *"Your mission as an analyst — before touching a single model —
> is to understand this data cold.*
> *By the end of the session you should be able to answer:*
> *Which features correlate most with price?*
> *Where in California are the most expensive houses?*
> *What does the distribution of house values look like?*
> *Are there any features we should be suspicious of?"*

---

## PHASE 1: Load and First Inspection (20 min)

Run together:
```bash
python3 housing_analysis.py
```

Let Section 1 print and pause there. Walk through the output together.

Point at the printed output when it appears:
```python
# What we're seeing in the output:
# df.shape: (20640, 9) — 20,640 rows, 9 columns (8 features + target)
# df.info(): all float64, no obvious type problems
# df.describe(): the summary statistics
```

**Walk through describe() together, calling on different people:**

> *"Look at AveRooms. Min is something like 0.8 rooms per household.*
> *Is that possible? An average of less than one room per house?*
> *That's a suspicious row — probably a very dense district or a data quirk."*

> *"Look at Population. What's the max? Compare it to the mean.*
> *Is that distribution symmetric or skewed?*
> *What does a max that's much larger than the mean tell you?"*

> *"Look at MedHouseVal — the target. Max is 5.0.*
> *Remember, values are in units of $100,000.*
> *So max = $500,000. This is 1990 California — that's actually the cap in the dataset.*
> *Any district with value = 5.0 might be censored — capped at $500K.*
> *That's a data quality issue we'll need to think about when we model."*

**Ask the room:** *"Without looking at any plots, what three things from describe()
already worry you or interest you?"*

Write answers on board. Good expected answers:
- AveRooms minimum is below 1 (suspicious)
- Population range is huge — likely skewed
- MedHouseVal caps at 5.0 ($500K) — censored data

---

## PHASE 2: Distributions (20 min)

Let Section 2 of the output print. Open the histogram visualization from `visuals/housing_analysis/`.

Walk through each feature:

> *"MedInc — income. See that right skew? Most districts have middle incomes,
> a few very high-income districts pull the tail right.*
> *Log(income) is often more useful for modeling — we'll see that in the prediction project."*

> *"HouseAge — interesting. That spike at 52 years.*
> *That's actually a censoring artifact — houses older than 52 years
> were all coded as 52 in the 1990 census.*
> *See how real data has weird artifacts you'd never know about without looking?"*

> *"Latitude and Longitude — uniform distributions.*
> *That's expected: we're looking at all of California.*
> *But when we combine them as a scatter plot, we'll see the actual map."*

**Ask the room:** *"For which features would you consider a log transformation before modeling?
Which ones look most skewed?"*

Expected: Population, AveOccup, AveRooms (heavily right-skewed).

Draw on board:
```
LOG TRANSFORM RULE OF THUMB:
  If max > 10× the median → consider log(x)
  Before:  population = [100, 500, 5000, 50000]   (hard to model)
  After:   log(pop)   = [4.6, 6.2, 8.5, 10.8]    (much more uniform)
```

---

## PHASE 3: Correlation Heatmap (20 min)

Let Section 3 (or correlation section) run. Open the heatmap visualization.

> *"Now the most important question in EDA: what actually predicts price?*
> *Read the MedHouseVal row of this heatmap."*

Walk through together on screen, writing findings on board:
```
CORRELATIONS WITH MedHouseVal:
  MedInc:    +0.69  ← strongest predictor by far
  Latitude:  -0.14  ← northern California is cheaper on average
  Longitude: -0.13  ← coastal is more expensive
  HouseAge:  +0.11  ← older districts slightly more expensive (urban cores)
  AveRooms:  +0.15  ← more rooms → more expensive
  Population: -0.02 ← essentially no linear correlation
```

> *"MedInc is the winner — 0.69 correlation with house value.*
> *Richer neighborhoods have more expensive houses. Not surprising.*
>
> *What IS surprising: Latitude and Longitude have low individual correlations.*
> *But geographically, coastal areas are dramatically more expensive.*
> *The correlation doesn't capture that because the pattern is spatial, not linear.*
>
> *This is why we need to look at the geographic plot."*

**Ask the room:** *"If Population has near-zero correlation with price,
should we exclude it from a regression model?
What might be wrong with using correlation alone to decide?"*

Key insight to draw out: correlation measures linear relationships only. Population might interact with other features in a non-linear way. Never exclude features based on correlation alone without thinking about domain knowledge.

---

## PHASE 4: Geographic Visualization (15 min)

Navigate to the geographic plot in `visuals/housing_analysis/` (the scatter plot with latitude/longitude colored by price).

> *"This is the most informative visualization in the whole module.*
> *What do you see?"*

Let them observe and comment. Guide them toward:

> *"Bay Area — San Francisco, Silicon Valley — highest prices.*
> *Los Angeles Basin — also very high.*
> *Central Valley — low prices.*
> *The coast vs inland divide is stark."*

> *"This is why latitude and longitude, combined, are more powerful than separately.*
> *A model that sees latitude and longitude as independent features
> has to figure out that the COMBINATION matters.*
>
> *A better approach: 'distance to coast' or 'distance to San Francisco'
> as engineered features. That's what we'll do in the prediction project."*

Draw on board:
```
GEOGRAPHIC FEATURE ENGINEERING IDEAS:
  latitude + longitude → cluster by region
  latitude + longitude → distance_to_coast
  latitude + longitude → distance_to_SF
  latitude + longitude → distance_to_LA

  (Linear regression can't 'see' a map — you have to help it)
```

---

## PHASE 5: Missing Values and Data Quality (10 min)

Let the missing values section of the output print.

> *"Good news for this dataset: no missing values.*
> *sklearn's built-in datasets are cleaned.*
>
> *But we talked about data quality issues:*
> *— HouseAge capped at 52*
> *— MedHouseVal capped at 5.0 ($500K)*
> *— AveRooms minimum below 1.0*
>
> *In a professional project, you'd document each of these decisions:
> Do we remove the capped values? Winsorize? Flag them?*
> *We'll address some of these in the prediction project."*

Write on board:
```
DATA QUALITY DECISIONS (document every one):
  MedHouseVal = 5.0 (2,290 rows, ~11%)
    → Option A: Remove them (might lose information)
    → Option B: Keep and acknowledge in limitations
    → Option C: Flag as a separate indicator feature

  AveRooms < 1.0
    → Option A: Investigate — might be legitimate dense housing
    → Option B: Remove if statistically impossible

  HouseAge = 52 (censored)
    → Flag as "age unknown or >52 years" indicator
```

---

## PHASE 6: EDA Summary and Handoff to Prediction Project (10 min)

> *"This is the deliverable of a housing analysis project.*
> *Not a model. Not predictions. A STORY about the data.*
>
> *If your manager asked you 'what did you learn about California housing in 1990?'
> you should be able to answer in three sentences with evidence."*

Ask the room to help build the summary on the board:
```
CALIFORNIA HOUSING EDA — FINDINGS:

  1. Income is the primary driver of house values (r=0.69).
     Districts with higher median income have dramatically higher prices.

  2. Location dominates at the regional level: Bay Area and LA
     command the highest prices. Central Valley is cheapest.
     Single lat/lon features miss this — need geographic engineering.

  3. House values are capped at $500K (11% of data). Models should
     be aware they cannot predict above this value from this dataset.

  Recommended for modeling:
    - Use MedInc (strongest predictor)
    - Engineer distance-to-coast or regional cluster features
    - Consider log(Population) and log(AveOccup) for skewed features
    - Flag or remove the capped MedHouseVal rows for regression
```

> *"This summary is the bridge between EDA and modeling.*
> *Next session: we'll actually build the model — with all these insights baked in."*

---

## CLOSING (10 min)

Board summary:
```
PROJECT 1 — HOUSING ANALYSIS:
  Dataset:       20,640 California districts, 1990 Census
  Key findings:
    Best predictor:    MedInc (r=0.69)
    Geographic effect: Coastal >> Inland (need spatial features)
    Data quality:      Value capping at $500K (11% of rows)

  EDA tools used:
    df.describe()     — caught suspicious min/max
    Histograms        — found skewed distributions, censored age
    Correlation map   — ranked feature importance
    Geographic plot   — revealed spatial patterns
    IQR / min-max     — flagged outliers and artifacts
```

**Homework:** Before the prediction session, write one paragraph answering:
*"If you were a real estate investor in 1990 California,
which three features from this dataset would you most want to know about a district,
and why? Cite specific numbers from the EDA."*

This forces them to connect data to decisions — the core skill.

---

## INSTRUCTOR TIPS

**"Why are we spending a whole session just looking at data?"**
> *"Because every real project does this, and almost every beginner skips it.*
> *The number of 'I trained a model and got weird results' problems that trace back
> to unchecked data quality issues is enormous.*
>
> *A team that EDA's properly builds one good model.*
> *A team that skips EDA builds five bad ones and then tries to figure out why."*

**"The geographic plot is cool but how does a model use lat/lon?"**
> *"As raw features, it can, but not well — linear regression sees them as independent
> linear contributors. In practice you'd engineer features like distance_to_coast
> or create a k-means cluster column for 'region'.*
> *This is the feature engineering step in the prediction project."*

**"Should we remove the 11% of rows where MedHouseVal = 5.0?"**
> *"That's a real modeling decision with tradeoffs.*
> *Keeping them: model trained on full data but target is 'censored' — we don't
> know true values above $500K. Predictions above that are unreliable.*
> *Removing them: we lose data, and our model can't handle luxury homes at all.*
>
> *Best practice: note the limitation, train on all data, and evaluate separately
> on the capped vs non-capped subsets to understand the impact."*

---

## Quick Reference

```
SESSION  (120 min)
├── Opening + dataset overview      15 min
├── Phase 1: Load + describe()      20 min
├── Phase 2: Distributions          20 min
├── Phase 3: Correlation heatmap    20 min
├── Phase 4: Geographic viz         15 min
├── Phase 5: Missing values         10 min
├── Phase 6: EDA summary            10 min
└── Closing + homework              10 min
```

---
*MLForBeginners · Part 1: Regression · Module 11*
