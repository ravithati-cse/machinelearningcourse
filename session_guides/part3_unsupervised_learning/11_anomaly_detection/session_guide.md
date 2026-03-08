# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 11: Anomaly Detection (Project + Part 3 Capstone)
### Single 120-Minute Session

> **The Part 3 graduation project.**
> "Find what doesn't belong" — without any labels to guide you.
> Fraud detection, equipment failure, network intrusion, medical outliers.
> Every major platform runs anomaly detection 24/7.

---

# SESSION (120 min)
## "Detect fraud without ever seeing a fraud example during training"

## Before They Arrive
- Terminal open in `unsupervised_learning/projects/`
- Credit card fraud statistic ready: ~0.1-1% of transactions are fraudulent

---

## OPENING (10 min)

> *"Your credit card gets declined on a trip.*
> *You call the bank: 'Why?'*
> *They say: 'Unusual activity — purchase in a country you've never visited,*
> *for an amount 5x your average.'*
>
> *How did they know it was suspicious?*
> *They didn't see a 'fraud' label on that transaction.*
> *They detected that it was UNUSUAL compared to your normal behavior.*
>
> *That's anomaly detection — unsupervised.*
> *No labels. Just: 'this doesn't look like the other 99.9%.'*
>
> *Used for: fraud (finance), intrusion detection (cybersecurity),*
> *equipment failure (manufacturing), cancer detection (medicine).*
> *Today: four different methods, one dataset, full comparison."*

Write on board:
```
ANOMALY DETECTION:
  Training: only normal data (no anomaly labels needed)
  Scoring: each new point gets an "anomaly score"
  Threshold: above score = anomaly, below = normal

  Why unsupervised?
  → Fraud patterns change constantly (adversarial)
  → Anomalies are RARE — not enough labels to train supervised
  → Truly novel anomalies: you've never seen them before
```

---

## SECTION 1: Run the Pipeline (20 min)

```bash
python3 anomaly_detection.py
```

While it loads:

> *"We have two datasets:*
> *1. Credit card transactions — 3.8% fraud*
> *2. Sensor readings — occasional temperature spikes*
>
> *Key: we train ONLY on normal data.*
> *Then test on a mix of normal + anomalies.*
> *Our known labels are only for EVALUATION — not training.*
> *That's what makes it truly unsupervised."*

---

## SECTION 2: Four Methods Side by Side (30 min)

Watch each method run:

**Method 1 — Statistical (Z-score + IQR):**
> *"Z-score: 'Is this point more than 3 standard deviations from the mean?'*
> *IQR: 'Is this point outside 1.5× the interquartile range?'*
> *Simple, fast, interpretable.*
> *Limitation: assumes Gaussian distribution. Fails on skewed data."*

**Method 2 — Isolation Forest:**
> *"Clever idea: anomalies are EASIER to isolate.*
> *Build random decision trees. Count how many splits needed to isolate each point.*
> *Normal point: buried deep in the tree, many splits needed.*
> *Anomaly: isolated quickly with few splits.*
> *Score: average tree depth. Low depth = anomaly.*
> *No assumptions about distribution. Works in high dimensions."*

**Method 3 — Local Outlier Factor:**
> *"Compares a point's density to its neighbors'.*
> *Normal: similar density to neighbors.*
> *Anomaly: much lower density than neighbors.*
> *Great for data with multiple different densities.*
> *Limitation: can't classify new points (transductive)."*

**Method 4 — One-Class SVM:**
> *"Learns a boundary around the normal data.*
> *Anything outside that boundary = anomaly.*
> *Like drawing a fence around what 'normal' looks like.*
> *Best when normal data has a clear geometric shape."*

---

## SECTION 3: The Comparison Table (15 min)

Fill in the table together from program output:

```
METHOD              Precision  Recall  F1    Train  Predict
─────────────────────────────────────────────────────────────
Z-score/IQR           ?         ?      ?    Fast    Fast
Isolation Forest      ?         ?      ?    Fast    Fast
Local Outlier Factor  ?         ?      ?    Med     Med
One-Class SVM         ?         ?      ?    Slow    Fast
Ensemble (majority)   ?         ?      ?    Med     Fast
```

> *"What tradeoffs are you seeing?*
> *Usually Isolation Forest and Ensemble perform best.*
> *Statistical methods: fastest but make assumptions.*
> *LOF: great for density-based anomalies, but slow on large datasets.*
> *One-Class SVM: sensitive to parameter choice, slower."*

---

## SECTION 4: The Business Cost Analysis (15 min)

> *"Accuracy metrics don't tell the whole story.*
> *What's the COST of getting it wrong?"*

Write on board:
```
FRAUD DETECTION ECONOMICS:

False Negative (miss a fraud):
  Cost = average fraud amount = $850 per case
  + customer calls, unhappy, may churn

False Positive (flag a real transaction):
  Cost = support call + customer frustration = $50 per case
  + some customers cancel their card

ASYMMETRIC COSTS:
  FN is 17× more expensive than FP
  → Lower your threshold to catch more fraud (reduce FN)
  → Accept more false positives

Total cost = FP × $50 + FN × $850
Optimize COST, not accuracy.
```

> *"This is why fraud systems err toward caution.*
> *Better to decline one real transaction than miss one fraud.*
> *The algorithm shows you the tradeoff.*
> *The business decides where to set the threshold."*

---

## SECTION 5: Ensemble — Combining Methods (10 min)

> *"The best systems combine multiple methods.*
> *If 2-of-3 methods flag a point as anomalous: anomaly.*
> *This reduces false positives from any single imperfect model.*
>
> *Real credit card fraud systems use 10-50 models.*
> *The vote threshold is tuned on business cost, not accuracy.*
> *Now you know how to build one."*

---

## PART 3 GRADUATION (20 min)

Write on board slowly:

```
PART 3 COMPLETE — UNSUPERVISED LEARNING MASTERED 🎓

Math Foundations:
  ✅ Distance metrics (Euclidean, Manhattan, Cosine)
  ✅ Variance and covariance — your data's shape
  ✅ Eigenvectors — directions of maximum spread
  ✅ Entropy and silhouette — evaluating without labels

Algorithms:
  ✅ K-Means from scratch — find k spherical clusters
  ✅ Hierarchical clustering — build the full cluster tree
  ✅ DBSCAN — density-based, any shape, finds noise
  ✅ PCA from scratch — linear dimensionality reduction
  ✅ t-SNE and UMAP — non-linear visualization

Projects:
  ✅ Customer segmentation — groups + business strategy
  ✅ Anomaly detection — find fraud without fraud labels

WHAT YOU CAN NOW DO:
  → Take any unlabeled dataset and find its structure
  → Choose the right clustering algorithm for the data shape
  → Reduce 1000 features to 2 for visualization
  → Detect outliers and anomalies without any labels
  → Build production-ready unsupervised ML pipelines

WHAT COMES NEXT:
  Part 4 (formerly Part 3): Deep Neural Networks
  → Where supervised + unsupervised meet
  → Autoencoders: unsupervised DNNs that compress data
  → The full journey continues
```

> *"You now have the full picture of classical machine learning:*
> *supervised (Parts 1-2) and unsupervised (Part 3).*
>
> *The supervised models tell you: 'Given X, predict Y.'*
> *The unsupervised models tell you: 'Given X, find structure.'*
>
> *Together: you can attack almost any real data problem.*
> *From here: deep learning makes everything bigger, better, and messier.*
> *Let's go."*

**Graduation moment:** Each person picks a fraud or not-fraud verdict on one test transaction. Run Isolation Forest. Who got it right?

---

## INSTRUCTOR TIPS

**"What contamination parameter should I use?"**
> *"contamination = expected fraction of anomalies in your data.*
> *For fraud: typically 0.001 to 0.05 (0.1% to 5%).*
> *If you don't know: start with 0.05 and tune based on the false positive rate.*
> *Isolation Forest with contamination='auto' uses 0.1 as default."*

**"What about supervised anomaly detection?"**
> *"If you have labeled fraud/normal examples: use them!*
> *XGBoost or Random Forest with class_weight='balanced' on the minority class.*
> *Supervised always beats unsupervised when you have good labels.*
> *Unsupervised shines when labels are sparse, expensive, or absent.*
> *Real systems often use BOTH: unsupervised to flag candidates,*
> *supervised to confirm."*

---

## Quick Reference
```
Single Session (120 min)
├── Opening — credit card story        10 min
├── Run pipeline                       20 min
├── Four methods walkthrough           30 min
├── Comparison table                   15 min
├── Business cost analysis             15 min
├── Ensemble approach                  10 min
└── Part 3 graduation                  20 min
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 11 (Capstone)*
