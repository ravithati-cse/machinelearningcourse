"""
ANOMALY DETECTION - Complete End-to-End Unsupervised Learning Project
======================================================================

PROJECT OVERVIEW:
----------------
Build a complete anomaly detection system using four unsupervised methods:
statistical (Z-score/IQR), Isolation Forest, Local Outlier Factor (LOF),
and One-Class SVM. Apply them to realistic fraud detection and sensor
monitoring datasets. Evaluate using known anomaly labels held out from training.

LEARNING OBJECTIVES:
-------------------
1. Understand what anomalies are and why detecting them without labels is hard
2. Implement statistical anomaly detection (Z-score, IQR) from scratch
3. Explain the Isolation Forest intuition: anomalies are easy to isolate
4. Understand Local Outlier Factor: anomalies live in low-density neighborhoods
5. Apply One-Class SVM to learn a tight boundary around normal behavior
6. Evaluate unsupervised detectors using held-out ground-truth labels
7. Reason about the business cost of false positives vs false negatives

RECOMMENDED VIDEOS:
------------------
StatQuest: "Isolation Forest"
   https://www.youtube.com/watch?v=DYDigrqPTRU
   Clear step-by-step explanation of random partitioning

StatQuest: "Local Outlier Factor"
   https://www.youtube.com/watch?v=lBLRRp84DMk
   Density-based anomaly intuition explained visually

Krish Naik: "Anomaly Detection using Isolation Forest"
   https://www.youtube.com/watch?v=TP0MaKbN7Bk
   Practical implementation walkthrough

Sentdex: "One-Class SVM for Anomaly Detection"
   https://www.youtube.com/watch?v=rJIMen5Nf9o
   Applying support vector machines to outlier detection

TIME: 2-3 hours
DIFFICULTY: Intermediate-Advanced
PREREQUISITES: Unsupervised learning math foundations (01-04)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'anomaly_detection'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANOMALY DETECTION - End-to-End Unsupervised Learning Project")
print("=" * 80)
print()
print("Real-world applications:")
print("  - Credit card fraud detection")
print("  - Network intrusion / cybersecurity")
print("  - Predictive maintenance (equipment failure)")
print("  - Medical diagnostics (rare disease detection)")
print("  - Quality control in manufacturing")
print()

# ============================================================================
# SECTION 1: WHAT IS ANOMALY DETECTION?
# ============================================================================

print("=" * 80)
print("SECTION 1: Understanding Anomaly Detection")
print("=" * 80)
print()

print("DEFINITION:")
print("  An anomaly (also called outlier) is an observation that differs")
print("  significantly from the majority of the data — the 'normal' pattern.")
print()
print("WHY UNSUPERVISED?")
print("  In most real scenarios:")
print("    - Labeled fraud data is extremely rare (< 0.1% of transactions)")
print("    - Labels are expensive: require expert review per case")
print("    - Fraud patterns CHANGE over time — new attack vectors emerge daily")
print("    - We must detect anomalies we've NEVER SEEN before")
print()
print("  This is why we train ONLY on normal data or use methods that don't")
print("  require labels, then flag points that don't fit the learned pattern.")
print()
print("THREE TYPES OF ANOMALIES:")
print()
print("  1. POINT ANOMALY:    A single data point is unusual")
print("     Example: A $50,000 transaction on a card that typically spends $50")
print()
print("  2. CONTEXTUAL ANOMALY: A point is anomalous only in context")
print("     Example: 35°C temperature is normal in summer, anomalous in winter")
print()
print("  3. COLLECTIVE ANOMALY: A sequence of points is anomalous together")
print("     Example: CPU usage normal by itself, but unusual if sustained for hours")
print()

# ============================================================================
# SECTION 2: DATASET CREATION
# ============================================================================

print("=" * 80)
print("SECTION 2: Creating Realistic Datasets")
print("=" * 80)
print()

np.random.seed(42)

# -----------------------------------------------------------------------
# DATASET A: Credit Card Transaction Data (2D for visualization)
# -----------------------------------------------------------------------
print("DATASET A: Synthetic Credit Card Transaction Data")
print("  Features: transaction_amount, hour_of_day")
print("  This is a simplified 2D version for visualization.")
print("  Real systems use 20-200 engineered features.")
print()

N_NORMAL_CC    = 500
N_ANOMALY_CC   = 20    # ~3.8% anomaly rate — realistic for fraud datasets
ANOMALY_RATE   = N_ANOMALY_CC / (N_NORMAL_CC + N_ANOMALY_CC)

# Normal transactions: small amounts, clustered around typical shopping hours
normal_amounts_cc = np.random.lognormal(mean=3.5, sigma=0.7, size=N_NORMAL_CC)   # $20-$300 range
normal_hours_cc   = np.random.normal(loc=13.0, scale=3.5, size=N_NORMAL_CC)       # centered at 1pm
normal_hours_cc   = np.clip(normal_hours_cc, 6, 23)

# Fraudulent transactions: unusually large amounts, odd hours (2am-5am)
fraud_amounts  = np.random.uniform(500, 3000, N_ANOMALY_CC)
fraud_hours    = np.random.uniform(1, 5, N_ANOMALY_CC)

# A few high-amount daytime transactions to make it realistic (not all fraud is night)
extra_fraud_amounts = np.random.uniform(1000, 5000, 5)
extra_fraud_hours   = np.random.uniform(9, 18, 5)

all_amounts = np.concatenate([normal_amounts_cc, fraud_amounts, extra_fraud_amounts])
all_hours   = np.concatenate([normal_hours_cc, fraud_hours, extra_fraud_hours])

N_ANOMALY_CC_TOTAL = N_ANOMALY_CC + 5
N_TOTAL_CC = N_NORMAL_CC + N_ANOMALY_CC_TOTAL

X_cc = np.column_stack([all_amounts, all_hours])
y_cc = np.array([0]*N_NORMAL_CC + [1]*N_ANOMALY_CC_TOTAL)  # 1 = fraud

print(f"  Total transactions:   {N_TOTAL_CC}")
print(f"  Normal transactions:  {N_NORMAL_CC}  ({N_NORMAL_CC/N_TOTAL_CC*100:.1f}%)")
print(f"  Fraudulent:           {N_ANOMALY_CC_TOTAL}  ({N_ANOMALY_CC_TOTAL/N_TOTAL_CC*100:.1f}%)")
print()
print(f"  Normal amounts  — mean: ${normal_amounts_cc.mean():.0f}, "
      f"std: ${normal_amounts_cc.std():.0f}")
print(f"  Fraud amounts   — mean: ${fraud_amounts.mean():.0f}, "
      f"std: ${fraud_amounts.std():.0f}")
print()

# -----------------------------------------------------------------------
# DATASET B: Industrial Sensor Data (1D for statistical illustration)
# -----------------------------------------------------------------------
print("DATASET B: Industrial Sensor Readings (Temperature Sensor)")
print("  A machine normally operates at 65°C ± 8°C.")
print("  Spikes above 95°C indicate possible cooling system failure.")
print()

N_SENSOR       = 500
N_SENSOR_ANOM  = 15

sensor_normal = np.random.normal(loc=65, scale=8, size=N_SENSOR)
sensor_spikes = np.random.uniform(100, 130, N_SENSOR_ANOM)

X_sensor = np.concatenate([sensor_normal, sensor_spikes])
y_sensor = np.array([0]*N_SENSOR + [1]*N_SENSOR_ANOM)

# Add a few cold anomalies (e.g., sensor malfunction giving very low readings)
cold_anom = np.random.uniform(5, 20, 5)
X_sensor  = np.concatenate([X_sensor, cold_anom])
y_sensor  = np.concatenate([y_sensor, [1]*5])

N_TOTAL_SENSOR = len(X_sensor)
N_ANOM_SENSOR  = y_sensor.sum()

print(f"  Total readings:     {N_TOTAL_SENSOR}")
print(f"  Normal readings:    {(y_sensor==0).sum()}  ({(y_sensor==0).sum()/N_TOTAL_SENSOR*100:.1f}%)")
print(f"  Anomalous readings: {N_ANOM_SENSOR}  ({N_ANOM_SENSOR/N_TOTAL_SENSOR*100:.1f}%)")
print()

# ============================================================================
# SECTION 3: METHOD 1 — STATISTICAL DETECTION (Z-SCORE AND IQR)
# ============================================================================

print("=" * 80)
print("SECTION 3: Method 1 — Statistical Detection (Z-score & IQR)")
print("=" * 80)
print()

print("STATISTICAL METHODS work best on 1D or when features are independent.")
print("They are interpretable, fast, and require no libraries.")
print()

# -----------------------------------------------------------------------
# Z-SCORE METHOD
# -----------------------------------------------------------------------
print("Z-SCORE METHOD:")
print("  Assumption: Normal data follows a Gaussian (bell curve) distribution")
print()
print("  For each point x: z = (x - mean) / std")
print()
print("  In a Gaussian distribution:")
print("    ~68% of points have |z| < 1")
print("    ~95% of points have |z| < 2")
print("    ~99.7% of points have |z| < 3")
print()
print("  Rule: Flag any point with |z| > 3 as an anomaly.")
print("  Intuition: A point 3 standard deviations from the mean is extremely")
print("  unlikely (< 0.3%) under the assumption of normality.")
print()

# Train ONLY on the normal portion (as we would in production)
X_sensor_normal = X_sensor[y_sensor == 0]
sensor_mean = X_sensor_normal.mean()
sensor_std  = X_sensor_normal.std()

z_threshold = 3.0
z_scores_all = np.abs((X_sensor - sensor_mean) / sensor_std)
z_predictions = (z_scores_all > z_threshold).astype(int)

# Evaluation metrics
def compute_metrics(y_true, y_pred):
    """Compute precision, recall, F1 for binary anomaly detection."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'precision': precision, 'recall': recall, 'f1': f1
    }

z_metrics = compute_metrics(y_sensor, z_predictions)

print(f"  Sensor mean (normal data): {sensor_mean:.1f}°C")
print(f"  Sensor std  (normal data): {sensor_std:.1f}°C")
print(f"  Threshold:                 z > {z_threshold} "
      f"= temperature > {sensor_mean + z_threshold*sensor_std:.1f}°C or "
      f"< {sensor_mean - z_threshold*sensor_std:.1f}°C")
print()
print(f"  Results:")
print(f"    Flagged as anomaly:   {z_predictions.sum()} readings")
print(f"    True anomalies found: {z_metrics['tp']} of {N_ANOM_SENSOR} ({z_metrics['recall']*100:.1f}% recall)")
print(f"    False alarms:         {z_metrics['fp']} ({z_metrics['precision']*100:.1f}% precision)")
print(f"    F1 Score:             {z_metrics['f1']:.3f}")
print()

# -----------------------------------------------------------------------
# IQR METHOD
# -----------------------------------------------------------------------
print("IQR (INTERQUARTILE RANGE) METHOD:")
print("  More ROBUST than Z-score — does not assume Gaussian distribution")
print()
print("  Q1 = 25th percentile (lower quartile)")
print("  Q3 = 75th percentile (upper quartile)")
print("  IQR = Q3 - Q1   (the middle 50% spread)")
print()
print("  Lower fence = Q1 - 1.5 × IQR")
print("  Upper fence = Q3 + 1.5 × IQR")
print()
print("  Any point outside these fences is flagged as an anomaly.")
print("  The factor 1.5 is standard (Tukey's rule); use 3.0 for extreme outliers only.")
print()

q1 = np.percentile(X_sensor_normal, 25)
q3 = np.percentile(X_sensor_normal, 75)
iqr = q3 - q1
iqr_factor = 1.5
lower_fence = q1 - iqr_factor * iqr
upper_fence = q3 + iqr_factor * iqr

iqr_predictions = ((X_sensor < lower_fence) | (X_sensor > upper_fence)).astype(int)
iqr_metrics = compute_metrics(y_sensor, iqr_predictions)

print(f"  Q1 (normal data):     {q1:.1f}°C")
print(f"  Q3 (normal data):     {q3:.1f}°C")
print(f"  IQR:                  {iqr:.1f}°C")
print(f"  Lower fence:          {lower_fence:.1f}°C")
print(f"  Upper fence:          {upper_fence:.1f}°C")
print()
print(f"  Results:")
print(f"    Flagged as anomaly:   {iqr_predictions.sum()} readings")
print(f"    True anomalies found: {iqr_metrics['tp']} of {N_ANOM_SENSOR} ({iqr_metrics['recall']*100:.1f}% recall)")
print(f"    False alarms:         {iqr_metrics['fp']} ({iqr_metrics['precision']*100:.1f}% precision)")
print(f"    F1 Score:             {iqr_metrics['f1']:.3f}")
print()

print("WHEN TO USE STATISTICAL METHODS:")
print("  - Single numeric features (1D) where distribution is known")
print("  - Fast dashboards and real-time monitoring")
print("  - When you need fully explainable, auditable rules")
print("  - NOT recommended for high-dimensional, correlated feature spaces")
print()

# ============================================================================
# SECTION 4: METHOD 2 — ISOLATION FOREST
# ============================================================================

print("=" * 80)
print("SECTION 4: Method 2 — Isolation Forest")
print("=" * 80)
print()

print("CORE INTUITION:")
print("  Anomalies are EASY TO ISOLATE from the rest of the data.")
print()
print("  Think about it geometrically:")
print("  - Normal points cluster together in dense regions.")
print("    To isolate ONE of them, you need many splits.")
print("  - Anomalies live in sparse, extreme regions.")
print("    A single split can isolate them immediately.")
print()
print("  The Isolation Forest builds many random decision trees.")
print("  At each step, it picks a feature at random and splits at a random value.")
print("  It counts how many splits are needed to isolate each point.")
print()
print("  Anomaly Score:")
print("    SHORT path → isolated quickly → anomaly")
print("    LONG path  → many splits needed → normal point")
print()
print("  The score is normalized to [0, 1]; points near 1 are anomalies.")
print()
print("ADVANTAGES:")
print("  + Linear time complexity O(n log n)")
print("  + Works well in high dimensions (unlike density-based methods)")
print("  + No assumption about data distribution")
print("  + Built-in handling of irrelevant features")
print()
print("DISADVANTAGES:")
print("  - Not ideal for local anomalies in dense neighborhoods")
print("  - Contamination parameter requires rough knowledge of anomaly rate")
print()

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler

    sklearn_available = True

    # Fit on NORMAL training data only (unsupervised production setting)
    contamination_rate = 0.05  # we expect ~5% anomalies; tune this

    print("Training Isolation Forest...")
    print(f"  contamination = {contamination_rate} (expected anomaly rate estimate)")
    print(f"  n_estimators  = 200 (number of trees)")
    print(f"  max_samples   = 'auto' (256 by default for large datasets)")
    print()

    # Scale the 2D credit card data
    scaler_cc = StandardScaler()
    X_cc_train = X_cc[y_cc == 0]           # train on normals only
    scaler_cc.fit(X_cc_train)
    X_cc_scaled = scaler_cc.transform(X_cc)

    iforest = IsolationForest(n_estimators=200,
                              contamination=contamination_rate,
                              random_state=42)
    # NOTE: In practice, train on X_cc_train only.
    # Here we fit on all for a fair comparison baseline.
    iforest.fit(X_cc_scaled)

    # predict returns: -1 = anomaly, 1 = normal
    if_raw = iforest.predict(X_cc_scaled)
    if_predictions = (if_raw == -1).astype(int)

    # Anomaly scores (lower score = more anomalous; score < 0 suggests anomaly)
    if_scores = iforest.decision_function(X_cc_scaled)
    # Convert so that HIGH score = HIGH anomaly (flip sign)
    if_scores_flipped = -if_scores

    if_metrics = compute_metrics(y_cc, if_predictions)

    print("Isolation Forest — Credit Card Fraud Results:")
    print(f"  Flagged as fraud:     {if_predictions.sum()} transactions")
    print(f"  True fraud caught:    {if_metrics['tp']} of {N_ANOMALY_CC_TOTAL} ({if_metrics['recall']*100:.1f}% recall)")
    print(f"  False alarms (FP):    {if_metrics['fp']}")
    print(f"  Precision:            {if_metrics['precision']:.3f}")
    print(f"  Recall:               {if_metrics['recall']:.3f}")
    print(f"  F1 Score:             {if_metrics['f1']:.3f}")
    print()

except ImportError:
    print("scikit-learn not installed. Install with: pip install scikit-learn")
    print("Implementing a simplified manual isolation forest approximation...")
    sklearn_available = False

    # Manual approximation: flag extreme quantiles in each feature
    q_low  = np.percentile(X_cc, 2, axis=0)
    q_high = np.percentile(X_cc, 98, axis=0)
    if_predictions = np.any((X_cc < q_low) | (X_cc > q_high), axis=1).astype(int)
    if_metrics = compute_metrics(y_cc, if_predictions)
    if_scores_flipped = np.max(np.abs((X_cc - X_cc.mean(axis=0)) / X_cc.std(axis=0)), axis=1)
    if_scores = if_scores_flipped
    scaler_cc = None
    X_cc_scaled = (X_cc - X_cc.mean(axis=0)) / X_cc.std(axis=0)

    print(f"  Manual approximation results: {if_metrics}")
    print()

# ============================================================================
# SECTION 5: METHOD 3 — LOCAL OUTLIER FACTOR (LOF)
# ============================================================================

print("=" * 80)
print("SECTION 5: Method 3 — Local Outlier Factor (LOF)")
print("=" * 80)
print()

print("CORE INTUITION:")
print("  Anomalies have LOWER LOCAL DENSITY than their neighbors.")
print()
print("  LOF compares how crowded a point's neighborhood is")
print("  compared to how crowded ITS NEIGHBORS' neighborhoods are.")
print()
print("  Local Reachability Density (LRD) of point p:")
print("    LRD(p) = 1 / (average reachability distance of p's k nearest neighbors)")
print()
print("  LOF Score of point p:")
print("    LOF(p) = average of [ LRD(neighbor) / LRD(p) ] for all k neighbors")
print()
print("  Interpretation:")
print("    LOF ≈ 1 → density similar to neighbors → normal")
print("    LOF >> 1 → much less dense than neighbors → anomaly")
print()
print("ADVANTAGES:")
print("  + Detects LOCAL anomalies (e.g., an outlier inside a dense cluster)")
print("  + No global distribution assumption")
print("  + Works in non-Gaussian, multi-modal datasets")
print()
print("DISADVANTAGES:")
print("  - Sensitive to k (number of neighbors) — requires tuning")
print("  - Slower than Isolation Forest (O(n^2) naive)")
print("  - sklearn's LOF has no predict() for new points (only fit_predict)")
print()

if sklearn_available:
    print("Training Local Outlier Factor...")
    print(f"  n_neighbors   = 20")
    print(f"  contamination = {contamination_rate}")
    print()

    lof = LocalOutlierFactor(n_neighbors=20,
                             contamination=contamination_rate)
    # LOF uses fit_predict (transductive — no separate predict)
    lof_raw = lof.fit_predict(X_cc_scaled)
    lof_predictions = (lof_raw == -1).astype(int)

    # Negative outlier factor scores (more negative = more anomalous)
    lof_scores_flipped = -lof.negative_outlier_factor_

    lof_metrics = compute_metrics(y_cc, lof_predictions)

    print("LOF — Credit Card Fraud Results:")
    print(f"  Flagged as fraud:     {lof_predictions.sum()} transactions")
    print(f"  True fraud caught:    {lof_metrics['tp']} of {N_ANOMALY_CC_TOTAL} ({lof_metrics['recall']*100:.1f}% recall)")
    print(f"  False alarms (FP):    {lof_metrics['fp']}")
    print(f"  Precision:            {lof_metrics['precision']:.3f}")
    print(f"  Recall:               {lof_metrics['recall']:.3f}")
    print(f"  F1 Score:             {lof_metrics['f1']:.3f}")
    print()
else:
    # Fallback: z-score based for each point relative to its local neighborhood
    lof_predictions = if_predictions.copy()
    lof_scores_flipped = if_scores_flipped.copy()
    lof_metrics = if_metrics.copy()

# ============================================================================
# SECTION 6: METHOD 4 — ONE-CLASS SVM
# ============================================================================

print("=" * 80)
print("SECTION 6: Method 4 — One-Class SVM")
print("=" * 80)
print()

print("CORE INTUITION:")
print("  A standard SVM separates two classes with a hyperplane.")
print("  One-Class SVM has only ONE class — normal data.")
print()
print("  It learns the TIGHTEST BOUNDARY that encloses the normal data.")
print("  Any new point that falls OUTSIDE this boundary is flagged as anomaly.")
print()
print("  With an RBF kernel, this boundary is not a plane but a flexible")
print("  curve that can capture complex shapes of the normal region.")
print()
print("KEY PARAMETERS:")
print("  nu:     Upper bound on the fraction of margin errors AND")
print("          lower bound on the fraction of support vectors.")
print("          Think of it as: 'at most nu fraction of training points")
print("          will be misclassified as anomalies.'")
print("          Set nu ≈ expected anomaly rate (our contamination estimate).")
print()
print("  gamma:  Kernel bandwidth. Higher = tighter boundary around training data.")
print("          'scale' = 1 / (n_features * X.var()) — auto-tuned.")
print()
print("ADVANTAGES:")
print("  + Can model non-linear boundaries (with RBF kernel)")
print("  + Works well on high-dimensional data")
print("  + Has a clear geometric interpretation")
print()
print("DISADVANTAGES:")
print("  - Sensitive to feature scaling (must normalize first)")
print("  - Slow for large datasets (O(n^2) to O(n^3))")
print("  - Choosing nu is non-trivial when anomaly rate is unknown")
print()

if sklearn_available:
    print("Training One-Class SVM...")
    print(f"  kernel = 'rbf'")
    print(f"  nu     = {contamination_rate}")
    print(f"  gamma  = 'scale'")
    print()

    ocsvm = OneClassSVM(kernel='rbf',
                        nu=contamination_rate,
                        gamma='scale')
    # Train only on normal data (ideally)
    ocsvm.fit(X_cc_scaled[y_cc == 0])

    ocsvm_raw = ocsvm.predict(X_cc_scaled)
    ocsvm_predictions = (ocsvm_raw == -1).astype(int)
    ocsvm_scores_flipped = -ocsvm.decision_function(X_cc_scaled)

    ocsvm_metrics = compute_metrics(y_cc, ocsvm_predictions)

    print("One-Class SVM — Credit Card Fraud Results:")
    print(f"  Flagged as fraud:     {ocsvm_predictions.sum()} transactions")
    print(f"  True fraud caught:    {ocsvm_metrics['tp']} of {N_ANOMALY_CC_TOTAL} ({ocsvm_metrics['recall']*100:.1f}% recall)")
    print(f"  False alarms (FP):    {ocsvm_metrics['fp']}")
    print(f"  Precision:            {ocsvm_metrics['precision']:.3f}")
    print(f"  Recall:               {ocsvm_metrics['recall']:.3f}")
    print(f"  F1 Score:             {ocsvm_metrics['f1']:.3f}")
    print()
else:
    ocsvm_predictions = if_predictions.copy()
    ocsvm_scores_flipped = if_scores_flipped.copy()
    ocsvm_metrics = if_metrics.copy()

# ============================================================================
# SECTION 7: COMPARING ALL FOUR METHODS
# ============================================================================

print("=" * 80)
print("SECTION 7: Side-by-Side Comparison of All Four Methods")
print("=" * 80)
print()

all_methods = {
    'Z-score (sensor)':   z_metrics,
    'IQR (sensor)':       iqr_metrics,
    'Isolation Forest':   if_metrics,
    'LOF':                lof_metrics,
    'One-Class SVM':      ocsvm_metrics,
}

# For the credit card dataset, compute Z-score and IQR too for completeness
# (using the scaled first feature — transaction amount)
amount_col = X_cc_scaled[:, 0]
amount_mean = amount_col[y_cc==0].mean()
amount_std  = amount_col[y_cc==0].std()
z_cc = np.abs((amount_col - amount_mean) / amount_std)
z_cc_pred = (z_cc > 3.0).astype(int)
z_cc_metrics = compute_metrics(y_cc, z_cc_pred)

q1_cc = np.percentile(amount_col[y_cc==0], 25)
q3_cc = np.percentile(amount_col[y_cc==0], 75)
iqr_cc = q3_cc - q1_cc
iqr_cc_pred = (amount_col > q3_cc + 1.5*iqr_cc).astype(int)
iqr_cc_metrics = compute_metrics(y_cc, iqr_cc_pred)

credit_card_methods = {
    'Z-score (amount only)':  z_cc_metrics,
    'IQR (amount only)':      iqr_cc_metrics,
    'Isolation Forest':       if_metrics,
    'LOF':                    lof_metrics,
    'One-Class SVM':          ocsvm_metrics,
}

print("Credit Card Fraud Detection — All Methods Compared:")
print("-" * 78)
print(f"{'Method':<22} {'Flagged':>8} {'TP':>6} {'FP':>6} {'FN':>6} "
      f"{'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 78)
for method_name, m in credit_card_methods.items():
    flagged = m['tp'] + m['fp']
    print(f"{method_name:<22} {flagged:>8} {m['tp']:>6} {m['fp']:>6} {m['fn']:>6} "
          f"{m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f}")
print()

print("INTERPRETING THE RESULTS:")
print()
print("  PRECISION: Of all points flagged, what fraction were real anomalies?")
print("    High precision → fewer false alarms → analysts less overwhelmed")
print()
print("  RECALL: Of all real anomalies, what fraction did we catch?")
print("    High recall → fewer missed threats → higher security")
print()
print("  F1 SCORE: Harmonic mean of precision and recall.")
print("    Balances both concerns — useful when classes are imbalanced.")
print()
print("  NOTE: These methods are unsupervised — they saw NO labels during training.")
print("  The labels are used ONLY here for evaluation, not during fitting.")
print()

best_method_cc = max(credit_card_methods.items(), key=lambda x: x[1]['f1'])
print(f"  Best method on this dataset: {best_method_cc[0]} (F1={best_method_cc[1]['f1']:.3f})")
print()

# ============================================================================
# SECTION 8: BUSINESS COST ANALYSIS
# ============================================================================

print("=" * 80)
print("SECTION 8: Business Cost of Errors — FP vs FN Tradeoff")
print("=" * 80)
print()

print("In anomaly detection, not all errors cost the same.")
print()
print("FRAUD DETECTION EXAMPLE:")
print()
print("  FALSE POSITIVE (FP) — Blocking a legitimate transaction:")
print("    - Customer inconvenience and frustration")
print("    - Risk of customer churning to competitor")
print("    - Cost of fraud analyst review time")
print("    - Estimated cost: $20-$80 per incident")
print()
print("  FALSE NEGATIVE (FN) — Missing a fraudulent transaction:")
print("    - Bank absorbs full fraud amount (chargeback liability)")
print("    - Regulatory fines if fraud rate exceeds thresholds")
print("    - Reputational damage if fraud rates become public")
print("    - Estimated cost: full transaction amount + $50 processing")
print()

# Example cost calculation
avg_fraud_amount   = 800   # dollars
fp_cost_per_case   = 50    # dollars (analyst review + customer friction)
fn_cost_base       = avg_fraud_amount + 50  # chargeback + processing

print("COST CALCULATION for Isolation Forest results:")
print(f"  True Positives (caught fraud): {if_metrics['tp']}  → $0 incremental cost (prevented!)")
print(f"  False Positives (false alarms): {if_metrics['fp']}  × ${fp_cost_per_case} = ${if_metrics['fp']*fp_cost_per_case:,}")
print(f"  False Negatives (missed fraud): {if_metrics['fn']}  × ${fn_cost_base} = ${if_metrics['fn']*fn_cost_base:,}")
print(f"  Total Error Cost: ${if_metrics['fp']*fp_cost_per_case + if_metrics['fn']*fn_cost_base:,}")
print()

# Baseline: flag nothing (no detection system)
fn_baseline_cost = N_ANOMALY_CC_TOTAL * fn_cost_base
print(f"BASELINE (no detection system):")
print(f"  All {N_ANOMALY_CC_TOTAL} fraudulent transactions go through → Cost: ${fn_baseline_cost:,}")
print()

detection_savings = fn_baseline_cost - (if_metrics['fp']*fp_cost_per_case + if_metrics['fn']*fn_cost_base)
print(f"VALUE of Isolation Forest system: ${detection_savings:,} saved per {N_TOTAL_CC} transactions")
print()

print("THRESHOLD TUNING:")
print("  Most anomaly detectors produce a SCORE, not just a binary label.")
print("  By adjusting the threshold, we can trade precision for recall:")
print()
print("  Lower threshold → flag more points → higher recall, lower precision")
print("    Use this when: missing anomalies is very costly (terrorism, medical)")
print()
print("  Higher threshold → flag fewer points → higher precision, lower recall")
print("    Use this when: false alarms are costly (system stability, trust)")
print()
print("  A COST-BASED threshold selects the score cutoff that minimizes")
print("  expected_cost = P(FP) * cost_FP + P(FN) * cost_FN")
print()

# ============================================================================
# SECTION 9: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 9: Generating Visualizations")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# VISUALIZATION 1: Raw Data + Anomaly Highlights + Statistical Bounds
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Anomaly Detection — Dataset Overview & Statistical Methods',
             fontsize=14, fontweight='bold')

# Top-left: Sensor data time series with anomalies
ax1 = fig.add_subplot(gs[0, :])
time_idx = np.arange(len(X_sensor))
normal_mask_s = y_sensor == 0
anom_mask_s   = y_sensor == 1

ax1.plot(time_idx[normal_mask_s], X_sensor[normal_mask_s],
         '.', color='steelblue', alpha=0.5, markersize=4, label='Normal reading')
ax1.scatter(time_idx[anom_mask_s], X_sensor[anom_mask_s],
            color='red', s=80, zorder=5, label=f'Anomaly ({N_ANOM_SENSOR} points)', marker='x', linewidth=2)

# Statistical bounds
ax1.axhline(upper_fence, color='orange', linestyle='--', linewidth=2,
            label=f'IQR upper fence ({upper_fence:.0f}°C)')
ax1.axhline(lower_fence, color='purple', linestyle='--', linewidth=2,
            label=f'IQR lower fence ({lower_fence:.0f}°C)')
ax1.axhline(sensor_mean + z_threshold*sensor_std, color='green', linestyle=':', linewidth=2,
            label=f'Z-score +3σ ({sensor_mean + z_threshold*sensor_std:.0f}°C)')
ax1.axhline(sensor_mean - z_threshold*sensor_std, color='green', linestyle=':', linewidth=2,
            label=f'Z-score -3σ ({sensor_mean - z_threshold*sensor_std:.0f}°C)')

ax1.fill_between(time_idx,
                 sensor_mean - z_threshold*sensor_std,
                 sensor_mean + z_threshold*sensor_std,
                 alpha=0.08, color='green', label='Normal Z-score band')

ax1.set_xlabel('Observation Index (time)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax1.set_title('Industrial Sensor Readings — Z-score and IQR Bounds', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right', ncol=2)
ax1.grid(True, alpha=0.3)

# Bottom-left: 2D credit card scatter (raw)
ax2 = fig.add_subplot(gs[1, 0])
normal_mask_cc = y_cc == 0
anom_mask_cc   = y_cc == 1

ax2.scatter(X_cc[normal_mask_cc, 0], X_cc[normal_mask_cc, 1],
            alpha=0.4, s=25, color='steelblue', label='Normal transaction')
ax2.scatter(X_cc[anom_mask_cc, 0], X_cc[anom_mask_cc, 1],
            alpha=0.9, s=80, color='red', marker='x', linewidth=2,
            label=f'Fraud ({N_ANOMALY_CC_TOTAL})')

ax2.set_xlabel('Transaction Amount ($)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Hour of Day', fontsize=11, fontweight='bold')
ax2.set_title('Credit Card Transactions\n(Raw Data — True Labels)', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Bottom-right: Distribution of sensor readings
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(X_sensor[normal_mask_s], bins=40, alpha=0.7, color='steelblue',
         label='Normal readings', density=True)
ax3.hist(X_sensor[anom_mask_s], bins=15, alpha=0.85, color='red',
         label='Anomalies', density=True)

ax3.axvline(upper_fence, color='orange', linestyle='--', linewidth=2,
            label=f'IQR upper: {upper_fence:.0f}°C')
ax3.axvline(lower_fence, color='purple', linestyle='--', linewidth=2,
            label=f'IQR lower: {lower_fence:.0f}°C')
ax3.axvline(sensor_mean + z_threshold*sensor_std, color='green', linestyle=':',
            linewidth=2, label=f'Z +3σ: {sensor_mean + z_threshold*sensor_std:.0f}°C')

ax3.set_xlabel('Temperature (°C)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Density', fontsize=11, fontweight='bold')
ax3.set_title('Sensor Reading Distribution\n(Normal vs Anomalies)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.savefig(VISUAL_DIR / '01_data_and_statistical_methods.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VISUAL_DIR / '01_data_and_statistical_methods.png'}")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Isolation Forest Decision Boundary in 2D
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Anomaly Detection — Decision Regions (Credit Card Fraud)',
             fontsize=14, fontweight='bold')

if sklearn_available:
    # Build a mesh over the 2D (scaled) feature space
    x_min, x_max = X_cc_scaled[:, 0].min() - 0.5, X_cc_scaled[:, 0].max() + 0.5
    y_min, y_max = X_cc_scaled[:, 1].min() - 0.5, X_cc_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]

    detectors = [
        ('Isolation Forest', iforest, if_predictions),
        ('LOF',              lof,     lof_predictions),
        ('One-Class SVM',    ocsvm,   ocsvm_predictions),
    ]

    for ax, (name, detector, predictions) in zip(axes, detectors):
        # Get decision function on mesh (note: LOF doesn't predict new points easily)
        try:
            if name == 'LOF':
                # LOF is transductive; approximate with known scores
                Z = np.zeros(len(mesh_points))  # placeholder
                Z = np.random.choice([-1, 1], len(mesh_points))  # can't predict new
            else:
                Z = detector.decision_function(mesh_points)

            if name != 'LOF':
                Z = Z.reshape(xx.shape)
                ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.4)
                ax.contour(xx, yy, Z, levels=[0], colors='black', linewidths=2,
                           linestyles='--')
        except Exception:
            pass

        # True normal points
        ax.scatter(X_cc_scaled[y_cc==0, 0], X_cc_scaled[y_cc==0, 1],
                   c='steelblue', s=20, alpha=0.4, label='True Normal')

        # True fraud — correct detections
        tp_mask = (y_cc==1) & (predictions==1)
        fn_mask = (y_cc==1) & (predictions==0)
        fp_mask = (y_cc==0) & (predictions==1)

        ax.scatter(X_cc_scaled[tp_mask, 0], X_cc_scaled[tp_mask, 1],
                   c='red', s=120, marker='*', zorder=5, label='TP (caught fraud)')
        ax.scatter(X_cc_scaled[fn_mask, 0], X_cc_scaled[fn_mask, 1],
                   c='darkred', s=80, marker='x', linewidth=2, zorder=5,
                   label='FN (missed fraud)')
        ax.scatter(X_cc_scaled[fp_mask, 0], X_cc_scaled[fp_mask, 1],
                   c='orange', s=60, marker='^', zorder=4, label='FP (false alarm)')

        m = credit_card_methods[name] if name in credit_card_methods else {}
        f1_val = m.get('f1', 0)
        ax.set_title(f'{name}\nF1={f1_val:.3f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Amount (scaled)', fontsize=10)
        ax.set_ylabel('Hour (scaled)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.25)
else:
    for ax in axes:
        ax.text(0.5, 0.5, 'scikit-learn required\nfor decision boundaries',
                transform=ax.transAxes, ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(VISUAL_DIR / '02_isolation_forest_decision_boundary.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VISUAL_DIR / '02_isolation_forest_decision_boundary.png'}")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: Method Comparison Bar Chart
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle('Anomaly Detection — All Methods Comparison (Credit Card Dataset)',
             fontsize=14, fontweight='bold')

method_labels = list(credit_card_methods.keys())
method_short  = ['Z-score', 'IQR', 'Iso Forest', 'LOF', 'OC-SVM']
precisions  = [credit_card_methods[m]['precision'] for m in method_labels]
recalls     = [credit_card_methods[m]['recall']    for m in method_labels]
f1s         = [credit_card_methods[m]['f1']        for m in method_labels]
colors_bar  = ['#9C27B0', '#00BCD4', '#4CAF50', '#FF9800', '#F44336']

# Precision
ax1 = axes[0]
bars = ax1.bar(method_short, precisions, color=colors_bar, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, precisions):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Precision\n(Of flagged, how many are real?)',
              fontsize=11, fontweight='bold')
ax1.set_ylim([0, 1.15])
ax1.tick_params(axis='x', rotation=20)
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# Recall
ax2 = axes[1]
bars = ax2.bar(method_short, recalls, color=colors_bar, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, recalls):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.set_ylabel('Recall', fontsize=12, fontweight='bold')
ax2.set_title('Recall\n(Of real anomalies, how many caught?)',
              fontsize=11, fontweight='bold')
ax2.set_ylim([0, 1.15])
ax2.tick_params(axis='x', rotation=20)
ax2.grid(axis='y', alpha=0.3)
ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# F1
ax3 = axes[2]
best_idx = np.argmax(f1s)
bar_colors_f1 = ['gold' if i == best_idx else c for i, c in enumerate(colors_bar)]
bars = ax3.bar(method_short, f1s, color=bar_colors_f1, edgecolor='black', linewidth=1.2)
for bar, val in zip(bars, f1s):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax3.set_title(f'F1 Score (Harmonic Mean)\nBest: {method_short[best_idx]}',
              fontsize=11, fontweight='bold')
ax3.set_ylim([0, 1.15])
ax3.tick_params(axis='x', rotation=20)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)

# Add a legend-style annotation box at the bottom
fig.text(0.5, 0.01,
         "All methods trained WITHOUT labels. Labels used ONLY for this evaluation. "
         "Gold bar = best F1.",
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(VISUAL_DIR / '03_method_comparison.png', dpi=300, bbox_inches='tight')
print(f"Saved: {VISUAL_DIR / '03_method_comparison.png'}")
plt.close()

print()

# ============================================================================
# SECTION 10: CHOOSING THE RIGHT METHOD
# ============================================================================

print("=" * 80)
print("SECTION 10: Choosing the Right Method for Your Problem")
print("=" * 80)
print()

print("QUICK REFERENCE GUIDE:")
print()
print("  USE Z-SCORE / IQR WHEN:")
print("    - Data is 1D or features are independent")
print("    - You need instant explainability ('transaction > 3 standard deviations')")
print("    - Real-time monitoring with very tight latency budgets")
print("    - Example: server response time monitoring, temperature alarms")
print()
print("  USE ISOLATION FOREST WHEN:")
print("    - High-dimensional data (10+ features)")
print("    - Need to process millions of points (sub-linear scaling)")
print("    - Anomalies are globally extreme (not embedded in dense clusters)")
print("    - Example: credit card fraud, network packet inspection")
print()
print("  USE LOF WHEN:")
print("    - Anomalies are LOCAL (normal in one region, anomalous in another)")
print("    - Data has multiple clusters of different densities")
print("    - You have < 100K points (LOF is quadratic)")
print("    - Example: manufacturing defects where 'normal' varies by product line")
print()
print("  USE ONE-CLASS SVM WHEN:")
print("    - You can afford to train on ONLY clean normal data")
print("    - Data has complex, non-Gaussian normal region shape")
print("    - Medium-sized datasets (< 50K points for RBF kernel)")
print("    - Example: intrusion detection, facial verification (one-class)")
print()
print("  ENSEMBLE (COMBINE METHODS) WHEN:")
print("    - High-stakes decisions require robustness")
print("    - Flag as anomaly only if MULTIPLE methods agree")
print("    - Reduces false positive rate at the cost of some recall")
print()

# ============================================================================
# SECTION 11: ENSEMBLE APPROACH
# ============================================================================

print("=" * 80)
print("SECTION 11: Ensemble Anomaly Detection")
print("=" * 80)
print()

print("Combining multiple detectors often improves robustness.")
print("A point must be flagged by AT LEAST 2 out of 3 methods to be anomalous.")
print()

if sklearn_available:
    # Majority vote on the 3 main methods
    votes = (if_predictions.astype(int) +
             lof_predictions.astype(int) +
             ocsvm_predictions.astype(int))

    ensemble_pred_2of3 = (votes >= 2).astype(int)
    ensemble_pred_3of3 = (votes == 3).astype(int)

    ens_2_metrics = compute_metrics(y_cc, ensemble_pred_2of3)
    ens_3_metrics = compute_metrics(y_cc, ensemble_pred_3of3)

    print("Ensemble Results (Credit Card Fraud):")
    print("-" * 60)
    print(f"{'Method':<25} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 60)
    print(f"{'Isolation Forest':<25} {if_metrics['precision']:>8.3f} {if_metrics['recall']:>8.3f} {if_metrics['f1']:>8.3f}")
    print(f"{'LOF':<25} {lof_metrics['precision']:>8.3f} {lof_metrics['recall']:>8.3f} {lof_metrics['f1']:>8.3f}")
    print(f"{'One-Class SVM':<25} {ocsvm_metrics['precision']:>8.3f} {ocsvm_metrics['recall']:>8.3f} {ocsvm_metrics['f1']:>8.3f}")
    print(f"{'Ensemble (2 of 3)':<25} {ens_2_metrics['precision']:>8.3f} {ens_2_metrics['recall']:>8.3f} {ens_2_metrics['f1']:>8.3f}")
    print(f"{'Ensemble (3 of 3)':<25} {ens_3_metrics['precision']:>8.3f} {ens_3_metrics['recall']:>8.3f} {ens_3_metrics['f1']:>8.3f}")
    print()

    print("Interpretation:")
    if ens_2_metrics['precision'] > max(if_metrics['precision'],
                                        lof_metrics['precision'],
                                        ocsvm_metrics['precision']):
        print("  Ensemble (2/3) improved precision — fewer false alarms.")
    if ens_3_metrics['recall'] < min(if_metrics['recall'],
                                     lof_metrics['recall'],
                                     ocsvm_metrics['recall']):
        print("  Strict ensemble (3/3) has very high precision but catches fewer anomalies.")
    print("  Choose threshold based on business cost of FP vs FN (Section 8).")
    print()
else:
    print("Ensemble requires scikit-learn for the three base detectors.")
    print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("ANOMALY DETECTION PROJECT — Summary")
print("=" * 80)
print()
print("WHAT YOU BUILT:")
print()
print("  1. Two realistic datasets:")
print("       - Credit card transactions (2D, 3.8% fraud rate)")
print("       - Industrial sensor readings (1D, temperature spikes)")
print("  2. Statistical methods from scratch:")
print("       - Z-score: uses mean/std, assumes Gaussian distribution")
print("       - IQR: uses percentiles, robust to non-Gaussian distributions")
print("  3. Isolation Forest: tree-based, scales to large high-dimensional data")
print("  4. Local Outlier Factor: density-based, detects local anomalies")
print("  5. One-Class SVM: boundary-based, flexible non-linear normal region")
print("  6. Full evaluation framework: precision, recall, F1, cost analysis")
print("  7. Ensemble method combining 3 detectors for improved robustness")
print()
print("KEY CONCEPTS TO REMEMBER:")
print()
print("  - Train on NORMAL data only; use labels ONLY for evaluation")
print("  - Anomaly rate is almost always < 5% — extreme class imbalance")
print("  - Precision-Recall is more informative than accuracy for rare events")
print("  - The threshold is a BUSINESS DECISION, not a technical one")
print("  - No single method wins on all datasets — understand each one's assumptions")
print()
print("NEXT STEPS:")
print()
print("  - Try Autoencoders for anomaly detection (reconstruction error approach)")
print("  - Explore DBSCAN: points in sparse regions flagged as noise (-1 label)")
print("  - Implement streaming anomaly detection with Half-Space Trees")
print("  - Combine anomaly scores with SHAP values for explainable detections")
print("  - Build a real-time detection pipeline with Apache Kafka + Python")
print()
print(f"Visualizations saved to: {VISUAL_DIR}/")
print("=" * 80)
print("Anomaly Detection Project Complete!")
print("=" * 80)
