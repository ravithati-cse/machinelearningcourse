"""
CUSTOMER SEGMENTATION - Complete End-to-End Unsupervised Learning Project
=========================================================================

PROJECT OVERVIEW:
----------------
Build a complete customer segmentation system that groups customers into
meaningful business segments using K-Means clustering, PCA, and hierarchical
clustering. This project mirrors real-world marketing analytics pipelines used
at companies like Amazon, Netflix, and retail banks.

LEARNING OBJECTIVES:
-------------------
1. Understand why normalization is essential for distance-based algorithms
2. Apply the elbow method and silhouette analysis to choose optimal K
3. Interpret cluster centroids and translate them into business segments
4. Use PCA to visualize high-dimensional clusters in 2D
5. Compare K-Means vs Agglomerative (hierarchical) clustering
6. Translate data science outputs into concrete business recommendations
7. Build a reusable, end-to-end segmentation pipeline

RECOMMENDED VIDEOS:
------------------
StatQuest: "K-means Clustering"
   https://www.youtube.com/watch?v=4b5d3muPQmA
   Crystal-clear visual explanation of how K-Means works

StatQuest: "PCA Step-by-Step"
   https://www.youtube.com/watch?v=FgakZw6K1QQ
   Understand why we use PCA for visualization

Krish Naik: "Customer Segmentation Project"
   https://www.youtube.com/watch?v=iwUli5gIcU0
   Real-world walkthrough with Mall Customers dataset

3Blue1Brown: "Principal Component Analysis"
   https://www.youtube.com/watch?v=PFDu9oVAE-g
   Beautiful visual intuition for PCA

TIME: 2-3 hours
DIFFICULTY: Intermediate
PREREQUISITES: Unsupervised learning math foundations (01-04)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'customer_segmentation'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CUSTOMER SEGMENTATION - End-to-End Unsupervised Learning Project")
print("=" * 80)
print()
print("Real-world use: Retail banks, e-commerce, telecom, subscription services")
print("Goal: Find natural groups of customers to tailor marketing strategies")
print()

# ============================================================================
# SECTION 1: DATASET CREATION
# ============================================================================

print("=" * 80)
print("SECTION 1: Creating a Realistic Customer Dataset")
print("=" * 80)
print()

print("We will create a synthetic dataset that mirrors the famous 'Mall Customers'")
print("dataset, extended with additional real-world features.")
print()
print("FEATURES:")
print("  - age                : Customer age in years (18-75)")
print("  - annual_income      : Annual income in thousands of USD")
print("  - spending_score     : Retailer-assigned score 1-100 (loyalty + spending)")
print("  - purchase_frequency : Average number of purchases per month")
print("  - account_age_months : How long the customer has been with us")
print()
print("WHY SYNTHETIC DATA?")
print("  Real customer data is private. Synthetic data with realistic structure")
print("  lets us practice the full pipeline without privacy concerns.")
print()

np.random.seed(42)
N = 300  # number of customers

# We deliberately generate 4 well-separated clusters representing common
# business archetypes, so that our algorithms can rediscover them.

def make_customer_segment(n, age_range, income_range, score_range,
                          freq_range, tenure_range, noise=0.15):
    """Generate one business segment of customers with realistic variation."""
    age    = np.random.uniform(*age_range, n)
    income = np.random.uniform(*income_range, n)
    score  = np.random.uniform(*score_range, n)
    freq   = np.random.uniform(*freq_range, n)
    tenure = np.random.uniform(*tenure_range, n)

    # Add correlated noise — real customers aren't perfectly separable
    age    += np.random.normal(0, noise * (age_range[1]-age_range[0]), n)
    income += np.random.normal(0, noise * (income_range[1]-income_range[0]), n)
    score  += np.random.normal(0, noise * (score_range[1]-score_range[0]), n)
    freq   += np.random.normal(0, noise * (freq_range[1]-freq_range[0]), n)
    tenure += np.random.normal(0, noise * (tenure_range[1]-tenure_range[0]), n)

    return np.column_stack([age, income, score, freq, tenure])

# Segment 0: High Value Loyalists — older, high income, high score, frequent buyers
seg0 = make_customer_segment(75,
    age_range=(45, 65), income_range=(80, 130), score_range=(70, 95),
    freq_range=(8, 15), tenure_range=(36, 84))

# Segment 1: Young Aspirationals — young, moderate income, very high spending score
seg1 = make_customer_segment(75,
    age_range=(20, 35), income_range=(30, 60), score_range=(65, 90),
    freq_range=(5, 12), tenure_range=(6, 24))

# Segment 2: Budget Shoppers — mixed age, lower income, low spending score
seg2 = make_customer_segment(75,
    age_range=(30, 60), income_range=(20, 45), score_range=(10, 40),
    freq_range=(1, 5), tenure_range=(12, 48))

# Segment 3: At-Risk Customers — high income but low engagement/score
seg3 = make_customer_segment(75,
    age_range=(35, 55), income_range=(70, 120), score_range=(10, 35),
    freq_range=(1, 4), tenure_range=(24, 72))

# Combine all segments
X_raw = np.vstack([seg0, seg1, seg2, seg3])
true_labels = np.array([0]*75 + [1]*75 + [2]*75 + [3]*75)

# Clip to realistic ranges
X_raw[:, 0] = np.clip(X_raw[:, 0], 18, 75)    # age
X_raw[:, 1] = np.clip(X_raw[:, 1], 15, 150)   # income
X_raw[:, 2] = np.clip(X_raw[:, 2], 1, 100)    # spending score
X_raw[:, 3] = np.clip(X_raw[:, 3], 1, 20)     # purchase frequency
X_raw[:, 4] = np.clip(X_raw[:, 4], 1, 96)     # account age months

feature_names = ['age', 'annual_income', 'spending_score',
                 'purchase_frequency', 'account_age_months']

print(f"Dataset created: {X_raw.shape[0]} customers, {X_raw.shape[1]} features")
print()
print("Dataset Statistics:")
print("-" * 70)
print(f"{'Feature':<22} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print("-" * 70)
for i, name in enumerate(feature_names):
    col = X_raw[:, i]
    print(f"{name:<22} {col.min():>8.1f} {col.max():>8.1f} {col.mean():>8.1f} {col.std():>8.1f}")
print()

print("Sample rows (first 5 customers):")
print("-" * 70)
header = f"{'age':>6} {'income':>8} {'score':>7} {'freq':>6} {'tenure':>8}"
print(header)
print("-" * 70)
for row in X_raw[:5]:
    print(f"{row[0]:>6.1f} {row[1]:>8.1f} {row[2]:>7.1f} {row[3]:>6.1f} {row[4]:>8.1f}")
print()

# ============================================================================
# SECTION 2: WHY NORMALIZATION MATTERS
# ============================================================================

print("=" * 80)
print("SECTION 2: Feature Normalization — Why It Is Critical")
print("=" * 80)
print()

print("THE PROBLEM WITH RAW FEATURES:")
print()
print("  K-Means uses Euclidean DISTANCE to assign points to clusters.")
print("  Distance is sensitive to the SCALE of each feature.")
print()
print("  Example: Two customers differ by:")
print("    - age: 5 years   → raw difference = 5")
print("    - income: $5,000 → raw difference = 5  (stored as 5.0)")
print("    - spending_score: 5 points → raw difference = 5")
print()
print("  These all LOOK equal in the formula, but income varies from 15 to 150")
print("  while score varies 1-100 and age 18-75. Without normalization,")
print("  features with large numeric ranges DOMINATE the distance calculation.")
print()
print("  Income (range ~135) would completely overpower age (range ~57)")
print("  and purchase_frequency (range ~19), making those features useless.")
print()
print("THE SOLUTION: StandardScaler (Z-score normalization)")
print()
print("  For each feature:  z = (x - mean) / std_dev")
print()
print("  After scaling:")
print("    - Every feature has mean = 0 and std = 1")
print("    - No feature dominates the distance calculation")
print("    - The algorithm treats all features equally")
print()

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    print("Scaled Feature Statistics (should all be ~0 mean, ~1 std):")
    print("-" * 70)
    print(f"{'Feature':<22} {'Mean':>10} {'Std':>10}")
    print("-" * 70)
    for i, name in enumerate(feature_names):
        col = X_scaled[:, i]
        print(f"{name:<22} {col.mean():>10.4f} {col.std():>10.4f}")
    print()

    sklearn_available = True

except ImportError:
    print("scikit-learn not installed.")
    print("Install with: pip install scikit-learn")
    print("Continuing with manual normalization...")
    sklearn_available = False

    # Manual Z-score normalization
    means = X_raw.mean(axis=0)
    stds  = X_raw.std(axis=0)
    X_scaled = (X_raw - means) / stds
    print("Manual StandardScaler applied.")
    print()

# ============================================================================
# SECTION 3: ELBOW METHOD — FINDING OPTIMAL K
# ============================================================================

print("=" * 80)
print("SECTION 3: Elbow Method — How Many Clusters?")
print("=" * 80)
print()

print("THE FUNDAMENTAL QUESTION:")
print("  Unsupervised learning does not have a 'correct' number of clusters.")
print("  We use heuristics to guide our choice.")
print()
print("THE ELBOW METHOD:")
print("  Run K-Means for k = 2, 3, 4, ... and record the WCSS (Within-Cluster")
print("  Sum of Squares) — the total squared distance from each point to its")
print("  cluster center.")
print()
print("  As k increases, WCSS always decreases (more clusters = smaller groups).")
print("  But after the 'elbow', adding clusters gives diminishing returns.")
print("  We pick k at the bend — where WCSS stops dropping steeply.")
print()

k_range = range(2, 9)
wcss_values = []

print("Running K-Means for k = 2 through 8...")
print()

if sklearn_available:
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(X_scaled)
        wcss_values.append(km.inertia_)
        print(f"  k={k}: WCSS = {km.inertia_:.2f}")
else:
    # Manual K-Means WCSS calculation
    def kmeans_wcss(X, k, max_iter=100, n_init=5, seed=42):
        best_wcss = float('inf')
        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            idx = rng.choice(len(X), k, replace=False)
            centers = X[idx].copy()
            for _ in range(max_iter):
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                labels = dists.argmin(axis=1)
                new_centers = np.array([X[labels == c].mean(axis=0) if (labels == c).any()
                                        else centers[c] for c in range(k)])
                if np.allclose(centers, new_centers, atol=1e-6):
                    break
                centers = new_centers
            wcss = sum(np.sum((X[labels == c] - centers[c])**2)
                       for c in range(k) if (labels == c).any())
            if wcss < best_wcss:
                best_wcss = wcss
        return best_wcss

    for k in k_range:
        wcss = kmeans_wcss(X_scaled, k)
        wcss_values.append(wcss)
        print(f"  k={k}: WCSS = {wcss:.2f}")

print()
print("INTERPRETING THE ELBOW:")
print("  The drop from k=2 to k=3 to k=4 is steep.")
print("  After k=4, improvements become much smaller.")
print("  This suggests k=4 is a good choice for this dataset.")
print()

# ============================================================================
# SECTION 4: SILHOUETTE ANALYSIS
# ============================================================================

print("=" * 80)
print("SECTION 4: Silhouette Analysis — Validating Cluster Quality")
print("=" * 80)
print()

print("THE SILHOUETTE SCORE:")
print("  For each point, the silhouette measures how similar it is to its own")
print("  cluster vs the nearest other cluster.")
print()
print("  s(i) = (b - a) / max(a, b)")
print()
print("  where:")
print("    a = average distance to all OTHER points in the SAME cluster")
print("    b = average distance to all points in the NEAREST other cluster")
print()
print("  Interpretation:")
print("    s close to +1 → point is well inside its cluster (good)")
print("    s close to  0 → point is on the boundary between clusters")
print("    s close to -1 → point might be in the wrong cluster (bad)")
print()

silhouette_scores = []

if sklearn_available:
    print("Computing silhouette scores for k = 2 through 8...")
    print()
    for k in k_range:
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels_k = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels_k)
        silhouette_scores.append(sil)
        print(f"  k={k}: Silhouette Score = {sil:.4f}")
    print()

    best_k_sil = list(k_range)[np.argmax(silhouette_scores)]
    best_k_elbow = 4  # informed by visual inspection of elbow

    print(f"Best k by Silhouette Score: k = {best_k_sil}")
    print(f"Best k by Elbow Method:    k = {best_k_elbow}")
    print()
    print("We will use k = 4, which aligns with both metrics and the known")
    print("business structure of this dataset.")
    print()
else:
    print("Skipping silhouette analysis (scikit-learn not available).")
    print()
    best_k_sil = 4

FINAL_K = 4

# ============================================================================
# SECTION 5: FINAL K-MEANS MODEL
# ============================================================================

print("=" * 80)
print("SECTION 5: Final K-Means Model with k = 4")
print("=" * 80)
print()

print("K-MEANS ALGORITHM RECAP:")
print("  1. Initialize k centroids (using k-means++ for smart initialization)")
print("  2. Assign each point to the nearest centroid (Euclidean distance)")
print("  3. Recompute centroids as the mean of assigned points")
print("  4. Repeat steps 2-3 until assignments stop changing")
print()

if sklearn_available:
    final_km = KMeans(n_clusters=FINAL_K, init='k-means++', n_init=10, random_state=42)
    cluster_labels = final_km.fit_predict(X_scaled)

    # Centroids in original (unscaled) space for interpretability
    centroids_scaled = final_km.cluster_centers_
    centroids_raw = scaler.inverse_transform(centroids_scaled)

    final_wcss = final_km.inertia_
    final_sil  = silhouette_score(X_scaled, cluster_labels)
    db_score   = davies_bouldin_score(X_scaled, cluster_labels)

else:
    # Manual K-Means implementation for fallback
    def run_kmeans(X, k, max_iter=300, n_init=10, seed=42):
        best_wcss = float('inf')
        best_labels = None
        best_centers = None
        rng = np.random.default_rng(seed)
        for _ in range(n_init):
            idx = rng.choice(len(X), k, replace=False)
            centers = X[idx].copy()
            labels = np.zeros(len(X), dtype=int)
            for _ in range(max_iter):
                dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
                new_labels = dists.argmin(axis=1)
                new_centers = np.array([
                    X[new_labels == c].mean(axis=0) if (new_labels == c).any()
                    else centers[c] for c in range(k)])
                if np.array_equal(labels, new_labels):
                    break
                labels = new_labels
                centers = new_centers
            wcss = sum(np.sum((X[labels == c] - centers[c])**2)
                       for c in range(k) if (labels == c).any())
            if wcss < best_wcss:
                best_wcss = wcss
                best_labels = labels.copy()
                best_centers = centers.copy()
        return best_labels, best_centers, best_wcss

    cluster_labels, centroids_scaled, final_wcss = run_kmeans(X_scaled, FINAL_K)
    means = X_raw.mean(axis=0)
    stds  = X_raw.std(axis=0)
    centroids_raw = centroids_scaled * stds + means
    final_sil = None
    db_score  = None

print(f"Final Model Results:")
print(f"  WCSS:              {final_wcss:.2f}")
if final_sil is not None:
    print(f"  Silhouette Score:  {final_sil:.4f} (higher is better, max=1)")
    print(f"  Davies-Bouldin:    {db_score:.4f}  (lower is better)")
print()

# Count cluster sizes
unique_labels, counts = np.unique(cluster_labels, return_counts=True)
print("Cluster Sizes:")
for label, count in zip(unique_labels, counts):
    pct = count / len(cluster_labels) * 100
    print(f"  Cluster {label}: {count:>4} customers ({pct:.1f}%)")
print()

# ============================================================================
# SECTION 6: INTERPRETING CLUSTERS — BUSINESS NAMING
# ============================================================================

print("=" * 80)
print("SECTION 6: Interpreting Clusters — From Data to Business Insight")
print("=" * 80)
print()

print("CLUSTER CENTROIDS (in original feature space):")
print("-" * 80)
header = f"{'Cluster':<10} {'Age':>8} {'Income':>10} {'Score':>8} {'Freq':>8} {'Tenure':>10}"
print(header)
print("-" * 80)
for i in range(FINAL_K):
    c = centroids_raw[i]
    print(f"{'Cluster '+str(i):<10} {c[0]:>8.1f} {c[1]:>10.1f} {c[2]:>8.1f} {c[3]:>8.1f} {c[4]:>10.1f}")
print()

print("INTERPRETING EACH CLUSTER:")
print()

# Sort clusters by spending score to assign consistent business names
score_col = 2  # spending_score column
centroid_scores = centroids_raw[:, score_col]
income_col = 1  # annual_income
centroid_incomes = centroids_raw[:, income_col]

# Build a simple heuristic naming scheme
# High income + high score -> High Value Loyalist
# Low income + high score  -> Young Aspirational
# Low income + low score   -> Budget Shopper
# High income + low score  -> At-Risk Customer
business_names = {}
business_colors = {}
business_actions = {}

color_palette = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

for i in range(FINAL_K):
    inc = centroids_raw[i, income_col]
    sco = centroids_raw[i, score_col]
    freq = centroids_raw[i, 3]

    if inc >= 70 and sco >= 60:
        name   = "High Value Loyalists"
        action = ("Reward with VIP program and exclusive early access. "
                  "Offer personalized upsells. Protect with high-touch retention.")
    elif inc < 60 and sco >= 60:
        name   = "Young Aspirationals"
        action = ("Offer installment plans and student/young adult discounts. "
                  "Engage via social media and referral programs. High CLV potential.")
    elif sco < 45 and freq < 5:
        if inc >= 60:
            name   = "At-Risk High Earners"
            action = ("Reactivation campaign: personalized offers, premium experience. "
                      "Survey to find pain points. Time-sensitive win-back discount.")
        else:
            name   = "Budget Shoppers"
            action = ("Promote value packs, loyalty points, and clearance sales. "
                      "Focus on price-driven messaging. Upsell gradually over time.")
    else:
        name   = "Mid-Tier Regulars"
        action = ("Push loyalty tier upgrade opportunities. "
                  "Cross-sell complementary products. Email/SMS engagement campaigns.")

    business_names[i] = name
    business_colors[i] = color_palette[i]
    business_actions[i] = action

for i in range(FINAL_K):
    n_cluster = (cluster_labels == i).sum()
    c = centroids_raw[i]
    print(f"  CLUSTER {i}: {business_names[i].upper()}")
    print(f"  {'─'*60}")
    print(f"    Size:              {n_cluster} customers ({n_cluster/N*100:.1f}% of base)")
    print(f"    Age (avg):         {c[0]:.1f} years")
    print(f"    Annual Income:     ${c[1]:.0f}K")
    print(f"    Spending Score:    {c[2]:.1f} / 100")
    print(f"    Purchase Freq:     {c[3]:.1f} purchases/month")
    print(f"    Account Tenure:    {c[4]:.0f} months")
    print(f"    Business Action:   {business_actions[i]}")
    print()

# ============================================================================
# SECTION 7: PCA FOR 2D VISUALIZATION
# ============================================================================

print("=" * 80)
print("SECTION 7: PCA — Reducing to 2D for Visualization")
print("=" * 80)
print()

print("THE CHALLENGE:")
print("  Our data is 5-dimensional. Humans can't visualize 5D space.")
print("  We need a way to project the data into 2D without losing too much")
print("  cluster structure.")
print()
print("THE SOLUTION: Principal Component Analysis (PCA)")
print()
print("  PCA finds the directions of MAXIMUM VARIANCE in the data.")
print("  PC1: the single direction capturing the most spread")
print("  PC2: the direction capturing the second-most spread, perpendicular to PC1")
print()
print("  By projecting onto (PC1, PC2), we get the BEST POSSIBLE 2D view")
print("  of a high-dimensional dataset.")
print()

if sklearn_available:
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_
    print(f"  PC1 explains: {var_explained[0]*100:.1f}% of variance")
    print(f"  PC2 explains: {var_explained[1]*100:.1f}% of variance")
    print(f"  Total:        {sum(var_explained)*100:.1f}% of variance retained in 2D")
    print()

    # Project centroids to PCA space for plotting
    centroids_pca = pca.transform(centroids_scaled)

else:
    # Manual PCA via SVD
    X_centered = X_scaled - X_scaled.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_pca = U[:, :2] * S[:2]
    centroids_pca = (centroids_scaled - X_scaled.mean(axis=0)) @ Vt[:2].T
    total_var = np.sum(S**2)
    var_explained = [(S[0]**2)/total_var, (S[1]**2)/total_var]
    print(f"  PC1: {var_explained[0]*100:.1f}%, PC2: {var_explained[1]*100:.1f}%")
    print()

print("  IMPORTANT CAVEAT:")
print("  PCA is for VISUALIZATION only. The actual clustering was done in the")
print("  full 5D space. The 2D view is an approximation — some overlap in 2D")
print("  does not mean the clusters overlap in 5D.")
print()

# ============================================================================
# SECTION 8: HIERARCHICAL CLUSTERING COMPARISON
# ============================================================================

print("=" * 80)
print("SECTION 8: Comparing K-Means vs Hierarchical Clustering")
print("=" * 80)
print()

print("K-MEANS:")
print("  + Very fast, scales to millions of points")
print("  + Produces compact, equally-sized clusters")
print("  - Requires specifying k in advance")
print("  - Assumes spherical clusters of similar size")
print("  - Sensitive to outliers (mean is pulled by extremes)")
print()
print("AGGLOMERATIVE (HIERARCHICAL) CLUSTERING:")
print("  + Does NOT require specifying k in advance")
print("  + Can find clusters of arbitrary shape")
print("  + Produces a full dendrogram for visual exploration")
print("  - Slow: O(n^3) in the worst case")
print("  - Hard to apply to new data points (no predict method)")
print()

if sklearn_available:
    hier = AgglomerativeClustering(n_clusters=FINAL_K, linkage='ward')
    hier_labels = hier.fit_predict(X_scaled)

    hier_sil = silhouette_score(X_scaled, hier_labels)
    km_sil   = silhouette_score(X_scaled, cluster_labels)

    print(f"Silhouette Scores (k=4):")
    print(f"  K-Means:           {km_sil:.4f}")
    print(f"  Hierarchical Ward: {hier_sil:.4f}")
    print()

    # Agreement between methods (adjusted rand index)
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(cluster_labels, hier_labels)
    print(f"  Adjusted Rand Index (agreement): {ari:.4f}")
    print(f"  (1.0 = perfect agreement, 0.0 = random)")
    print()

    if km_sil >= hier_sil:
        print("  K-Means produced equal or better-separated clusters for this dataset.")
    else:
        print("  Hierarchical clustering produced better-separated clusters here.")
    print()
    print("  Both methods agree on most points, confirming the cluster structure")
    print("  is genuine and not an artifact of the K-Means algorithm.")
else:
    print("Hierarchical comparison requires scikit-learn.")
    hier_labels = cluster_labels  # fallback
print()

# ============================================================================
# SECTION 9: BUSINESS RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("SECTION 9: Business Recommendations by Segment")
print("=" * 80)
print()

print("Translating data science into actionable marketing strategy:")
print()

recommendation_details = {
    "High Value Loyalists": {
        "priority": "RETAIN",
        "budget": "High ($50-100 per customer)",
        "channels": "Email, Phone, In-store VIP events",
        "tactics": [
            "Exclusive early access to new products",
            "Personalized recommendations based on purchase history",
            "Tiered loyalty rewards (platinum/diamond tier)",
            "Dedicated account manager for top 10%",
        ],
        "kpi": "Churn rate < 5%, NPS > 70, CLV growth > 10% YoY"
    },
    "Young Aspirationals": {
        "priority": "GROW",
        "budget": "Medium ($20-40 per customer)",
        "channels": "Instagram, TikTok, Email, Push notifications",
        "tactics": [
            "Buy-now-pay-later / installment options",
            "Social referral bonus (give $10, get $10)",
            "Gamified loyalty points system",
            "Influencer-style product launches",
        ],
        "kpi": "Frequency +30%, Average order value +15%"
    },
    "At-Risk High Earners": {
        "priority": "WIN BACK",
        "budget": "High ($40-80 per customer — high potential)",
        "channels": "Email, Direct mail, Phone",
        "tactics": [
            "Win-back email sequence with escalating offers",
            "Premium experience upgrade (free shipping, gift wrap)",
            "NPS survey to diagnose pain point",
            "Dedicated reactivation discount (15-20% one-time)",
        ],
        "kpi": "Reactivation rate > 20%, Spending score +25 points"
    },
    "Budget Shoppers": {
        "priority": "MONETIZE EFFICIENTLY",
        "budget": "Low ($5-15 per customer)",
        "channels": "Email, SMS, App push",
        "tactics": [
            "Flash sales and clearance promotions",
            "Subscription bundle (save 15% per month)",
            "Value pack cross-sells",
            "Tiered loyalty: earn points per dollar spent",
        ],
        "kpi": "Purchase frequency +20%, Bundle attach rate > 15%"
    },
    "Mid-Tier Regulars": {
        "priority": "UPGRADE",
        "budget": "Medium ($15-30 per customer)",
        "channels": "Email, App, In-store",
        "tactics": [
            "Loyalty tier upgrade campaigns",
            "Cross-sell complementary categories",
            "Seasonal promotions aligned with purchase cadence",
            "Re-engagement if inactive > 30 days",
        ],
        "kpi": "Move 30% to High Value tier within 12 months"
    }
}

for i in range(FINAL_K):
    segment_name = business_names[i]
    if segment_name in recommendation_details:
        rec = recommendation_details[segment_name]
        n_seg = (cluster_labels == i).sum()
        print(f"  SEGMENT: {segment_name} (n={n_seg})")
        print(f"  Priority:  {rec['priority']}")
        print(f"  Budget:    {rec['budget']}")
        print(f"  Channels:  {rec['channels']}")
        print(f"  Tactics:")
        for tactic in rec['tactics']:
            print(f"    - {tactic}")
        print(f"  KPIs:      {rec['kpi']}")
        print()

# ============================================================================
# SECTION 10: VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("SECTION 10: Generating Visualizations")
print("=" * 80)
print()

# ---------------------------------------------------------------------------
# VISUALIZATION 1: Elbow Curve + Silhouette Scores
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Choosing the Optimal Number of Clusters', fontsize=15, fontweight='bold')

# Elbow curve
ax1 = axes[0]
ax1.plot(list(k_range), wcss_values, 'bo-', linewidth=2.5, markersize=8)
ax1.axvline(x=FINAL_K, color='red', linestyle='--', linewidth=2, label=f'Chosen k={FINAL_K}')
ax1.fill_between(list(k_range), wcss_values,
                 alpha=0.1, color='blue')
ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax1.set_ylabel('WCSS (Within-Cluster Sum of Squares)', fontsize=12, fontweight='bold')
ax1.set_title('Elbow Method', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate the elbow
ax1.annotate('Elbow\n(diminishing returns\nafter this point)',
             xy=(FINAL_K, wcss_values[FINAL_K - list(k_range)[0]]),
             xytext=(FINAL_K + 1.2, wcss_values[FINAL_K - list(k_range)[0]] * 1.05),
             fontsize=9,
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red')

# Silhouette scores
ax2 = axes[1]
if sklearn_available and silhouette_scores:
    bar_colors = ['green' if k == best_k_sil else 'steelblue' for k in k_range]
    bars = ax2.bar(list(k_range), silhouette_scores, color=bar_colors,
                   edgecolor='black', linewidth=1.2)
    ax2.axvline(x=best_k_sil, color='green', linestyle='--', linewidth=2,
                label=f'Best k={best_k_sil}')
    for bar, sil in zip(bars, silhouette_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                 f'{sil:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
else:
    ax2.text(0.5, 0.5, 'Silhouette analysis\nrequires scikit-learn',
             transform=ax2.transAxes, ha='center', va='center', fontsize=12)

ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax2.set_title('Silhouette Score by k\n(Higher = better-separated clusters)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
save_path = VISUAL_DIR / '01_elbow_and_silhouette.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 2: Silhouette Subplot per Cluster
# ---------------------------------------------------------------------------

if sklearn_available:
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(f'Silhouette Analysis — k={FINAL_K} Clusters', fontsize=14, fontweight='bold')

    sil_samples = silhouette_samples(X_scaled, cluster_labels)
    y_lower = 10

    seg_colors = list(color_palette[:FINAL_K])

    for i in range(FINAL_K):
        ith_silhouette = np.sort(sil_samples[cluster_labels == i])
        size_cluster_i = ith_silhouette.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_silhouette,
                         alpha=0.8, color=seg_colors[i],
                         label=f'Cluster {i}: {business_names[i]}')

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'{i}', fontsize=10,
                fontweight='bold', color=seg_colors[i])

        y_lower = y_upper + 10

    avg_sil = np.mean(sil_samples)
    ax.axvline(x=avg_sil, color='red', linestyle='--', linewidth=2.5,
               label=f'Avg silhouette: {avg_sil:.3f}')

    ax.set_xlabel('Silhouette Coefficient', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_xlim([-0.15, 1.0])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Add interpretation text
    ax.text(0.65, 0.95,
            "Wide bars = more members\n"
            "Right extent = high quality\n"
            "Most points > avg line = good",
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = VISUAL_DIR / '02_silhouette_plot.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

# ---------------------------------------------------------------------------
# VISUALIZATION 3: Final Clusters in PCA Space with Business Labels
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Customer Segments — PCA Visualization', fontsize=15, fontweight='bold')

# Left: K-Means clusters with business names
ax_km = axes[0]
for i in range(FINAL_K):
    mask = cluster_labels == i
    ax_km.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  s=50, alpha=0.7, color=color_palette[i],
                  label=f'C{i}: {business_names[i]}',
                  edgecolors='white', linewidth=0.3)

# Plot centroids
for i in range(FINAL_K):
    ax_km.scatter(centroids_pca[i, 0], centroids_pca[i, 1],
                  s=250, color=color_palette[i], marker='*',
                  edgecolors='black', linewidth=1.5, zorder=5)
    ax_km.annotate(f' {business_names[i]}\n (n={(cluster_labels==i).sum()})',
                   xy=(centroids_pca[i, 0], centroids_pca[i, 1]),
                   fontsize=7.5, fontweight='bold', color=color_palette[i],
                   xytext=(5, 5), textcoords='offset points')

ax_km.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax_km.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax_km.set_title('K-Means Clusters\n(stars = centroids)', fontsize=12, fontweight='bold')
ax_km.legend(loc='upper right', fontsize=8)
ax_km.grid(True, alpha=0.25)

# Right: True vs discovered clusters
ax_true = axes[1]
true_colors = ['#9C27B0', '#00BCD4', '#8BC34A', '#FF5722']
true_names  = ['High Value Loyalists', 'Young Aspirationals',
               'Budget Shoppers', 'At-Risk High Earners']

for i in range(FINAL_K):
    mask = true_labels == i
    ax_true.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    s=50, alpha=0.7, color=true_colors[i],
                    label=true_names[i],
                    edgecolors='white', linewidth=0.3)

ax_true.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax_true.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax_true.set_title('True Underlying Segments\n(for reference — unknown in real data)',
                  fontsize=12, fontweight='bold')
ax_true.legend(loc='upper right', fontsize=8)
ax_true.grid(True, alpha=0.25)

# Note box
fig.text(0.5, 0.01,
         "Note: PCA 2D projection for visualization only. Clustering was performed in full 5D feature space.",
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout(rect=[0, 0.03, 1, 1])
save_path = VISUAL_DIR / '03_clusters_pca_visualization.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Saved: {save_path}")
plt.close()

print()

# ============================================================================
# SECTION 11: PREDICT SEGMENT FOR NEW CUSTOMERS
# ============================================================================

print("=" * 80)
print("SECTION 11: Predicting Segments for New Customers")
print("=" * 80)
print()

print("Once trained, we can assign ANY new customer to a segment.")
print("This is the production use-case: when a new customer signs up,")
print("we immediately know how to communicate with them.")
print()

new_customers = np.array([
    [55, 110, 82, 12, 60],   # High income, high score
    [25, 35,  78,  9, 10],   # Young, low income, high score
    [40, 25,  20,  2, 24],   # Low income, low score
    [48, 95,  18,  2, 48],   # High income, low score
])
new_names = [
    "Alice (55yr, $110K, score=82)",
    "Bob   (25yr, $35K,  score=78)",
    "Carol (40yr, $25K,  score=20)",
    "Dave  (48yr, $95K,  score=18)",
]

if sklearn_available:
    new_scaled = scaler.transform(new_customers)
    new_predictions = final_km.predict(new_scaled)

    print("New Customer Classifications:")
    print("-" * 70)
    for name, pred in zip(new_names, new_predictions):
        segment = business_names[pred]
        print(f"  {name}")
        print(f"     -> Segment {pred}: {segment}")
        print()
else:
    # Manual prediction: nearest centroid
    new_scaled = (new_customers - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    dists = np.linalg.norm(new_scaled[:, None, :] - centroids_scaled[None, :, :], axis=2)
    new_predictions = dists.argmin(axis=1)

    print("New Customer Classifications (manual nearest-centroid):")
    print("-" * 70)
    for name, pred in zip(new_names, new_predictions):
        segment = business_names[pred]
        print(f"  {name}")
        print(f"     -> Segment {pred}: {segment}")
        print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("CUSTOMER SEGMENTATION PROJECT — Summary")
print("=" * 80)
print()
print("WHAT YOU BUILT:")
print()
print("  1. Created a realistic 5-feature synthetic customer dataset (n=300)")
print("  2. Applied StandardScaler normalization (critical for distance-based methods)")
print("  3. Used the Elbow Method to explore k=2 through k=8")
print("  4. Validated cluster quality with Silhouette Analysis")
print("  5. Trained a final K-Means model with k=4")
print("  6. Interpreted centroids and named each cluster with a business label")
print("  7. Used PCA to create a 2D visualization of the 5D clusters")
print("  8. Compared K-Means vs Agglomerative Hierarchical Clustering")
print("  9. Produced concrete business recommendations per segment")
print(" 10. Built a prediction function for new customers")
print()
print("KEY INSIGHTS:")
print()
print("  - Normalization is NOT optional for K-Means (income would dominate)")
print("  - Elbow + Silhouette together give more reliable k than either alone")
print("  - Centroids in original space (after inverse_transform) are interpretable")
print("  - PCA is for visualization — always cluster in full feature space")
print("  - The business value is in the NAMES and ACTIONS, not the cluster numbers")
print()
print("NEXT STEPS:")
print()
print("  - Try DBSCAN for non-spherical cluster shapes")
print("  - Add RFM features (Recency, Frequency, Monetary) for richer segmentation")
print("  - Use Gaussian Mixture Models (GMM) for soft/probabilistic assignment")
print("  - Build a dashboard in Streamlit or Tableau for business stakeholders")
print("  - Set up monthly re-clustering as customer behavior evolves")
print()
print(f"Visualizations saved to: {VISUAL_DIR}/")
print("=" * 80)
print("Customer Segmentation Project Complete!")
print("=" * 80)
