"""
🎲 INFORMATION THEORY BASICS - Entropy, Surprise, and Cluster Quality

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. Entropy: the mathematical measure of uncertainty or "surprise" in a distribution
2. High vs low entropy distributions and what they tell you about data
3. How entropy connects to information content (bits)
4. Joint entropy and mutual information: how much two variables share
5. How entropy helps evaluate clustering quality and detect anomalies
6. The silhouette score: an intuitive preview of how we measure cluster tightness
7. KL divergence: measuring how different two distributions are

================================================================================
RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
ABSOLUTE MUST WATCH:
   - 3Blue1Brown: "Entropy in Compression"
     https://www.youtube.com/watch?v=zHbfceWe-qY
     (Brilliant visual treatment of entropy and information)

   - StatQuest: "Entropy (for Decision Trees)"
     https://www.youtube.com/watch?v=YtebGVx-Fxw
     (Clear, concrete examples with categories)

Also Recommended:
   - Aurélien Géron: "Information Theory Basics"
     https://www.youtube.com/watch?v=ErfnhcEV1O8
     (Graduate-level but excellent intuition building)

================================================================================
OVERVIEW
================================================================================
Why Does a ML Course Need Information Theory?

Unsupervised learning = finding STRUCTURE in data without labels.
But how do we know if we found GOOD structure?

Information theory gives us the tools:
  * Entropy tells us how "mixed up" (uncertain) a set of labels is
  * Low entropy after clustering -> clusters are PURE, well-separated
  * High entropy -> clusters are mixed -> bad clustering
  * KL divergence -> how different is one distribution from another?
  * Mutual information -> do two features carry the same information?

These concepts also appear in:
  * Decision trees (split on lowest entropy)
  * VAEs and generative models (KL divergence in the loss)
  * Feature selection (mutual information)
  * Anomaly detection (points with surprising, high-entropy neighborhoods)

This is the mathematical language of "information" and "surprise".
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/04_information_theory/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("INFORMATION THEORY BASICS")
print("   Entropy, Surprise, and Measuring Cluster Quality")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: INTUITION — WHAT IS INFORMATION?
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Intuition — What Is 'Information'?")
print("=" * 80)
print()

print("THE KEY INSIGHT:")
print("-" * 70)
print("  'Information' in the mathematical sense is about SURPRISE.")
print()
print("  SCENARIO A: You flip a fair coin. Result: Heads.")
print("    -> You were NOT sure (50/50). This is INFORMATIVE! (1 bit of info)")
print()
print("  SCENARIO B: You flip a two-headed coin. Result: Heads.")
print("    -> You KNEW it would be heads. This is NOT informative. (0 bits of info)")
print()
print("  SCENARIO C: Tomorrow the sun rises. Result: It rises.")
print("    -> You were 100% sure. Zero information.")
print()
print("FORMAL DEFINITION: Self-information (surprise) of event x:")
print("-" * 70)
print("  I(x) = -log2(P(x))   [measured in bits]")
print()
print("  P(x) = 1.0 -> I(x) = -log2(1.0) = 0 bits  (certain, no surprise)")
print("  P(x) = 0.5 -> I(x) = -log2(0.5) = 1 bit   (fair coin flip)")
print("  P(x) = 0.25-> I(x) = -log2(0.25)= 2 bits  (1 of 4 equally likely)")
print("  P(x) = 0.01-> I(x) = -log2(0.01)= 6.6 bits (very rare = very surprising!)")
print()

# Calculate and display self-information
probs = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001]
print(f"  {'P(x)':<12} {'I(x) bits':<15} {'Surprise level'}")
print(f"  {'-'*11} {'-'*14} {'-'*20}")
for p in probs:
    info = -np.log2(p)
    if info < 0.5:
        level = "No surprise (boring!)"
    elif info < 1.5:
        level = "Low surprise"
    elif info < 3.0:
        level = "Moderate surprise"
    elif info < 5.0:
        level = "High surprise"
    else:
        level = "Very surprising (rare event!)"
    print(f"  {p:<12.3f} {info:<15.4f} {level}")
print()

# ============================================================================
# SECTION 2: ENTROPY — AVERAGE SURPRISE OF A DISTRIBUTION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Shannon Entropy — The Average Surprise")
print("=" * 80)
print()

print("ENTROPY = Expected (average) information content of a distribution")
print()
print("FORMULA:")
print("-" * 70)
print("  H(X) = -sum( P(xi) * log2(P(xi)) )")
print()
print("  For each possible outcome xi:")
print("    * P(xi) is its probability")
print("    * log2(P(xi)) is its self-information (negative because log is negative for P<1)")
print("    * We weight by P(xi) to get the AVERAGE")
print()
print("  Convention: 0 * log(0) = 0 (limit, since P -> 0 faster than log -> -inf)")
print()
print("  Units: bits (when using log base 2) or nats (when using natural log)")
print()

def entropy_scratch(probs, base=2):
    """
    Compute Shannon entropy from a probability distribution.
    probs: array of probabilities (must sum to 1)
    base: log base (2 = bits, e = nats, 10 = dits)
    Returns entropy in the specified units.
    """
    probs = np.array(probs, dtype=float)
    probs = probs[probs > 0]  # Remove zeros (0*log(0) = 0 by convention)
    if base == 2:
        return -np.sum(probs * np.log2(probs))
    elif base == np.e:
        return -np.sum(probs * np.log(probs))
    else:
        return -np.sum(probs * np.log(probs)) / np.log(base)

def normalize(counts):
    """Convert counts to probabilities."""
    counts = np.array(counts, dtype=float)
    return counts / counts.sum()

# Show entropy of different distributions
print("EXAMPLE: Rolling a die — different distributions")
print("-" * 70)
print()

distributions = {
    "Fair die (6 sides)":           [1/6]*6,
    "Loaded die (6 always wins)":   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "Slightly biased (6 favored)":  [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],
    "Two outcomes equally likely":  [0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    "Uniform over 2 outcomes":      [0.5, 0.5],
}

for name, dist in distributions.items():
    dist_arr = np.array(dist)
    H = entropy_scratch(dist_arr)
    max_H = np.log2(len(dist_arr[dist_arr > 0]))  # max entropy for this many outcomes
    print(f"  {name}:")
    print(f"    Distribution: {[f'{p:.2f}' for p in dist_arr]}")
    print(f"    Entropy H = {H:.4f} bits  (max possible = {max_H:.4f} bits)")
    pct = (H / max_H * 100) if max_H > 0 else 0
    bar_len = int(H / max_H * 30) if max_H > 0 else 0
    print(f"    {'[' + '#'*bar_len + ' '*(30-bar_len) + ']'} {pct:.0f}% of max entropy")
    print()

print("KEY INSIGHT:")
print("-" * 70)
print("  MAXIMUM ENTROPY: Uniform distribution (all outcomes equally likely)")
print("    -> Most uncertain, most informative, highest disorder")
print()
print("  MINIMUM ENTROPY (= 0): Certain outcome (one probability = 1)")
print("    -> No uncertainty, no information, no disorder")
print()
print("  Entropy measures UNCERTAINTY or DISORDER in a distribution.")
print()

# Numerical property: max entropy for N outcomes
print("MAX ENTROPY SCALES WITH NUMBER OF OUTCOMES:")
print("-" * 70)
print(f"  {'Outcomes (N)':<15} {'Max Entropy (bits)':<22} {'= log2(N)'}")
print(f"  {'-'*14} {'-'*21} {'-'*10}")
for n in [2, 4, 8, 16, 64, 256, 1024]:
    max_h = np.log2(n)
    print(f"  {n:<15} {max_h:<22.4f} = log2({n})")
print()

# ============================================================================
# SECTION 3: ENTROPY IN CLUSTERING — EVALUATING CLUSTER PURITY
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Entropy for Clustering — Are Our Clusters Pure?")
print("=" * 80)
print()

print("THE CLUSTERING CHALLENGE:")
print("-" * 70)
print("  After running K-Means or another algorithm, we get cluster assignments.")
print("  But how do we know if the clusters are GOOD?")
print()
print("  If we have ground-truth labels (even for evaluation only), we can ask:")
print("  'Within each cluster, how MIXED are the true labels?'")
print()
print("  Low entropy cluster  = one class dominates -> GOOD cluster")
print("  High entropy cluster = all classes mixed up -> BAD cluster")
print()

def cluster_entropy(cluster_labels):
    """
    Compute entropy of class labels within a cluster.
    cluster_labels: array of class labels for points in one cluster.
    Returns entropy in bits.
    """
    unique, counts = np.unique(cluster_labels, return_counts=True)
    probs = counts / counts.sum()
    return entropy_scratch(probs)

# Simulate three clustering scenarios
np.random.seed(42)
n_pts = 100

# Perfect clustering: each cluster has one class
perfect_cluster1 = ['cat'] * 50
perfect_cluster2 = ['dog'] * 50

# Good clustering: one class dominates
good_cluster1 = ['cat'] * 45 + ['dog'] * 5
good_cluster2 = ['dog'] * 42 + ['cat'] * 8

# Bad clustering: classes are mixed
bad_cluster1 = ['cat'] * 25 + ['dog'] * 25
bad_cluster2 = ['cat'] * 25 + ['dog'] * 25

print("CLUSTER PURITY EXAMPLES:")
print("-" * 70)
print()
scenarios = [
    ("PERFECT clustering",
     [perfect_cluster1, perfect_cluster2], "Entropy = 0, perfect separation"),
    ("GOOD clustering",
     [good_cluster1, good_cluster2],       "Low entropy, mostly pure"),
    ("BAD clustering",
     [bad_cluster1, bad_cluster2],         "High entropy, totally mixed"),
]

for scenario_name, clusters, verdict in scenarios:
    print(f"  {scenario_name}:")
    entropies = []
    for i, cluster in enumerate(clusters):
        H = cluster_entropy(cluster)
        entropies.append(H)
        unique, counts = np.unique(cluster, return_counts=True)
        dist_str = ", ".join([f"{u}:{c}" for u, c in zip(unique, counts)])
        print(f"    Cluster {i+1} ({dist_str}): H = {H:.4f} bits")
    avg_H = np.mean(entropies)
    print(f"    -> Average entropy: {avg_H:.4f} bits  | {verdict}")
    print()

print("WEIGHTED AVERAGE ENTROPY (proper metric):")
print("-" * 70)
print("  H_weighted = sum( |cluster_i| / |total| * H(cluster_i) )")
print("  This weights larger clusters more.")
print()

# ============================================================================
# SECTION 4: SILHOUETTE SCORE PREVIEW
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Silhouette Score — Geometric Cluster Quality (Preview)")
print("=" * 80)
print()

print("While entropy measures cluster PURITY (needs labels), the silhouette score")
print("measures cluster quality using only the DATA GEOMETRY — no labels needed!")
print()
print("SILHOUETTE SCORE for a single point i:")
print("-" * 70)
print("  a(i) = mean distance from point i to all OTHER points in the same cluster")
print("  b(i) = mean distance from point i to all points in the NEAREST other cluster")
print()
print("  silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))")
print()
print("  Range: -1 to +1")
print("   +1: Perfect — point is very close to its cluster, far from others")
print("    0: On the boundary — point could belong to either cluster")
print("   -1: Misclassified — point is closer to a different cluster!")
print()

def silhouette_single(X, labels, point_idx):
    """
    Compute silhouette score for a single point.
    X: (n, 2) data array
    labels: cluster label for each point
    point_idx: index of the point to evaluate
    Returns: silhouette score in [-1, 1]
    """
    current_label = labels[point_idx]
    # a(i): mean distance to same-cluster points
    same_cluster_mask = (labels == current_label) & (np.arange(len(labels)) != point_idx)
    if not np.any(same_cluster_mask):
        return 0.0
    same_dists = np.linalg.norm(X[same_cluster_mask] - X[point_idx], axis=1)
    a = np.mean(same_dists)
    # b(i): min mean distance to any OTHER cluster
    other_labels = np.unique(labels[labels != current_label])
    if len(other_labels) == 0:
        return 0.0
    b_values = []
    for other_lab in other_labels:
        other_mask = labels == other_lab
        other_dists = np.linalg.norm(X[other_mask] - X[point_idx], axis=1)
        b_values.append(np.mean(other_dists))
    b = np.min(b_values)
    return (b - a) / max(a, b)

# Generate two clear clusters and one ambiguous cluster
np.random.seed(7)
c1 = np.random.randn(30, 2) * 0.5 + np.array([0.0, 0.0])
c2 = np.random.randn(30, 2) * 0.5 + np.array([4.0, 4.0])
c3 = np.random.randn(20, 2) * 1.5 + np.array([2.0, 2.0])  # ambiguous

X_sil = np.vstack([c1, c2, c3])
labels_sil = np.array([0]*30 + [1]*30 + [2]*20)

# Compute silhouette for a sample of points
print("SILHOUETTE SCORES FOR SAMPLE POINTS:")
print("-" * 70)
print(f"  {'Point':<8} {'Cluster':<10} {'a(i)':<10} {'b(i)':<10} {'Sil Score':<12} {'Verdict'}")
print(f"  {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*11} {'-'*15}")
sample_indices = [0, 15, 30, 45, 60, 72]
for idx in sample_indices:
    s = silhouette_single(X_sil, labels_sil, idx)
    cl = labels_sil[idx]
    same_mask = (labels_sil == cl) & (np.arange(len(labels_sil)) != idx)
    a_val = np.mean(np.linalg.norm(X_sil[same_mask] - X_sil[idx], axis=1))
    other_labs = np.unique(labels_sil[labels_sil != cl])
    b_vals = [np.mean(np.linalg.norm(X_sil[labels_sil==ol] - X_sil[idx], axis=1)) for ol in other_labs]
    b_val = np.min(b_vals)
    verdict = "Well-placed" if s > 0.5 else ("Boundary" if s > 0 else "Misclassified?")
    print(f"  {idx:<8} {cl:<10} {a_val:<10.3f} {b_val:<10.3f} {s:<12.4f} {verdict}")

# Overall silhouette
all_sil = [silhouette_single(X_sil, labels_sil, i) for i in range(len(X_sil))]
avg_sil = np.mean(all_sil)
print(f"\n  Average silhouette score: {avg_sil:.4f}")
print()
print("SILHOUETTE SCORE GUIDE:")
print("-" * 70)
print(f"  {'Range':<20} {'Interpretation'}")
print(f"  {'-'*19} {'-'*30}")
print(f"  {'0.71 to 1.00':<20} Strong, well-separated clusters")
print(f"  {'0.51 to 0.70':<20} Reasonable structure found")
print(f"  {'0.26 to 0.50':<20} Weak structure, may be artificial")
print(f"  {'< 0.25':<20} No substantial structure found")
print()
print("  In K-Means module: We'll use this to choose the best K!")
print()

# sklearn verification
try:
    from sklearn.metrics import silhouette_score, silhouette_samples
    sk_avg = silhouette_score(X_sil, labels_sil)
    print(f"SKLEARN VERIFICATION:")
    print(f"-" * 70)
    print(f"  sklearn silhouette_score: {sk_avg:.4f}")
    print(f"  Our from-scratch:         {avg_sil:.4f}")
    print(f"  (Small diff expected — sklearn uses vectorized computation)")
    print()
except ImportError:
    print("  sklearn not installed. Run: pip install scikit-learn")

# ============================================================================
# SECTION 5: KL DIVERGENCE — HOW DIFFERENT ARE TWO DISTRIBUTIONS?
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: KL Divergence — Comparing Two Distributions")
print("=" * 80)
print()

print("MOTIVATION: We often need to measure how DIFFERENT two distributions are.")
print()
print("Examples in unsupervised learning:")
print("  * Is this new data point from the same distribution as training data?")
print("    -> If KL divergence is large, it might be an ANOMALY!")
print("  * VAE (Variational Autoencoders): loss includes KL divergence to enforce")
print("    that the learned latent space matches a Gaussian distribution")
print()

print("FORMULA (KL Divergence):")
print("-" * 70)
print("  KL(P || Q) = sum( P(x) * log2(P(x) / Q(x)) )")
print()
print("  Read as: 'KL divergence FROM Q TO P' or 'how much P differs from Q'")
print()
print("  PROPERTIES:")
print("    * KL >= 0 always (Gibbs inequality)")
print("    * KL = 0 only if P = Q (identical distributions)")
print("    * NOT symmetric: KL(P||Q) != KL(Q||P) in general!")
print("    * 'Relative entropy' — extra bits needed to encode P using Q's code")
print()

def kl_divergence(p, q, epsilon=1e-12):
    """
    KL divergence KL(P || Q).
    P: true distribution
    Q: approximate distribution
    Both must sum to 1. Adds epsilon for numerical stability.
    """
    p = np.array(p, dtype=float) + epsilon
    q = np.array(q, dtype=float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log2(p / q))

# Examples
print("EXAMPLES:")
print("-" * 70)
# 6-outcome distributions
P_uniform = np.array([1/6]*6)       # fair die
Q_loaded   = np.array([0.05, 0.05, 0.10, 0.10, 0.20, 0.50])  # loaded die
P_peaked   = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])        # always 6
P_medium   = np.array([0.1, 0.1, 0.15, 0.15, 0.25, 0.25])    # moderate bias

print(f"  P = Fair die:     {[f'{x:.2f}' for x in P_uniform]}")
print(f"  Q = Loaded die:   {[f'{x:.2f}' for x in Q_loaded]}")
print(f"  P = Always six:   {[f'{x:.2f}' for x in P_peaked]}")
print(f"  P = Medium bias:  {[f'{x:.2f}' for x in P_medium]}")
print()

pairs = [
    ("KL(Fair || Loaded)",         P_uniform, Q_loaded),
    ("KL(Loaded || Fair)",         Q_loaded, P_uniform),
    ("KL(Fair || Fair)",           P_uniform, P_uniform),
    ("KL(AlwaysSix || Fair)",      P_peaked, P_uniform),
    ("KL(MediumBias || Loaded)",   P_medium, Q_loaded),
]

for name, P, Q in pairs:
    kl = kl_divergence(P, Q)
    print(f"  {name:<40} = {kl:.4f} bits")

print()
print("OBSERVATIONS:")
print("  KL(P||P) = 0          (a distribution is 'zero bits away' from itself)")
print("  KL(Fair||Loaded) != KL(Loaded||Fair)  (not symmetric!)")
print("  KL(AlwaysSix||Fair) = large  (very different distributions)")
print()

# scipy.stats.entropy verification
try:
    from scipy.stats import entropy as scipy_entropy
    # scipy entropy(p, q) computes KL divergence
    kl_scipy = scipy_entropy(P_uniform, Q_loaded, base=2)
    kl_ours  = kl_divergence(P_uniform, Q_loaded)
    print(f"SCIPY VERIFICATION:")
    print(f"  scipy.stats.entropy(P, Q, base=2): {kl_scipy:.4f}")
    print(f"  Our kl_divergence(P, Q):           {kl_ours:.6f}")
    print(f"  Note: scipy adds epsilon automatically, tiny numerical difference")
    print()
    print("SCIPY FOR ENTROPY:")
    H_scipy = scipy_entropy(P_uniform, base=2)
    H_ours  = entropy_scratch(P_uniform)
    print(f"  scipy.stats.entropy(P, base=2) = {H_scipy:.6f}")
    print(f"  Our entropy_scratch(P)          = {H_ours:.6f}")
except ImportError:
    print("  scipy not installed. Run: pip install scipy")
    print("  (Our from-scratch implementations are correct!)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Visualizations")
print("=" * 80)
print()
print("Generating Visualization 1: Entropy of Different Distributions...")

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
fig.suptitle('INFORMATION THEORY: Entropy — Measuring Uncertainty and Disorder',
             fontsize=14, fontweight='bold', y=1.01)

# Row 1: Distributions and their entropy
categories = ['A', 'B', 'C', 'D', 'E']
dist_examples = [
    ([1.0, 0.0, 0.0, 0.0, 0.0], 'Certain\n(P=1 for A)', 'tomato'),
    ([0.7, 0.2, 0.05, 0.03, 0.02], 'Dominant\n(A likely)', 'orange'),
    ([0.4, 0.3, 0.15, 0.1, 0.05], 'Moderate\nbias', 'gold'),
    ([0.2, 0.2, 0.2, 0.2, 0.2], 'Uniform\n(max entropy)', 'seagreen'),
]

entropies_row1 = []
for ax, (dist, title, color) in zip(axes[0], dist_examples):
    H = entropy_scratch(dist)
    entropies_row1.append(H)
    ax.bar(categories, dist, color=color, edgecolor='black', alpha=0.85)
    ax.set_title(f'{title}\nH = {H:.3f} bits', fontsize=10, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    # Entropy bar
    max_H = np.log2(5)
    bar_pct = H / max_H
    ax.text(0.5, 0.95, f'Uncertainty: {bar_pct*100:.0f}%',
            transform=ax.transAxes, ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Row 2: Self-information curve, entropy vs N outcomes, cluster entropy, KL divergence
# --- Entropy as a function of probability (binary case) ---
ax = axes[1, 0]
p_vals = np.linspace(0.001, 0.999, 500)
H_binary = -(p_vals * np.log2(p_vals) + (1 - p_vals) * np.log2(1 - p_vals))
ax.plot(p_vals, H_binary, 'b-', lw=3, label='H(p) binary entropy')
ax.scatter([0.5], [1.0], s=200, c='red', zorder=5, edgecolors='black')
ax.annotate('Max entropy = 1 bit\nat p = 0.5', xy=(0.5, 1.0), xytext=(0.15, 0.85),
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='black'))
ax.scatter([0.001, 0.999], [H_binary[0], H_binary[-1]], s=100, c='green', zorder=5)
ax.set_xlabel('Probability p')
ax.set_ylabel('Entropy H(p) [bits]')
ax.set_title('Binary Entropy Function\nH(p) = -p*log2(p) - (1-p)*log2(1-p)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# --- Max entropy vs number of outcomes ---
ax = axes[1, 1]
n_outcomes = np.arange(2, 33)
max_entropies = np.log2(n_outcomes)
ax.plot(n_outcomes, max_entropies, 'go-', markersize=6, lw=2.5, label='Max H = log2(N)')
ax.fill_between(n_outcomes, max_entropies, alpha=0.15, color='green')
ax.set_xlabel('Number of Equally-Likely Outcomes (N)')
ax.set_ylabel('Maximum Entropy (bits)')
ax.set_title('Max Entropy Grows with N\nH_max = log2(N)',
             fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3)
for n_mark in [2, 4, 8, 16, 32]:
    ax.scatter([n_mark], [np.log2(n_mark)], s=80, c='darkgreen', zorder=5)
    ax.text(n_mark+0.3, np.log2(n_mark)+0.05, f'N={n_mark}\n{np.log2(n_mark):.1f}b',
            fontsize=7, color='darkgreen')
ax.legend(fontsize=9)

# --- Cluster entropy comparison ---
ax = axes[1, 2]
cluster_scenarios = {
    'Perfect\ncluster': [1.0, 0.0, 0.0],
    'Good\ncluster':    [0.8, 0.15, 0.05],
    'Mediocre\ncluster':[0.5, 0.3, 0.2],
    'Bad\ncluster':     [0.34, 0.33, 0.33],
}
sc_names = list(cluster_scenarios.keys())
sc_entropies = [entropy_scratch(v) for v in cluster_scenarios.values()]
colors_bar = ['seagreen', 'steelblue', 'orange', 'tomato']
bars = ax.bar(sc_names, sc_entropies, color=colors_bar, edgecolor='black', alpha=0.85)
ax.axhline(y=np.log2(3), color='gray', lw=2, ls='--', label=f'Max H = {np.log2(3):.2f} bits')
for bar, H in zip(bars, sc_entropies):
    ax.text(bar.get_x() + bar.get_width()/2, H + 0.03, f'{H:.2f}',
            ha='center', fontsize=9, fontweight='bold')
ax.set_title('Cluster Entropy\n(Low entropy = pure cluster = GOOD)',
             fontsize=10, fontweight='bold')
ax.set_ylabel('Entropy (bits)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# --- KL divergence: effect of diverging from uniform ---
ax = axes[1, 3]
# Vary how much one class dominates
dominant_probs = np.linspace(1/5, 0.999, 100)
kl_from_uniform = []
for dp in dominant_probs:
    rest = (1.0 - dp) / 4
    q = np.array([dp, rest, rest, rest, rest])
    kl_from_uniform.append(kl_divergence(q, P_uniform))
ax.plot(dominant_probs, kl_from_uniform, 'purple', lw=3, label='KL(biased || uniform)')
ax.axvline(x=1/5, color='green', lw=2, ls='--', label='Uniform (KL=0)')
ax.axvline(x=0.999, color='red', lw=2, ls='--', label='Almost certain (KL large)')
ax.set_xlabel('Probability of dominant category')
ax.set_ylabel('KL(biased || uniform) [bits]')
ax.set_title('KL Divergence FROM Uniform\n(0 = identical, large = very different)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_entropy_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_entropy_distributions.png")

# ============================================================================
# VISUALIZATION 2: Silhouette analysis
# ============================================================================
print("Generating Visualization 2: Silhouette Score Analysis...")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle('SILHOUETTE SCORE — Measuring Cluster Quality Without Labels',
             fontsize=13, fontweight='bold')

# Three clustering scenarios: well-separated, overlapping, and bad
np.random.seed(42)
scenarios_sil = [
    {
        'name': 'Well-Separated\nClusters',
        'data': [np.random.randn(40, 2)*0.4 + offset for offset in [[-4, 0], [0, 3], [4, 0]]],
        'colors': ['tomato', 'steelblue', 'seagreen'],
    },
    {
        'name': 'Overlapping\nClusters',
        'data': [np.random.randn(40, 2)*1.2 + offset for offset in [[-1, 0], [1, 0], [0, 1.5]]],
        'colors': ['tomato', 'steelblue', 'seagreen'],
    },
    {
        'name': 'Terrible\nClustering',
        'data': [np.random.randn(40, 2)*3.0 + offset for offset in [[0, 0], [0.5, 0.5], [1, 1]]],
        'colors': ['tomato', 'steelblue', 'seagreen'],
    },
]

for ax, scenario in zip(axes, scenarios_sil):
    X_s = np.vstack(scenario['data'])
    labs_s = np.concatenate([np.full(40, i) for i in range(3)])

    # Compute silhouette scores
    sil_scores = np.array([silhouette_single(X_s, labs_s, i) for i in range(len(X_s))])
    avg_sil_s  = np.mean(sil_scores)

    for i, (data, color) in enumerate(zip(scenario['data'], scenario['colors'])):
        mask = labs_s == i
        sc = ax.scatter(X_s[mask, 0], X_s[mask, 1], c=sil_scores[mask],
                        cmap='RdYlGn', vmin=-1, vmax=1, s=50,
                        edgecolors=color, linewidths=1.5, alpha=0.85)

    plt.colorbar(sc, ax=ax, label='Silhouette score')
    ax.set_title(f'{scenario["name"]}\nAvg Silhouette = {avg_sil_s:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.grid(True, alpha=0.3)
    qual = "GOOD" if avg_sil_s > 0.5 else ("MEDIOCRE" if avg_sil_s > 0.25 else "POOR")
    ax.text(0.5, 0.04, f'Cluster quality: {qual}', transform=ax.transAxes,
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round',
                      facecolor='lightgreen' if qual == 'GOOD' else
                                'lightyellow' if qual == 'MEDIOCRE' else 'lightcoral',
                      alpha=0.9))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_silhouette_score_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_silhouette_score_analysis.png")

# ============================================================================
# VISUALIZATION 3: Information theory summary reference card
# ============================================================================
print("Generating Visualization 3: Information Theory Reference Card...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('INFORMATION THEORY: Complete Reference Card for ML',
             fontsize=14, fontweight='bold')

# --- Panel 1: Entropy curves ---
ax = axes[0]
p = np.linspace(0.001, 0.999, 400)

# Binary entropy
H_bin = -(p * np.log2(p) + (1-p) * np.log2(1-p))
# Self-information
I_self = -np.log2(p)
ax.plot(p, H_bin, 'b-', lw=3, label='H(p,1-p): binary entropy')
ax.plot(p, np.minimum(I_self, 5), 'r--', lw=2.5, label='-log2(p): self-information (clipped)')
ax.axvline(0.5, color='gray', lw=1.5, ls=':', alpha=0.7)
ax.scatter([0.5], [1.0], s=150, c='blue', zorder=5)
ax.text(0.52, 1.02, 'max H\n= 1 bit', fontsize=9, color='blue', fontweight='bold')
ax.set_xlabel('Probability p')
ax.set_ylabel('Bits')
ax.set_title('Entropy and Self-Information\nversus Probability', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 4.5)

# --- Panel 2: KL divergence examples (asymmetry) ---
ax = axes[1]
n_cat = 6
x_pos = np.arange(n_cat)
width = 0.35
P_ex = np.array([0.05, 0.05, 0.10, 0.20, 0.30, 0.30])
Q_ex = np.array([0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
kl_pq = kl_divergence(P_ex, Q_ex)
kl_qp = kl_divergence(Q_ex, P_ex)

ax.bar(x_pos - width/2, P_ex, width, label=f'P: right-skewed', color='steelblue', alpha=0.85, edgecolor='black')
ax.bar(x_pos + width/2, Q_ex, width, label=f'Q: left-skewed', color='tomato', alpha=0.85, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'x{i+1}' for i in range(n_cat)])
ax.set_ylabel('Probability')
ax.set_title(f'KL Divergence is NOT Symmetric!\n'
             f'KL(P||Q) = {kl_pq:.3f} bits\n'
             f'KL(Q||P) = {kl_qp:.3f} bits',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.text(0.5, 0.55, f'KL(P||Q) = {kl_pq:.2f}\nKL(Q||P) = {kl_qp:.2f}\nNot the same!',
        transform=ax.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

# --- Panel 3: Text reference ---
ax = axes[2]
ax.text(0.5, 0.98, 'INFORMATION THEORY CHEAT SHEET', fontsize=12,
        fontweight='bold', ha='center', transform=ax.transAxes)
ref_lines = [
    "",
    "SELF-INFORMATION:",
    "  I(x) = -log2(P(x))  [bits]",
    "  Rare event -> high information",
    "",
    "ENTROPY (Shannon):",
    "  H(X) = -sum(P(x) * log2(P(x)))",
    "  = average self-information",
    "  H=0: perfectly certain",
    "  H=log2(N): uniform (max disorder)",
    "",
    "NUMPY: entropy_scratch(probs)",
    "SCIPY: scipy.stats.entropy(p, base=2)",
    "",
    "KL DIVERGENCE:",
    "  KL(P||Q) = sum(P * log2(P/Q))",
    "  KL >= 0 always",
    "  KL(P||P) = 0",
    "  NOT symmetric! KL(P||Q)!=KL(Q||P)",
    "",
    "SILHOUETTE SCORE:",
    "  s(i) = (b - a) / max(a, b)",
    "  a = avg dist to same-cluster pts",
    "  b = avg dist to nearest cluster",
    "  +1=perfect, 0=boundary, -1=wrong",
    "  SKLEARN: silhouette_score(X, lbls)",
    "",
    "WHERE THESE APPEAR IN ML:",
    "  * K-Means: silhouette to choose K",
    "  * Decision trees: entropy splitting",
    "  * VAE: KL divergence in loss",
    "  * Anomaly detection: high entropy",
    "    neighborhood -> outlier!",
    "  * Feature selection: mutual info",
]
y = 0.94
for line in ref_lines:
    bold = any(line.startswith(kw) for kw in [
        'SELF-', 'ENTROPY', 'KL ', 'SILHOUETTE', 'WHERE', 'NUMPY', 'SCIPY'])
    ax.text(0.04, y, line, fontsize=8, transform=ax.transAxes,
            family='monospace', fontweight='bold' if bold else 'normal')
    y -= 0.032
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_information_theory_reference.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_information_theory_reference.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Information Theory Basics")
print("=" * 80)
print()
print("WHAT WE LEARNED:")
print("-" * 70)
print()
print("1. SELF-INFORMATION:")
print("   I(x) = -log2(P(x))")
print("   Rare events carry MORE information (are more surprising)")
print("   Certain events carry ZERO information")
print()
print("2. SHANNON ENTROPY:")
print("   H(X) = -sum( P(x) * log2(P(x)) )")
print("   Average uncertainty / surprise of a distribution")
print("   Low H -> concentrated (certain) | High H -> spread (uncertain)")
print("   numpy: -(probs * np.log2(probs)).sum()   or   scipy.stats.entropy(p, base=2)")
print()
print("3. CLUSTER QUALITY (Entropy):")
print("   Compute label distribution within each cluster")
print("   Low entropy cluster = pure = GOOD")
print("   High entropy cluster = mixed = BAD")
print("   Weighted average entropy = overall clustering quality metric")
print()
print("4. KL DIVERGENCE:")
print("   KL(P||Q) = sum( P * log2(P/Q) )")
print("   Measures how different P is from Q")
print("   KL >= 0, KL = 0 iff P = Q, NOT symmetric")
print("   scipy.stats.entropy(P, Q, base=2)")
print()
print("5. SILHOUETTE SCORE PREVIEW:")
print("   s(i) = (b - a) / max(a, b)")
print("   Measures how well each point fits its cluster (geometry only, no labels!)")
print("   Range [-1, 1] | Goal: maximize average silhouette score")
print("   sklearn.metrics.silhouette_score(X, labels)")
print("   We'll use this in the K-Means module to choose the best K!")
print()
print("=" * 80)
print("Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("  01_entropy_distributions.png")
print("  02_silhouette_score_analysis.png")
print("  03_information_theory_reference.png")
print("=" * 80)
print()
print("NEXT STEPS:")
print("  1. Look at visualization 2 — notice how the silhouette score drops when")
print("     clusters overlap! Green = well-placed, Red = possibly misclassified.")
print("  2. Modify the probability distributions and watch entropy change.")
print("  3. Next: Move to the Algorithms section!")
print("     -> K-Means Clustering (uses distance metrics + silhouette to choose K)")
print("     -> PCA (uses eigenvectors from Module 03)")
print()
print("=" * 80)
print("INFORMATION THEORY BASICS MASTERED!")
print("   You understand entropy, KL divergence, and cluster quality metrics!")
print("   These appear in every advanced ML paper you'll ever read!")
print("=" * 80)
