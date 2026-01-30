"""
📊 METRICS DEEP DIVE - ROC, AUC, and Advanced Metrics
===============================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. ROC curves - visualizing classifier performance
2. AUC (Area Under Curve) - single number metric
3. Precision-Recall curves - when classes are imbalanced
4. Multi-class classification metrics
5. Cost-sensitive learning
6. Choosing the right threshold
7. Which metric to use when

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "ROC and AUC, Clearly Explained!"
   https://www.youtube.com/watch?v=4jRBRDbJemM
   THE BEST explanation of ROC curves!

⭐ StatQuest: "Sensitivity and Specificity"
   https://www.youtube.com/watch?v=vP06aMoz4v8
   Foundation for understanding ROC curves

📚 Josh Starmer: "Precision and Recall"
   Understanding the precision-recall tradeoff

📚 Luis Serrano: "ROC Curves and AUC"
   https://www.youtube.com/watch?v=OAl6eAyP-yo
   Visual intuitive explanation

TIME: 75-90 minutes
DIFFICULTY: Intermediate-Advanced
PREREQUISITES: 04_confusion_matrix.py (CRITICAL!)

KEY CONCEPTS:
------------
- ROC Curve: Receiver Operating Characteristic
- TPR: True Positive Rate (Recall/Sensitivity)
- FPR: False Positive Rate (1 - Specificity)
- AUC: Area Under ROC Curve (0.5 = random, 1.0 = perfect)
- Precision-Recall Curve: For imbalanced classes
- Threshold Tuning: Adjusting classification threshold
- Multi-class Metrics: One-vs-Rest, Macro/Micro averaging
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'metrics_deep_dive'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📊 METRICS DEEP DIVE - ROC, AUC, and Advanced Metrics")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: REVIEW - CONFUSION MATRIX
# ============================================================================

print("=" * 80)
print("SECTION 1: Quick Review - Confusion Matrix")
print("=" * 80)
print()

print("Everything starts with the Confusion Matrix!")
print()
print("                    PREDICTED")
print("                Positive    Negative")
print("              ┌──────────┬──────────┐")
print("   ACTUAL   P │    TP    │    FN    │")
print("            O │          │          │")
print("            S ├──────────┼──────────┤")
print("              │    FP    │    TN    │")
print("   ACTUAL   N │          │          │")
print("            E │          │          │")
print("            G └──────────┴──────────┘")
print()

print("KEY METRICS (from confusion matrix):")
print("-" * 70)
print()
print("1. TRUE POSITIVE RATE (TPR) = Recall = Sensitivity")
print("   = TP / (TP + FN)")
print("   = Of all actual positives, how many did we catch?")
print()
print("2. FALSE POSITIVE RATE (FPR) = 1 - Specificity")
print("   = FP / (FP + TN)")
print("   = Of all actual negatives, how many did we wrongly flag?")
print()
print("3. PRECISION = Positive Predictive Value")
print("   = TP / (TP + FP)")
print("   = Of all predicted positives, how many were correct?")
print()
print("4. SPECIFICITY = True Negative Rate")
print("   = TN / (TN + FP)")
print("   = Of all actual negatives, how many did we correctly identify?")
print()

# Example with actual numbers
print("Example: Medical Test")
print("-" * 70)
print("100 patients: 20 sick, 80 healthy")
print()

# Scenario 1
tp, fn, fp, tn = 18, 2, 10, 70
print(f"Test Results:")
print(f"   TP={tp} (correctly identified sick)")
print(f"   FN={fn} (missed sick patients)")
print(f"   FP={fp} (false alarms)")
print(f"   TN={tn} (correctly identified healthy)")
print()

tpr = tp / (tp + fn)
fpr = fp / (fp + tn)
precision = tp / (tp + fp)
specificity = tn / (tn + fp)

print(f"Calculated Metrics:")
print(f"   TPR (Sensitivity) = {tp}/{tp+fn} = {tpr:.2f} ({tpr*100:.0f}% of sick patients caught)")
print(f"   FPR              = {fp}/{fp+tn} = {fpr:.2f} ({fpr*100:.0f}% false alarm rate)")
print(f"   Precision        = {tp}/{tp+fp} = {precision:.2f} ({precision*100:.0f}% of positive tests correct)")
print(f"   Specificity      = {tn}/{tn+fp} = {specificity:.2f} ({specificity*100:.0f}% of healthy correctly id'd)")
print()

# ============================================================================
# SECTION 2: ROC CURVE
# ============================================================================

print("=" * 80)
print("SECTION 2: ROC Curve - The Big Picture")
print("=" * 80)
print()

print("ROC = Receiver Operating Characteristic")
print()

print("THE PROBLEM:")
print("-" * 70)
print("Most classifiers output PROBABILITIES, not just yes/no")
print()
print("Example: Medical test")
print("   Patient A: 0.9 probability of disease")
print("   Patient B: 0.7 probability")
print("   Patient C: 0.3 probability")
print()
print("Where do we draw the line? (the THRESHOLD)")
print()
print("Option 1: Threshold = 0.5")
print("   → Patients A and B: Positive")
print("   → Patient C: Negative")
print()
print("Option 2: Threshold = 0.8 (more conservative)")
print("   → Only Patient A: Positive")
print("   → Fewer false alarms, but might miss some sick patients")
print()
print("Option 3: Threshold = 0.2 (more aggressive)")
print("   → All three: Positive")
print("   → Catch more sick patients, but more false alarms")
print()

print("THE SOLUTION: ROC Curve")
print("-" * 70)
print("• Try MANY different thresholds")
print("• For each threshold, calculate TPR and FPR")
print("• Plot TPR vs FPR")
print("• This is the ROC curve!")
print()

print("ROC CURVE INTERPRETATION:")
print()
print("   Y-axis (TPR): How many positives did we catch?")
print("   X-axis (FPR): How many false alarms did we make?")
print()
print("   GOAL: High TPR (catch positives) with Low FPR (few false alarms)")
print()

# Generate sample predictions
np.random.seed(42)
n_samples = 200

# Sick patients (class 1): higher probabilities
y_true_sick = np.ones(100)
y_proba_sick = np.random.beta(5, 2, 100)  # Concentrated near 1

# Healthy patients (class 0): lower probabilities
y_true_healthy = np.zeros(100)
y_proba_healthy = np.random.beta(2, 5, 100)  # Concentrated near 0

# Combine
y_true = np.concatenate([y_true_sick, y_true_healthy])
y_proba = np.concatenate([y_proba_sick, y_proba_healthy])

print("Example: Medical test on 200 patients")
print(f"   100 sick (class 1)")
print(f"   100 healthy (class 0)")
print()
print(f"Probability range: {y_proba.min():.3f} to {y_proba.max():.3f}")
print()

# Calculate ROC curve manually
thresholds = np.linspace(0, 1, 100)
tpr_list = []
fpr_list = []

print("Calculating TPR and FPR for different thresholds...")
print("-" * 70)
print(f"{'Threshold':<12} {'TPR':<12} {'FPR':<12} {'Interpretation'}")
print("-" * 70)

# Show some sample thresholds
for i, threshold in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    if threshold == 0.1:
        interp = "Aggressive (catch everything)"
    elif threshold == 0.5:
        interp = "Balanced"
    elif threshold == 0.9:
        interp = "Conservative (few false alarms)"
    else:
        interp = ""

    print(f"{threshold:<12.1f} {tpr:<12.2f} {fpr:<12.2f} {interp}")

# Calculate full ROC curve
for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    tpr_list.append(tpr)
    fpr_list.append(fpr)

tpr_array = np.array(tpr_list)
fpr_array = np.array(fpr_list)

print()
print(f"✓ Calculated {len(thresholds)} points for ROC curve")
print()

# ============================================================================
# SECTION 3: AUC (AREA UNDER CURVE)
# ============================================================================

print("=" * 80)
print("SECTION 3: AUC - One Number to Rule Them All")
print("=" * 80)
print()

print("AUC = Area Under the ROC Curve")
print()

# Calculate AUC using trapezoidal rule
auc = np.trapz(tpr_array, fpr_array)
auc = abs(auc)  # Make positive

print("HOW TO INTERPRET AUC:")
print("-" * 70)
print()
print(f"   AUC = 0.5  →  Random guessing (coin flip)")
print(f"   AUC = 0.7  →  Okay model")
print(f"   AUC = 0.8  →  Good model")
print(f"   AUC = 0.9  →  Excellent model")
print(f"   AUC = 1.0  →  Perfect model")
print()
print(f"Our model AUC: {auc:.3f}")
print()

print("WHAT AUC MEANS:")
print()
print("AUC = Probability that a random positive example")
print("      ranks higher than a random negative example")
print()
print("Example: AUC = 0.85")
print("   → 85% chance that a sick patient has higher")
print("     probability than a healthy patient")
print()

print("WHY AUC IS USEFUL:")
print()
print("✓ Single number (easy to compare models)")
print("✓ Threshold-independent (evaluates all thresholds)")
print("✓ Balanced (works well for imbalanced classes)")
print("✓ Intuitive interpretation")
print()

print("WHEN AUC CAN BE MISLEADING:")
print()
print("✗ Doesn't tell you optimal threshold")
print("✗ Might not match your business metric")
print("✗ Can hide poor performance at specific thresholds")
print("✗ Not ideal for VERY imbalanced classes (use PR curve)")
print()

# ============================================================================
# SECTION 4: PRECISION-RECALL CURVE
# ============================================================================

print("=" * 80)
print("SECTION 4: Precision-Recall Curve - For Imbalanced Data")
print("=" * 80)
print()

print("WHEN CLASSES ARE IMBALANCED:")
print("-" * 70)
print()
print("Example: Fraud detection")
print("   • 10,000 transactions")
print("   • 9,900 legitimate (99%)")
print("   • 100 fraudulent (1%)")
print()

print("Problem with ROC/AUC:")
print("   • FPR = FP / (FP + TN)")
print("   • Large TN (9,900) makes FPR look small even with many FP")
print("   • Can have AUC=0.95 but miss most fraud!")
print()

print("Solution: Precision-Recall Curve")
print("   • Focuses on positive class performance")
print("   • Ignores TN (true negatives)")
print("   • Better for imbalanced classes")
print()

print("METRICS:")
print()
print("   Recall (TPR) = TP / (TP + FN)")
print("      → Of actual positives, how many did we catch?")
print()
print("   Precision = TP / (TP + FP)")
print("      → Of predicted positives, how many were correct?")
print()

# Calculate Precision-Recall curve
precision_list = []
recall_list = []

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # 1.0 if no positives predicted

    recall_list.append(recall)
    precision_list.append(precision)

precision_array = np.array(precision_list)
recall_array = np.array(recall_list)

print("TRADEOFF:")
print()
print("   High Recall (catch all positives)")
print("      → Low Precision (many false alarms)")
print()
print("   High Precision (few false alarms)")
print("      → Low Recall (miss some positives)")
print()

print("Examples:")
print("-" * 70)
print()
print("Cancer Screening:")
print("   → Prioritize RECALL (don't miss any cancer)")
print("   → Accept lower precision (some false positives okay)")
print()
print("Email Spam Filter:")
print("   → Prioritize PRECISION (don't block important emails)")
print("   → Accept lower recall (some spam gets through)")
print()

# ============================================================================
# SECTION 5: CHOOSING THE RIGHT THRESHOLD
# ============================================================================

print("=" * 80)
print("SECTION 5: Finding the Optimal Threshold")
print("=" * 80)
print()

print("Default threshold = 0.5, but this is often NOT optimal!")
print()

print("STRATEGIES:")
print()

print("1. MAXIMIZE F1 SCORE")
print("   F1 = 2 × (Precision × Recall) / (Precision + Recall)")
print("   → Balances precision and recall")
print()

# Find threshold that maximizes F1
f1_list = []
for i, threshold in enumerate(thresholds):
    y_pred = (y_proba >= threshold).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    f1_list.append(f1)

f1_array = np.array(f1_list)
best_f1_idx = np.argmax(f1_array)
best_f1_threshold = thresholds[best_f1_idx]

print(f"   Best F1 threshold: {best_f1_threshold:.3f}")
print(f"   F1 score: {f1_array[best_f1_idx]:.3f}")
print()

print("2. TARGET SPECIFIC RECALL")
print("   Example: 'We must catch 95% of positives'")
print("   → Find threshold that gives Recall ≥ 0.95")
print()

# Find threshold for 95% recall
target_recall = 0.95
recall_95_idx = np.argmin(np.abs(recall_array - target_recall))
recall_95_threshold = thresholds[recall_95_idx]

print(f"   Threshold for 95% recall: {recall_95_threshold:.3f}")
print(f"   Actual recall: {recall_array[recall_95_idx]:.3f}")
print(f"   Precision at this threshold: {precision_array[recall_95_idx]:.3f}")
print()

print("3. COST-BASED THRESHOLD")
print("   Assign costs to FP and FN, minimize total cost")
print()
print("   Example: Medical test")
print("   • Cost of FN (missing sick patient) = $100,000")
print("   • Cost of FP (unnecessary treatment) = $1,000")
print("   • FN is 100× more expensive!")
print("   → Use lower threshold (catch more positives)")
print()

# Cost-sensitive threshold
cost_fn = 100  # Relative cost
cost_fp = 1

min_cost = float('inf')
best_cost_threshold = 0.5

for i, threshold in enumerate(thresholds):
    y_pred = (y_proba >= threshold).astype(int)

    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    total_cost = (fn * cost_fn) + (fp * cost_fp)

    if total_cost < min_cost:
        min_cost = total_cost
        best_cost_threshold = threshold

print(f"   Cost-optimal threshold: {best_cost_threshold:.3f}")
print(f"   (Given FN cost = {cost_fn}×, FP cost = {cost_fp}×)")
print()

# ============================================================================
# SECTION 6: USING SCIKIT-LEARN
# ============================================================================

print("=" * 80)
print("SECTION 6: Using Scikit-Learn for ROC and PR Curves")
print("=" * 80)
print()

try:
    from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                  average_precision_score, roc_auc_score)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Create simple dataset
    np.random.seed(42)
    n_samples = 500

    # Class 1: higher values
    X_class1 = np.random.randn(n_samples//2, 2) * 1.0 + np.array([2, 2])
    # Class 0: lower values
    X_class0 = np.random.randn(n_samples//2, 2) * 1.0 + np.array([0, 0])

    X = np.vstack([X_class1, X_class0])
    y = np.array([1]*(n_samples//2) + [0]*(n_samples//2))

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get predicted probabilities
    y_proba_test = model.predict_proba(X_test)[:, 1]

    # Calculate ROC curve
    fpr_sk, tpr_sk, thresholds_roc = roc_curve(y_test, y_proba_test)
    roc_auc = auc(fpr_sk, tpr_sk)

    print("ROC Curve:")
    print(f"   AUC = {roc_auc:.3f}")
    print()

    # Calculate Precision-Recall curve
    precision_sk, recall_sk, thresholds_pr = precision_recall_curve(y_test, y_proba_test)
    avg_precision = average_precision_score(y_test, y_proba_test)

    print("Precision-Recall Curve:")
    print(f"   Average Precision = {avg_precision:.3f}")
    print()

    # Using roc_auc_score (simpler)
    auc_simple = roc_auc_score(y_test, y_proba_test)
    print(f"ROC AUC (using roc_auc_score): {auc_simple:.3f}")
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not installed")
    sklearn_available = False

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: ROC Curve with interpretation
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('ROC Curve: Complete Guide', fontsize=16, fontweight='bold')

# Plot 1: Basic ROC curve
ax1 = axes[0, 0]
ax1.plot(fpr_array, tpr_array, 'b-', linewidth=3, label=f'Our Model (AUC={auc:.3f})')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC=0.5)')
ax1.fill_between(fpr_array, tpr_array, alpha=0.3)

ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=12, fontweight='bold')
ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])

# Add annotations
ax1.annotate('Perfect Classifier\n(TPR=1, FPR=0)', xy=(0, 1), xytext=(0.3, 0.85),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))

ax1.annotate('Random\nGuessing', xy=(0.5, 0.5), xytext=(0.6, 0.3),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Plot 2: Different model comparisons
ax2 = axes[0, 1]

# Generate different quality models
# Perfect model
ax2.plot([0, 0, 1], [0, 1, 1], 'g-', linewidth=2, label='Perfect (AUC=1.0)', alpha=0.7)
# Good model
ax2.plot(fpr_array, tpr_array, 'b-', linewidth=2, label=f'Good (AUC={auc:.2f})', alpha=0.7)
# Random
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random (AUC=0.5)', alpha=0.7)
# Poor model
ax2.plot([0, 1, 1], [0, 0, 1], color='orange', linewidth=2, label='Poor (AUC=0.7)', alpha=0.7)

ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('Comparing Different Models', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Threshold visualization
ax3 = axes[1, 0]

# Show how threshold affects TPR and FPR
ax3.plot(thresholds, tpr_array, 'b-', linewidth=2, label='TPR (Recall)', marker='o', markersize=3)
ax3.plot(thresholds, fpr_array, 'r-', linewidth=2, label='FPR', marker='s', markersize=3)

# Mark some key thresholds
ax3.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
ax3.axvline(best_f1_threshold, color='green', linestyle='--', alpha=0.5, label=f'Best F1 ({best_f1_threshold:.2f})')

ax3.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold')
ax3.set_ylabel('Rate', fontsize=11, fontweight='bold')
ax3.set_title('Effect of Threshold on TPR and FPR', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Text explanation
ax4 = axes[1, 1]
ax4.axis('off')

explanation_text = """
READING THE ROC CURVE
═══════════════════════════════════════════════════════════════

AXES:
• X-axis (FPR): False Positive Rate
  - Proportion of negatives misclassified as positive
  - Want this LOW (fewer false alarms)

• Y-axis (TPR): True Positive Rate (Recall)
  - Proportion of positives correctly identified
  - Want this HIGH (catch more positives)

THE IDEAL POINT: Top-left corner (0, 1)
  → TPR = 1 (catch all positives)
  → FPR = 0 (no false alarms)

AUC INTERPRETATION:
• AUC = 1.0 → Perfect model
• AUC = 0.9-1.0 → Excellent
• AUC = 0.8-0.9 → Good
• AUC = 0.7-0.8 → Okay
• AUC = 0.5 → Random guessing (no better than coin flip)
• AUC < 0.5 → Worse than random (flip predictions!)

USING ROC FOR THRESHOLD SELECTION:
1. Pick a point on the curve = picking a threshold
2. Moving right on curve = lower threshold
   → Catch more positives (↑TPR)
   → But also more false alarms (↑FPR)
3. Moving left on curve = higher threshold
   → Fewer false alarms (↓FPR)
   → But miss more positives (↓TPR)

WHEN TO USE ROC/AUC:
✓ Balanced classes (roughly 50/50)
✓ Both FP and FN are equally important
✓ Want threshold-independent evaluation
✗ Very imbalanced classes (use PR curve instead)

═══════════════════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, explanation_text.strip(),
        transform=ax4.transAxes,
        fontsize=8,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_roc_curve_complete.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_roc_curve_complete.png")
plt.close()

# Visualization 2: Precision-Recall Curve
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Precision-Recall Curve: Complete Guide', fontsize=16, fontweight='bold')

# Plot 1: Basic PR curve
ax1 = axes[0, 0]
ax1.plot(recall_array, precision_array, 'b-', linewidth=3, label='Our Model')
baseline_precision = np.sum(y_true == 1) / len(y_true)
ax1.axhline(baseline_precision, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_precision:.3f})')
ax1.fill_between(recall_array, precision_array, alpha=0.3)

ax1.set_xlabel('Recall (TPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax1.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])

# Add annotations
ax1.annotate('Perfect Classifier\n(Precision=1, Recall=1)', xy=(1, 1), xytext=(0.5, 0.85),
            fontsize=10, ha='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'))

# Plot 2: F1 score vs threshold
ax2 = axes[0, 1]
ax2.plot(thresholds, f1_array, 'purple', linewidth=3, marker='o', markersize=3)
ax2.axvline(best_f1_threshold, color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Best F1 threshold')
ax2.axhline(f1_array[best_f1_idx], color='green', linestyle='--', linewidth=1, alpha=0.5)

ax2.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
ax2.set_title(f'F1 Score vs Threshold (Best: {best_f1_threshold:.3f})', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Precision and Recall vs Threshold
ax3 = axes[1, 0]
ax3.plot(thresholds, precision_array, 'b-', linewidth=2, label='Precision', marker='o', markersize=3)
ax3.plot(thresholds, recall_array, 'r-', linewidth=2, label='Recall', marker='s', markersize=3)
ax3.plot(thresholds, f1_array, 'purple', linewidth=2, label='F1 Score', marker='^', markersize=3)

ax3.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
ax3.axvline(best_f1_threshold, color='green', linestyle='--', alpha=0.5)

ax3.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold')
ax3.set_ylabel('Score', fontsize=11, fontweight='bold')
ax3.set_title('Precision-Recall-F1 Tradeoff', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Text explanation
ax4 = axes[1, 1]
ax4.axis('off')

pr_explanation = """
PRECISION-RECALL CURVE
═══════════════════════════════════════════════════════════════

WHEN TO USE:
✓ Imbalanced classes (e.g., fraud detection: 1% fraud)
✓ Care more about positive class performance
✓ False negatives are costly
✗ Balanced classes (use ROC instead)

METRICS:
• Precision = TP / (TP + FP)
  - Of predictions we made, how many were correct?
  - "How precise are our positive predictions?"

• Recall = TP / (TP + FN)
  - Of actual positives, how many did we find?
  - "How many positives did we recall/find?"

THE TRADEOFF:
  High Recall → Catch more positives
               → But more false alarms (low precision)

  High Precision → Few false alarms
                  → But miss more positives (low recall)

F1 SCORE: Harmonic mean of Precision and Recall
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  → Balances both metrics
  → Good default objective

CHOOSING THRESHOLD:
• Max F1: Balance precision and recall
• High Recall: Medical diagnosis (don't miss disease)
• High Precision: Spam filter (don't block real emails)
• Cost-based: Assign $ costs to FP and FN

BASELINE:
  Random classifier precision = % of positive class
  Example: 5% fraud → baseline precision = 0.05
  Your model should beat this!

═══════════════════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, pr_explanation.strip(),
        transform=ax4.transAxes,
        fontsize=8,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/02_precision_recall_complete.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/02_precision_recall_complete.png")
plt.close()

# Visualization 3: Metric comparison flowchart
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.axis('off')
fig.suptitle('Which Metric Should I Use? Decision Flowchart', fontsize=16, fontweight='bold', y=0.98)

flowchart = """
                         START: Evaluating Classification Model
                                        │
                                        ▼
                    ┌───────────────────────────────────────────┐
                    │  Are your classes balanced?               │
                    │  (roughly 50/50 or at least 40/60)        │
                    └─────────────┬──────────────────┬──────────┘
                                  │                  │
                          YES     │                  │    NO
                                  │                  │
                                  ▼                  ▼
                    ┌──────────────────┐  ┌──────────────────────────┐
                    │  ROC/AUC         │  │  Which class matters      │
                    │                  │  │  more to you?             │
                    │  ✓ Easy to       │  └───────────┬──────────────┘
                    │    interpret     │              │
                    │  ✓ Threshold-    │              ▼
                    │    independent   │     ┌────────┴──────────┐
                    │  ✓ Standard      │     │                   │
                    │    metric        │  POSITIVE            BOTH
                    └──────────────────┘     │                   │
                                             ▼                   ▼
                              ┌──────────────────────┐  ┌───────────────┐
                              │ Precision-Recall     │  │ Use F1 Score  │
                              │ Curve                │  │ or Balanced   │
                              │                      │  │ Accuracy      │
                              │ ✓ Focuses on         │  └───────────────┘
                              │   positive class     │
                              │ ✓ Ignores TN         │
                              │ ✓ Better for         │
                              │   rare events        │
                              └──────────────────────┘
                                        │
                                        ▼
                    ┌────────────────────────────────────────┐
                    │  What's more important?                │
                    └───┬────────────┬───────────┬───────────┘
                        │            │           │
                   Precision     Recall      Both Equal
                        │            │           │
                        ▼            ▼           ▼
                  ┌──────────┐ ┌──────────┐ ┌──────────┐
                  │ Maximize │ │ Maximize │ │ Maximize │
                  │ Precision│ │ Recall   │ │ F1 Score │
                  │          │ │          │ │          │
                  │ Example: │ │ Example: │ │ Example: │
                  │ - Spam   │ │ - Cancer │ │ - General│
                  │   filter │ │   screen │ │   purpose│
                  │ - Don't  │ │ - Don't  │ │ - Trade- │
                  │   block  │ │   miss   │ │   off    │
                  │   real   │ │   disease│ │   both   │
                  │   emails │ │          │ │          │
                  └──────────┘ └──────────┘ └──────────┘

SPECIAL CASES:
═══════════════════════════════════════════════════════════════════════
• Multi-class classification: Use macro/micro-averaged metrics
• Cost-sensitive: Define costs for FP and FN, minimize total cost
• Multiple objectives: Report multiple metrics (precision, recall, F1)
• Production: Monitor actual business metric (revenue, user satisfaction)
═══════════════════════════════════════════════════════════════════════

QUICK REFERENCE:
─────────────────────────────────────────────────────────────────────
Accuracy:     Good for balanced classes, overall performance
Precision:    "Of my positive predictions, how many were right?"
Recall:       "Of actual positives, how many did I find?"
F1:           Harmonic mean of Precision and Recall
ROC AUC:      Threshold-independent, balanced classes
PR AUC:       Imbalanced classes, focus on positive class
═══════════════════════════════════════════════════════════════════════
"""

ax.text(0.5, 0.5, flowchart,
       transform=ax.transAxes,
       fontsize=7.5,
       verticalalignment='center',
       horizontalalignment='center',
       fontfamily='monospace',
       bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.2))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/03_metric_selection_flowchart.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/03_metric_selection_flowchart.png")
plt.close()

print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("📊 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ ROC CURVE: Plots TPR vs FPR for all thresholds")
print("   • Shows tradeoff between catching positives and false alarms")
print("   • Ideal point: top-left corner (TPR=1, FPR=0)")
print()
print("✓ AUC (Area Under ROC Curve): Single-number metric")
print("   • 0.5 = random, 1.0 = perfect")
print("   • Probability that positive ranks higher than negative")
print("   • Threshold-independent evaluation")
print()
print("✓ PRECISION-RECALL CURVE: For imbalanced classes")
print("   • Focuses on positive class performance")
print("   • Ignores true negatives")
print("   • Better than ROC for rare events")
print()
print("✓ THRESHOLD SELECTION:")
print("   • Default (0.5) is often NOT optimal")
print("   • Maximize F1: Balance precision and recall")
print("   • Target specific recall: Medical screening")
print("   • Cost-based: Assign costs to FP and FN")
print()
print("✓ WHICH METRIC TO USE:")
print("   • Balanced classes → ROC/AUC")
print("   • Imbalanced classes → Precision-Recall")
print("   • Equal importance → F1 Score")
print("   • Don't miss positives → Maximize Recall")
print("   • Few false alarms → Maximize Precision")
print()
print("✓ KEY TAKEAWAYS:")
print("   • No single 'best' metric - depends on problem")
print("   • Always visualize (ROC/PR curves)")
print("   • Consider business costs in threshold selection")
print("   • Report multiple metrics")
print("   • Understand tradeoffs")
print()
print("=" * 80)
print("📊 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
