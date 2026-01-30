"""
🎯 CONFUSION MATRIX - The Foundation of All Classification Metrics
====================================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. What a confusion matrix is and how to read it
2. True Positives, False Positives, True Negatives, False Negatives
3. How ALL classification metrics derive from the confusion matrix
4. Accuracy, Precision, Recall (Sensitivity), Specificity, F1 Score
5. When to use which metric based on your problem
6. Why accuracy can be misleading

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Confusion Matrix"
   https://www.youtube.com/watch?v=Kdsp6soqA7o
   THE best explanation of confusion matrices

⭐ StatQuest: "Sensitivity and Specificity"
   https://www.youtube.com/watch?v=vP06aMoz4v8
   Clear explanation of these important metrics

⭐ StatQuest: "Precision and Recall"
   https://www.youtube.com/watch?v=qWfzIYCvBqo
   When to use precision vs recall

📚 Krish Naik: "Confusion Matrix Explained"
   https://www.youtube.com/watch?v=FAr2GmWNbT0
   Practical examples and interpretations

TIME: 45-60 minutes
DIFFICULTY: Intermediate
PREREQUISITES: 01_sigmoid_function.py, 02_probability_for_classification.py

KEY CONCEPTS:
------------
- Confusion Matrix: 2x2 table showing actual vs predicted
- TP, FP, TN, FN: The four outcomes
- Accuracy: Overall correctness
- Precision: Of predicted positive, how many correct?
- Recall (Sensitivity): Of actual positive, how many found?
- F1 Score: Harmonic mean of precision and recall
- Specificity: Of actual negative, how many correct?

THIS IS THE MOST IMPORTANT MODULE IN CLASSIFICATION!
Every metric you'll ever use comes from the confusion matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / '04_confusion_matrix'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎯 CONFUSION MATRIX - Foundation of Classification Metrics")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE CONFUSION MATRIX
# ============================================================================

print("=" * 80)
print("SECTION 1: What is a Confusion Matrix?")
print("=" * 80)
print()

print("A confusion matrix is a 2x2 table that describes classification performance:")
print()
print("                    PREDICTED")
print("                 Positive  Negative")
print("              ┌──────────┬──────────┐")
print("      ACTUAL  │          │          │")
print("    Positive  │    TP    │    FN    │")
print("              │          │          │")
print("              ├──────────┼──────────┤")
print("      ACTUAL  │          │          │")
print("    Negative  │    FP    │    TN    │")
print("              │          │          │")
print("              └──────────┴──────────┘")
print()

print("THE FOUR OUTCOMES:")
print()
print("TP (True Positive):")
print("   • Actual = Positive")
print("   • Predicted = Positive")
print("   • ✓ Correct prediction!")
print("   • Example: Email is spam, we predicted spam")
print()

print("TN (True Negative):")
print("   • Actual = Negative")
print("   • Predicted = Negative")
print("   • ✓ Correct prediction!")
print("   • Example: Email is not spam, we predicted not spam")
print()

print("FP (False Positive) - TYPE I ERROR:")
print("   • Actual = Negative")
print("   • Predicted = Positive")
print("   • ✗ Wrong prediction (false alarm!)")
print("   • Example: Email is NOT spam, but we predicted spam")
print()

print("FN (False Negative) - TYPE II ERROR:")
print("   • Actual = Positive")
print("   • Predicted = Negative")
print("   • ✗ Wrong prediction (missed detection!)")
print("   • Example: Email IS spam, but we predicted not spam")
print()

# ============================================================================
# SECTION 2: EXAMPLE - SPAM DETECTION
# ============================================================================

print("=" * 80)
print("SECTION 2: Real Example - Spam Email Detection")
print("=" * 80)
print()

print("Scenario: We tested our spam filter on 100 emails")
print()

# Example confusion matrix
TP = 35  # Correctly identified spam
FN = 5   # Spam we missed
FP = 10  # False alarms (good email marked as spam)
TN = 50  # Correctly identified good email

total = TP + FN + FP + TN

print(f"True Positives (TP)  = {TP}  (Spam correctly detected)")
print(f"False Negatives (FN) = {FN}  (Spam we missed)")
print(f"False Positives (FP) = {FP}  (Good email marked as spam)")
print(f"True Negatives (TN)  = {TN}  (Good email correctly identified)")
print(f"Total                = {total} emails")
print()

print("Confusion Matrix:")
print()
print("                    PREDICTED")
print("                  Spam    Not Spam")
print("              ┌─────────┬─────────┐")
print(f"      ACTUAL  │         │         │")
print(f"       Spam   │   {TP:2d}    │   {FN:2d}    │  {TP+FN} actual spam")
print(f"              │         │         │")
print("              ├─────────┼─────────┤")
print(f"      ACTUAL  │         │         │")
print(f"    Not Spam  │   {FP:2d}    │   {TN:2d}    │  {FP+TN} actual not spam")
print(f"              │         │         │")
print("              └─────────┴─────────┘")
print(f"                {TP+FP}       {FN+TN}")
print("           predicted  predicted")
print("              spam    not spam")
print()

# ============================================================================
# SECTION 3: METRICS FROM CONFUSION MATRIX
# ============================================================================

print("=" * 80)
print("SECTION 3: Metrics Derived from Confusion Matrix")
print("=" * 80)
print()

print("ALL classification metrics come from TP, TN, FP, FN!")
print()

# Calculate metrics
accuracy = (TP + TN) / total
precision = TP / (TP + FP)
recall = TP / (TP + FN)  # Also called Sensitivity
specificity = TN / (TN + FP)
f1_score = 2 * (precision * recall) / (precision + recall)

print("1. ACCURACY")
print("   Formula: (TP + TN) / Total")
print("   Question: What % of predictions were correct?")
print(f"   Calculation: ({TP} + {TN}) / {total} = {accuracy:.3f}")
print(f"   Interpretation: {accuracy*100:.1f}% of predictions were correct")
print()

print("2. PRECISION (Positive Predictive Value)")
print("   Formula: TP / (TP + FP)")
print("   Question: Of emails we marked as spam, what % were actually spam?")
print(f"   Calculation: {TP} / ({TP} + {FP}) = {precision:.3f}")
print(f"   Interpretation: {precision*100:.1f}% of spam predictions were correct")
print("   → High precision means few false alarms")
print()

print("3. RECALL (Sensitivity, True Positive Rate)")
print("   Formula: TP / (TP + FN)")
print("   Question: Of actual spam emails, what % did we catch?")
print(f"   Calculation: {TP} / ({TP} + {FN}) = {recall:.3f}")
print(f"   Interpretation: We caught {recall*100:.1f}% of all spam")
print("   → High recall means we miss few spam emails")
print()

print("4. SPECIFICITY (True Negative Rate)")
print("   Formula: TN / (TN + FP)")
print("   Question: Of actual good emails, what % did we correctly identify?")
print(f"   Calculation: {TN} / ({TN} + {FP}) = {specificity:.3f}")
print(f"   Interpretation: {specificity*100:.1f}% of good emails identified correctly")
print("   → High specificity means few false alarms")
print()

print("5. F1 SCORE (Harmonic Mean of Precision and Recall)")
print("   Formula: 2 × (Precision × Recall) / (Precision + Recall)")
print("   Question: What's the balance between precision and recall?")
print(f"   Calculation: 2 × ({precision:.3f} × {recall:.3f}) / ({precision:.3f} + {recall:.3f}) = {f1_score:.3f}")
print(f"   Interpretation: F1 = {f1_score:.3f} (scale: 0 to 1, higher is better)")
print("   → Balances precision and recall into one metric")
print()

# ============================================================================
# SECTION 4: WHY ACCURACY CAN BE MISLEADING
# ============================================================================

print("=" * 80)
print("SECTION 4: The Accuracy Paradox")
print("=" * 80)
print()

print("WARNING: Accuracy can be misleading with imbalanced classes!")
print()

print("Example: Rare Disease Detection (1% of population has disease)")
print()
print("Scenario 1: Lazy Model - Always predict 'No Disease'")
print()

# Imbalanced example - always predict negative
tp_lazy = 0    # Never detects disease
fn_lazy = 10   # Misses all diseased
fp_lazy = 0    # Never predicts disease
tn_lazy = 990  # Correctly identifies healthy
total_lazy = 1000

accuracy_lazy = (tp_lazy + tn_lazy) / total_lazy
precision_lazy = 0  # Can't divide by zero
recall_lazy = 0 if (tp_lazy + fn_lazy) > 0 else 0

print(f"TP = {tp_lazy}, FN = {fn_lazy}, FP = {fp_lazy}, TN = {tn_lazy}")
print()
print(f"Accuracy  = {accuracy_lazy:.1%}  ← Looks great!")
print(f"Precision = N/A (never predicted positive)")
print(f"Recall    = {recall_lazy:.1%}    ← But catches ZERO diseases!")
print()
print("This model is USELESS but has 99% accuracy!")
print()

print("Scenario 2: Good Model - Actually detects disease")
print()
tp_good = 8    # Detects 8 out of 10
fn_good = 2    # Misses 2
fp_good = 50   # 50 false alarms
tn_good = 940  # 940 correct negatives
total_good = 1000

accuracy_good = (tp_good + tn_good) / total_good
precision_good = tp_good / (tp_good + fp_good)
recall_good = tp_good / (tp_good + fn_good)
f1_good = 2 * (precision_good * recall_good) / (precision_good + recall_good)

print(f"TP = {tp_good}, FN = {fn_good}, FP = {fp_good}, TN = {tn_good}")
print()
print(f"Accuracy  = {accuracy_good:.1%}  ← Lower accuracy!")
print(f"Precision = {precision_good:.1%} ← Some false alarms")
print(f"Recall    = {recall_good:.1%}  ← But catches most diseases!")
print(f"F1 Score  = {f1_good:.3f}")
print()
print("This model is MUCH better, but has lower accuracy!")
print()

print("LESSON: For imbalanced data, use Precision/Recall/F1, not Accuracy!")
print()

# ============================================================================
# SECTION 5: PRECISION VS RECALL TRADEOFF
# ============================================================================

print("=" * 80)
print("SECTION 5: The Precision-Recall Tradeoff")
print("=" * 80)
print()

print("You can't maximize both precision AND recall simultaneously.")
print("There's a tradeoff!")
print()

print("SCENARIO: Adjusting threshold in spam filter")
print()

# Simulate different thresholds
print("Threshold = 0.3 (Low - mark as spam if 30% confident)")
print("   More emails marked as spam")
print("   → Higher Recall (catches more spam)")
print("   → Lower Precision (more false alarms)")
print("   Example: TP=38, FP=20, FN=2, TN=40")
tp_low, fp_low, fn_low, tn_low = 38, 20, 2, 40
prec_low = tp_low / (tp_low + fp_low)
rec_low = tp_low / (tp_low + fn_low)
print(f"   Precision = {prec_low:.2f}, Recall = {rec_low:.2f}")
print()

print("Threshold = 0.7 (High - mark as spam if 70% confident)")
print("   Fewer emails marked as spam")
print("   → Lower Recall (misses some spam)")
print("   → Higher Precision (fewer false alarms)")
print("   Example: TP=30, FP=5, FN=10, TN=55")
tp_high, fp_high, fn_high, tn_high = 30, 5, 10, 55
prec_high = tp_high / (tp_high + fp_high)
rec_high = tp_high / (tp_high + fn_high)
print(f"   Precision = {prec_high:.2f}, Recall = {rec_high:.2f}")
print()

print("WHEN TO PRIORITIZE WHAT:")
print()
print("Prioritize RECALL (High Sensitivity):")
print("   • Medical diagnosis (can't miss diseases)")
print("   • Fraud detection (can't miss fraud)")
print("   • Security threats (can't miss attacks)")
print("   → Okay to have false alarms, can't miss true positives")
print()

print("Prioritize PRECISION:")
print("   • Spam filtering (avoid filtering good emails)")
print("   • Recommender systems (don't recommend bad items)")
print("   • Loan approval (don't approve bad loans)")
print("   → Want high confidence in positive predictions")
print()

print("Balance Both (Use F1 Score):")
print("   • When both false positives and false negatives matter")
print("   • Search engines, text classification")
print()

# ============================================================================
# SECTION 6: MANUAL CALCULATIONS
# ============================================================================

print("=" * 80)
print("SECTION 6: Practice - Calculate Metrics Yourself!")
print("=" * 80)
print()

print("New Example: Cancer Screening (200 patients)")
print()

# New example for practice
tp_cancer = 45
fn_cancer = 5
fp_cancer = 15
tn_cancer = 135
total_cancer = tp_cancer + fn_cancer + fp_cancer + tn_cancer

print("Given confusion matrix:")
print()
print("                PREDICTED")
print("              Cancer  No Cancer")
print("           ┌────────┬────────┐")
print(f"   ACTUAL  │        │        │")
print(f"   Cancer  │   {tp_cancer:2d}   │   {fn_cancer:2d}   │")
print("           ├────────┼────────┤")
print(f"   ACTUAL  │        │        │")
print(f" No Cancer │   {fp_cancer:2d}   │  {tn_cancer:3d}   │")
print("           └────────┴────────┘")
print()

print("Calculate the metrics step-by-step:")
print()

# Show calculations
acc_cancer = (tp_cancer + tn_cancer) / total_cancer
prec_cancer = tp_cancer / (tp_cancer + fp_cancer)
rec_cancer = tp_cancer / (tp_cancer + fn_cancer)
spec_cancer = tn_cancer / (tn_cancer + fp_cancer)
f1_cancer = 2 * (prec_cancer * rec_cancer) / (prec_cancer + rec_cancer)

print(f"1. Accuracy   = (TP + TN) / Total")
print(f"              = ({tp_cancer} + {tn_cancer}) / {total_cancer}")
print(f"              = {acc_cancer:.3f} or {acc_cancer*100:.1f}%")
print()

print(f"2. Precision  = TP / (TP + FP)")
print(f"              = {tp_cancer} / ({tp_cancer} + {fp_cancer})")
print(f"              = {prec_cancer:.3f} or {prec_cancer*100:.1f}%")
print(f"   → Of predicted cancer, {prec_cancer*100:.1f}% actually had cancer")
print()

print(f"3. Recall     = TP / (TP + FN)")
print(f"              = {tp_cancer} / ({tp_cancer} + {fn_cancer})")
print(f"              = {rec_cancer:.3f} or {rec_cancer*100:.1f}%")
print(f"   → We caught {rec_cancer*100:.1f}% of all cancer cases")
print()

print(f"4. Specificity = TN / (TN + FP)")
print(f"               = {tn_cancer} / ({tn_cancer} + {fp_cancer})")
print(f"               = {spec_cancer:.3f} or {spec_cancer*100:.1f}%")
print(f"   → {spec_cancer*100:.1f}% of healthy patients correctly identified")
print()

print(f"5. F1 Score   = 2 × (Precision × Recall) / (Precision + Recall)")
print(f"              = 2 × ({prec_cancer:.3f} × {rec_cancer:.3f}) / ({prec_cancer:.3f} + {rec_cancer:.3f})")
print(f"              = {f1_cancer:.3f}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Confusion Matrix Components
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Understanding the Confusion Matrix', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Basic confusion matrix structure
ax1 = axes[0, 0]
ax1.axis('off')

structure_text = """
CONFUSION MATRIX STRUCTURE
═══════════════════════════════════════════════════

                    PREDICTED
                Positive    Negative
              ┌───────────┬───────────┐
      ACTUAL  │           │           │
    Positive  │    TP     │    FN     │
              │  (True    │  (False   │
              │ Positive) │ Negative) │
              ├───────────┼───────────┤
      ACTUAL  │           │           │
    Negative  │    FP     │    TN     │
              │  (False   │   (True   │
              │ Positive) │ Negative) │
              └───────────┴───────────┘

THE FOUR OUTCOMES:

✓ TP (True Positive):
  Actual = Positive, Predicted = Positive
  CORRECT! Found what we were looking for.

✓ TN (True Negative):
  Actual = Negative, Predicted = Negative
  CORRECT! Correctly identified negative.

✗ FP (False Positive) - Type I Error:
  Actual = Negative, Predicted = Positive
  WRONG! False alarm.

✗ FN (False Negative) - Type II Error:
  Actual = Positive, Predicted = Negative
  WRONG! Missed it!

═══════════════════════════════════════════════════
"""

ax1.text(0.5, 0.5, structure_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

# Plot 2: Example confusion matrix heatmap
ax2 = axes[0, 1]
cm = np.array([[TP, FN], [FP, TN]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax2,
            xticklabels=['Predicted\nSpam', 'Predicted\nNot Spam'],
            yticklabels=['Actual\nSpam', 'Actual\nNot Spam'],
            annot_kws={"size": 16, "weight": "bold"})
ax2.set_title('Spam Detection Example\n(TP=35, FN=5, FP=10, TN=50)', fontsize=12, fontweight='bold')

# Add cell labels
ax2.text(0.5, 0.25, 'TP', ha='center', va='bottom', fontsize=10, color='red', weight='bold')
ax2.text(1.5, 0.25, 'FN', ha='center', va='bottom', fontsize=10, color='red', weight='bold')
ax2.text(0.5, 1.25, 'FP', ha='center', va='bottom', fontsize=10, color='red', weight='bold')
ax2.text(1.5, 1.25, 'TN', ha='center', va='bottom', fontsize=10, color='red', weight='bold')

# Plot 3: Metrics formulas
ax3 = axes[1, 0]
ax3.axis('off')

formulas_text = f"""
METRICS FROM CONFUSION MATRIX
═══════════════════════════════════════════════════

All metrics derive from TP, TN, FP, FN!

1. ACCURACY
   Formula: (TP + TN) / (TP + TN + FP + FN)
   Question: What % of ALL predictions were correct?
   Example: ({TP} + {TN}) / {total} = {accuracy:.1%}

2. PRECISION (Positive Predictive Value)
   Formula: TP / (TP + FP)
   Question: Of predicted POSITIVE, how many correct?
   Example: {TP} / ({TP} + {FP}) = {precision:.1%}

3. RECALL (Sensitivity, TPR)
   Formula: TP / (TP + FN)
   Question: Of actual POSITIVE, how many found?
   Example: {TP} / ({TP} + {FN}) = {recall:.1%}

4. SPECIFICITY (True Negative Rate)
   Formula: TN / (TN + FP)
   Question: Of actual NEGATIVE, how many correct?
   Example: {TN} / ({TN} + {FP}) = {specificity:.1%}

5. F1 SCORE
   Formula: 2 × (Prec × Rec) / (Prec + Rec)
   Question: What's the balance?
   Example: {f1_score:.3f}

═══════════════════════════════════════════════════
"""

ax3.text(0.5, 0.5, formulas_text,
        transform=ax3.transAxes,
        fontsize=9.5,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

# Plot 4: When to use which metric
ax4 = axes[1, 1]
ax4.axis('off')

guidance_text = """
WHICH METRIC TO USE?
═══════════════════════════════════════════════════

ACCURACY:
✓ Use when: Classes are balanced
✓ Example: Coin flip prediction (50/50 split)
✗ Don't use: Imbalanced data (rare disease)

PRECISION:
✓ Use when: False positives are costly
✓ Example: Spam filter (avoid blocking good email)
✓ Example: Loan approval (don't approve bad loans)

RECALL:
✓ Use when: False negatives are costly
✓ Example: Disease screening (can't miss cases)
✓ Example: Fraud detection (must catch fraud)
✓ Example: Security threats (can't miss attacks)

SPECIFICITY:
✓ Use when: True negatives matter
✓ Example: Showing healthy people are healthy

F1 SCORE:
✓ Use when: Need balance of precision & recall
✓ Use when: Classes are imbalanced
✓ Example: Text classification, search engines

PRECISION-RECALL TRADEOFF:
• High precision → Low recall (strict predictions)
• Low precision → High recall (lenient predictions)
• Can't maximize both! Choose based on problem.

═══════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, guidance_text,
        transform=ax4.transAxes,
        fontsize=9,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_confusion_matrix_explained.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_confusion_matrix_explained.png")
plt.close()

# Visualization 2: Metrics comparison for different scenarios
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparing Metrics Across Different Scenarios', fontsize=16, fontweight='bold', y=0.995)

# Create different scenarios
scenarios = {
    'Balanced\nGood Model': {'TP': 40, 'FN': 10, 'FP': 8, 'TN': 42},
    'Imbalanced\nLazy Model': {'TP': 0, 'FN': 10, 'FP': 0, 'TN': 990},
    'Imbalanced\nGood Model': {'TP': 8, 'FN': 2, 'FP': 50, 'TN': 940},
    'High Precision\nLow Recall': {'TP': 30, 'FN': 20, 'FP': 5, 'TN': 45},
    'Low Precision\nHigh Recall': {'TP': 48, 'FN': 2, 'FP': 25, 'TN': 25},
}

metrics_data = {'Scenario': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []}

for scenario_name, values in scenarios.items():
    tp, fn, fp, tn = values['TP'], values['FN'], values['FP'], values['TN']
    total = tp + fn + fp + tn

    acc = (tp + tn) / total
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    metrics_data['Scenario'].append(scenario_name)
    metrics_data['Accuracy'].append(acc)
    metrics_data['Precision'].append(prec)
    metrics_data['Recall'].append(rec)
    metrics_data['F1'].append(f1)

# Plot each metric
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
colors_metric = ['blue', 'green', 'orange', 'red']

for idx, (metric_name, color) in enumerate(zip(metrics_to_plot, colors_metric)):
    ax = axes[idx // 2, idx % 2]
    values = metrics_data[metric_name]
    scenarios_names = metrics_data['Scenario']

    bars = ax.bar(range(len(scenarios_names)), values, color=color, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(range(len(scenarios_names)))
    ax.set_xticklabels(scenarios_names, rotation=0, ha='center', fontsize=9)
    ax.set_ylabel(f'{metric_name} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_name} Across Scenarios', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/02_metrics_comparison_scenarios.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/02_metrics_comparison_scenarios.png")
plt.close()

# Visualization 3: Precision-Recall Tradeoff
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Precision-Recall Tradeoff', fontsize=16, fontweight='bold', y=1.02)

# Generate precision-recall curve for different thresholds
# Simulated probabilities and true labels
np.random.seed(42)
n_samples = 100
# 40 positive, 60 negative
y_true = np.array([1]*40 + [0]*60)
# Positive class gets higher scores on average
y_scores = np.concatenate([
    np.random.beta(8, 2, 40),  # Positive samples
    np.random.beta(2, 8, 60)   # Negative samples
])

thresholds = np.linspace(0, 1, 50)
precisions = []
recalls = []

for thresh in thresholds:
    y_pred = (y_scores >= thresh).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    precisions.append(prec)
    recalls.append(rec)

# Plot 1: Precision and Recall vs Threshold
ax1 = axes[0]
ax1.plot(thresholds, precisions, linewidth=3, color='blue', label='Precision', marker='o', markersize=4)
ax1.plot(thresholds, recalls, linewidth=3, color='green', label='Recall', marker='s', markersize=4)
ax1.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Precision and Recall vs Threshold', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12)
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Default threshold')

# Annotations
ax1.annotate('Low threshold\n→ High Recall\n→ Low Precision',
            xy=(0.2, 0.9), xytext=(0.1, 0.5),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax1.annotate('High threshold\n→ Low Recall\n→ High Precision',
            xy=(0.8, 0.9), xytext=(0.7, 0.5),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Plot 2: Precision-Recall curve
ax2 = axes[1]
ax2.plot(recalls, precisions, linewidth=3, color='purple', marker='o', markersize=4)
ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1.05)
ax2.set_ylim(0, 1.05)

# Highlight specific points
# Find balanced point (closest to precision=recall)
diffs = [abs(p - r) for p, r in zip(precisions, recalls)]
balanced_idx = diffs.index(min(diffs))
ax2.plot(recalls[balanced_idx], precisions[balanced_idx], 'ro', markersize=15, label='Balanced point')
ax2.annotate(f'Balanced\nP≈R≈{precisions[balanced_idx]:.2f}',
            xy=(recalls[balanced_idx], precisions[balanced_idx]),
            xytext=(recalls[balanced_idx]-0.2, precisions[balanced_idx]+0.15),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

ax2.legend(fontsize=12)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/03_precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/03_precision_recall_tradeoff.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🎯 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ CONFUSION MATRIX is the foundation of ALL classification metrics!")
print()
print("✓ THE FOUR OUTCOMES:")
print("   TP (True Positive)  - Correctly predicted positive")
print("   TN (True Negative)  - Correctly predicted negative")
print("   FP (False Positive) - Incorrectly predicted positive (Type I error)")
print("   FN (False Negative) - Incorrectly predicted negative (Type II error)")
print()
print("✓ KEY METRICS:")
print("   Accuracy   = (TP + TN) / Total     → Overall correctness")
print("   Precision  = TP / (TP + FP)        → Of predicted +, how many correct?")
print("   Recall     = TP / (TP + FN)        → Of actual +, how many found?")
print("   Specificity = TN / (TN + FP)       → Of actual -, how many correct?")
print("   F1 Score   = 2×(P×R)/(P+R)         → Harmonic mean of P and R")
print()
print("✓ WHEN TO USE WHICH:")
print("   Accuracy     → Balanced classes")
print("   Precision    → When false positives are costly")
print("   Recall       → When false negatives are costly")
print("   F1 Score     → When you need balance, or imbalanced data")
print()
print("✓ PRECISION-RECALL TRADEOFF:")
print("   Can't maximize both!")
print("   Low threshold → High Recall, Low Precision")
print("   High threshold → High Precision, Low Recall")
print()
print("✓ ACCURACY PARADOX:")
print("   High accuracy doesn't mean good model with imbalanced data!")
print()
print("NEXT: We'll learn about decision boundaries - visualizing how")
print("      classifiers separate classes in feature space!")
print()
print("=" * 80)
print("🎯 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
