"""
📉 LOG LOSS (BINARY CROSS-ENTROPY)
===================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. Why we can't use MSE for classification
2. What log loss (cross-entropy) is and how it works
3. How to calculate log loss manually
4. Why log loss penalizes confident wrong predictions heavily
5. The connection between log loss and maximum likelihood

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Cross Entropy"
   https://www.youtube.com/watch?v=6ArSys5qHAU
   THE best explanation of cross-entropy/log loss

⭐ StatQuest: "Logistic Regression Details Pt2: Maximum Likelihood"
   https://www.youtube.com/watch?v=BfKanl1aSG0
   Shows why we use log loss (maximum likelihood principle)

📚 3Blue1Brown: "Neural Networks Chapter 3"
   https://www.youtube.com/watch?v=tIeHLnjs5U8
   Beautiful visual explanation of cost functions

TIME: 45-60 minutes
DIFFICULTY: Intermediate
PREREQUISITES: 01_sigmoid_function.py, 02_probability_for_classification.py

KEY CONCEPTS:
------------
- Log Loss (Binary Cross-Entropy): Cost function for binary classification
- Why MSE doesn't work well for classification
- Log function penalizes confident wrong predictions
- Formula: -[y·log(p) + (1-y)·log(1-p)]
- Lower log loss = better predictions
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / '03_log_loss'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📉 LOG LOSS (BINARY CROSS-ENTROPY)")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHY NOT MSE FOR CLASSIFICATION?
# ============================================================================

print("=" * 80)
print("SECTION 1: Why MSE Doesn't Work Well for Classification")
print("=" * 80)
print()

print("In LINEAR REGRESSION, we used Mean Squared Error (MSE):")
print("   MSE = (1/n) × Σ(y_actual - y_predicted)²")
print()
print("Example for house prices:")
print("   Actual: $200,000")
print("   Predicted: $250,000")
print("   Error²: ($200,000 - $250,000)² = $2,500,000,000")
print("   This makes sense! Large errors are heavily penalized.")
print()

print("But for CLASSIFICATION, this doesn't work as well:")
print()
print("Example for spam classification:")
print("   Actual: 1 (spam)")
print("   Predicted Probability: 0.9 (90% confident it's spam)")
print()
print("   MSE approach:")
print("   Error² = (1 - 0.9)² = 0.01")
print()
print("   Different prediction:")
print("   Predicted Probability: 0.6 (60% confident it's spam)")
print("   Error² = (1 - 0.6)² = 0.16")
print()

print("PROBLEM with MSE for classification:")
print("   1. Doesn't account for the probabilistic nature")
print("   2. Gradient descent works poorly (flat gradients)")
print("   3. Doesn't heavily penalize confident WRONG predictions")
print()
print("We need a better cost function → LOG LOSS!")
print()

# ============================================================================
# SECTION 2: INTRODUCING LOG LOSS
# ============================================================================

print("=" * 80)
print("SECTION 2: Log Loss (Binary Cross-Entropy)")
print("=" * 80)
print()

print("LOG LOSS FORMULA:")
print("   For a single prediction:")
print("   L = -[y·log(p) + (1-y)·log(1-p)]")
print()
print("   Where:")
print("   • y = actual class (0 or 1)")
print("   • p = predicted probability (0 to 1)")
print("   • log = natural logarithm (ln)")
print()

print("Let's break this down by cases:")
print()
print("CASE 1: Actual class is 1 (y=1)")
print("   L = -[1·log(p) + 0·log(1-p)]")
print("   L = -log(p)")
print()
print("   If p=0.9 (confident and CORRECT):")
print(f"   L = -log(0.9) = {-np.log(0.9):.3f}  → Small loss ✓")
print()
print("   If p=0.1 (confident but WRONG):")
print(f"   L = -log(0.1) = {-np.log(0.1):.3f}  → Large loss! ✗")
print()

print("CASE 2: Actual class is 0 (y=0)")
print("   L = -[0·log(p) + 1·log(1-p)]")
print("   L = -log(1-p)")
print()
print("   If p=0.1 (confident and CORRECT):")
print(f"   L = -log(1-0.1) = -log(0.9) = {-np.log(0.9):.3f}  → Small loss ✓")
print()
print("   If p=0.9 (confident but WRONG):")
print(f"   L = -log(1-0.9) = -log(0.1) = {-np.log(0.1):.3f}  → Large loss! ✗")
print()

# ============================================================================
# SECTION 3: CALCULATING LOG LOSS - EXAMPLES
# ============================================================================

print("=" * 80)
print("SECTION 3: Calculating Log Loss - Step by Step")
print("=" * 80)
print()

def log_loss_single(y_actual, y_pred):
    """Calculate log loss for a single prediction"""
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    if y_actual == 1:
        return -np.log(y_pred)
    else:
        return -np.log(1 - y_pred)

# Example predictions
examples = [
    # (actual, predicted, description)
    (1, 0.99, "Actual=Spam, Predicted=0.99 (very confident, CORRECT)"),
    (1, 0.90, "Actual=Spam, Predicted=0.90 (confident, CORRECT)"),
    (1, 0.60, "Actual=Spam, Predicted=0.60 (somewhat confident, CORRECT)"),
    (1, 0.51, "Actual=Spam, Predicted=0.51 (barely confident, CORRECT)"),
    (1, 0.50, "Actual=Spam, Predicted=0.50 (uncertain)"),
    (1, 0.10, "Actual=Spam, Predicted=0.10 (confident, WRONG!)"),
    (1, 0.01, "Actual=Spam, Predicted=0.01 (very confident, WRONG!)"),
    (0, 0.01, "Actual=Not Spam, Predicted=0.01 (very confident, CORRECT)"),
    (0, 0.10, "Actual=Not Spam, Predicted=0.10 (confident, CORRECT)"),
    (0, 0.50, "Actual=Not Spam, Predicted=0.50 (uncertain)"),
    (0, 0.90, "Actual=Not Spam, Predicted=0.90 (confident, WRONG!)"),
    (0, 0.99, "Actual=Not Spam, Predicted=0.99 (very confident, WRONG!)"),
]

print("Calculating log loss for different predictions:")
print("-" * 85)
print(f"{'Actual':<10} {'Predicted':<12} {'Log Loss':<12} {'Description'}")
print("-" * 85)

for actual, pred, desc in examples:
    loss = log_loss_single(actual, pred)
    print(f"{actual:<10} {pred:<12.2f} {loss:<12.3f} {desc}")
print()

print("KEY INSIGHTS:")
print("   ✓ Correct confident predictions → Low loss (< 0.1)")
print("   ✓ Uncertain predictions (p≈0.5) → Medium loss (≈ 0.69)")
print("   ✗ Wrong confident predictions → High loss (> 2.0)")
print()
print("   The MORE confident you are when WRONG, the HIGHER the penalty!")
print()

# ============================================================================
# SECTION 4: LOG LOSS FOR MULTIPLE PREDICTIONS
# ============================================================================

print("=" * 80)
print("SECTION 4: Average Log Loss for Multiple Predictions")
print("=" * 80)
print()

print("For a dataset with n examples:")
print("   Average Log Loss = (1/n) × Σ Loss_i")
print()
print("   Where Loss_i = -[y_i·log(p_i) + (1-y_i)·log(1-p_i)]")
print()

# Example dataset
np.random.seed(42)
n_examples = 10
y_actual = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # True labels

# Good model predictions
y_pred_good = np.array([0.9, 0.1, 0.85, 0.95, 0.05, 0.88, 0.12, 0.08, 0.92, 0.15])

# Poor model predictions
y_pred_poor = np.array([0.6, 0.4, 0.55, 0.65, 0.45, 0.58, 0.48, 0.42, 0.62, 0.47])

# Calculate log loss for each example
print("Example Dataset (10 emails):")
print("-" * 80)
print(f"{'Email':<8} {'Actual':<10} {'Good Model':<15} {'Loss':<10} {'Poor Model':<15} {'Loss'}")
print("-" * 80)

losses_good = []
losses_poor = []

for i in range(n_examples):
    loss_good = log_loss_single(y_actual[i], y_pred_good[i])
    loss_poor = log_loss_single(y_actual[i], y_pred_poor[i])
    losses_good.append(loss_good)
    losses_poor.append(loss_poor)

    print(f"{i+1:<8} {y_actual[i]:<10} {y_pred_good[i]:<15.2f} {loss_good:<10.3f} "
          f"{y_pred_poor[i]:<15.2f} {loss_poor:<10.3f}")

avg_loss_good = np.mean(losses_good)
avg_loss_poor = np.mean(losses_poor)

print("-" * 80)
print(f"{'AVERAGE:':<8} {'':<10} {'':<15} {avg_loss_good:<10.3f} {'':<15} {avg_loss_poor:<10.3f}")
print()

print("INTERPRETATION:")
print(f"   Good Model: Average Log Loss = {avg_loss_good:.3f}")
print("   → Low loss means good predictions!")
print()
print(f"   Poor Model: Average Log Loss = {avg_loss_poor:.3f}")
print("   → High loss means poor predictions")
print()
print("GOAL: Minimize average log loss during training!")
print()

# ============================================================================
# SECTION 5: WHY DOES LOG WORK THIS WAY?
# ============================================================================

print("=" * 80)
print("SECTION 5: Understanding the Logarithm")
print("=" * 80)
print()

print("The natural logarithm (ln or log) has special properties:")
print()
print("   log(1) = 0")
print("   log(0.5) ≈ -0.69")
print("   log(0.1) ≈ -2.30")
print("   log(0.01) ≈ -4.61")
print("   log(0) = -∞ (undefined, but approaches -∞)")
print()

print("When we use -log(p):")
print("   -log(1) = 0      → Perfect prediction, zero loss")
print("   -log(0.9) ≈ 0.11  → Good prediction, small loss")
print("   -log(0.5) ≈ 0.69  → Uncertain, medium loss")
print("   -log(0.1) ≈ 2.30  → Bad prediction, large loss")
print("   -log(0.01) ≈ 4.61 → Very bad, very large loss")
print("   -log(0) = ∞      → Impossible, infinite loss")
print()

print("This creates the property we want:")
print("   • Correct and confident → Small loss")
print("   • Uncertain → Medium loss")
print("   • Wrong and confident → Large loss (grows rapidly!)")
print()

# Show log values
prob_values = [1.0, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
print("How -log(p) grows:")
print("-" * 50)
print(f"{'Probability':<15} {'-log(p)':<15} {'Interpretation'}")
print("-" * 50)
for p in prob_values:
    if p == 0:
        log_val = "∞"
        interp = "Infinite penalty"
    else:
        log_val = f"{-np.log(p):.3f}"
        if p >= 0.9:
            interp = "Small loss ✓"
        elif p >= 0.5:
            interp = "Medium loss"
        else:
            interp = "Large loss ✗"
    print(f"{p:<15.3f} {log_val:<15} {interp}")
print()

# ============================================================================
# SECTION 6: LOG LOSS VS MSE COMPARISON
# ============================================================================

print("=" * 80)
print("SECTION 6: Log Loss vs MSE - Side by Side")
print("=" * 80)
print()

print("For a positive class (y=1), comparing cost functions:")
print("-" * 70)
print(f"{'Predicted P':<15} {'MSE':<15} {'Log Loss':<15} {'Which is better?'}")
print("-" * 70)

test_probs = [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]
for p in test_probs:
    mse = (1 - p) ** 2
    logloss = -np.log(p)

    if p >= 0.7:
        comment = "Both low (good prediction)"
    elif p >= 0.5:
        comment = "Log loss penalizes more"
    else:
        comment = "Log loss MUCH higher ✓"

    print(f"{p:<15.2f} {mse:<15.3f} {logloss:<15.3f} {comment}")
print()

print("ADVANTAGES OF LOG LOSS over MSE for classification:")
print("   1. Heavily penalizes confident wrong predictions")
print("   2. Better gradient descent properties (steeper gradients)")
print("   3. Derived from maximum likelihood principle")
print("   4. Probabilistic interpretation")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Log Loss Curves
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Understanding Log Loss (Binary Cross-Entropy)', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Log loss when actual=1
ax1 = axes[0, 0]
p_range = np.linspace(0.01, 0.99, 100)
loss_y1 = -np.log(p_range)  # When y=1

ax1.plot(p_range, loss_y1, linewidth=3, color='blue', label='Log Loss when y=1')
ax1.fill_between(p_range, 0, loss_y1, alpha=0.3, color='blue')
ax1.set_xlabel('Predicted Probability P(y=1)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
ax1.set_title('Log Loss When Actual Class = 1 (Positive)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 5)

# Add annotations
ax1.annotate('Perfect prediction\nP=1.0 → Loss=0',
            xy=(0.99, 0.01), xytext=(0.7, 1),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax1.annotate('Wrong & confident\nP=0.1 → Loss≈2.3',
            xy=(0.1, 2.3), xytext=(0.3, 3.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

ax1.axvspan(0.8, 1.0, alpha=0.2, color='green', label='Good predictions')
ax1.axvspan(0.0, 0.2, alpha=0.2, color='red', label='Bad predictions')
ax1.legend()

# Plot 2: Log loss when actual=0
ax2 = axes[0, 1]
loss_y0 = -np.log(1 - p_range)  # When y=0

ax2.plot(p_range, loss_y0, linewidth=3, color='red', label='Log Loss when y=0')
ax2.fill_between(p_range, 0, loss_y0, alpha=0.3, color='red')
ax2.set_xlabel('Predicted Probability P(y=1)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
ax2.set_title('Log Loss When Actual Class = 0 (Negative)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 5)

# Add annotations
ax2.annotate('Perfect prediction\nP=0.0 → Loss=0',
            xy=(0.01, 0.01), xytext=(0.3, 1),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

ax2.annotate('Wrong & confident\nP=0.9 → Loss≈2.3',
            xy=(0.9, 2.3), xytext=(0.6, 3.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))

ax2.axvspan(0.0, 0.2, alpha=0.2, color='green', label='Good predictions')
ax2.axvspan(0.8, 1.0, alpha=0.2, color='red', label='Bad predictions')
ax2.legend()

# Plot 3: Log Loss vs MSE comparison
ax3 = axes[1, 0]
mse_loss = (1 - p_range) ** 2  # MSE when y=1
log_loss_comp = -np.log(p_range)  # Log loss when y=1

ax3.plot(p_range, mse_loss, linewidth=3, color='orange', label='MSE (squared error)')
ax3.plot(p_range, log_loss_comp, linewidth=3, color='blue', label='Log Loss')
ax3.set_xlabel('Predicted Probability (when y=1)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Loss Value', fontsize=12, fontweight='bold')
ax3.set_title('Log Loss vs MSE Comparison (y=1)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(0, 5)

# Add annotation
ax3.annotate('Log loss penalizes\nwrong predictions\nMUCH more heavily',
            xy=(0.15, 1.9), xytext=(0.4, 4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Plot 4: The -log function
ax4 = axes[1, 1]
x_log = np.linspace(0.01, 1, 100)
y_log = -np.log(x_log)

ax4.plot(x_log, y_log, linewidth=3, color='purple')
ax4.set_xlabel('x', fontsize=12, fontweight='bold')
ax4.set_ylabel('-log(x)', fontsize=12, fontweight='bold')
ax4.set_title('The Logarithm Function: y = -log(x)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_ylim(0, 5)

# Add key points
key_points = [(1, 0), (0.9, -np.log(0.9)), (0.5, -np.log(0.5)), (0.1, -np.log(0.1))]
for x, y in key_points:
    ax4.plot(x, y, 'ro', markersize=10)
    ax4.annotate(f'({x:.1f}, {y:.2f})',
                xy=(x, y), xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=9)

ax4.annotate('Asymptote:\nAs x→0, -log(x)→∞',
            xy=(0.05, 3), xytext=(0.3, 4),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_log_loss_curves.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_log_loss_curves.png")
plt.close()

# Visualization 2: Comparing different models
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Log Loss: Comparing Model Performance', fontsize=16, fontweight='bold', y=0.995)

# Create sample predictions for 3 models
n_samples = 20
np.random.seed(42)
true_labels = np.random.randint(0, 2, n_samples)

# Perfect model
perfect_model = true_labels.astype(float)

# Good model (some uncertainty)
good_model = np.where(true_labels == 1,
                     np.random.uniform(0.7, 0.95, n_samples),
                     np.random.uniform(0.05, 0.3, n_samples))

# Poor model (random)
poor_model = np.random.uniform(0.3, 0.7, n_samples)

# Plot 1: Perfect Model
ax1 = axes[0, 0]
indices = np.arange(n_samples)
colors_true = ['red' if label == 0 else 'green' for label in true_labels]

ax1.scatter(indices, true_labels, c=colors_true, s=200, alpha=0.3, label='True Labels', marker='s', edgecolors='black', linewidth=2)
ax1.scatter(indices, perfect_model, c='blue', s=100, alpha=0.8, label='Predictions', marker='o')
ax1.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_title('Perfect Model (Log Loss = 0.0)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Calculate and show loss
perfect_losses = [log_loss_single(true_labels[i], np.clip(perfect_model[i], 1e-15, 1-1e-15)) for i in range(n_samples)]
avg_perfect = np.mean(perfect_losses)
ax1.text(0.5, 0.95, f'Avg Log Loss: {avg_perfect:.4f}',
        transform=ax1.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 2: Good Model
ax2 = axes[0, 1]
ax2.scatter(indices, true_labels, c=colors_true, s=200, alpha=0.3, label='True Labels', marker='s', edgecolors='black', linewidth=2)
ax2.scatter(indices, good_model, c='blue', s=100, alpha=0.8, label='Predictions', marker='o')
ax2.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
ax2.set_title('Good Model', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.1, 1.1)

# Calculate and show loss
good_losses = [log_loss_single(true_labels[i], good_model[i]) for i in range(n_samples)]
avg_good = np.mean(good_losses)
ax2.text(0.5, 0.95, f'Avg Log Loss: {avg_good:.4f}',
        transform=ax2.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Plot 3: Poor Model
ax3 = axes[1, 0]
ax3.scatter(indices, true_labels, c=colors_true, s=200, alpha=0.3, label='True Labels', marker='s', edgecolors='black', linewidth=2)
ax3.scatter(indices, poor_model, c='blue', s=100, alpha=0.8, label='Predictions', marker='o')
ax3.set_xlabel('Sample Index', fontsize=12, fontweight='bold')
ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
ax3.set_title('Poor Model (Random Guessing)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-0.1, 1.1)

# Calculate and show loss
poor_losses = [log_loss_single(true_labels[i], poor_model[i]) for i in range(n_samples)]
avg_poor = np.mean(poor_losses)
ax3.text(0.5, 0.95, f'Avg Log Loss: {avg_poor:.4f}',
        transform=ax3.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

# Plot 4: Loss Comparison
ax4 = axes[1, 1]
models = ['Perfect\nModel', 'Good\nModel', 'Poor\nModel']
avg_losses = [avg_perfect, avg_good, avg_poor]
colors_bars = ['green', 'yellow', 'red']

bars = ax4.bar(models, avg_losses, color=colors_bars, alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Average Log Loss', fontsize=12, fontweight='bold')
ax4.set_title('Model Comparison: Lower is Better!', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, loss in zip(bars, avg_losses):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{loss:.4f}',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax4.axhline(y=0.693, color='purple', linestyle='--', linewidth=2, label='Random Guessing (≈0.693)')
ax4.legend()

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/02_model_comparison_log_loss.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/02_model_comparison_log_loss.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("📉 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ LOG LOSS (Binary Cross-Entropy) is the cost function for classification")
print()
print("✓ FORMULA:")
print("   L = -[y·log(p) + (1-y)·log(1-p)]")
print()
print("✓ WHY NOT MSE?")
print("   - MSE doesn't heavily penalize confident wrong predictions")
print("   - Log loss has better gradient descent properties")
print("   - Log loss has probabilistic interpretation")
print()
print("✓ KEY PROPERTIES:")
print("   - Perfect prediction (p=y) → Loss = 0")
print("   - Uncertain (p=0.5) → Loss ≈ 0.693")
print("   - Confident wrong prediction → Loss → ∞")
print()
print("✓ TRAINING GOAL:")
print("   Minimize average log loss across all training examples")
print()
print("✓ INTERPRETATION:")
print("   Lower log loss = Better model")
print("   Log loss of 0 = Perfect predictions")
print("   Log loss > 0.693 = Worse than random guessing")
print()
print("NEXT: We'll learn about the Confusion Matrix - the foundation of all")
print("      classification metrics!")
print()
print("=" * 80)
print("📉 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
