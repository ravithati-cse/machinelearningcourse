"""
📊 PROBABILITY FOR CLASSIFICATION
==================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. How classification outputs probabilities, not just classes
2. Why 0.5 is the default decision threshold
3. How to interpret probability predictions
4. The relationship between odds and probability
5. Why we use log-odds in logistic regression

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Odds and Log(Odds), Clearly Explained"
   https://www.youtube.com/watch?v=ARfXDSkQf1Y
   Best explanation of odds and log-odds for ML

⭐ StatQuest: "Logistic Regression Details Pt1: Coefficients"
   https://www.youtube.com/watch?v=vN5cNN2-HWE
   Shows how probability connects to logistic regression

📚 Khan Academy: "Probability"
   https://www.youtube.com/watch?v=uzkc-qNVoOk
   Foundational probability concepts

TIME: 30-45 minutes
DIFFICULTY: Beginner
PREREQUISITES: 01_sigmoid_function.py

KEY CONCEPTS:
------------
- Probability: P(event) - value between 0 and 1
- Odds: P(event) / P(not event) - value from 0 to infinity
- Log-Odds: ln(odds) - value from -∞ to +∞
- Decision Threshold: cutoff for classification (usually 0.5)
- Class Prediction: Converting probability to discrete class
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / '02_probability'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📊 PROBABILITY FOR CLASSIFICATION")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: PROBABILITY BASICS FOR CLASSIFICATION
# ============================================================================

print("=" * 80)
print("SECTION 1: Probability in Classification")
print("=" * 80)
print()

print("In REGRESSION, we predict a number:")
print("   Example: House price = $245,000")
print()
print("In CLASSIFICATION, we predict a PROBABILITY:")
print("   Example: P(email is spam) = 0.87 (87% confident it's spam)")
print()
print("KEY INSIGHT: Classification models output probabilities FIRST,")
print("             then we convert to class labels (spam/not spam)")
print()

# Example probabilities
examples = [
    ("Email A", 0.95, "Very confident it's spam"),
    ("Email B", 0.73, "Probably spam"),
    ("Email C", 0.51, "Slightly more likely spam"),
    ("Email D", 0.50, "Completely uncertain"),
    ("Email E", 0.49, "Slightly more likely NOT spam"),
    ("Email F", 0.23, "Probably NOT spam"),
    ("Email G", 0.05, "Very confident it's NOT spam"),
]

print("Examples of probability predictions:")
print("-" * 70)
print(f"{'Email':<10} {'P(Spam)':<12} {'Interpretation'}")
print("-" * 70)
for email, prob, interpretation in examples:
    print(f"{email:<10} {prob:<12.2f} {interpretation}")
print()

print("PROBABILITY SCALE:")
print("   0.0 = 0% confident  → Definitely NOT spam")
print("   0.5 = 50% confident → Completely uncertain")
print("   1.0 = 100% confident → Definitely spam")
print()

# ============================================================================
# SECTION 2: DECISION THRESHOLDS
# ============================================================================

print("=" * 80)
print("SECTION 2: Converting Probabilities to Classes")
print("=" * 80)
print()

print("We need a THRESHOLD to convert probability → class:")
print()
print("DEFAULT THRESHOLD: 0.5")
print("   If P(spam) >= 0.5  → Predict 'Spam'")
print("   If P(spam) < 0.5   → Predict 'Not Spam'")
print()

# Apply threshold
threshold = 0.5
print(f"Applying threshold = {threshold}:")
print("-" * 70)
print(f"{'Email':<10} {'P(Spam)':<12} {'Prediction':<15} {'Reasoning'}")
print("-" * 70)
for email, prob, interpretation in examples:
    prediction = "SPAM" if prob >= threshold else "NOT SPAM"
    reasoning = f"P={prob:.2f} >= {threshold}" if prob >= threshold else f"P={prob:.2f} < {threshold}"
    print(f"{email:<10} {prob:<12.2f} {prediction:<15} {reasoning}")
print()

print("IMPORTANT: The threshold doesn't have to be 0.5!")
print()
print("Example - Conservative Spam Filter (threshold = 0.7):")
print("   Only mark as spam if 70%+ confident")
print("   Avoids false positives (real email marked as spam)")
print()
print("Example - Aggressive Spam Filter (threshold = 0.3):")
print("   Mark as spam if 30%+ confident")
print("   Catches more spam, but more false positives")
print()

# Show different thresholds
print("How threshold affects Email C (P = 0.51):")
print("-" * 50)
print(f"{'Threshold':<15} {'Prediction':<15} {'Reasoning'}")
print("-" * 50)
for thresh in [0.3, 0.5, 0.7, 0.9]:
    pred = "SPAM" if 0.51 >= thresh else "NOT SPAM"
    print(f"{thresh:<15.1f} {pred:<15} {'0.51 >= ' + str(thresh) if 0.51 >= thresh else '0.51 < ' + str(thresh)}")
print()

# ============================================================================
# SECTION 3: ODDS AND LOG-ODDS
# ============================================================================

print("=" * 80)
print("SECTION 3: From Probability to Odds to Log-Odds")
print("=" * 80)
print()

print("WHY DO WE NEED ODDS?")
print("   Probability: 0 to 1 (limited range)")
print("   Odds: 0 to ∞ (unlimited on high end)")
print("   Log-Odds: -∞ to +∞ (unlimited both directions)")
print()
print("This is important for logistic regression!")
print()

print("FORMULAS:")
print("   Odds = P / (1 - P)")
print("   Log-Odds = ln(Odds) = ln(P / (1 - P))")
print()

# Calculate odds and log-odds for our examples
print("Converting Probabilities:")
print("-" * 80)
print(f"{'P(Spam)':<12} {'P(Not Spam)':<15} {'Odds':<15} {'Log-Odds':<15} {'Interpretation'}")
print("-" * 80)

prob_values = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
for p in prob_values:
    p_not = 1 - p
    odds = p / p_not
    log_odds = np.log(odds)

    if p < 0.5:
        interp = f"Favors NOT spam ({odds:.2f}:1 against)"
    elif p == 0.5:
        interp = "Completely uncertain (1:1 odds)"
    else:
        interp = f"Favors spam ({odds:.2f}:1 for)"

    print(f"{p:<12.2f} {p_not:<15.2f} {odds:<15.2f} {log_odds:<15.2f} {interp}")
print()

print("UNDERSTANDING ODDS:")
print("   Odds = 1     → Equal probability (50/50)")
print("   Odds = 2     → 2:1 in favor (67% probability)")
print("   Odds = 9     → 9:1 in favor (90% probability)")
print("   Odds = 0.5   → 1:2 against (33% probability)")
print()

print("UNDERSTANDING LOG-ODDS:")
print("   Log-Odds = 0       → P = 0.5 (uncertain)")
print("   Log-Odds > 0       → P > 0.5 (favors positive class)")
print("   Log-Odds < 0       → P < 0.5 (favors negative class)")
print("   Log-Odds = +2.2    → P ≈ 0.9 (90% confident)")
print("   Log-Odds = -2.2    → P ≈ 0.1 (10% confident)")
print()

print("WHY LOG-ODDS IN LOGISTIC REGRESSION?")
print("   Linear regression: y = β₀ + β₁x  (outputs any number)")
print("   Logistic regression: log-odds = β₀ + β₁x  (then convert to probability)")
print()
print("   log-odds can be any number (-∞ to +∞) ✓")
print("   Then sigmoid converts log-odds → probability (0 to 1)")
print()

# ============================================================================
# SECTION 4: SIGMOID CONNECTS LOG-ODDS TO PROBABILITY
# ============================================================================

print("=" * 80)
print("SECTION 4: The Connection - Sigmoid Function")
print("=" * 80)
print()

print("Remember sigmoid from the previous module:")
print("   σ(z) = 1 / (1 + e^(-z))")
print()
print("In logistic regression:")
print("   z = log-odds = β₀ + β₁x")
print("   P(y=1) = σ(z) = 1 / (1 + e^(-z))")
print()
print("So sigmoid transforms: log-odds → probability")
print()

# Show the transformation
log_odds_examples = [-3, -2, -1, 0, 1, 2, 3]
print("Examples:")
print("-" * 60)
print(f"{'Log-Odds (z)':<20} {'Sigmoid(z)':<20} {'Probability'}")
print("-" * 60)
for z in log_odds_examples:
    sigmoid_z = 1 / (1 + np.exp(-z))
    print(f"{z:<20} {sigmoid_z:<20.4f} {sigmoid_z*100:.1f}%")
print()

print("INTERPRETATION:")
print("   Logistic regression gives us log-odds")
print("   Sigmoid converts log-odds to probability")
print("   Threshold converts probability to class")
print()
print("Full Pipeline:")
print("   Features (x) → Linear model (β₀ + β₁x) → Log-Odds (z)")
print("   → Sigmoid(z) → Probability → Threshold → Class")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Probability Scale and Decision Threshold
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Understanding Probability in Classification', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Probability scale
ax1 = axes[0, 0]
probs = np.array([ex[1] for ex in examples])
labels = [ex[0] for ex in examples]

colors = ['red' if p < 0.5 else 'green' for p in probs]
ax1.scatter(probs, np.zeros_like(probs), c=colors, s=200, alpha=0.6, edgecolors='black', linewidth=2)
ax1.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold = 0.5')

for i, (prob, label) in enumerate(zip(probs, labels)):
    ax1.annotate(label, (prob, 0), xytext=(0, 20 if i % 2 == 0 else -30),
                textcoords='offset points', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow' if prob == 0.5 else 'lightgray', alpha=0.7),
                fontsize=9)

ax1.set_xlim(-0.05, 1.05)
ax1.set_ylim(-0.1, 0.1)
ax1.set_xlabel('Probability P(Spam)', fontsize=12, fontweight='bold')
ax1.set_title('Probability Scale: Red = Not Spam, Green = Spam', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yticks([])

# Add regions
ax1.axvspan(0, 0.5, alpha=0.1, color='red', label='Predict: NOT Spam')
ax1.axvspan(0.5, 1, alpha=0.1, color='green', label='Predict: Spam')
ax1.legend(loc='upper left')

# Plot 2: Different thresholds
ax2 = axes[0, 1]
thresholds = [0.3, 0.5, 0.7]
test_probs = np.linspace(0, 1, 100)

for thresh in thresholds:
    predictions = (test_probs >= thresh).astype(int)
    ax2.plot(test_probs, predictions, label=f'Threshold = {thresh}', linewidth=3)

ax2.set_xlabel('Probability P(Spam)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Predicted Class (0=Not Spam, 1=Spam)', fontsize=12, fontweight='bold')
ax2.set_title('How Threshold Affects Predictions', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(-0.1, 1.1)

# Add annotation
ax2.annotate('Lower threshold\n→ More spam detected\n→ More false positives',
            xy=(0.4, 1), xytext=(0.15, 0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax2.annotate('Higher threshold\n→ Less spam detected\n→ Fewer false positives',
            xy=(0.75, 0), xytext=(0.85, 0.3),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Plot 3: Probability to Odds transformation
ax3 = axes[1, 0]
prob_range = np.linspace(0.01, 0.99, 100)
odds_range = prob_range / (1 - prob_range)

ax3.plot(prob_range, odds_range, linewidth=3, color='purple')
ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Odds = 1 (P=0.5)')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2)

ax3.set_xlabel('Probability', fontsize=12, fontweight='bold')
ax3.set_ylabel('Odds', fontsize=12, fontweight='bold')
ax3.set_title('Probability → Odds Transformation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Add annotations
ax3.annotate('P=0.5 → Odds=1\n(50/50 chance)',
            xy=(0.5, 1), xytext=(0.3, 5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax3.annotate('P=0.9 → Odds=9\n(9:1 in favor)',
            xy=(0.9, 9), xytext=(0.7, 12),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

# Plot 4: Probability to Log-Odds transformation
ax4 = axes[1, 1]
log_odds_range = np.log(odds_range)

ax4.plot(prob_range, log_odds_range, linewidth=3, color='orange')
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Log-Odds = 0 (P=0.5)')
ax4.axvline(x=0.5, color='red', linestyle='--', linewidth=2)

ax4.set_xlabel('Probability', fontsize=12, fontweight='bold')
ax4.set_ylabel('Log-Odds', fontsize=12, fontweight='bold')
ax4.set_title('Probability → Log-Odds Transformation', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Add annotations
ax4.annotate('P=0.5 → Log-Odds=0\n(Uncertain)',
            xy=(0.5, 0), xytext=(0.25, 2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

ax4.annotate('P>0.5 → Log-Odds>0\n(Positive class)',
            xy=(0.75, np.log(3)), xytext=(0.8, 3),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_probability_thresholds_transformations.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_probability_thresholds_transformations.png")
plt.close()

# Visualization 2: Complete Pipeline from Log-Odds to Class
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('The Complete Classification Pipeline: Log-Odds → Probability → Class',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Linear model produces log-odds
ax1 = axes[0, 0]
x_values = np.linspace(-3, 3, 100)
# Simple linear model: log-odds = -1 + 2*x
log_odds_values = -1 + 2 * x_values

ax1.plot(x_values, log_odds_values, linewidth=3, color='blue', label='Log-Odds = β₀ + β₁x')
ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Log-Odds = 0')
ax1.set_xlabel('Feature Value (x)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Log-Odds', fontsize=12, fontweight='bold')
ax1.set_title('Step 1: Linear Model → Log-Odds', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax1.annotate('Example: β₀=-1, β₁=2\nLog-Odds = -1 + 2x',
            xy=(1.5, 2), xytext=(0, 4),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

# Plot 2: Sigmoid converts to probability
ax2 = axes[0, 1]
z_range = np.linspace(-6, 6, 100)
prob_from_sigmoid = 1 / (1 + np.exp(-z_range))

ax2.plot(z_range, prob_from_sigmoid, linewidth=3, color='green', label='P = σ(z)')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='P = 0.5')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='z = 0')
ax2.set_xlabel('Log-Odds (z)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Probability P(y=1)', fontsize=12, fontweight='bold')
ax2.set_title('Step 2: Sigmoid → Probability', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax2.annotate('Sigmoid transforms\nlog-odds to probability',
            xy=(0, 0.5), xytext=(3, 0.2),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

# Plot 3: Threshold creates classes
ax3 = axes[1, 0]
prob_test = np.linspace(0, 1, 100)
class_pred = (prob_test >= 0.5).astype(int)

ax3.plot(prob_test, class_pred, linewidth=3, color='purple')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
ax3.set_xlabel('Probability P(y=1)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Predicted Class', fontsize=12, fontweight='bold')
ax3.set_title('Step 3: Threshold → Class Prediction', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(-0.1, 1.1)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Class 0\n(Not Spam)', 'Class 1\n(Spam)'])

# Plot 4: Complete pipeline
ax4 = axes[1, 1]
ax4.axis('off')

# Create flowchart
pipeline_text = """
COMPLETE CLASSIFICATION PIPELINE
═══════════════════════════════════════════════════════

Input: Features (x)
    ↓
Linear Model: z = β₀ + β₁x₁ + β₂x₂ + ...
    ↓
Output: Log-Odds (z)   [-∞ to +∞]
    ↓
Sigmoid Function: P = 1 / (1 + e⁻ᶻ)
    ↓
Output: Probability P(y=1)   [0 to 1]
    ↓
Decision Rule: If P ≥ threshold → Class 1
               If P < threshold → Class 0
    ↓
Final Output: Class Prediction

═══════════════════════════════════════════════════════

EXAMPLE:
Features: [email_length=100, num_links=5]
Linear Model: z = -2 + 0.01×100 + 0.3×5 = -2 + 1 + 1.5 = 0.5
Sigmoid: P = 1/(1+e⁻⁰·⁵) = 0.62
Decision: 0.62 ≥ 0.5 → SPAM

═══════════════════════════════════════════════════════

KEY INSIGHTS:
• Linear model works with log-odds (unlimited range)
• Sigmoid converts to probability (0-1 range)
• Threshold converts to discrete class
• Default threshold is 0.5, but adjustable!
"""

ax4.text(0.5, 0.5, pipeline_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/02_complete_classification_pipeline.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/02_complete_classification_pipeline.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("📊 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ Classification outputs PROBABILITIES (0 to 1), not just classes")
print("✓ Threshold (usually 0.5) converts probability → class prediction")
print("✓ You can adjust threshold based on your application needs")
print()
print("✓ TRANSFORMATIONS:")
print("   Probability (0 to 1) → Odds (0 to ∞) → Log-Odds (-∞ to +∞)")
print()
print("✓ WHY THIS MATTERS:")
print("   - Linear model produces log-odds (can be any number)")
print("   - Sigmoid converts log-odds → probability (0 to 1)")
print("   - Threshold converts probability → class (0 or 1)")
print()
print("✓ FORMULAS TO REMEMBER:")
print("   Odds = P / (1 - P)")
print("   Log-Odds = ln(Odds)")
print("   Probability = Sigmoid(Log-Odds) = 1 / (1 + e^(-z))")
print()
print("NEXT: We'll learn about log-loss (the cost function for classification)")
print()
print("=" * 80)
print("📊 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
