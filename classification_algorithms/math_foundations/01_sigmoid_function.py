"""
📈 THE SIGMOID FUNCTION - Gateway to Classification

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. Why we need the sigmoid function for classification
2. How sigmoid converts any number to 0-1 (probability)
3. The mathematical formula and its shape
4. Relationship to logistic regression
5. Comparison with other activation functions
6. Why it's called "logistic" function

This is THE key to understanding classification!

================================================================================
📺 RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
⭐ ABSOLUTE MUST WATCH:
   - StatQuest: "Logistic Regression"
     https://www.youtube.com/watch?v=yIYKR4sgzI8
     (Best explanation of sigmoid and logistic regression!)

   - StatQuest: "Odds and Log(Odds) Clearly Explained"
     https://www.youtube.com/watch?v=ARfXDSkQf1Y
     (Understand the math behind sigmoid)

Also Recommended:
   - 3Blue1Brown: "But what is a neural network?"
     https://www.youtube.com/watch?v=aircAruvnKk
     (Shows sigmoid in context of neural networks)

================================================================================
OVERVIEW
================================================================================
The Problem:
- Linear regression gives us ANY number: -1000, 0.5, 237.8, etc.
- Classification needs probabilities: values between 0 and 1
- How do we convert unlimited range → 0 to 1?

The Solution: SIGMOID FUNCTION!
- Takes any input (−∞ to +∞)
- Outputs a probability (0 to 1)
- Smooth, differentiable (can use gradient descent!)
- S-shaped curve

This is the bridge from regression to classification!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/01_sigmoid/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("📈 THE SIGMOID FUNCTION")
print("   Converting Unlimited Numbers to Probabilities")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE PROBLEM - WHY WE NEED SIGMOID
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: The Problem with Linear Models for Classification")
print("=" * 80)
print()

print("SCENARIO: Email Spam Detection")
print("-" * 70)
print("We want to predict: Is email spam? (Yes=1, No=0)")
print()
print("Using features like:")
print("  • Number of exclamation marks")
print("  • Contains word 'free'")
print("  • Sender is unknown")
print()

print("ATTEMPT 1: Use Linear Regression")
print("-" * 70)
print("Linear model: y = β₀ + β₁×(exclamation_marks) + β₂×(has_free) + ...")
print()

# Example predictions from linear model
linear_predictions = np.array([-0.5, 0.2, 0.8, 1.3, 2.1])
print("Example predictions from linear model:")
print(f"{'Input':<15} {'Linear Output':<20} {'Problem'}")
print("-" * 65)
for i, pred in enumerate(linear_predictions):
    if pred < 0:
        problem = "Negative! Can't be a probability"
    elif pred > 1:
        problem = "Greater than 1! Invalid probability"
    else:
        problem = "OK (but rare)"
    print(f"Email {i+1:<9} {pred:<20.2f} {problem}")

print()
print("❌ PROBLEM: Linear models give us ANY number!")
print("   But we need probabilities: 0 ≤ P ≤ 1")
print()

print("WHAT WE NEED:")
print("-" * 70)
print("A function that:")
print("  ✓ Takes ANY input (−∞ to +∞)")
print("  ✓ Always outputs between 0 and 1")
print("  ✓ Smooth (differentiable for gradient descent)")
print("  ✓ Interpretable as probability")
print()
print("SOLUTION: The Sigmoid Function! 📈")
print()

# ============================================================================
# SECTION 2: THE SIGMOID FUNCTION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: The Sigmoid (Logistic) Function")
print("=" * 80)
print()

print("THE FORMULA:")
print("-" * 70)
print("  σ(z) = 1 / (1 + e^(-z))")
print()
print("Where:")
print("  • σ (sigma) = sigmoid function")
print("  • z = input (any real number)")
print("  • e = Euler's number ≈ 2.71828")
print("  • e^(-z) = exponential function")
print()

print("ALTERNATIVE FORMS (same thing!):")
print("-" * 70)
print("  σ(z) = 1 / (1 + exp(-z))")
print("  σ(z) = exp(z) / (1 + exp(z))")
print()

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

print("KEY PROPERTIES:")
print("-" * 70)
print("1. OUTPUT RANGE: Always between 0 and 1")

# Test various inputs
test_inputs = [-10, -5, -2, 0, 2, 5, 10]
print(f"\n{'Input (z)':<12} {'σ(z)':<15} {'Interpretation'}")
print("-" * 60)
for z in test_inputs:
    sig_z = sigmoid(z)
    if sig_z < 0.3:
        interp = "Very unlikely (close to 0)"
    elif sig_z < 0.7:
        interp = "Uncertain (around 0.5)"
    else:
        interp = "Very likely (close to 1)"
    print(f"{z:<12} {sig_z:<15.6f} {interp}")

print()
print("2. SYMMETRIC around z=0: σ(0) = 0.5")
print(f"   σ(0) = {sigmoid(0):.6f} ✓")
print()

print("3. SATURATION:")
print(f"   • Large positive z → σ(z) ≈ 1")
print(f"     σ(10) = {sigmoid(10):.10f} ≈ 1")
print(f"   • Large negative z → σ(z) ≈ 0")
print(f"     σ(-10) = {sigmoid(-10):.10f} ≈ 0")
print()

print("4. SMOOTH S-CURVE: Gradual transition from 0 to 1")
print()

# ============================================================================
# SECTION 3: VISUALIZING SIGMOID
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Visualizing the Sigmoid Function")
print("=" * 80)
print()

print("📊 Generating Visualization 1: The Sigmoid Curve...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('📈 THE SIGMOID FUNCTION: Converting Numbers to Probabilities',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: The sigmoid curve
ax = axes[0, 0]
z = np.linspace(-10, 10, 500)
sig_z = sigmoid(z)

ax.plot(z, sig_z, 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='y = 0')
ax.axhline(y=1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='y = 1')
ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='y = 0.5 (threshold)')
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Annotate key points
ax.scatter([0], [0.5], s=200, color='green', zorder=5, edgecolor='black', linewidth=2)
ax.annotate('σ(0) = 0.5\n(Decision boundary)',
            xy=(0, 0.5), xytext=(2, 0.3),
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.scatter([2], [sigmoid(2)], s=100, color='blue', zorder=5)
ax.annotate(f'σ(2) = {sigmoid(2):.3f}\n(Likely positive)',
            xy=(2, sigmoid(2)), xytext=(4, sigmoid(2)+0.15),
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='blue', lw=1))

ax.scatter([-2], [sigmoid(-2)], s=100, color='orange', zorder=5)
ax.annotate(f'σ(-2) = {sigmoid(-2):.3f}\n(Likely negative)',
            xy=(-2, sigmoid(-2)), xytext=(-6, sigmoid(-2)-0.15),
            fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='orange', lw=1))

ax.set_xlabel('Input (z)', fontsize=11)
ax.set_ylabel('Output σ(z)', fontsize=11)
ax.set_title('The Sigmoid Curve\n(S-shaped)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 10)
ax.set_ylim(-0.1, 1.1)

# Plot 2: Comparison with linear
ax = axes[0, 1]
z = np.linspace(-5, 5, 200)

# Linear (clipped)
linear = z / 10 + 0.5
linear_clipped = np.clip(linear, 0, 1)

ax.plot(z, sigmoid(z), 'b-', linewidth=3, label='Sigmoid (smooth)', zorder=3)
ax.plot(z, linear_clipped, 'r--', linewidth=2, label='Linear (clipped)', alpha=0.7)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Input (z)', fontsize=11)
ax.set_ylabel('Output', fontsize=11)
ax.set_title('Sigmoid vs Linear\n(Why Sigmoid is Better)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

ax.text(0, -0.05, 'Sigmoid is SMOOTH\n→ Gradient descent works!',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# Plot 3: The exponential component
ax = axes[1, 0]
z = np.linspace(-5, 5, 200)

# Show e^(-z)
exp_neg_z = np.exp(-z)
denominator = 1 + exp_neg_z

ax.plot(z, exp_neg_z, 'r-', linewidth=2, label='e^(-z)', alpha=0.7)
ax.plot(z, denominator, 'g-', linewidth=2, label='1 + e^(-z)', alpha=0.7)
ax.plot(z, sigmoid(z), 'b-', linewidth=3, label='σ(z) = 1/(1+e^(-z))')

ax.set_xlabel('Input (z)', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Breaking Down the Formula', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.5, 5)

# Plot 4: Key facts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'SIGMOID FUNCTION: KEY FACTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

facts = [
    "📐 FORMULA:",
    "   σ(z) = 1 / (1 + e^(-z))",
    "",
    "📊 OUTPUT RANGE:",
    "   Always: 0 < σ(z) < 1",
    "   Never exactly 0 or 1!",
    "",
    "🎯 KEY POINTS:",
    "   • σ(0) = 0.5",
    "   • σ(z) + σ(-z) = 1",
    "   • σ(−∞) → 0",
    "   • σ(+∞) → 1",
    "",
    "💡 INTERPRETATION:",
    "   Output is PROBABILITY!",
    "   • σ(z) = 0.7 means 70% confident",
    "   • σ(z) = 0.3 means 30% confident",
    "",
    "✅ WHY IT WORKS:",
    "   • Smooth curve (differentiable)",
    "   • Bounded (0 to 1)",
    "   • Monotonic (always increasing)",
    "   • Easy to interpret",
]

y_pos = 0.87
for line in facts:
    if line.startswith(('📐', '📊', '🎯', '💡', '✅')):
        weight = 'bold'
        size = 9.5
    elif line.startswith('•'):
        weight = 'normal'
        size = 8.5
    else:
        weight = 'normal'
        size = 9
    ax.text(0.5, y_pos, line, fontsize=size, ha='center',
            transform=ax.transAxes, family='monospace', fontweight=weight)
    y_pos -= 0.037

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_sigmoid_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 01_sigmoid_curve.png")
print()

# ============================================================================
# SECTION 4: THE DERIVATIVE OF SIGMOID
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: The Derivative - Why Sigmoid is Perfect for ML")
print("=" * 80)
print()

print("REMEMBER GRADIENT DESCENT?")
print("-" * 70)
print("We need derivatives to optimize!")
print()

print("SIGMOID DERIVATIVE:")
print("-" * 70)
print("  σ'(z) = σ(z) × (1 - σ(z))")
print()
print("Amazing property: Derivative is in terms of sigmoid itself!")
print("  If σ(z) = 0.7, then σ'(z) = 0.7 × 0.3 = 0.21")
print()

def sigmoid_derivative(z):
    """Derivative of sigmoid"""
    sig = sigmoid(z)
    return sig * (1 - sig)

print("EXAMPLES:")
print(f"{'z':<10} {'σ(z)':<15} {'σ'(z)':<15} {'Gradient'}")
print("-" * 60)
test_z = [-5, -2, 0, 2, 5]
for z in test_z:
    sig = sigmoid(z)
    deriv = sigmoid_derivative(z)
    if abs(deriv) < 0.1:
        grad_str = "Small (saturated)"
    else:
        grad_str = "Large (steep region)"
    print(f"{z:<10} {sig:<15.4f} {deriv:<15.4f} {grad_str}")

print()
print("KEY OBSERVATION:")
print("  • Largest derivative at z=0 (steepest part of curve)")
print("  • Derivative approaches 0 at extremes (flat regions)")
print("  • This is why sigmoid saturates!")
print()

# ============================================================================
# SECTION 5: USING SIGMOID IN LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Sigmoid in Logistic Regression")
print("=" * 80)
print()

print("COMBINING LINEAR MODEL + SIGMOID:")
print("-" * 70)
print("1. Linear part: z = β₀ + β₁x₁ + β₂x₂ + ...")
print("   (This can be any value)")
print()
print("2. Sigmoid transformation: P(y=1) = σ(z)")
print("   (Converts to probability)")
print()
print("Complete: P(y=1|x) = σ(β₀ + β₁x₁ + β₂x₂ + ...)")
print()

print("EXAMPLE: Email Spam Detection")
print("-" * 70)

# Simulate email features
beta_0 = -2.0  # Intercept
beta_1 = 1.5   # Coefficient for exclamation marks
beta_2 = 2.0   # Coefficient for contains 'free'

print(f"Model: P(spam) = σ({beta_0} + {beta_1}×exclamation + {beta_2}×has_free)")
print()

emails = [
    ("Normal email", 0, 0),
    ("Promotional", 1, 1),
    ("Suspicious", 3, 1),
    ("Very spammy", 5, 1),
]

print(f"{'Email':<20} {'Exclam!':<10} {'Has Free':<10} {'z':<10} {'P(spam)':<12} {'Prediction'}")
print("-" * 85)

for name, exclam, has_free in emails:
    z = beta_0 + beta_1 * exclam + beta_2 * has_free
    prob = sigmoid(z)
    pred = "SPAM" if prob > 0.5 else "NOT SPAM"
    print(f"{name:<20} {exclam:<10} {has_free:<10} {z:<10.2f} {prob:<12.4f} {pred}")

print()
print("✅ Sigmoid converts model output to probability!")
print("   We can set threshold (usually 0.5) to make decisions")
print()

# ============================================================================
# VISUALIZATION 2: Sigmoid in Action
# ============================================================================
print("📊 Generating Visualization 2: Sigmoid in Logistic Regression...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎯 SIGMOID IN LOGISTIC REGRESSION',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Linear vs Logistic
ax = axes[0, 0]

# Generate data
np.random.seed(42)
x_data = np.random.randn(100) * 2
y_data = (x_data + np.random.randn(100) * 0.5 > 0).astype(int)

# Linear regression attempt
linear_pred = 0.5 + 0.2 * x_data
linear_pred_clipped = np.clip(linear_pred, 0, 1)

# Logistic regression
z_logistic = 0.5 * x_data
logistic_pred = sigmoid(z_logistic)

# Plot data
ax.scatter(x_data[y_data==0], y_data[y_data==0], alpha=0.6, s=50, label='Class 0', color='blue')
ax.scatter(x_data[y_data==1], y_data[y_data==1], alpha=0.6, s=50, label='Class 1', color='red')

# Plot predictions
x_range = np.linspace(-5, 5, 200)
z_range = 0.5 * x_range
ax.plot(x_range, sigmoid(z_range), 'g-', linewidth=3, label='Logistic (sigmoid)', zorder=3)

ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
ax.set_xlabel('Feature (x)', fontsize=11)
ax.set_ylabel('P(y=1)', fontsize=11)
ax.set_title('Classification with Sigmoid', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.1, 1.1)

# Plot 2: Decision boundary
ax = axes[0, 1]

# 2D decision boundary visualization
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)

# Decision function: z = 0.5*x1 + 0.8*x2 - 1
Z = 0.5*X1 + 0.8*X2 - 1
P = sigmoid(Z)

# Plot contours
contour = ax.contourf(X1, X2, P, levels=20, cmap='RdYlBu_r', alpha=0.6)
ax.contour(X1, X2, P, levels=[0.5], colors='black', linewidths=3, label='Decision boundary')

# Sample points
np.random.seed(42)
n_points = 50
class_0_x1 = np.random.randn(n_points) - 1
class_0_x2 = np.random.randn(n_points) - 1
class_1_x1 = np.random.randn(n_points) + 1
class_1_x2 = np.random.randn(n_points) + 1

ax.scatter(class_0_x1, class_0_x2, s=50, c='blue', edgecolor='black', linewidth=0.5, label='Class 0', alpha=0.8)
ax.scatter(class_1_x1, class_1_x2, s=50, c='red', edgecolor='black', linewidth=0.5, label='Class 1', alpha=0.8)

cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('P(y=1)', fontsize=10)

ax.set_xlabel('Feature 1', fontsize=11)
ax.set_ylabel('Feature 2', fontsize=11)
ax.set_title('2D Decision Boundary\n(Sigmoid Creates Smooth Probabilities)', fontsize=11, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 3: Sigmoid derivative
ax = axes[1, 0]

z = np.linspace(-6, 6, 200)
sig = sigmoid(z)
sig_deriv = sigmoid_derivative(z)

ax.plot(z, sig, 'b-', linewidth=3, label='σ(z)', alpha=0.7)
ax.plot(z, sig_deriv, 'r-', linewidth=3, label="σ'(z)")

ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Input (z)', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Sigmoid and Its Derivative', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(0, 0.3, "Max derivative\nat z=0",
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 4: Summary
ax = axes[1, 1]
ax.text(0.5, 0.95, 'SIGMOID SUMMARY', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

summary = [
    "🎯 PURPOSE:",
    "   Convert any number → probability",
    "",
    "📐 IN LOGISTIC REGRESSION:",
    "   P(y=1) = σ(β₀ + β₁x₁ + ... + βₙxₙ)",
    "",
    "✅ ADVANTAGES:",
    "   • Output always 0 to 1",
    "   • Smooth (differentiable)",
    "   • Easy derivative: σ'(z) = σ(z)(1-σ(z))",
    "   • Probabilistic interpretation",
    "",
    "📊 DECISION MAKING:",
    "   • If σ(z) > 0.5 → Predict class 1",
    "   • If σ(z) < 0.5 → Predict class 0",
    "   • Can adjust threshold for business needs",
    "",
    "🔑 KEY INSIGHT:",
    "   Logistic Regression =",
    "   Linear Regression + Sigmoid!",
    "",
    "Next: Learn how to train the model",
    "and find optimal β values!"
]

y_pos = 0.87
for line in summary:
    if line.startswith(('🎯', '📐', '✅', '📊', '🔑', 'Next')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center',
            transform=ax.transAxes, family='monospace', fontweight=weight)
    y_pos -= 0.037

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_sigmoid_in_action.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 02_sigmoid_in_action.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: The Sigmoid Function")
print("=" * 80)
print()

print("🎯 WHAT WE LEARNED:")
print("-" * 70)
print("1. THE PROBLEM:")
print("   Linear models give unlimited outputs")
print("   Classification needs probabilities (0 to 1)")
print()

print("2. THE SOLUTION:")
print("   σ(z) = 1 / (1 + e^(-z))")
print("   • Input: any number")
print("   • Output: 0 to 1")
print()

print("3. KEY PROPERTIES:")
print("   • S-shaped curve")
print("   • σ(0) = 0.5")
print("   • Symmetric")
print("   • Saturates at extremes")
print()

print("4. THE DERIVATIVE:")
print("   σ'(z) = σ(z) × (1 - σ(z))")
print("   • Easy to compute")
print("   • Maximum at z = 0")
print("   • Enables gradient descent")
print()

print("5. IN LOGISTIC REGRESSION:")
print("   P(y=1|x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)")
print("   • Linear model inside sigmoid")
print("   • Output is probability")
print("   • Threshold at 0.5 for decisions")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 01_sigmoid_curve.png")
print("✅ 02_sigmoid_in_action.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review visualizations - see the S-curve!")
print("   2. Watch StatQuest video on Logistic Regression")
print("   3. Experiment: Change the code, try different β values")
print("   4. Next: 02_probability_for_classification.py")
print()

print("💡 REMEMBER:")
print("   Sigmoid is the BRIDGE from regression to classification!")
print("   Master this, and logistic regression is easy!")
print()

print("=" * 80)
print("🎉 SIGMOID FUNCTION MASTERED!")
print("   You now understand the key to classification!")
print("=" * 80)
