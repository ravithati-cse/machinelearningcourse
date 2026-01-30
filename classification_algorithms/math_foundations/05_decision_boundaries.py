"""
🎯 DECISION BOUNDARIES - Visualizing How Classifiers Separate Classes
======================================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. What a decision boundary is
2. How linear classifiers create straight line boundaries
3. How non-linear classifiers create curved boundaries
4. The relationship between decision boundary and model complexity
5. How to visualize decision boundaries in 2D feature space
6. Overfitting vs underfitting through decision boundaries

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Machine Learning Fundamentals: Decision Boundaries"
   (Part of various ML algorithm videos)

📚 Serrano.Academy: "A visual guide to classification"
   Beautiful visual explanations of decision boundaries

⭐ 3Blue1Brown: "Neural Networks Chapter 1"
   https://www.youtube.com/watch?v=aircAruvnKk
   Shows how boundaries emerge from learning

TIME: 45 minutes
DIFFICULTY: Intermediate
PREREQUISITES: 01_sigmoid_function.py, 02_probability_for_classification.py

KEY CONCEPTS:
------------
- Decision Boundary: Line/curve separating classes
- Linear Boundary: Straight line (logistic regression, linear SVM)
- Non-linear Boundary: Curved (polynomial, trees, neural networks)
- Model Complexity: Simple boundaries vs complex boundaries
- Overfitting: Boundary too complex, memorizes training data
- Underfitting: Boundary too simple, can't capture pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / '05_decision_boundaries'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎯 DECISION BOUNDARIES - Visualizing Classification")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHAT IS A DECISION BOUNDARY?
# ============================================================================

print("=" * 80)
print("SECTION 1: What is a Decision Boundary?")
print("=" * 80)
print()

print("A DECISION BOUNDARY is the line (or surface) that separates different classes.")
print()
print("In 2D (two features):")
print("   • Decision boundary is a LINE or CURVE")
print("   • One side: Class 0 (e.g., Not Spam)")
print("   • Other side: Class 1 (e.g., Spam)")
print()

print("Example: Email Classification")
print("   Feature 1 (x₁): Number of exclamation marks")
print("   Feature 2 (x₂): Number of suspicious links")
print()
print("   If x₁ + x₂ > 5  → Spam")
print("   If x₁ + x₂ ≤ 5  → Not Spam")
print()
print("   The boundary: x₁ + x₂ = 5 (a straight line)")
print()

print("WHY DECISION BOUNDARIES MATTER:")
print("   • Visualize what the model learned")
print("   • Understand model behavior")
print("   • Diagnose overfitting/underfitting")
print("   • Compare different models")
print()

# ============================================================================
# SECTION 2: LINEAR DECISION BOUNDARIES
# ============================================================================

print("=" * 80)
print("SECTION 2: Linear Decision Boundaries")
print("=" * 80)
print()

print("LINEAR CLASSIFIERS create STRAIGHT LINE boundaries:")
print()
print("Equation: β₀ + β₁x₁ + β₂x₂ = 0")
print()
print("   • Logistic Regression → Linear boundary")
print("   • Linear SVM → Linear boundary")
print("   • Perceptron → Linear boundary")
print()

print("Example with coefficients:")
print("   β₀ = -5, β₁ = 1, β₂ = 1")
print("   Decision boundary: -5 + x₁ + x₂ = 0")
print("   Simplified: x₁ + x₂ = 5")
print()

print("Points on the boundary:")
print("   (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)")
print()

# Example linear boundary
x1_boundary = np.linspace(0, 10, 100)
x2_boundary = 5 - x1_boundary  # From x1 + x2 = 5

print("Prediction for new points:")
print("-" * 60)
print(f"{'Point (x₁, x₂)':<20} {'x₁ + x₂':<12} {'Prediction'}")
print("-" * 60)

test_points = [(1, 1), (2, 4), (3, 3), (4, 5), (6, 2)]
for x1, x2 in test_points:
    sum_val = x1 + x2
    pred = "Spam" if sum_val > 5 else "Not Spam"
    print(f"({x1}, {x2})              {sum_val:<12} {pred}")
print()

print("PROPERTIES OF LINEAR BOUNDARIES:")
print("   ✓ Simple and interpretable")
print("   ✓ Fast to compute")
print("   ✓ Works well for linearly separable data")
print("   ✗ Can't capture complex patterns")
print("   ✗ Underfits if data is non-linear")
print()

# ============================================================================
# SECTION 3: NON-LINEAR DECISION BOUNDARIES
# ============================================================================

print("=" * 80)
print("SECTION 3: Non-Linear Decision Boundaries")
print("=" * 80)
print()

print("NON-LINEAR CLASSIFIERS create CURVED boundaries:")
print()
print("Methods:")
print("   • Polynomial features: x₁², x₁x₂, x₂² added")
print("   • Decision Trees: Rectangle boundaries")
print("   • Random Forests: Complex curved boundaries")
print("   • Neural Networks: Very flexible boundaries")
print("   • Kernel SVM: Curved boundaries")
print()

print("Example with polynomial features:")
print("   Original features: x₁, x₂")
print("   Add polynomial: x₁², x₁x₂, x₂²")
print("   Boundary: β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₁x₂ + β₅x₂² = 0")
print()
print("   This creates a CURVED boundary (parabola, circle, ellipse, etc.)")
print()

print("Example - Circular boundary:")
print("   Equation: (x₁ - 5)² + (x₂ - 5)² = 16")
print("   This is a circle centered at (5, 5) with radius 4")
print("   Inside circle: Class 0")
print("   Outside circle: Class 1")
print()

# ============================================================================
# SECTION 4: MODEL COMPLEXITY AND BOUNDARIES
# ============================================================================

print("=" * 80)
print("SECTION 4: Model Complexity and Decision Boundaries")
print("=" * 80)
print()

print("The complexity of the decision boundary reflects model complexity:")
print()

print("UNDERFITTING (Too Simple):")
print("   • Boundary is too simple")
print("   • Can't capture the pattern")
print("   • High training error, high test error")
print("   • Example: Linear boundary for circular data")
print()

print("GOOD FIT (Just Right):")
print("   • Boundary captures the true pattern")
print("   • Generalizes well to new data")
print("   • Low training error, low test error")
print("   • Example: Slightly curved boundary for slightly non-linear data")
print()

print("OVERFITTING (Too Complex):")
print("   • Boundary is too complex")
print("   • Memorizes training data, including noise")
print("   • Low training error, HIGH test error")
print("   • Example: Wiggly boundary that wraps around every training point")
print()

print("HOW TO CONTROL COMPLEXITY:")
print("   • Regularization (L1, L2)")
print("   • Limit polynomial degree")
print("   • Limit tree depth")
print("   • Dropout in neural networks")
print("   • Early stopping")
print()

# ============================================================================
# SECTION 5: CREATING SYNTHETIC DATA
# ============================================================================

print("=" * 80)
print("SECTION 5: Creating Example Datasets")
print("=" * 80)
print()

# Set random seed for reproducibility
np.random.seed(42)

# Dataset 1: Linearly separable
print("Dataset 1: Linearly Separable")
n_samples = 100
# Class 0: bottom-left
X_class0_linear = np.random.randn(n_samples//2, 2) * 0.8 + np.array([2, 2])
# Class 1: top-right
X_class1_linear = np.random.randn(n_samples//2, 2) * 0.8 + np.array([5, 5])
X_linear = np.vstack([X_class0_linear, X_class1_linear])
y_linear = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
print(f"   Created {n_samples} samples")
print(f"   Class 0: {(y_linear == 0).sum()} samples")
print(f"   Class 1: {(y_linear == 1).sum()} samples")
print("   Perfect for linear classifier!")
print()

# Dataset 2: Circular pattern (non-linear)
print("Dataset 2: Circular Pattern (Non-linear)")
n_samples = 200
# Class 0: inside circle
angle_inner = np.random.uniform(0, 2*np.pi, n_samples//2)
radius_inner = np.random.uniform(0, 2, n_samples//2)
X_class0_circle = np.column_stack([
    5 + radius_inner * np.cos(angle_inner),
    5 + radius_inner * np.sin(angle_inner)
])
# Class 1: outside circle
angle_outer = np.random.uniform(0, 2*np.pi, n_samples//2)
radius_outer = np.random.uniform(3, 5, n_samples//2)
X_class1_circle = np.column_stack([
    5 + radius_outer * np.cos(angle_outer),
    5 + radius_outer * np.sin(angle_outer)
])
X_circle = np.vstack([X_class0_circle, X_class1_circle])
y_circle = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
print(f"   Created {n_samples} samples")
print(f"   Class 0: Inside circle ({(y_circle == 0).sum()} samples)")
print(f"   Class 1: Outside circle ({(y_circle == 1).sum()} samples)")
print("   Requires non-linear classifier!")
print()

# Dataset 3: XOR pattern (very non-linear)
print("Dataset 3: XOR Pattern (Very Non-linear)")
n_samples = 200
# Class 0: top-left and bottom-right
X_class0_xor = np.vstack([
    np.random.randn(n_samples//4, 2) * 0.5 + np.array([2, 6]),
    np.random.randn(n_samples//4, 2) * 0.5 + np.array([6, 2])
])
# Class 1: top-right and bottom-left
X_class1_xor = np.vstack([
    np.random.randn(n_samples//4, 2) * 0.5 + np.array([6, 6]),
    np.random.randn(n_samples//4, 2) * 0.5 + np.array([2, 2])
])
X_xor = np.vstack([X_class0_xor, X_class1_xor])
y_xor = np.array([0]*(n_samples//2) + [1]*(n_samples//2))
print(f"   Created {n_samples} samples")
print(f"   Class 0: Diagonal corners ({(y_xor == 0).sum()} samples)")
print(f"   Class 1: Other diagonal ({(y_xor == 1).sum()} samples)")
print("   Very challenging! Needs complex boundary!")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['red', 'blue']

# Visualization 1: Three datasets with their ideal boundaries
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Decision Boundaries for Different Data Patterns', fontsize=16, fontweight='bold')

# Plot 1: Linear data
ax1 = axes[0]
ax1.scatter(X_linear[y_linear==0, 0], X_linear[y_linear==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax1.scatter(X_linear[y_linear==1, 0], X_linear[y_linear==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')

# Draw linear boundary
x_boundary = np.linspace(0, 8, 100)
y_boundary = x_boundary  # Simple diagonal line
ax1.plot(x_boundary, y_boundary, 'g-', linewidth=3, label='Decision Boundary')
ax1.fill_between(x_boundary, 0, y_boundary, alpha=0.1, color='red')
ax1.fill_between(x_boundary, y_boundary, 8, alpha=0.1, color='blue')

ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax1.set_title('Linearly Separable Data\n(Linear Boundary Works!)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 8)
ax1.set_ylim(0, 8)

# Plot 2: Circular data
ax2 = axes[1]
ax2.scatter(X_circle[y_circle==0, 0], X_circle[y_circle==0, 1],
           c='red', s=50, alpha=0.6, edgecolors='black', linewidth=1, label='Class 0 (Inside)')
ax2.scatter(X_circle[y_circle==1, 0], X_circle[y_circle==1, 1],
           c='blue', s=50, alpha=0.6, edgecolors='black', linewidth=1, label='Class 1 (Outside)')

# Draw circular boundary
theta = np.linspace(0, 2*np.pi, 100)
radius = 2.5
x_circle_bound = 5 + radius * np.cos(theta)
y_circle_bound = 5 + radius * np.sin(theta)
ax2.plot(x_circle_bound, y_circle_bound, 'g-', linewidth=3, label='Decision Boundary (Circle)')

ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax2.set_title('Circular Pattern\n(Needs Non-linear Boundary)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Plot 3: XOR data
ax3 = axes[2]
ax3.scatter(X_xor[y_xor==0, 0], X_xor[y_xor==0, 1],
           c='red', s=50, alpha=0.6, edgecolors='black', linewidth=1, label='Class 0')
ax3.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
           c='blue', s=50, alpha=0.6, edgecolors='black', linewidth=1, label='Class 1')

# Draw complex boundary (approximate)
ax3.plot([1, 4], [4, 1], 'g-', linewidth=3, label='Complex Boundary')
ax3.plot([4, 7], [7, 4], 'g-', linewidth=3)
ax3.plot([1, 4], [4, 7], 'g-', linewidth=3)
ax3.plot([4, 7], [1, 4], 'g-', linewidth=3)

ax3.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax3.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax3.set_title('XOR Pattern\n(Needs Very Complex Boundary)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 8)
ax3.set_ylim(0, 8)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_decision_boundaries_patterns.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_decision_boundaries_patterns.png")
plt.close()

# Visualization 2: Underfitting vs Good Fit vs Overfitting
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Complexity: Underfitting → Good Fit → Overfitting', fontsize=16, fontweight='bold')

# Generate slightly non-linear data
np.random.seed(42)
n = 60
X_demo = np.random.randn(n, 2) * 1.5
y_demo = (X_demo[:, 0]**2 + X_demo[:, 1]**2 > 2).astype(int)

# Plot 1: Underfitting (linear boundary)
ax1 = axes[0]
ax1.scatter(X_demo[y_demo==0, 0], X_demo[y_demo==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax1.scatter(X_demo[y_demo==1, 0], X_demo[y_demo==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')

# Linear boundary
x_line = np.linspace(-4, 4, 100)
y_line = x_line * 0.5  # Simple linear
ax1.plot(x_line, y_line, 'orange', linewidth=3, label='Linear Boundary')

ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax1.set_title('UNDERFITTING\n(Too Simple)', fontsize=12, fontweight='bold', color='orange')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-4, 4)
ax1.set_ylim(-4, 4)
ax1.text(0.5, 0.05, 'High Training Error\nHigh Test Error',
        transform=ax1.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Plot 2: Good fit (circular boundary)
ax2 = axes[1]
ax2.scatter(X_demo[y_demo==0, 0], X_demo[y_demo==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax2.scatter(X_demo[y_demo==1, 0], X_demo[y_demo==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')

# Circular boundary
theta = np.linspace(0, 2*np.pi, 100)
r = 1.5
x_circ = r * np.cos(theta)
y_circ = r * np.sin(theta)
ax2.plot(x_circ, y_circ, 'green', linewidth=3, label='Circular Boundary')

ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax2.set_title('GOOD FIT\n(Just Right!)', fontsize=12, fontweight='bold', color='green')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-4, 4)
ax2.text(0.5, 0.05, 'Low Training Error\nLow Test Error',
        transform=ax2.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# Plot 3: Overfitting (complex wiggly boundary)
ax3 = axes[2]
ax3.scatter(X_demo[y_demo==0, 0], X_demo[y_demo==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax3.scatter(X_demo[y_demo==1, 0], X_demo[y_demo==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')

# Complex wiggly boundary (overfitting)
theta_over = np.linspace(0, 2*np.pi, 200)
r_over = 1.5 + 0.5*np.sin(10*theta_over)  # Wiggly
x_over = r_over * np.cos(theta_over)
y_over = r_over * np.sin(theta_over)
ax3.plot(x_over, y_over, 'red', linewidth=3, label='Complex Boundary')

ax3.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax3.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax3.set_title('OVERFITTING\n(Too Complex)', fontsize=12, fontweight='bold', color='red')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-4, 4)
ax3.set_ylim(-4, 4)
ax3.text(0.5, 0.05, 'Low Training Error\nHigh Test Error!',
        transform=ax3.transAxes, ha='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/02_model_complexity_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/02_model_complexity_comparison.png")
plt.close()

# Visualization 3: How decision boundary equation works
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Understanding Decision Boundary Mathematics', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: The boundary equation
ax1 = axes[0, 0]
ax1.axis('off')

boundary_math = """
DECISION BOUNDARY MATHEMATICS
═══════════════════════════════════════════════════

For 2 features (x₁, x₂):

LINEAR BOUNDARY:
   β₀ + β₁x₁ + β₂x₂ = 0

   This is a STRAIGHT LINE!

   Example: -5 + x₁ + x₂ = 0
   Rearrange: x₂ = 5 - x₁

   Predictions:
   • β₀ + β₁x₁ + β₂x₂ > 0  → Class 1
   • β₀ + β₁x₁ + β₂x₂ < 0  → Class 0
   • β₀ + β₁x₁ + β₂x₂ = 0  → On boundary

NON-LINEAR BOUNDARY (Polynomial):
   β₀ + β₁x₁ + β₂x₂ + β₃x₁² + β₄x₂² + β₅x₁x₂ = 0

   This creates CURVES!

   Example (circle): x₁² + x₂² = 16
   Predictions:
   • x₁² + x₂² > 16  → Outside (Class 1)
   • x₁² + x₂² < 16  → Inside (Class 0)
   • x₁² + x₂² = 16  → On boundary

COEFFICIENTS DETERMINE SHAPE:
   β₁, β₂ → Direction/slope of boundary
   β₀     → Position of boundary
   β₃, β₄, β₅ → Curvature (if non-linear)

═══════════════════════════════════════════════════
"""

ax1.text(0.5, 0.5, boundary_math,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

# Plot 2: Different slopes (β₁, β₂)
ax2 = axes[0, 1]
x_range = np.linspace(0, 10, 100)

# Different slopes
slopes = [(1, 1), (2, 1), (1, 2), (0.5, 1)]
colors_slopes = ['blue', 'green', 'red', 'orange']
labels_slopes = ['x₁ + x₂ = 5', '2x₁ + x₂ = 10', 'x₁ + 2x₂ = 10', '0.5x₁ + x₂ = 5']

for (b1, b2), color, label in zip(slopes, colors_slopes, labels_slopes):
    if b2 != 0:
        c = 5 if (b1, b2) == (1, 1) or (b1, b2) == (0.5, 1) else 10
        y_vals = (c - b1 * x_range) / b2
        ax2.plot(x_range, y_vals, linewidth=3, color=color, label=label)

ax2.set_xlabel('x₁', fontsize=12, fontweight='bold')
ax2.set_ylabel('x₂', fontsize=12, fontweight='bold')
ax2.set_title('Effect of Coefficients (β₁, β₂) on Boundary Slope', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)

# Plot 3: Different intercepts (β₀)
ax3 = axes[1, 0]
# Same slope, different intercepts
intercepts = [0, 2, 4, 6]
colors_int = ['blue', 'green', 'orange', 'red']

for c, color in zip(intercepts, colors_int):
    y_vals = c - x_range
    ax3.plot(x_range, y_vals, linewidth=3, color=color, label=f'x₁ + x₂ = {c}')

ax3.set_xlabel('x₁', fontsize=12, fontweight='bold')
ax3.set_ylabel('x₂', fontsize=12, fontweight='bold')
ax3.set_title('Effect of Intercept (β₀) on Boundary Position', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)
ax3.set_ylim(-5, 10)

# Plot 4: Key insights
ax4 = axes[1, 1]
ax4.axis('off')

insights_text = """
KEY INSIGHTS
═══════════════════════════════════════════════════

INTERPRETING BOUNDARIES:

1. SIMPLE BOUNDARY (Linear):
   ✓ Fast to train and predict
   ✓ Easy to interpret
   ✓ Works for linearly separable data
   ✗ Limited flexibility

2. COMPLEX BOUNDARY (Non-linear):
   ✓ Can capture complex patterns
   ✓ Better for real-world data
   ✗ Harder to interpret
   ✗ Risk of overfitting

3. DISTANCE FROM BOUNDARY:
   • Far from boundary → High confidence
   • Close to boundary → Low confidence
   • On boundary → 50/50 (p = 0.5)

4. CHOOSING COMPLEXITY:
   • Start simple (linear)
   • Add complexity if needed
   • Use validation set to check
   • Regularize to prevent overfitting

5. VISUALIZATION HELPS:
   • See what model learned
   • Spot overfitting visually
   • Compare different models
   • Debug poor performance

═══════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, insights_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/03_boundary_mathematics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/03_boundary_mathematics.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🎯 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ DECISION BOUNDARY separates different classes in feature space")
print()
print("✓ LINEAR BOUNDARIES:")
print("   • Created by logistic regression, linear SVM")
print("   • Equation: β₀ + β₁x₁ + β₂x₂ = 0")
print("   • Straight line in 2D, plane in 3D")
print("   • Good for linearly separable data")
print()
print("✓ NON-LINEAR BOUNDARIES:")
print("   • Created by polynomial features, trees, neural networks")
print("   • Curved lines, circles, complex shapes")
print("   • Can capture complex patterns")
print()
print("✓ MODEL COMPLEXITY:")
print("   • Underfitting: Boundary too simple → High error")
print("   • Good fit: Boundary captures pattern → Low error")
print("   • Overfitting: Boundary too complex → Memorizes noise")
print()
print("✓ KEY INSIGHTS:")
print("   • Visualize boundaries to understand models")
print("   • Distance from boundary = confidence")
print("   • Start simple, add complexity if needed")
print("   • Use validation to prevent overfitting")
print()
print("NEXT: We'll implement LOGISTIC REGRESSION - the main classification")
print("      algorithm that creates linear decision boundaries!")
print()
print("=" * 80)
print("🎯 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
