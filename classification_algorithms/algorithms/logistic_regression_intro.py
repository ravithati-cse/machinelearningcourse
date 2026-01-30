"""
🤖 LOGISTIC REGRESSION - The Foundation of Classification
==========================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. How logistic regression extends linear regression
2. The complete pipeline: Linear model → Log-odds → Sigmoid → Probability
3. How to implement logistic regression from scratch
4. How to use scikit-learn's LogisticRegression
5. How to interpret coefficients
6. How to visualize decision boundaries
7. How to make predictions with probabilities

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Logistic Regression"
   https://www.youtube.com/watch?v=yIYKR4sgzI8
   THE BEST introduction - MUST WATCH!

⭐ StatQuest: "Logistic Regression Details Pt1: Coefficients"
   https://www.youtube.com/watch?v=vN5cNN2-HWE
   How coefficients work in logistic regression

⭐ StatQuest: "Logistic Regression Details Pt2: Maximum Likelihood"
   https://www.youtube.com/watch?v=BfKanl1aSG0
   Why we use log loss (from likelihood perspective)

📚 StatQuest: "Logistic Regression Details Pt3: R-squared and p-value"
   https://www.youtube.com/watch?v=xxFYro8QuXA
   Evaluating logistic regression

TIME: 75-90 minutes
DIFFICULTY: Intermediate
PREREQUISITES: All 5 math foundation modules

KEY CONCEPTS:
------------
- Logistic Regression = Linear Regression + Sigmoid
- Log-odds: z = β₀ + β₁x₁ + β₂x₂ + ...
- Probability: P(y=1) = sigmoid(z) = 1 / (1 + e^(-z))
- Trained using gradient descent to minimize log loss
- Coefficients show how features affect log-odds
- Decision boundary: Where P(y=1) = 0.5
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pathlib import Path

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'logistic_regression'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🤖 LOGISTIC REGRESSION - The Main Classification Algorithm")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: FROM LINEAR TO LOGISTIC REGRESSION
# ============================================================================

print("=" * 80)
print("SECTION 1: From Linear to Logistic Regression")
print("=" * 80)
print()

print("REMEMBER LINEAR REGRESSION?")
print("   ŷ = β₀ + β₁x₁ + β₂x₂ + ...")
print("   Output: Any number (-∞ to +∞)")
print("   Used for: Predicting continuous values (prices, temperatures)")
print()

print("THE PROBLEM FOR CLASSIFICATION:")
print("   We need probabilities (0 to 1), not any number!")
print()

print("THE SOLUTION: LOGISTIC REGRESSION")
print()
print("Step 1: Use linear model to get log-odds")
print("   z = β₀ + β₁x₁ + β₂x₂ + ...")
print("   z can be any number (-∞ to +∞) ✓")
print()

print("Step 2: Apply sigmoid to convert log-odds → probability")
print("   P(y=1) = σ(z) = 1 / (1 + e^(-z))")
print("   P(y=1) is between 0 and 1 ✓")
print()

print("Step 3: Apply threshold to get class")
print("   If P(y=1) ≥ 0.5 → Predict class 1")
print("   If P(y=1) < 0.5  → Predict class 0")
print()

print("COMPLETE PIPELINE:")
print("   Features → Linear model → Log-odds → Sigmoid → Probability → Threshold → Class")
print()

# Example
print("Example with 2 features:")
print("-" * 70)
x1_ex, x2_ex = 3, 4
beta0, beta1, beta2 = -5, 1, 1

z_ex = beta0 + beta1*x1_ex + beta2*x2_ex
prob_ex = 1 / (1 + np.exp(-z_ex))
pred_ex = 1 if prob_ex >= 0.5 else 0

print(f"Features: x₁ = {x1_ex}, x₂ = {x2_ex}")
print(f"Coefficients: β₀ = {beta0}, β₁ = {beta1}, β₂ = {beta2}")
print()
print(f"Step 1 - Linear model:")
print(f"   z = {beta0} + {beta1}×{x1_ex} + {beta2}×{x2_ex}")
print(f"   z = {beta0} + {beta1*x1_ex} + {beta2*x2_ex}")
print(f"   z = {z_ex}")
print()
print(f"Step 2 - Sigmoid:")
print(f"   P(y=1) = 1 / (1 + e^(-{z_ex}))")
print(f"   P(y=1) = {prob_ex:.4f} ({prob_ex*100:.2f}%)")
print()
print(f"Step 3 - Threshold (0.5):")
print(f"   {prob_ex:.4f} ≥ 0.5? {'Yes' if prob_ex >= 0.5 else 'No'}")
print(f"   Prediction: Class {pred_ex}")
print()

# ============================================================================
# SECTION 2: THE SIGMOID FUNCTION (RECAP)
# ============================================================================

print("=" * 80)
print("SECTION 2: The Sigmoid Function (Quick Recap)")
print("=" * 80)
print()

def sigmoid(z):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-z))

print("The sigmoid function is the KEY to logistic regression:")
print()
print("   σ(z) = 1 / (1 + e^(-z))")
print()

# Show sigmoid values
z_values = [-5, -2, -1, 0, 1, 2, 5]
print("Sigmoid transforms log-odds to probabilities:")
print("-" * 60)
print(f"{'Log-Odds (z)':<15} {'σ(z)':<15} {'Probability':<15} {'Prediction'}")
print("-" * 60)
for z in z_values:
    sig = sigmoid(z)
    pred = "Class 1" if sig >= 0.5 else "Class 0"
    print(f"{z:<15} {sig:<15.4f} {sig*100:<15.1f}% {pred}")
print()

print("KEY PROPERTIES:")
print("   • z = 0  → P = 0.5  (decision boundary)")
print("   • z > 0  → P > 0.5  (predicts class 1)")
print("   • z < 0  → P < 0.5  (predicts class 0)")
print("   • Large |z| → Very confident prediction")
print()

# ============================================================================
# SECTION 3: TRAINING LOGISTIC REGRESSION
# ============================================================================

print("=" * 80)
print("SECTION 3: How Logistic Regression Learns")
print("=" * 80)
print()

print("TRAINING PROCESS:")
print()
print("1. START with random coefficients β₀, β₁, β₂, ...")
print()
print("2. FOR EACH training example:")
print("   a. Calculate log-odds: z = β₀ + β₁x₁ + β₂x₂ + ...")
print("   b. Calculate probability: P(y=1) = sigmoid(z)")
print("   c. Calculate log loss: L = -[y·log(P) + (1-y)·log(1-P)]")
print()
print("3. Calculate AVERAGE log loss over all examples")
print()
print("4. Use GRADIENT DESCENT to update coefficients:")
print("   • Compute gradient (how to change β to reduce loss)")
print("   • Update: β_new = β_old - learning_rate × gradient")
print()
print("5. REPEAT steps 2-4 until log loss stops decreasing")
print()

print("GOAL: Find coefficients that minimize log loss")
print()
print("Result: Model that makes accurate probabilistic predictions!")
print()

# ============================================================================
# SECTION 4: SIMPLE IMPLEMENTATION FROM SCRATCH
# ============================================================================

print("=" * 80)
print("SECTION 4: Implementing Logistic Regression from Scratch")
print("=" * 80)
print()

print("Let's implement a simple version to understand the math!")
print()

# Generate simple dataset
np.random.seed(42)
n_samples = 100

# Class 0: points around (2, 2)
X_class0 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([2, 2])
# Class 1: points around (5, 5)
X_class1 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([5, 5])

X = np.vstack([X_class0, X_class1])
y = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

print(f"Created dataset: {n_samples} samples")
print(f"   Class 0: {(y==0).sum()} samples")
print(f"   Class 1: {(y==1).sum()} samples")
print()

# Add intercept term (column of ones)
X_with_intercept = np.column_stack([np.ones(len(X)), X])

print("Features shape:", X.shape)
print("With intercept:", X_with_intercept.shape)
print()

# Initialize coefficients
beta = np.random.randn(3) * 0.01
print(f"Initial coefficients: β = {beta}")
print()

# Training loop (simplified gradient descent)
learning_rate = 0.1
n_iterations = 1000

print("Training...")
print("-" * 70)
print(f"{'Iteration':<15} {'Log Loss':<20} {'Accuracy':<20}")
print("-" * 70)

for iteration in range(n_iterations):
    # Forward pass
    z = X_with_intercept @ beta  # Linear combination
    probabilities = sigmoid(z)    # Sigmoid

    # Calculate log loss
    epsilon = 1e-15  # Avoid log(0)
    probabilities_clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    log_loss = -np.mean(y * np.log(probabilities_clipped) +
                        (1 - y) * np.log(1 - probabilities_clipped))

    # Predictions
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = np.mean(predictions == y)

    # Gradient descent
    error = probabilities - y
    gradient = X_with_intercept.T @ error / len(y)
    beta = beta - learning_rate * gradient

    # Print progress
    if iteration % 100 == 0 or iteration == n_iterations - 1:
        print(f"{iteration:<15} {log_loss:<20.4f} {accuracy*100:<20.1f}%")

print()
print(f"Final coefficients: β = {beta}")
print(f"   β₀ (intercept) = {beta[0]:.4f}")
print(f"   β₁ (feature 1) = {beta[1]:.4f}")
print(f"   β₂ (feature 2) = {beta[2]:.4f}")
print()

# Make predictions on new points
print("Testing predictions on new points:")
print("-" * 70)
print(f"{'Point (x₁, x₂)':<20} {'Log-odds (z)':<15} {'P(y=1)':<15} {'Prediction'}")
print("-" * 70)

test_points = [(1, 1), (3, 3), (4, 4), (6, 6), (7, 7)]
for x1_test, x2_test in test_points:
    x_test = np.array([1, x1_test, x2_test])  # Add intercept
    z_test = x_test @ beta
    p_test = sigmoid(z_test)
    pred_test = "Class 1" if p_test >= 0.5 else "Class 0"
    print(f"({x1_test}, {x2_test})              {z_test:<15.4f} {p_test:<15.4f} {pred_test}")
print()

# ============================================================================
# SECTION 5: USING SCIKIT-LEARN
# ============================================================================

print("=" * 80)
print("SECTION 5: Using Scikit-Learn (The Professional Way)")
print("=" * 80)
print()

print("Our implementation works, but scikit-learn is optimized and feature-rich!")
print()

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X, y)

    print("✓ Trained LogisticRegression model")
    print()

    # Get coefficients
    print("Sklearn coefficients:")
    print(f"   Intercept (β₀): {model.intercept_[0]:.4f}")
    print(f"   Coefficients (β₁, β₂): {model.coef_[0]}")
    print()

    print("Compare with our implementation:")
    print(f"   Our β₀: {beta[0]:.4f}")
    print(f"   Our β₁, β₂: {beta[1:]}")
    print()
    print("Similar values! ✓")
    print()

    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]  # Probability of class 1

    # Evaluate
    acc = accuracy_score(y, y_pred)
    loss = log_loss(y, y_pred_proba)

    print(f"Sklearn Performance:")
    print(f"   Accuracy:  {acc*100:.2f}%")
    print(f"   Log Loss:  {loss:.4f}")
    print()

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()
    print(f"   True Negatives:  {cm[0, 0]}")
    print(f"   False Positives: {cm[0, 1]}")
    print(f"   False Negatives: {cm[1, 0]}")
    print(f"   True Positives:  {cm[1, 1]}")
    print()

    # Test on new points
    print("Testing on new points with sklearn:")
    print("-" * 70)
    print(f"{'Point (x₁, x₂)':<20} {'P(y=1)':<15} {'Prediction'}")
    print("-" * 70)

    for x1_test, x2_test in test_points:
        x_test = np.array([[x1_test, x2_test]])
        p_test = model.predict_proba(x_test)[0, 1]
        pred_test = model.predict(x_test)[0]
        pred_label = f"Class {pred_test}"
        print(f"({x1_test}, {x2_test})              {p_test:<15.4f} {pred_label}")
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not installed. Install with: pip install scikit-learn")
    print()
    sklearn_available = False

# ============================================================================
# SECTION 6: INTERPRETING COEFFICIENTS
# ============================================================================

print("=" * 80)
print("SECTION 6: Interpreting Coefficients")
print("=" * 80)
print()

print("Coefficients show how features affect the LOG-ODDS:")
print()

if sklearn_available:
    print(f"β₀ (intercept) = {model.intercept_[0]:.4f}")
    print(f"   When x₁=0 and x₂=0, log-odds = {model.intercept_[0]:.4f}")
    print()

    print(f"β₁ (x₁ coefficient) = {model.coef_[0][0]:.4f}")
    print(f"   For each unit increase in x₁:")
    print(f"   • Log-odds increase by {model.coef_[0][0]:.4f}")
    if model.coef_[0][0] > 0:
        print(f"   • Positive → Higher x₁ makes class 1 MORE likely")
    else:
        print(f"   • Negative → Higher x₁ makes class 1 LESS likely")
    print()

    print(f"β₂ (x₂ coefficient) = {model.coef_[0][1]:.4f}")
    print(f"   For each unit increase in x₂:")
    print(f"   • Log-odds increase by {model.coef_[0][1]:.4f}")
    if model.coef_[0][1] > 0:
        print(f"   • Positive → Higher x₂ makes class 1 MORE likely")
    else:
        print(f"   • Negative → Higher x₂ makes class 1 LESS likely")
    print()

print("CONVERTING TO ODDS RATIO:")
print("   Odds Ratio = e^β")
print()

if sklearn_available:
    for i, coef in enumerate(model.coef_[0], 1):
        odds_ratio = np.exp(coef)
        print(f"   Feature {i}: Odds Ratio = e^{coef:.4f} = {odds_ratio:.4f}")
        if odds_ratio > 1:
            print(f"      → Each unit increase multiplies odds by {odds_ratio:.2f}")
        else:
            print(f"      → Each unit increase divides odds by {1/odds_ratio:.2f}")
        print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: The Complete Pipeline
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Logistic Regression: From Features to Predictions', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Data distribution
ax1 = axes[0, 0]
ax1.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, alpha=0.6,
           edgecolors='black', linewidth=1.5, label='Class 0')
ax1.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, alpha=0.6,
           edgecolors='black', linewidth=1.5, label='Class 1')
ax1.set_xlabel('Feature 1 (x₁)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature 2 (x₂)', fontsize=12, fontweight='bold')
ax1.set_title('Training Data', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Decision boundary
ax2 = axes[0, 1]
ax2.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, alpha=0.6,
           edgecolors='black', linewidth=1.5, label='Class 0')
ax2.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, alpha=0.6,
           edgecolors='black', linewidth=1.5, label='Class 1')

# Draw decision boundary
if sklearn_available:
    # Create mesh
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                           np.linspace(x2_min, x2_max, 200))
    Z = model.predict_proba(np.c_[xx1.ravel(), xx2.ravel()])[:, 1]
    Z = Z.reshape(xx1.shape)

    # Plot decision boundary and regions
    ax2.contourf(xx1, xx2, Z, alpha=0.3, levels=np.linspace(0, 1, 11), cmap='RdBu_r')
    contour = ax2.contour(xx1, xx2, Z, levels=[0.5], colors='green', linewidths=3)
    ax2.clabel(contour, inline=True, fontsize=10, fmt='Decision\nBoundary')

ax2.set_xlabel('Feature 1 (x₁)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Feature 2 (x₂)', fontsize=12, fontweight='bold')
ax2.set_title('Decision Boundary (P=0.5)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sigmoid function
ax3 = axes[1, 0]
z_range = np.linspace(-10, 10, 100)
probs = sigmoid(z_range)

ax3.plot(z_range, probs, linewidth=3, color='purple', label='Sigmoid(z)')
ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
ax3.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Decision boundary (z=0)')
ax3.fill_between(z_range, 0, probs, where=(z_range<0), alpha=0.2, color='red', label='Predict Class 0')
ax3.fill_between(z_range, 0, probs, where=(z_range>=0), alpha=0.2, color='blue', label='Predict Class 1')

ax3.set_xlabel('Log-Odds (z = β₀ + β₁x₁ + β₂x₂)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Probability P(y=1)', fontsize=12, fontweight='bold')
ax3.set_title('Sigmoid Transformation', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Pipeline diagram
ax4 = axes[1, 1]
ax4.axis('off')

pipeline_text = """
LOGISTIC REGRESSION PIPELINE
═══════════════════════════════════════════════════

Input: Features (x₁, x₂, ...)

    ↓

Step 1: Linear Combination
   z = β₀ + β₁x₁ + β₂x₂ + ...
   (Log-odds, can be any number)

    ↓

Step 2: Sigmoid Function
   P(y=1) = 1 / (1 + e^(-z))
   (Probability, between 0 and 1)

    ↓

Step 3: Decision Rule
   If P(y=1) ≥ 0.5 → Predict Class 1
   If P(y=1) < 0.5  → Predict Class 0

    ↓

Output: Class prediction + Probability

═══════════════════════════════════════════════════

TRAINING:
• Initialize β randomly
• For each example:
  - Compute prediction
  - Calculate log loss
• Update β using gradient descent
• Repeat until convergence

GOAL: Minimize log loss

═══════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, pipeline_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_logistic_regression_complete.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_logistic_regression_complete.png")
plt.close()

# Visualization 2: Probability surface
if sklearn_available:
    fig = plt.figure(figsize=(16, 6))

    # 2D probability heatmap
    ax1 = fig.add_subplot(1, 2, 1)
    contourf = ax1.contourf(xx1, xx2, Z, alpha=0.8, levels=20, cmap='RdBu_r')
    ax1.scatter(X[y==0, 0], X[y==0, 1], c='red', s=100, alpha=0.8,
               edgecolors='black', linewidth=2, label='Class 0')
    ax1.scatter(X[y==1, 0], X[y==1, 1], c='blue', s=100, alpha=0.8,
               edgecolors='black', linewidth=2, label='Class 1')
    ax1.contour(xx1, xx2, Z, levels=[0.5], colors='green', linewidths=4)

    cbar = plt.colorbar(contourf, ax=ax1)
    cbar.set_label('P(y=1)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Feature 1 (x₁)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Feature 2 (x₂)', fontsize=12, fontweight='bold')
    ax1.set_title('Probability Surface (2D View)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 3D probability surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.plot_surface(xx1, xx2, Z, cmap='RdBu_r', alpha=0.7, edgecolor='none')
    ax2.scatter(X[y==0, 0], X[y==0, 1], 0, c='red', s=50, marker='o', alpha=0.8, edgecolors='black')
    ax2.scatter(X[y==1, 0], X[y==1, 1], 1, c='blue', s=50, marker='o', alpha=0.8, edgecolors='black')

    ax2.set_xlabel('Feature 1 (x₁)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Feature 2 (x₂)', fontsize=10, fontweight='bold')
    ax2.set_zlabel('P(y=1)', fontsize=10, fontweight='bold')
    ax2.set_title('Probability Surface (3D View)', fontsize=12, fontweight='bold')

    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_probability_surface.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_probability_surface.png")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🤖 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ LOGISTIC REGRESSION extends linear regression for classification")
print()
print("✓ THE PIPELINE:")
print("   1. Linear model: z = β₀ + β₁x₁ + β₂x₂ + ...")
print("   2. Sigmoid: P(y=1) = 1 / (1 + e^(-z))")
print("   3. Threshold: If P ≥ 0.5 → Class 1, else Class 0")
print()
print("✓ TRAINING:")
print("   • Minimize log loss using gradient descent")
print("   • Find coefficients that best separate classes")
print()
print("✓ COEFFICIENTS:")
print("   • Show how features affect log-odds")
print("   • Positive β → Feature increases P(y=1)")
print("   • Negative β → Feature decreases P(y=1)")
print("   • Magnitude → Strength of effect")
print()
print("✓ DECISION BOUNDARY:")
print("   • Where P(y=1) = 0.5")
print("   • Defined by β₀ + β₁x₁ + β₂x₂ + ... = 0")
print("   • Straight line for 2 features")
print()
print("✓ IMPLEMENTATION:")
print("   • Can implement from scratch (for learning)")
print("   • Use scikit-learn in practice (optimized)")
print()
print("NEXT: We'll explore other classification algorithms like KNN,")
print("      Decision Trees, and Random Forests!")
print()
print("=" * 80)
print("🤖 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
