"""
🎯 K-NEAREST NEIGHBORS (KNN) CLASSIFIER
========================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. How KNN makes predictions using "similarity"
2. The concept of distance in feature space
3. How the K parameter affects predictions
4. Implementing KNN from scratch
5. Using scikit-learn's KNeighborsClassifier
6. When to use KNN vs Logistic Regression
7. Advantages and limitations of KNN

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "K-nearest neighbors, Clearly Explained"
   https://www.youtube.com/watch?v=HVXime0nQeI
   THE BEST explanation of KNN!

📚 Serrano.Academy: "K Nearest Neighbors"
   Clear visual walkthrough

⭐ Josh Starmer: "KNN Step-by-Step"
   Detailed implementation walkthrough

TIME: 60-75 minutes
DIFFICULTY: Beginner/Intermediate
PREREQUISITES: 01_sigmoid_function.py, 04_confusion_matrix.py

KEY CONCEPTS:
------------
- Instance-Based Learning: No training, just store data
- Distance Metrics: Euclidean, Manhattan
- K Parameter: Number of neighbors to consider
- Voting: Majority class wins
- Decision Boundaries: Complex, non-linear
- Lazy Learning: Computation happens at prediction time
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import Counter

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'knn_classifier'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🎯 K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE KNN CONCEPT
# ============================================================================

print("=" * 80)
print("SECTION 1: What is K-Nearest Neighbors?")
print("=" * 80)
print()

print("KNN is the SIMPLEST machine learning algorithm!")
print()
print("THE INTUITION:")
print("   'You are the average of your 5 closest friends'")
print()
print("In ML terms:")
print("   'A data point's class is determined by its K nearest neighbors'")
print()

print("HOW IT WORKS:")
print()
print("Step 1: Store all training data (no actual 'training')")
print()
print("Step 2: To predict a new point:")
print("   a. Calculate distance to ALL training points")
print("   b. Find the K closest points (K nearest neighbors)")
print("   c. Look at their classes")
print("   d. Majority vote wins!")
print()

print("Example with K=3:")
print("-" * 70)
print("New point: (3, 4)")
print()
print("Find 3 nearest neighbors:")
print("   Neighbor 1: (2.5, 4.2) → Class: Spam     (distance: 0.54)")
print("   Neighbor 2: (3.1, 3.8) → Class: Spam     (distance: 0.22)")
print("   Neighbor 3: (2.8, 4.5) → Class: Not Spam (distance: 0.54)")
print()
print("Vote: Spam=2, Not Spam=1")
print("Prediction: SPAM (majority vote)")
print()

print("KEY DIFFERENCE FROM LOGISTIC REGRESSION:")
print("   • Logistic Regression: Learns coefficients, uses equation")
print("   • KNN: No learning! Just remembers training data")
print()

# ============================================================================
# SECTION 2: DISTANCE METRICS
# ============================================================================

print("=" * 80)
print("SECTION 2: Measuring Distance")
print("=" * 80)
print()

print("To find 'nearest' neighbors, we need to measure DISTANCE.")
print()

print("1. EUCLIDEAN DISTANCE (Most Common)")
print("   Formula: d = √[(x₁-x₂)² + (y₁-y₂)²]")
print("   Think: Straight-line distance (as the crow flies)")
print()

# Example calculation
point_a = np.array([1, 2])
point_b = np.array([4, 6])

euclidean = np.sqrt(np.sum((point_a - point_b)**2))

print(f"   Example: Distance from {point_a} to {point_b}")
print(f"   d = √[(1-4)² + (2-6)²]")
print(f"   d = √[9 + 16]")
print(f"   d = √25")
print(f"   d = {euclidean:.2f}")
print()

print("2. MANHATTAN DISTANCE (City Block)")
print("   Formula: d = |x₁-x₂| + |y₁-y₂|")
print("   Think: Distance if you can only move along streets (grid)")
print()

manhattan = np.sum(np.abs(point_a - point_b))

print(f"   Example: Distance from {point_a} to {point_b}")
print(f"   d = |1-4| + |2-6|")
print(f"   d = 3 + 4")
print(f"   d = {manhattan:.2f}")
print()

print("When to use which:")
print("   • Euclidean: Most common, works well in general")
print("   • Manhattan: Better for high-dimensional data")
print()

# ============================================================================
# SECTION 3: THE K PARAMETER
# ============================================================================

print("=" * 80)
print("SECTION 3: Choosing K - The Magic Number")
print("=" * 80)
print()

print("K is the NUMBER OF NEIGHBORS to consider.")
print()

print("SMALL K (e.g., K=1):")
print("   • Very sensitive to noise")
print("   • Complex decision boundaries (overfitting)")
print("   • Training accuracy: HIGH")
print("   • Test accuracy: May be LOW")
print()

print("LARGE K (e.g., K=100):")
print("   • Smooth decision boundaries")
print("   • May ignore local patterns (underfitting)")
print("   • Training accuracy: LOWER")
print("   • Test accuracy: May be LOW")
print()

print("GOOD K (e.g., K=5 or K=7):")
print("   • Balance between overfitting and underfitting")
print("   • Usually odd numbers (avoids ties)")
print("   • Found through cross-validation")
print()

print("RULE OF THUMB:")
print("   • Start with K = √(number of samples)")
print("   • Try odd values: 3, 5, 7, 9")
print("   • Use cross-validation to find optimal K")
print()

# ============================================================================
# SECTION 4: KNN FROM SCRATCH
# ============================================================================

print("=" * 80)
print("SECTION 4: Implementing KNN from Scratch")
print("=" * 80)
print()

class KNNClassifier:
    """Simple KNN Classifier implementation"""

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data"""
        self.X_train = X
        self.y_train = y
        print(f"Stored {len(X)} training samples")

    def predict(self, X):
        """Predict class for each sample in X"""
        predictions = []
        for x in X:
            # Calculate distances to all training points
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

            # Get indices of K nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Get classes of K nearest neighbors
            k_nearest_classes = self.y_train[k_indices]

            # Majority vote
            most_common = Counter(k_nearest_classes).most_common(1)[0][0]
            predictions.append(most_common)

        return np.array(predictions)

    def predict_with_details(self, x):
        """Predict with detailed information"""
        # Calculate distances
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

        # Get K nearest
        k_indices = np.argsort(distances)[:self.k]
        k_distances = distances[k_indices]
        k_classes = self.y_train[k_indices]

        # Vote
        votes = Counter(k_classes)
        prediction = votes.most_common(1)[0][0]

        return prediction, k_indices, k_distances, k_classes, votes

print("Created KNNClassifier class!")
print()

# Generate sample data
np.random.seed(42)
n_samples = 50

# Class 0: bottom-left
X_class0 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([2, 2])
# Class 1: top-right
X_class1 = np.random.randn(n_samples//2, 2) * 0.8 + np.array([5, 5])

X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

print(f"Created training data: {len(X_train)} samples")
print(f"   Class 0: {(y_train==0).sum()} samples")
print(f"   Class 1: {(y_train==1).sum()} samples")
print()

# Train KNN (just store data)
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
print()

# Make a prediction with details
test_point = np.array([3.5, 3.5])
pred, neighbor_idx, neighbor_dist, neighbor_classes, votes = knn.predict_with_details(test_point)

print(f"Predicting for point: {test_point}")
print()
print(f"K = {knn.k} nearest neighbors:")
print("-" * 70)
print(f"{'Neighbor':<12} {'Location':<20} {'Distance':<12} {'Class'}")
print("-" * 70)
for i, (idx, dist, cls) in enumerate(zip(neighbor_idx, neighbor_dist, neighbor_classes), 1):
    loc = X_train[idx]
    print(f"Neighbor {i:<3} ({loc[0]:.2f}, {loc[1]:.2f})         {dist:<12.3f} {cls}")
print()
print(f"Votes: {dict(votes)}")
print(f"Prediction: Class {pred}")
print()

# Test on multiple points
test_points = np.array([[1, 1], [3, 3], [4, 4], [6, 6]])
predictions = knn.predict(test_points)

print("Testing on multiple points:")
print("-" * 70)
print(f"{'Test Point':<20} {'Prediction':<15} {'Interpretation'}")
print("-" * 70)
for point, pred in zip(test_points, predictions):
    interp = "Class 0 (like bottom-left cluster)" if pred == 0 else "Class 1 (like top-right cluster)"
    print(f"({point[0]:.1f}, {point[1]:.1f})            Class {pred:<10} {interp}")
print()

# Calculate accuracy
train_predictions = knn.predict(X_train)
accuracy = np.mean(train_predictions == y_train)
print(f"Training accuracy: {accuracy*100:.2f}%")
print()

# ============================================================================
# SECTION 5: USING SCIKIT-LEARN
# ============================================================================

print("=" * 80)
print("SECTION 5: Using Scikit-Learn's KNeighborsClassifier")
print("=" * 80)
print()

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.model_selection import cross_val_score

    # Train KNN with different K values
    print("Training KNN with different K values...")
    print()

    k_values = [1, 3, 5, 7, 9, 15]
    print("-" * 70)
    print(f"{'K Value':<12} {'Training Acc':<15} {'CV Score (mean)':<20} {'Interpretation'}")
    print("-" * 70)

    best_k = None
    best_score = 0

    for k in k_values:
        # Train model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)

        # Training accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        # Cross-validation score (better estimate)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()

        # Interpretation
        if k == 1:
            interp = "Too sensitive (overfitting)"
        elif k >= 15:
            interp = "Too smooth (underfitting)"
        else:
            interp = "Good balance"

        print(f"{k:<12} {train_acc*100:<15.1f}% {cv_mean*100:<20.1f}% {interp}")

        if cv_mean > best_score:
            best_score = cv_mean
            best_k = k

    print()
    print(f"Best K = {best_k} with CV score = {best_score*100:.1f}%")
    print()

    # Train final model with best K
    final_model = KNeighborsClassifier(n_neighbors=best_k)
    final_model.fit(X_train, y_train)

    print(f"Final model trained with K={best_k}")
    print()

    # Make predictions
    y_pred = final_model.predict(X_train)

    # Evaluation
    print("Model Performance:")
    print("-" * 70)
    acc = accuracy_score(y_train, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_train, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print()

    # Classification report
    print("Classification Report:")
    print(classification_report(y_train, y_pred, target_names=['Class 0', 'Class 1']))

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not installed")
    sklearn_available = False

# ============================================================================
# SECTION 6: KNN VS LOGISTIC REGRESSION
# ============================================================================

print("=" * 80)
print("SECTION 6: KNN vs Logistic Regression")
print("=" * 80)
print()

print("When to use which algorithm?")
print()

print("Use KNN when:")
print("   ✓ Decision boundary is very non-linear")
print("   ✓ You have lots of training data")
print("   ✓ You need simple implementation")
print("   ✓ You don't need to understand 'why' (interpretability not needed)")
print("   ✓ Prediction time is not critical")
print()

print("Use Logistic Regression when:")
print("   ✓ Decision boundary is mostly linear")
print("   ✓ You need fast predictions")
print("   ✓ You need to interpret results (coefficients)")
print("   ✓ You have limited training data")
print("   ✓ You need probabilistic predictions")
print()

print("COMPARISON TABLE:")
print("-" * 70)
print(f"{'Aspect':<25} {'KNN':<25} {'Logistic Regression'}")
print("-" * 70)
print(f"{'Training Time':<25} {'Fast (just store)':<25} {'Slower (optimization)'}")
print(f"{'Prediction Time':<25} {'Slow (calc distances)':<25} {'Fast (just equation)'}")
print(f"{'Memory Usage':<25} {'High (stores all data)':<25} {'Low (just coefficients)'}")
print(f"{'Decision Boundary':<25} {'Non-linear, complex':<25} {'Linear'}")
print(f"{'Interpretability':<25} {'Low':<25} {'High (coefficients)'}")
print(f"{'Probabilistic':<25} {'No (hard voting)':<25} {'Yes (sigmoid)'}")
print(f"{'Overfitting Risk':<25} {'High (small K)':<25} {'Lower (with regularization)'}")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: How KNN Works
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('How K-Nearest Neighbors Works', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Data and test point
ax1 = axes[0, 0]
ax1.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax1.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')

# Test point
test_pt = np.array([3.5, 3.5])
ax1.scatter(*test_pt, c='green', s=300, marker='*', edgecolors='black',
           linewidth=2, label='New Point', zorder=5)

ax1.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax1.set_title('Step 1: Plot Training Data and New Point', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Show distances
ax2 = axes[0, 1]
ax2.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
           c='red', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 0')
ax2.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
           c='blue', s=100, alpha=0.6, edgecolors='black', linewidth=1.5, label='Class 1')
ax2.scatter(*test_pt, c='green', s=300, marker='*', edgecolors='black',
           linewidth=2, label='New Point', zorder=5)

# Draw circles showing distances
for radius in [0.5, 1.0, 1.5, 2.0]:
    circle = plt.Circle(test_pt, radius, fill=False, color='green',
                       linestyle='--', alpha=0.3, linewidth=1)
    ax2.add_patch(circle)

ax2.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax2.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax2.set_title('Step 2: Calculate Distances to All Points', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 7)
ax2.set_ylim(0, 7)

# Plot 3: Highlight K nearest
ax3 = axes[1, 0]
ax3.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
           c='red', s=100, alpha=0.3, edgecolors='black', linewidth=1, label='Class 0')
ax3.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
           c='blue', s=100, alpha=0.3, edgecolors='black', linewidth=1, label='Class 1')
ax3.scatter(*test_pt, c='green', s=300, marker='*', edgecolors='black',
           linewidth=2, label='New Point', zorder=5)

# Highlight K=5 nearest neighbors
if sklearn_available:
    knn_temp = KNeighborsClassifier(n_neighbors=5)
    knn_temp.fit(X_train, y_train)
    distances, indices = knn_temp.kneighbors([test_pt])

    for idx in indices[0]:
        point = X_train[idx]
        color = 'red' if y_train[idx] == 0 else 'blue'
        ax3.scatter(*point, c=color, s=200, edgecolors='yellow', linewidth=3, zorder=4)
        ax3.plot([test_pt[0], point[0]], [test_pt[1], point[1]],
                'g--', alpha=0.5, linewidth=2)

ax3.set_xlabel('Feature 1', fontsize=12, fontweight='bold')
ax3.set_ylabel('Feature 2', fontsize=12, fontweight='bold')
ax3.set_title('Step 3: Find K=5 Nearest Neighbors (yellow edges)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Decision summary
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = """
KNN ALGORITHM SUMMARY
═══════════════════════════════════════════════════

STEP-BY-STEP PROCESS:

1. STORE training data (no actual training!)

2. FOR each new point to predict:
   a. Calculate distance to ALL training points
   b. Sort by distance
   c. Select K nearest neighbors
   d. Count classes of K neighbors
   e. Majority class wins!

EXAMPLE (K=5):
   New Point: (3.5, 3.5)

   5 Nearest Neighbors:
   • 3 points are Class 1 (blue)
   • 2 points are Class 0 (red)

   Vote: Class 1 = 3, Class 0 = 2
   Prediction: Class 1 (majority)

KEY PARAMETERS:
• K: Number of neighbors
  - Small K → sensitive to noise
  - Large K → too smooth
  - Typical: 3, 5, 7 (odd numbers)

• Distance Metric:
  - Euclidean (default): √[(x₁-x₂)² + (y₁-y₂)²]
  - Manhattan: |x₁-x₂| + |y₁-y₂|

PROS:
✓ Simple to understand and implement
✓ No training time
✓ Naturally handles multi-class
✓ Works with non-linear boundaries

CONS:
✗ Slow predictions (calculate all distances)
✗ Memory intensive (stores all data)
✗ Sensitive to feature scaling
✗ Curse of dimensionality

═══════════════════════════════════════════════════
"""

ax4.text(0.5, 0.5, summary_text,
        transform=ax4.transAxes,
        fontsize=9.5,
        verticalalignment='center',
        horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_knn_explained.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_knn_explained.png")
plt.close()

# Visualization 2: Effect of K parameter
if sklearn_available:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Effect of K Parameter on Decision Boundaries', fontsize=16, fontweight='bold')

    k_values_viz = [1, 3, 5, 7, 15, 30]

    # Create mesh for decision boundary
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    for idx, k in enumerate(k_values_viz):
        ax = axes[idx // 3, idx % 3]

        # Train model
        knn_viz = KNeighborsClassifier(n_neighbors=k)
        knn_viz.fit(X_train, y_train)

        # Predict on mesh
        Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['red', 'blue'])
        ax.contour(xx, yy, Z, colors='green', linewidths=2, levels=1)

        # Plot training points
        ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
                  c='red', s=50, alpha=0.8, edgecolors='black', linewidth=1)
        ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
                  c='blue', s=50, alpha=0.8, edgecolors='black', linewidth=1)

        # Calculate accuracy
        train_pred = knn_viz.predict(X_train)
        acc = accuracy_score(y_train, train_pred)

        # Determine if good/bad
        if k == 1:
            quality = "OVERFITTING"
            color = 'red'
        elif k >= 15:
            quality = "UNDERFITTING"
            color = 'orange'
        else:
            quality = "GOOD FIT"
            color = 'green'

        ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
        ax.set_title(f'K={k} | Accuracy={acc*100:.1f}%\n{quality}',
                    fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_k_parameter_effect.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_k_parameter_effect.png")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🎯 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ KNN is a SIMPLE, INTUITIVE algorithm")
print()
print("✓ HOW IT WORKS:")
print("   1. Store all training data")
print("   2. For new point: find K nearest neighbors")
print("   3. Majority vote determines class")
print()
print("✓ KEY CONCEPTS:")
print("   • Distance: Euclidean or Manhattan")
print("   • K parameter: Balance between sensitivity and smoothness")
print("   • No training phase (lazy learning)")
print("   • Prediction is expensive (calculate all distances)")
print()
print("✓ CHOOSING K:")
print("   • Small K → Overfitting (too sensitive)")
print("   • Large K → Underfitting (too smooth)")
print("   • Use cross-validation to find optimal K")
print("   • Prefer odd numbers (avoid ties)")
print()
print("✓ KNN vs LOGISTIC REGRESSION:")
print("   • KNN: Non-linear boundaries, slow predictions, no interpretability")
print("   • LogReg: Linear boundaries, fast predictions, interpretable")
print()
print("✓ WHEN TO USE KNN:")
print("   • Non-linear decision boundaries")
print("   • Don't need fast predictions")
print("   • Interpretability not important")
print()
print("NEXT: We'll learn Decision Trees - another intuitive algorithm!")
print()
print("=" * 80)
print("🎯 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
