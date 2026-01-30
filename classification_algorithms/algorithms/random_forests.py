"""
🌲 RANDOM FORESTS - Wisdom of the Crowd
===============================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. Ensemble learning: combining multiple models
2. Bootstrap aggregating (bagging)
3. Feature randomness in Random Forests
4. Why many weak trees beat one strong tree
5. Out-of-bag (OOB) error estimation
6. Feature importance in Random Forests
7. Implementing and using Random Forests

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Random Forests Part 1 - Building, Using and Evaluating"
   https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
   THE BEST explanation of Random Forests!

⭐ StatQuest: "Random Forests Part 2 - Missing Data and Clustering"
   https://www.youtube.com/watch?v=nyxTdL_4Q-Q
   Advanced Random Forest topics

📚 Josh Starmer: "Bootstrap Aggregating (Bagging)"
   Foundation for understanding Random Forests

📚 Luis Serrano: "Random Forests"
   https://www.youtube.com/watch?v=v6VJ2RO66Ag
   Visual intuitive explanation

TIME: 75-90 minutes
DIFFICULTY: Intermediate-Advanced
PREREQUISITES: decision_trees.py (MUST complete first!)

KEY CONCEPTS:
------------
- Ensemble Learning: Combining multiple models
- Bagging: Bootstrap + Aggregating
- Bootstrap Sample: Random sampling with replacement
- Feature Randomness: Each tree sees random feature subset
- Out-of-Bag Error: Free validation score
- Voting: Democracy of trees
- Feature Importance: Which features matter most
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import Counter

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'random_forests'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🌲 RANDOM FORESTS - The Wisdom of the Crowd")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: ENSEMBLE LEARNING CONCEPT
# ============================================================================

print("=" * 80)
print("SECTION 1: What is Ensemble Learning?")
print("=" * 80)
print()

print("THE BIG IDEA: Many weak learners → One strong learner!")
print()

print("Real-World Analogy:")
print("-" * 70)
print("❓ Question: How many jellybeans are in this jar?")
print()
print("   Option 1: Ask ONE expert")
print("      → Might be good, might be biased")
print()
print("   Option 2: Ask 100 random people and AVERAGE their guesses")
print("      → Usually more accurate! (Wisdom of crowds)")
print()

print("In Machine Learning:")
print("-" * 70)
print("   Option 1: Train ONE perfect decision tree")
print("      → Often overfits to training data")
print("      → Unstable (small changes = different tree)")
print()
print("   Option 2: Train 100 different trees and VOTE")
print("      → More accurate!")
print("      → More stable!")
print("      → Less overfitting!")
print()

print("KEY INSIGHT:")
print("   • Individual trees can be WRONG")
print("   • But they're wrong in DIFFERENT ways")
print("   • Average their predictions → errors cancel out!")
print()

print("ENSEMBLE METHODS:")
print()
print("1. BAGGING (Bootstrap Aggregating)")
print("   • Train multiple models on DIFFERENT random subsets")
print("   • Each model sees slightly different data")
print("   • VOTE or AVERAGE their predictions")
print("   • Random Forests use this!")
print()
print("2. BOOSTING (Sequential learning)")
print("   • Train models ONE AT A TIME")
print("   • Each new model focuses on mistakes of previous")
print("   • Examples: AdaBoost, Gradient Boosting, XGBoost")
print()
print("3. STACKING (Meta-learning)")
print("   • Train different types of models")
print("   • Use another model to combine their predictions")
print()

# ============================================================================
# SECTION 2: HOW RANDOM FORESTS WORK
# ============================================================================

print("=" * 80)
print("SECTION 2: Random Forest Algorithm")
print("=" * 80)
print()

print("Random Forest = Decision Trees + Bagging + Feature Randomness")
print()

print("ALGORITHM:")
print("-" * 70)
print()
print("1. BOOTSTRAP SAMPLING (for each tree):")
print("   • Randomly sample N data points WITH REPLACEMENT")
print("   • Some samples appear multiple times")
print("   • Some samples don't appear at all (~37% left out)")
print()
print("   Example: Original data = [A, B, C, D, E]")
print("   Tree 1 might get: [A, A, C, D, E]  (B missing)")
print("   Tree 2 might get: [A, B, B, C, D]  (E missing)")
print("   Tree 3 might get: [B, C, D, D, E]  (A missing)")
print()

print("2. FEATURE RANDOMNESS (at each split in each tree):")
print("   • Don't consider ALL features")
print("   • Only consider RANDOM SUBSET of features")
print("   • Common choice: √(n_features) for classification")
print()
print("   Example: If you have 16 features")
print("   → Each split considers only √16 = 4 random features")
print()

print("3. GROW TREES:")
print("   • Build each tree to maximum depth (or until pure)")
print("   • No pruning needed!")
print("   • Individual trees will overfit, but ensemble won't")
print()

print("4. MAKE PREDICTIONS:")
print("   • Each tree votes for a class")
print("   • MAJORITY VOTE wins")
print()
print("   Example: 100 trees")
print("   • 73 trees vote 'Class 1'")
print("   • 27 trees vote 'Class 0'")
print("   → Prediction: Class 1 (with 73% confidence)")
print()

print("WHY THIS WORKS:")
print("-" * 70)
print("   ✓ Bootstrap sampling → Different training data → Diversity")
print("   ✓ Feature randomness → More diversity → Less correlation")
print("   ✓ Deep trees → Low bias (captures patterns)")
print("   ✓ Averaging → Low variance (reduces overfitting)")
print()

# ============================================================================
# SECTION 3: SIMPLE RANDOM FOREST IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 3: Simple Random Forest from Scratch")
print("=" * 80)
print()

class SimpleDecisionTree:
    """Very simple decision tree for Random Forest"""

    def __init__(self, max_depth=None, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree = None

    def _gini(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y, feature_indices):
        """Find best split considering only specified features"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                n = len(y)
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask)/n) * gini_left + (np.sum(right_mask)/n) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth):
        """Recursively grow the tree"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'value': leaf_value}

        # Random feature subset
        if self.max_features is not None:
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)

        # Find best split
        feature_idx, threshold = self._best_split(X, y, feature_indices)

        if feature_idx is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'value': leaf_value}

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build subtrees
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'type': 'split',
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def fit(self, X, y):
        """Build the tree"""
        self.tree = self._grow_tree(X, y, depth=0)
        return self

    def _predict_sample(self, x, node):
        """Predict single sample"""
        if node['type'] == 'leaf':
            return node['value']

        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        """Predict for multiple samples"""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class SimpleRandomForest:
    """Simple Random Forest implementation"""

    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        """Train Random Forest"""
        print(f"Training {self.n_trees} trees...")
        self.trees = []
        n_samples = len(X)

        for i in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Train tree
            tree = SimpleDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"   Tree {i+1}/{self.n_trees} complete")

        print(f"✓ Trained {self.n_trees} trees!")
        return self

    def predict(self, X):
        """Predict by majority vote"""
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote for each sample
        predictions = []
        for i in range(len(X)):
            votes = tree_predictions[:, i]
            majority_vote = Counter(votes).most_common(1)[0][0]
            predictions.append(majority_vote)

        return np.array(predictions)

    def predict_proba(self, X):
        """Predict class probabilities"""
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        probabilities = []
        for i in range(len(X)):
            votes = tree_predictions[:, i]
            # Proportion of votes for each class
            prob_class_1 = np.sum(votes == 1) / len(self.trees)
            probabilities.append([1 - prob_class_1, prob_class_1])

        return np.array(probabilities)


print("Created SimpleRandomForest class!")
print()

# Generate sample data
np.random.seed(42)
n_samples = 200

# Class 0: bottom-left
X_class0 = np.random.randn(n_samples//2, 2) * 1.0 + np.array([2, 2])
# Class 1: top-right
X_class1 = np.random.randn(n_samples//2, 2) * 1.0 + np.array([5, 5])

X_train = np.vstack([X_class0, X_class1])
y_train = np.array([0]*(n_samples//2) + [1]*(n_samples//2))

print(f"Created training data: {len(X_train)} samples")
print(f"   Class 0: {(y_train==0).sum()} samples")
print(f"   Class 1: {(y_train==1).sum()} samples")
print()

# Train Random Forest
n_features = X_train.shape[1]
max_features = int(np.sqrt(n_features))  # Common choice for classification
print(f"Number of features: {n_features}")
print(f"Max features per split: {max_features}")
print()

rf = SimpleRandomForest(n_trees=50, max_depth=10, max_features=max_features)
rf.fit(X_train, y_train)
print()

# Make predictions
y_pred = rf.predict(X_train)
accuracy = np.mean(y_pred == y_train)
print(f"Training accuracy: {accuracy*100:.2f}%")
print()

# Show probability predictions
y_proba = rf.predict_proba(X_train[:5])
print("Sample probability predictions (first 5 samples):")
print("-" * 70)
for i in range(5):
    print(f"Sample {i+1}: Class 0: {y_proba[i, 0]:.2f}, Class 1: {y_proba[i, 1]:.2f}")
    print(f"           True class: {y_train[i]}, Predicted: {y_pred[i]}")
print()

# ============================================================================
# SECTION 4: USING SCIKIT-LEARN
# ============================================================================

print("=" * 80)
print("SECTION 4: Using Scikit-Learn's RandomForestClassifier")
print("=" * 80)
print()

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # Split data for better evaluation
    X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(
        X_train, y_train, test_size=0.3, random_state=42
    )

    print(f"Train set: {len(X_train_sk)} samples")
    print(f"Test set:  {len(X_test_sk)} samples")
    print()

    # Train single decision tree for comparison
    print("Training single Decision Tree...")
    single_tree = DecisionTreeClassifier(random_state=42)
    single_tree.fit(X_train_sk, y_train_sk)
    tree_train_acc = accuracy_score(y_train_sk, single_tree.predict(X_train_sk))
    tree_test_acc = accuracy_score(y_test_sk, single_tree.predict(X_test_sk))
    print(f"   Train accuracy: {tree_train_acc*100:.2f}%")
    print(f"   Test accuracy:  {tree_test_acc*100:.2f}%")
    print()

    # Train Random Forest
    print("Training Random Forest (100 trees)...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_sk, y_train_sk)
    rf_train_acc = accuracy_score(y_train_sk, rf_model.predict(X_train_sk))
    rf_test_acc = accuracy_score(y_test_sk, rf_model.predict(X_test_sk))
    print(f"   Train accuracy: {rf_train_acc*100:.2f}%")
    print(f"   Test accuracy:  {rf_test_acc*100:.2f}%")
    print()

    print("COMPARISON:")
    print("-" * 70)
    print(f"{'Model':<20} {'Train Acc':<15} {'Test Acc':<15} {'Gap':<15}")
    print("-" * 70)
    tree_gap = tree_train_acc - tree_test_acc
    rf_gap = rf_train_acc - rf_test_acc
    print(f"{'Single Tree':<20} {tree_train_acc*100:<15.2f}% {tree_test_acc*100:<15.2f}% {tree_gap*100:<15.2f}%")
    print(f"{'Random Forest':<20} {rf_train_acc*100:<15.2f}% {rf_test_acc*100:<15.2f}% {rf_gap*100:<15.2f}%")
    print()

    if rf_gap < tree_gap:
        print("✓ Random Forest has SMALLER gap → Less overfitting!")
    print()

    # Feature importance
    print("FEATURE IMPORTANCE:")
    print("-" * 70)
    print("Random Forest automatically calculates feature importance!")
    print(f"   Feature 1 importance: {rf_model.feature_importances_[0]:.3f}")
    print(f"   Feature 2 importance: {rf_model.feature_importances_[1]:.3f}")
    print()
    print("Higher value = more important for predictions")
    print()

    # Classification report
    y_pred_test = rf_model.predict(X_test_sk)
    print("Classification Report:")
    print(classification_report(y_test_sk, y_pred_test, target_names=['Class 0', 'Class 1']))

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not installed")
    sklearn_available = False

# ============================================================================
# SECTION 5: OUT-OF-BAG ERROR
# ============================================================================

print("=" * 80)
print("SECTION 5: Out-of-Bag (OOB) Error - Free Validation!")
print("=" * 80)
print()

print("CLEVER TRICK: Free validation without a separate test set!")
print()

print("HOW IT WORKS:")
print("-" * 70)
print()
print("1. When training each tree with bootstrap sampling:")
print("   • ~63% of samples are USED (in-bag)")
print("   • ~37% of samples are LEFT OUT (out-of-bag)")
print()
print("2. For each sample:")
print("   • Find all trees that DIDN'T use it in training")
print("   • Get predictions from those trees")
print("   • This is like validation (unbiased estimate)")
print()
print("3. Average OOB predictions across all samples")
print("   → Free estimate of test error!")
print()

print("WHY THIS IS USEFUL:")
print("   ✓ Don't need separate validation set")
print("   ✓ Use all data for training")
print("   ✓ Still get unbiased error estimate")
print("   ✓ Great for small datasets")
print()

if sklearn_available:
    # Train with OOB score
    print("Training Random Forest with OOB estimation...")
    rf_oob = RandomForestClassifier(n_estimators=100, max_depth=10,
                                     oob_score=True, random_state=42)
    rf_oob.fit(X_train, y_train)

    print(f"OOB Score: {rf_oob.oob_score_*100:.2f}%")
    print("   (This is unbiased estimate of test accuracy)")
    print()

# ============================================================================
# SECTION 6: HYPERPARAMETERS
# ============================================================================

print("=" * 80)
print("SECTION 6: Random Forest Hyperparameters")
print("=" * 80)
print()

print("KEY HYPERPARAMETERS:")
print()

print("1. n_estimators (number of trees)")
print("   • Default: 100")
print("   • More trees = better performance (up to a point)")
print("   • More trees = slower training")
print("   • Rule: Start with 100, increase if needed")
print()

print("2. max_depth (maximum tree depth)")
print("   • Default: None (trees grow until pure)")
print("   • Lower depth = faster, less overfitting")
print("   • Higher depth = more complex patterns")
print("   • Rule: Try None first, limit if overfitting")
print()

print("3. max_features (features per split)")
print("   • Default: 'sqrt' for classification")
print("   • 'sqrt': √(n_features) - recommended for classification")
print("   • 'log2': log₂(n_features)")
print("   • int: exact number")
print("   • Lower = more diversity = less overfitting")
print()

print("4. min_samples_split")
print("   • Minimum samples to split a node")
print("   • Default: 2")
print("   • Higher = more regularization")
print()

print("5. min_samples_leaf")
print("   • Minimum samples in a leaf")
print("   • Default: 1")
print("   • Higher = smoother decision boundaries")
print()

print("6. bootstrap")
print("   • Whether to use bootstrap sampling")
print("   • Default: True (recommended)")
print("   • False = use all data (becomes 'Extra Trees')")
print()

print("RECOMMENDED STARTING POINT:")
print("-" * 70)
print("RandomForestClassifier(")
print("    n_estimators=100,      # Good balance")
print("    max_depth=None,        # Let trees grow")
print("    max_features='sqrt',   # Standard for classification")
print("    min_samples_split=2,   # Default is fine")
print("    bootstrap=True,        # Essential for RF")
print("    oob_score=True,        # Get free validation")
print("    random_state=42        # Reproducibility")
print(")")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Decision boundaries comparison
if sklearn_available:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Decision Boundaries: Single Tree vs Random Forest', fontsize=16, fontweight='bold')

    # Create mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    models = [
        ("Single Deep Tree\n(Overfits)", DecisionTreeClassifier(random_state=42)),
        ("Single Pruned Tree\n(Underfits)", DecisionTreeClassifier(max_depth=3, random_state=42)),
        ("Random Forest (100 trees)\n(Just Right!)", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ]

    for idx, (title, model) in enumerate(models):
        ax = axes[idx]

        # Train model
        model.fit(X_train_sk, y_train_sk)

        # Predict on mesh
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['red', 'blue'])
        ax.contour(xx, yy, Z, colors='green', linewidths=2, levels=1)

        # Plot training points
        ax.scatter(X_train_sk[y_train_sk==0, 0], X_train_sk[y_train_sk==0, 1],
                  c='red', s=50, alpha=0.8, edgecolors='black', linewidth=1, label='Class 0')
        ax.scatter(X_train_sk[y_train_sk==1, 0], X_train_sk[y_train_sk==1, 1],
                  c='blue', s=50, alpha=0.8, edgecolors='black', linewidth=1, label='Class 1')

        # Calculate accuracies
        train_pred = model.predict(X_train_sk)
        test_pred = model.predict(X_test_sk)
        train_acc = accuracy_score(y_train_sk, train_pred)
        test_acc = accuracy_score(y_test_sk, test_pred)

        ax.set_xlabel('Feature 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\nTrain: {train_acc*100:.1f}% | Test: {test_acc*100:.1f}%',
                    fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/01_comparison_boundaries.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/01_comparison_boundaries.png")
    plt.close()

# Visualization 2: Ensemble voting illustration
if sklearn_available:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Random Forest: Individual Trees vs Ensemble', fontsize=16, fontweight='bold')

    # Train Random Forest to access individual trees
    rf_viz = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_viz.fit(X_train_sk, y_train_sk)

    # Create mesh
    x_min, x_max = X_train_sk[:, 0].min() - 1, X_train_sk[:, 0].max() + 1
    y_min, y_max = X_train_sk[:, 1].min() - 1, X_train_sk[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))

    # Show 5 individual trees
    tree_indices = [0, 10, 25, 50, 75]
    for i, tree_idx in enumerate(tree_indices):
        ax = axes[i // 3, i % 3]

        # Get predictions from individual tree
        tree = rf_viz.estimators_[tree_idx]
        Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['red', 'blue'])
        ax.contour(xx, yy, Z, colors='gray', linewidths=1, levels=1, alpha=0.5)

        ax.scatter(X_train_sk[y_train_sk==0, 0], X_train_sk[y_train_sk==0, 1],
                  c='red', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax.scatter(X_train_sk[y_train_sk==1, 0], X_train_sk[y_train_sk==1, 1],
                  c='blue', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Accuracy of this single tree
        tree_pred = tree.predict(X_test_sk)
        tree_acc = accuracy_score(y_test_sk, tree_pred)

        ax.set_xlabel('Feature 1', fontsize=10)
        ax.set_ylabel('Feature 2', fontsize=10)
        ax.set_title(f'Tree {tree_idx+1}/100\nTest Acc: {tree_acc*100:.1f}%',
                    fontsize=10, fontweight='bold', color='orange')
        ax.grid(True, alpha=0.3)

    # Show ensemble result
    ax = axes[1, 2]

    # Ensemble predictions
    Z_ensemble = rf_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_ensemble = Z_ensemble.reshape(xx.shape)

    ax.contourf(xx, yy, Z_ensemble, alpha=0.3, levels=1, colors=['red', 'blue'])
    ax.contour(xx, yy, Z_ensemble, colors='green', linewidths=2, levels=1)

    ax.scatter(X_train_sk[y_train_sk==0, 0], X_train_sk[y_train_sk==0, 1],
              c='red', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.scatter(X_train_sk[y_train_sk==1, 0], X_train_sk[y_train_sk==1, 1],
              c='blue', s=30, alpha=0.8, edgecolors='black', linewidth=0.5)

    ensemble_acc = rf_test_acc
    ax.set_xlabel('Feature 1', fontsize=10)
    ax.set_ylabel('Feature 2', fontsize=10)
    ax.set_title(f'ENSEMBLE (100 trees)\nTest Acc: {ensemble_acc*100:.1f}%\n→ Better than individual trees!',
                fontsize=10, fontweight='bold', color='green')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_individual_vs_ensemble.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_individual_vs_ensemble.png")
    plt.close()

# Visualization 3: Feature importance
if sklearn_available:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle('Feature Importance in Random Forest', fontsize=14, fontweight='bold')

    # For demonstration, create data with more features
    np.random.seed(42)
    n_samples_demo = 200
    # Create 8 features
    X_demo = np.random.randn(n_samples_demo, 8)
    # Make features 0, 2, 5 important for the target
    y_demo = ((X_demo[:, 0] + X_demo[:, 2] * 2 + X_demo[:, 5] * 1.5) > 0).astype(int)

    # Train Random Forest
    rf_importance = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_importance.fit(X_demo, y_demo)

    # Get feature importances
    importances = rf_importance.feature_importances_
    feature_names = [f'Feature {i}' for i in range(8)]

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    # Plot
    colors = ['green' if importances[i] > 0.1 else 'lightblue' for i in indices]
    ax.barh(range(len(importances)), importances[indices], color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Which Features Matter Most?', fontsize=12, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add importance values on bars
    for i, (idx, imp) in enumerate(zip(indices, importances[indices])):
        ax.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10, fontweight='bold')

    # Add interpretation text
    text = """
INTERPRETATION:
═══════════════════════════════════════════════
• Higher score = more important for predictions
• Calculated by: contribution to Gini reduction
• Features with high importance drive decisions
• Low importance features can be removed
═══════════════════════════════════════════════
    """
    ax.text(0.5, -0.15, text.strip(),
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           horizontalalignment='left',
           fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/03_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/03_feature_importance.png")
    plt.close()

print()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🌲 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ ENSEMBLE LEARNING: Many weak learners → One strong learner")
print()
print("✓ RANDOM FOREST = Decision Trees + Bagging + Feature Randomness")
print()
print("✓ HOW IT WORKS:")
print("   1. Bootstrap sampling → Different training data for each tree")
print("   2. Feature randomness → Only consider random subset at each split")
print("   3. Grow deep trees → Low bias")
print("   4. Vote on predictions → Low variance")
print()
print("✓ KEY ADVANTAGES:")
print("   ✓ More accurate than single trees")
print("   ✓ Resistant to overfitting")
print("   ✓ Handles non-linear relationships")
print("   ✓ Automatic feature importance")
print("   ✓ Works well with default parameters")
print("   ✓ Can handle missing data")
print()
print("✓ KEY DISADVANTAGES:")
print("   ✗ Less interpretable than single tree")
print("   ✗ Slower to train and predict")
print("   ✗ Larger model size")
print("   ✗ Not great for extrapolation")
print()
print("✓ OUT-OF-BAG ERROR:")
print("   • Free validation estimate")
print("   • Uses samples not in bootstrap")
print("   • ~37% of data per tree")
print()
print("✓ FEATURE IMPORTANCE:")
print("   • Automatic calculation")
print("   • Based on Gini reduction")
print("   • Helps understand what drives predictions")
print()
print("✓ KEY HYPERPARAMETERS:")
print("   • n_estimators: Number of trees (start with 100)")
print("   • max_depth: Tree depth (None = unlimited)")
print("   • max_features: Features per split ('sqrt' for classification)")
print()
print("WHEN TO USE:")
print("   • Need high accuracy")
print("   • Have enough computational resources")
print("   • Don't need strict interpretability")
print("   • Have non-linear relationships")
print()
print("WHEN NOT TO USE:")
print("   • Need fast predictions (millions per second)")
print("   • Need to explain every decision")
print("   • Have very limited memory")
print("   • Need to extrapolate beyond training range")
print()
print("=" * 80)
print("🌲 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
