"""
🌳 DECISION TREES - Intuitive Classification Through Questions
===============================================================

LEARNING OBJECTIVES:
-------------------
After this module, you'll understand:
1. How decision trees make predictions using if-then rules
2. Splitting criteria: Gini impurity and Information Gain
3. How trees grow (recursive splitting)
4. Overfitting and the importance of pruning
5. Implementing a simple decision tree from scratch
6. Using scikit-learn's DecisionTreeClassifier
7. Visualizing tree structure and decision boundaries

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Decision Trees"
   https://www.youtube.com/watch?v=7VeUPuFGJHk
   THE BEST explanation of decision trees!

⭐ StatQuest: "Decision Trees, Part 2 - Feature Selection and Missing Data"
   https://www.youtube.com/watch?v=wpNl-JwwplA
   How trees choose which feature to split on

📚 Josh Starmer: "Gini Impurity and Information Gain"
   Mathematical details of splitting criteria

TIME: 60-75 minutes
DIFFICULTY: Intermediate
PREREQUISITES: 04_confusion_matrix.py, 05_decision_boundaries.py

KEY CONCEPTS:
------------
- Decision Tree: Series of if-then questions
- Node: Decision point or leaf (final prediction)
- Splitting: Choosing best feature and threshold
- Gini Impurity: Measure of class mixture
- Information Gain: Reduction in impurity after split
- Overfitting: Trees that are too deep
- Pruning: Limiting tree depth to prevent overfitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
from collections import Counter

# Setup visualization directory
VISUAL_DIR = Path(__file__).parent.parent / 'visuals' / 'decision_trees'
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("🌳 DECISION TREES - Classification Through Questions")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: THE DECISION TREE CONCEPT
# ============================================================================

print("=" * 80)
print("SECTION 1: What is a Decision Tree?")
print("=" * 80)
print()

print("A decision tree is like playing '20 Questions' to classify data!")
print()

print("EXAMPLE: Should I go outside?")
print()
print("                    Is it raining?")
print("                    /            \\")
print("                  YES             NO")
print("                  /                \\")
print("            Stay Inside      Is it hot?")
print("                             /        \\")
print("                           YES        NO")
print("                           /            \\")
print("                    Go Swimming    Go for Walk")
print()

print("In ML terms:")
print("   • Each question is a SPLIT on a feature")
print("   • Each answer leads to another question or a PREDICTION")
print("   • The path from top to bottom is a RULE")
print()

print("Email Spam Classification Example:")
print()
print("                Contains 'FREE'?")
print("                /               \\")
print("              YES                NO")
print("              /                   \\")
print("         SPAM             Many exclamation marks?")
print("                          /                    \\")
print("                        YES                    NO")
print("                        /                        \\")
print("                    SPAM                    NOT SPAM")
print()

print("KEY ADVANTAGES:")
print("   ✓ Easy to understand (visual rules)")
print("   ✓ No feature scaling needed")
print("   ✓ Handles non-linear relationships")
print("   ✓ Can explain every prediction")
print()

print("KEY DISADVANTAGES:")
print("   ✗ Prone to overfitting")
print("   ✗ Unstable (small data changes = big tree changes)")
print("   ✗ Not as accurate as ensemble methods")
print()

# ============================================================================
# SECTION 2: SPLITTING CRITERIA
# ============================================================================

print("=" * 80)
print("SECTION 2: How Trees Decide Where to Split")
print("=" * 80)
print()

print("The key question: Which feature and threshold gives the BEST split?")
print()

print("GINI IMPURITY (Most Common)")
print("   Measures how 'mixed' the classes are")
print()
print("   Formula: Gini = 1 - Σ(p_i²)")
print("   where p_i is proportion of class i")
print()

def gini_impurity(classes):
    """Calculate Gini impurity"""
    if len(classes) == 0:
        return 0
    counts = np.bincount(classes)
    probabilities = counts / len(classes)
    return 1 - np.sum(probabilities ** 2)

print("Examples:")
print("-" * 70)

# Pure node (all same class)
pure = np.array([0, 0, 0, 0, 0])
gini_pure = gini_impurity(pure)
print(f"All Class 0: {pure}")
print(f"   Gini = {gini_pure:.3f} (Perfect! No mixing)")
print()

# Balanced mix
balanced = np.array([0, 0, 0, 1, 1, 1])
gini_bal = gini_impurity(balanced)
print(f"50/50 mix: {balanced}")
print(f"   Gini = {gini_bal:.3f} (Maximum impurity for 2 classes)")
print()

# Slightly mixed
slight_mix = np.array([0, 0, 0, 0, 1])
gini_slight = gini_impurity(slight_mix)
print(f"80/20 mix: {slight_mix}")
print(f"   Gini = {gini_slight:.3f} (Some impurity)")
print()

print("GINI INTERPRETATION:")
print("   • Gini = 0.0 → Pure node (all same class)")
print("   • Gini = 0.5 → Maximum mixing (50/50 for binary)")
print("   • Lower Gini = Better split")
print()

print("INFORMATION GAIN (Alternative)")
print("   Based on entropy from information theory")
print("   Measures uncertainty reduction")
print()
print("   Formula: Entropy = -Σ(p_i × log₂(p_i))")
print("   Information Gain = Entropy_before - Weighted_Entropy_after")
print()
print("   Both Gini and Entropy give similar results!")
print()

# ============================================================================
# SECTION 3: HOW TREES GROW
# ============================================================================

print("=" * 80)
print("SECTION 3: Growing a Decision Tree")
print("=" * 80)
print()

print("ALGORITHM (Recursive):")
print()
print("1. START with all data at root node")
print()
print("2. FOR each feature:")
print("   • Try different thresholds")
print("   • Calculate Gini impurity after split")
print("   • Choose split that MINIMIZES impurity")
print()
print("3. SPLIT data into left and right child nodes")
print()
print("4. REPEAT steps 2-3 for each child (recursion)")
print()
print("5. STOP when:")
print("   • Node is pure (all same class)")
print("   • Reached maximum depth")
print("   • Too few samples to split")
print()

print("Example: Binary split on feature x₁")
print("-" * 70)
print()
print("Parent node: [0, 0, 0, 1, 1, 1, 1, 1]")
print(f"   Gini = {gini_impurity(np.array([0,0,0,1,1,1,1,1])):.3f}")
print()
print("Try split: x₁ <= 5")
print("   Left:  [0, 0, 0, 1, 1]")
print(f"   Gini_left = {gini_impurity(np.array([0,0,0,1,1])):.3f}")
print("   Right: [1, 1, 1]")
print(f"   Gini_right = {gini_impurity(np.array([1,1,1])):.3f}")
print()
print("Weighted Gini after split:")
weighted = (5/8) * gini_impurity(np.array([0,0,0,1,1])) + (3/8) * gini_impurity(np.array([1,1,1]))
print(f"   = (5/8) × {gini_impurity(np.array([0,0,0,1,1])):.3f} + (3/8) × {gini_impurity(np.array([1,1,1])):.3f}")
print(f"   = {weighted:.3f}")
print()
print(f"Information Gain = {gini_impurity(np.array([0,0,0,1,1,1,1,1])):.3f} - {weighted:.3f} = {gini_impurity(np.array([0,0,0,1,1,1,1,1])) - weighted:.3f}")
print("   → This split REDUCES impurity!")
print()

# ============================================================================
# SECTION 4: SIMPLE TREE IMPLEMENTATION
# ============================================================================

print("=" * 80)
print("SECTION 4: Simple Decision Tree from Scratch")
print("=" * 80)
print()

class SimpleDecisionTree:
    """Very simple decision tree (for education only!)"""

    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """Build the tree"""
        self.tree = self._grow_tree(X, y, depth=0)

    def _gini(self, y):
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _best_split(self, X, y):
        """Find best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            # Try different thresholds (use unique values)
            thresholds = np.unique(X[:, feature_idx])

            for threshold in thresholds:
                # Split
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate weighted Gini
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
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            # Leaf node: return most common class
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'value': leaf_value}

        # Find best split
        feature_idx, threshold = self._best_split(X, y)

        if feature_idx is None:
            # Can't split, make leaf
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'value': leaf_value}

        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        # Recursively build left and right subtrees
        left_subtree = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._grow_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            'type': 'split',
            'feature': feature_idx,
            'threshold': threshold,
            'left': left_subtree,
            'right': right_subtree
        }

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

print("Created SimpleDecisionTree class!")
print()

# Generate sample data
np.random.seed(42)
n_samples = 100

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

# Train tree
tree = SimpleDecisionTree(max_depth=3)
tree.fit(X_train, y_train)
print("Trained decision tree with max_depth=3")
print()

# Make predictions
y_pred = tree.predict(X_train)
accuracy = np.mean(y_pred == y_train)
print(f"Training accuracy: {accuracy*100:.2f}%")
print()

# ============================================================================
# SECTION 5: USING SCIKIT-LEARN
# ============================================================================

print("=" * 80)
print("SECTION 5: Using Scikit-Learn's DecisionTreeClassifier")
print("=" * 80)
print()

try:
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("Training trees with different depths...")
    print()

    depths = [1, 2, 3, 5, 10, None]  # None = unlimited
    print("-" * 70)
    print(f"{'Max Depth':<15} {'Train Acc':<15} {'Nodes':<15} {'Interpretation'}")
    print("-" * 70)

    best_depth = None
    best_acc = 0
    models = {}

    for depth in depths:
        # Train model
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, y_train)

        # Accuracy
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        # Number of nodes
        n_nodes = model.tree_.node_count

        # Interpretation
        if depth == 1:
            interp = "Too simple (underfitting)"
        elif depth is None or depth >= 10:
            interp = "Too complex (overfitting)"
        else:
            interp = "Good balance"

        depth_str = str(depth) if depth is not None else "Unlimited"
        print(f"{depth_str:<15} {train_acc*100:<15.1f}% {n_nodes:<15} {interp}")

        models[depth] = model

        if depth in [2, 3, 5] and train_acc > best_acc:
            best_acc = train_acc
            best_depth = depth

    print()
    print(f"Best depth = {best_depth} with accuracy = {best_acc*100:.1f}%")
    print()

    # Use best model
    final_model = models[best_depth]

    print("Final Model Details:")
    print("-" * 70)
    print(f"Max Depth: {final_model.max_depth}")
    print(f"Number of Nodes: {final_model.tree_.node_count}")
    print(f"Number of Leaves: {final_model.get_n_leaves()}")
    print(f"Feature Importances: {final_model.feature_importances_}")
    print()

    # Predictions
    y_pred = final_model.predict(X_train)

    print("Classification Report:")
    print(classification_report(y_train, y_pred, target_names=['Class 0', 'Class 1']))

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not installed")
    sklearn_available = False

# ============================================================================
# SECTION 6: OVERFITTING AND PRUNING
# ============================================================================

print("=" * 80)
print("SECTION 6: Overfitting and Tree Depth")
print("=" * 80)
print()

print("Decision trees naturally overfit if left unpruned!")
print()

print("OVERFITTING happens when:")
print("   • Tree is too deep")
print("   • Tree memorizes training data (including noise)")
print("   • Each leaf has very few samples")
print()

print("PREVENTING OVERFITTING:")
print()
print("1. Limit max_depth")
print("   • Depth 3-5 often works well")
print("   • Deeper = more complex = more overfitting")
print()

print("2. Set min_samples_split")
print("   • Don't split if node has < N samples")
print("   • Forces larger leaves")
print()

print("3. Set min_samples_leaf")
print("   • Each leaf must have >= N samples")
print("   • Prevents tiny leaves")
print()

print("4. Use max_leaf_nodes")
print("   • Limit total number of leaves")
print("   • Controls overall complexity")
print()

print("5. Use ensemble methods")
print("   • Random Forests (next module!)")
print("   • Average multiple trees")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Tree structure and splits
if sklearn_available:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Decision Tree Structure and Visualization', fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Shallow tree (depth=1)
    ax1 = axes[0, 0]
    tree_d1 = DecisionTreeClassifier(max_depth=1, random_state=42)
    tree_d1.fit(X_train, y_train)
    plot_tree(tree_d1, ax=ax1, filled=True, feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1'], fontsize=10)
    ax1.set_title('Depth = 1 (Stump - Underfitting)', fontsize=12, fontweight='bold', color='orange')

    # Plot 2: Good tree (depth=3)
    ax2 = axes[0, 1]
    tree_d3 = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_d3.fit(X_train, y_train)
    plot_tree(tree_d3, ax=ax2, filled=True, feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1'], fontsize=8)
    ax2.set_title('Depth = 3 (Good Balance)', fontsize=12, fontweight='bold', color='green')

    # Plot 3: Deep tree (depth=5)
    ax3 = axes[1, 0]
    tree_d5 = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree_d5.fit(X_train, y_train)
    plot_tree(tree_d5, ax=ax3, filled=True, feature_names=['Feature 1', 'Feature 2'],
             class_names=['Class 0', 'Class 1'], fontsize=6)
    ax3.set_title('Depth = 5 (Starting to Overfit)', fontsize=12, fontweight='bold', color='red')

    # Plot 4: Text explanation
    ax4 = axes[1, 1]
    ax4.axis('off')

    tree_text = """
READING THE TREE
═══════════════════════════════════════════════════

EACH NODE SHOWS:
• Feature and threshold (e.g., "x₁ <= 3.5")
• gini: Impurity measure (0 = pure)
• samples: Number of data points
• value: [class 0 count, class 1 count]
• class: Predicted class (majority)

COLOR CODING:
• Orange shade → More Class 0
• Blue shade → More Class 1
• Darker → More confident (purer)

HOW TO FOLLOW A PATH:
1. Start at top (root)
2. Check condition
3. Go left if TRUE, right if FALSE
4. Repeat until reaching a leaf
5. Leaf's class is the prediction

EXAMPLE PATH (Depth 3 tree):
Point (2.5, 2.5):
→ x₁ <= 3.5? YES → Go left
→ x₂ <= 3.0? YES → Go left
→ Predict: Class 0

Point (6.0, 6.0):
→ x₁ <= 3.5? NO → Go right
→ Predict: Class 1

TREE DEPTH EFFECTS:
• Depth 1: Too simple, can't capture pattern
• Depth 3: Good balance
• Depth 5+: Too complex, overfits noise

═══════════════════════════════════════════════════
"""

    ax4.text(0.5, 0.5, tree_text,
            transform=ax4.transAxes,
            fontsize=9,
            verticalalignment='center',
            horizontalalignment='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/01_tree_structure.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/01_tree_structure.png")
    plt.close()

# Visualization 2: Decision boundaries with different depths
if sklearn_available:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Decision Boundaries: Effect of Tree Depth', fontsize=16, fontweight='bold')

    depths_viz = [1, 2, 3, 5, 10, None]

    # Create mesh
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    for idx, depth in enumerate(depths_viz):
        ax = axes[idx // 3, idx % 3]

        # Train model
        tree_viz = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree_viz.fit(X_train, y_train)

        # Predict on mesh
        Z = tree_viz.predict(np.c_[xx.ravel(), yy.ravel()])
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
        train_pred = tree_viz.predict(X_train)
        acc = accuracy_score(y_train, train_pred)
        n_leaves = tree_viz.get_n_leaves()

        # Determine quality
        if depth == 1:
            quality = "UNDERFITTING"
            color = 'orange'
        elif depth is None or depth >= 10:
            quality = "OVERFITTING"
            color = 'red'
        else:
            quality = "GOOD FIT"
            color = 'green'

        depth_str = str(depth) if depth is not None else "∞"
        ax.set_xlabel('Feature 1', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature 2', fontsize=10, fontweight='bold')
        ax.set_title(f'Depth={depth_str} | Acc={acc*100:.1f}% | Leaves={n_leaves}\n{quality}',
                    fontsize=11, fontweight='bold', color=color)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_decision_boundaries_depth.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_decision_boundaries_depth.png")
    plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print()
print("=" * 80)
print("🌳 SUMMARY: What You Learned")
print("=" * 80)
print()
print("✓ DECISION TREES use if-then rules for classification")
print()
print("✓ HOW THEY WORK:")
print("   1. Start with all data at root")
print("   2. Find best feature and threshold to split")
print("   3. Recursively split child nodes")
print("   4. Stop when pure or max depth reached")
print()
print("✓ SPLITTING CRITERIA:")
print("   • Gini Impurity: Measures class mixture (0 = pure)")
print("   • Information Gain: Reduction in impurity after split")
print("   • Goal: Minimize impurity in child nodes")
print()
print("✓ KEY PARAMETERS:")
print("   • max_depth: Limits tree depth (prevent overfitting)")
print("   • min_samples_split: Minimum samples to split node")
print("   • min_samples_leaf: Minimum samples in leaf")
print()
print("✓ ADVANTAGES:")
print("   ✓ Easy to understand and visualize")
print("   ✓ No feature scaling needed")
print("   ✓ Handles non-linear relationships")
print("   ✓ Interpretable (can explain every decision)")
print()
print("✓ DISADVANTAGES:")
print("   ✗ Prone to overfitting (need pruning)")
print("   ✗ Unstable (small changes → different tree)")
print("   ✗ Not as accurate as ensemble methods")
print()
print("✓ OVERFITTING PREVENTION:")
print("   • Limit tree depth")
print("   • Set minimum samples per split/leaf")
print("   • Use ensemble methods (Random Forests!)")
print()
print("NEXT: We'll learn Random Forests - combining multiple trees!")
print()
print("=" * 80)
print("🌳 Module Complete! Check the visualizations:")
print(f"   {VISUAL_DIR}/")
print("=" * 80)
