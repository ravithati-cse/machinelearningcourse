"""
⚖️ MODEL COMPARISON - Which Algorithm Works Best?
===============================================================

PROJECT OVERVIEW:
----------------
Compare ALL classification algorithms you've learned on the same dataset!
Learn when to use which algorithm and understand their tradeoffs.

LEARNING OBJECTIVES:
-------------------
1. Comparing multiple algorithms systematically
2. Understanding algorithm strengths and weaknesses
3. Performance vs complexity tradeoffs
4. Training time vs accuracy tradeoffs
5. Interpretability vs performance
6. Choosing the right algorithm for your problem

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Machine Learning Fundamentals"
   https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF
   Compare different algorithms

📚 Krish Naik: "Complete Machine Learning Algorithms Comparison"
   Comprehensive algorithm overview

📚 Data School: "Comparing ML Algorithms"
   Practical algorithm selection guide

TIME: 2-3 hours
DIFFICULTY: Intermediate-Advanced
PREREQUISITES: ALL classification algorithm modules!

ALGORITHMS COMPARED:
-------------------
1. Logistic Regression - Linear decision boundary
2. K-Nearest Neighbors - Distance-based
3. Decision Tree - Tree-based rules
4. Random Forest - Ensemble of trees
5. (Bonus) Naive Bayes - Probabilistic

EVALUATION DIMENSIONS:
---------------------
• Accuracy, Precision, Recall, F1
• ROC AUC
• Training time
• Prediction time
• Model complexity
• Interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'model_comparison'
DATA_DIR = PROJECT_DIR / 'data'

VISUAL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("⚖️ MODEL COMPARISON PROJECT")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("SECTION 1: Loading Dataset")
print("=" * 80)
print()

try:
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, confusion_matrix,
                                  classification_report, roc_curve, auc)
    import pandas as pd

    print("Creating synthetic classification dataset...")
    print()

    # Create challenging but realistic dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        weights=[0.7, 0.3],  # Imbalanced
        flip_y=0.05,  # 5% label noise
        random_state=42
    )

    print("DATASET CHARACTERISTICS:")
    print("-" * 70)
    print(f"   Samples: {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: 2 (binary classification)")
    print(f"   Class 0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"   Class 1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
    print(f"   Imbalanced: Yes (70-30 split)")
    print(f"   Noise: 5% (realistic)")
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not available")
    sklearn_available = False
    exit(1)

# ============================================================================
# SECTION 2: TRAIN-TEST SPLIT
# ============================================================================

print("=" * 80)
print("SECTION 2: Train-Test Split and Preprocessing")
print("=" * 80)
print()

# Split data (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train):,} samples")
print(f"Test set:     {len(X_test):,} samples")
print()

# Scale features (important for some algorithms)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (StandardScaler)")
print()

# ============================================================================
# SECTION 3: TRAIN ALL MODELS
# ============================================================================

print("=" * 80)
print("SECTION 3: Training All Classification Algorithms")
print("=" * 80)
print()

print("Training 5 different classification algorithms...")
print("(This may take a minute...)")
print()

# Define all models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Naive Bayes': GaussianNB()
}

results = {}

print("-" * 90)
print(f"{'Model':<22} {'Train Time':<12} {'Predict Time':<14} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1'}")
print("-" * 90)

for name, model in models.items():
    # Train and time it
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time

    # Predict and time it
    start_time = time.time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.time() - start_time

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_proba = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC
    if len(np.unique(y_proba)) > 2:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = roc_auc_score(y_test, y_pred)

    # Store results
    results[name] = {
        'model': model,
        'train_time': train_time,
        'predict_time': predict_time,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }

    print(f"{name:<22} {train_time:<11.3f}s {predict_time:<13.6f}s {accuracy:<9.3f} {precision:<9.3f} {recall:<9.3f} {f1:.3f}")

print()

# ============================================================================
# SECTION 4: DETAILED COMPARISON
# ============================================================================

print("=" * 80)
print("SECTION 4: Detailed Performance Comparison")
print("=" * 80)
print()

# Find best models for each metric
best_accuracy = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_precision = max(results.keys(), key=lambda k: results[k]['precision'])
best_recall = max(results.keys(), key=lambda k: results[k]['recall'])
best_f1 = max(results.keys(), key=lambda k: results[k]['f1'])
best_roc_auc = max(results.keys(), key=lambda k: results[k]['roc_auc'])
fastest_train = min(results.keys(), key=lambda k: results[k]['train_time'])
fastest_predict = min(results.keys(), key=lambda k: results[k]['predict_time'])

print("BEST PERFORMERS:")
print("-" * 70)
print(f"   Highest Accuracy:  {best_accuracy:<25} ({results[best_accuracy]['accuracy']:.3f})")
print(f"   Highest Precision: {best_precision:<25} ({results[best_precision]['precision']:.3f})")
print(f"   Highest Recall:    {best_recall:<25} ({results[best_recall]['recall']:.3f})")
print(f"   Highest F1:        {best_f1:<25} ({results[best_f1]['f1']:.3f})")
print(f"   Highest ROC AUC:   {best_roc_auc:<25} ({results[best_roc_auc]['roc_auc']:.3f})")
print(f"   Fastest Training:  {fastest_train:<25} ({results[fastest_train]['train_time']:.3f}s)")
print(f"   Fastest Predict:   {fastest_predict:<25} ({results[fastest_predict]['predict_time']:.6f}s)")
print()

# ============================================================================
# SECTION 5: ALGORITHM CHARACTERISTICS
# ============================================================================

print("=" * 80)
print("SECTION 5: Algorithm Characteristics and Tradeoffs")
print("=" * 80)
print()

characteristics = {
    'Logistic Regression': {
        'interpretability': 'High',
        'training_speed': 'Fast',
        'prediction_speed': 'Very Fast',
        'handles_non_linear': 'No',
        'needs_scaling': 'Yes',
        'prone_to_overfit': 'Low',
        'good_for': 'Linear relationships, interpretability'
    },
    'K-Nearest Neighbors': {
        'interpretability': 'Medium',
        'training_speed': 'Very Fast (lazy)',
        'prediction_speed': 'Slow',
        'handles_non_linear': 'Yes',
        'needs_scaling': 'Yes',
        'prone_to_overfit': 'Medium',
        'good_for': 'Small datasets, non-linear patterns'
    },
    'Decision Tree': {
        'interpretability': 'Very High',
        'training_speed': 'Fast',
        'prediction_speed': 'Very Fast',
        'handles_non_linear': 'Yes',
        'needs_scaling': 'No',
        'prone_to_overfit': 'High',
        'good_for': 'Explainability, categorical features'
    },
    'Random Forest': {
        'interpretability': 'Low',
        'training_speed': 'Slow',
        'prediction_speed': 'Medium',
        'handles_non_linear': 'Yes',
        'needs_scaling': 'No',
        'prone_to_overfit': 'Low',
        'good_for': 'High accuracy, robust predictions'
    },
    'Naive Bayes': {
        'interpretability': 'High',
        'training_speed': 'Very Fast',
        'prediction_speed': 'Very Fast',
        'handles_non_linear': 'Limited',
        'needs_scaling': 'No',
        'prone_to_overfit': 'Low',
        'good_for': 'Text classification, fast predictions'
    }
}

print(f"{'Algorithm':<22} {'Interpretability':<18} {'Training':<15} {'Overfitting Risk'}")
print("-" * 80)
for name, chars in characteristics.items():
    print(f"{name:<22} {chars['interpretability']:<18} {chars['training_speed']:<15} {chars['prone_to_overfit']}")

print()

# ============================================================================
# SECTION 6: WHEN TO USE WHICH ALGORITHM
# ============================================================================

print("=" * 80)
print("SECTION 6: Algorithm Selection Guide")
print("=" * 80)
print()

print("LOGISTIC REGRESSION:")
print("-" * 70)
print("   USE WHEN:")
print("      • Need interpretable model (explain coefficients)")
print("      • Linear relationship between features and target")
print("      • Want fast training and prediction")
print("      • Need probability estimates")
print()
print("   AVOID WHEN:")
print("      • Relationships are highly non-linear")
print("      • Features are not independent")
print()

print("K-NEAREST NEIGHBORS:")
print("-" * 70)
print("   USE WHEN:")
print("      • Small to medium dataset")
print("      • Non-linear decision boundaries")
print("      • Simple baseline needed")
print()
print("   AVOID WHEN:")
print("      • Large dataset (slow predictions)")
print("      • High-dimensional data (curse of dimensionality)")
print("      • Need fast predictions")
print()

print("DECISION TREE:")
print("-" * 70)
print("   USE WHEN:")
print("      • Need maximum interpretability")
print("      • Have categorical features")
print("      • Don't want to scale features")
print("      • Want to visualize decision process")
print()
print("   AVOID WHEN:")
print("      • Data is noisy (prone to overfitting)")
print("      • Need most accurate model")
print()

print("RANDOM FOREST:")
print("-" * 70)
print("   USE WHEN:")
print("      • Need high accuracy (often best out-of-box)")
print("      • Have enough computational resources")
print("      • Don't need strict interpretability")
print("      • Want robust, stable predictions")
print()
print("   AVOID WHEN:")
print("      • Need fast predictions (real-time systems)")
print("      • Must explain every decision")
print("      • Limited memory/computing")
print()

print("NAIVE BAYES:")
print("-" * 70)
print("   USE WHEN:")
print("      • Text classification (spam, sentiment)")
print("      • Need very fast training and prediction")
print("      • Features are independent")
print("      • Baseline model needed")
print()
print("   AVOID WHEN:")
print("      • Features are strongly correlated")
print("      • Need highest possible accuracy")
print()

# ============================================================================
# SECTION 7: CROSS-VALIDATION
# ============================================================================

print("=" * 80)
print("SECTION 7: Cross-Validation for Robust Evaluation")
print("=" * 80)
print()

print("Single train-test split can be misleading.")
print("Using 5-fold cross-validation for more reliable estimates...")
print()

print("-" * 70)
print(f"{'Model':<22} {'CV Mean':<12} {'CV Std':<12} {'Interpretation'}")
print("-" * 70)

for name, model in models.items():
    # 5-fold cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    interpretation = "Stable" if cv_std < 0.02 else "Variable"

    print(f"{name:<22} {cv_mean:<11.3f} {cv_std:<11.3f} {interpretation}")

print()
print("Lower std = more stable/reliable model")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Comprehensive comparison
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
fig.suptitle('Classification Algorithm Comparison', fontsize=16, fontweight='bold')

# Plot 1: Accuracy comparison
ax1 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
colors = ['green' if name == best_accuracy else 'lightblue' for name in names]
bars = ax1.bar(range(len(names)), accuracies, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax1.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax1.set_title('Accuracy Comparison', fontsize=11, fontweight='bold')
ax1.set_ylim([min(accuracies) - 0.05, 1.0])
ax1.grid(axis='y', alpha=0.3)

# Plot 2: F1 Score comparison
ax2 = fig.add_subplot(gs[0, 1])
f1_scores = [results[name]['f1'] for name in names]
colors = ['green' if name == best_f1 else 'lightblue' for name in names]
bars = ax2.bar(range(len(names)), f1_scores, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(len(names)))
ax2.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax2.set_ylabel('F1 Score', fontsize=10, fontweight='bold')
ax2.set_title('F1 Score Comparison', fontsize=11, fontweight='bold')
ax2.set_ylim([min(f1_scores) - 0.05, 1.0])
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Training time comparison
ax3 = fig.add_subplot(gs[0, 2])
train_times = [results[name]['train_time'] for name in names]
colors = ['green' if name == fastest_train else 'lightcoral' for name in names]
bars = ax3.bar(range(len(names)), train_times, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(names)))
ax3.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
ax3.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
ax3.set_title('Training Time', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Plot 4: ROC Curves
ax4 = fig.add_subplot(gs[1, :2])
for name in names:
    y_proba = results[name]['y_proba']
    if len(np.unique(y_proba)) > 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        linestyle = '-' if name == best_roc_auc else '--'
        linewidth = 3 if name == best_roc_auc else 2
        ax4.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
                label=f'{name} (AUC={roc_auc:.3f})')

ax4.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax4.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax4.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax4.set_title('ROC Curves - All Models', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)

# Plot 5: Precision-Recall tradeoff
ax5 = fig.add_subplot(gs[1, 2])
precisions = [results[name]['precision'] for name in names]
recalls = [results[name]['recall'] for name in names]
ax5.scatter(recalls, precisions, s=200, c=range(len(names)), cmap='viridis',
           edgecolors='black', linewidth=2)
for i, name in enumerate(names):
    ax5.annotate(name.split()[0], (recalls[i], precisions[i]),
                fontsize=8, ha='center', va='center', fontweight='bold', color='white')
ax5.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax5.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax5.set_title('Precision-Recall Tradeoff', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([min(recalls) - 0.05, max(recalls) + 0.05])
ax5.set_ylim([min(precisions) - 0.05, max(precisions) + 0.05])

# Plot 6: Speed comparison (log scale)
ax6 = fig.add_subplot(gs[2, :])
x_pos = np.arange(len(names))
width = 0.35

train_bars = ax6.bar(x_pos - width/2, train_times, width, label='Training Time',
                     color='steelblue', edgecolor='black', linewidth=1.5)
predict_times = [results[name]['predict_time'] * 1000 for name in names]  # Convert to ms
predict_bars = ax6.bar(x_pos + width/2, predict_times, width, label='Prediction Time (ms)',
                      color='coral', edgecolor='black', linewidth=1.5)

ax6.set_xticks(x_pos)
ax6.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=10)
ax6.set_ylabel('Time (seconds / milliseconds)', fontsize=11, fontweight='bold')
ax6.set_title('Speed Comparison: Training vs Prediction', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3)

plt.savefig(f'{VISUAL_DIR}/01_complete_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_complete_comparison.png")
plt.close()

print()

# ============================================================================
# PROJECT SUMMARY
# ============================================================================

print()
print("=" * 80)
print("⚖️ PROJECT COMPLETE: Algorithm Comparison Summary")
print("=" * 80)
print()

print("✓ ALGORITHMS COMPARED: 5")
print("   • Logistic Regression")
print("   • K-Nearest Neighbors")
print("   • Decision Tree")
print("   • Random Forest")
print("   • Naive Bayes")
print()

print("✓ WINNER BY CATEGORY:")
print("-" * 70)
print(f"   🏆 Best Overall (F1):       {best_f1}")
print(f"   🎯 Best Accuracy:           {best_accuracy}")
print(f"   ⚡ Fastest Training:        {fastest_train}")
print(f"   🚀 Fastest Prediction:      {fastest_predict}")
print(f"   📊 Best ROC AUC:            {best_roc_auc}")
print()

print("✓ KEY INSIGHTS:")
print()
print("   1. NO SINGLE 'BEST' ALGORITHM:")
print("      • Performance depends on your specific problem")
print("      • Different metrics favor different algorithms")
print("      • Always try multiple approaches")
print()

print("   2. TRADEOFFS EVERYWHERE:")
print("      • Accuracy vs Speed")
print("      • Complexity vs Interpretability")
print("      • Training time vs Prediction time")
print()

print("   3. GENERAL RECOMMENDATIONS:")
print("      • Start simple: Logistic Regression")
print("      • Need accuracy: Random Forest")
print("      • Need speed: Naive Bayes")
print("      • Need interpretability: Decision Tree")
print("      • Need flexibility: KNN")
print()

print("✓ DECISION FRAMEWORK:")
print()
print("   1. Try Logistic Regression (baseline)")
print("   2. If not satisfactory:")
print("      → Linear data: Try regularized regression")
print("      → Non-linear data: Try Random Forest or KNN")
print("      → Need interpretability: Try Decision Tree")
print("   3. Optimize best performer with hyperparameter tuning")
print("   4. Ensemble multiple models if possible")
print()

print("=" * 80)
print("⚖️ Model Comparison Project Complete!")
print(f"   Visualizations: {VISUAL_DIR}/")
print("=" * 80)
