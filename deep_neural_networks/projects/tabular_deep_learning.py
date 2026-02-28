"""
🏆 DEEP NEURAL NETWORKS — Project 2: Deep Learning on Tabular Data
====================================================================

Learning Objectives:
  1. Apply DNN to real structured/tabular data (breast cancer dataset)
  2. Build and compare 3 models: Logistic Regression, Random Forest, DNN
  3. Properly preprocess tabular data for neural networks (StandardScaler)
  4. Compare models on accuracy, precision, recall, F1, and AUC
  5. Plot ROC curves for all 3 models together
  6. Understand WHEN deep learning beats traditional ML (and when it doesn't)
  7. Compute feature importance for DNN via permutation importance

YouTube Resources:
  ⭐ StatQuest - Random Forests https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
  ⭐ StatQuest - ROC curves https://www.youtube.com/watch?v=4jRBRDbJemM
  📚 Andrej Karpathy - "Most of the time, don't use DNNs on tabular data"

Time Estimate: 60-75 minutes
Difficulty: Intermediate
Prerequisites: All math foundations + algorithms 1-4 + Project 1 (MNIST)
Key Concepts: tabular data, ROC-AUC, model comparison, feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "tabular_deep_learning")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🏆 PROJECT 2: DEEP LEARNING ON TABULAR DATA")
print("=" * 70)
print()
print("When should you use a Deep Neural Network vs traditional ML?")
print()
print("  Use DNN when:")
print("    - Very large datasets (>100k samples)")
print("    - Complex non-linear interactions between many features")
print("    - Unstructured data (images, text, audio)")
print()
print("  Stick with Random Forest / XGBoost when:")
print("    - Tabular data with <100k samples")
print("    - Need interpretability")
print("    - Need fast training and inference")
print("    - Features have clear independent meanings")
print()
print("Let's test this empirically on real medical data!")
print()


# ======================================================================
# SECTION 1: Load Data
# ======================================================================
print("=" * 70)
print("SECTION 1: BREAST CANCER DATASET")
print("=" * 70)
print()

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

print(f"  Dataset: Breast Cancer Wisconsin (Diagnostic)")
print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"  Classes: {class_names}  (0=malignant, 1=benign)")
print(f"  Class balance: {y.mean():.1%} benign, {(1-y.mean()):.1%} malignant")
print()
print("  Features (30 total):")
for i, fname in enumerate(feature_names):
    print(f"    {i+1:2d}. {fname}")
print()

# Split: 70% train, 15% val, 15% test
X_tr_raw, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(X_tr_raw, y_tr, test_size=0.15, random_state=42)

scaler = StandardScaler()
X_tr  = scaler.fit_transform(X_tr_raw)
X_val = scaler.transform(X_val_raw)
X_te  = scaler.transform(X_te)

print(f"  After split + StandardScaler:")
print(f"    Train: {X_tr.shape[0]} samples")
print(f"    Val:   {X_val.shape[0]} samples")
print(f"    Test:  {X_te.shape[0]} samples")
print()


# ======================================================================
# SECTION 2: Model 1 — Logistic Regression (Baseline)
# ======================================================================
print("=" * 70)
print("SECTION 2: MODEL 1 — LOGISTIC REGRESSION (BASELINE)")
print("=" * 70)
print()
print("  The simplest classifier. Linear decision boundary.")
print("  Fast, interpretable, great baseline.")
print()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix, roc_curve)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_tr, y_tr)

lr_pred  = lr_model.predict(X_te)
lr_prob  = lr_model.predict_proba(X_te)[:, 1]

print(f"  Logistic Regression — Test Results:")
print(f"    Accuracy:  {accuracy_score(y_te, lr_pred):.4f}")
print(f"    Precision: {precision_score(y_te, lr_pred):.4f}")
print(f"    Recall:    {recall_score(y_te, lr_pred):.4f}")
print(f"    F1 Score:  {f1_score(y_te, lr_pred):.4f}")
print(f"    ROC-AUC:   {roc_auc_score(y_te, lr_prob):.4f}")
print()


# ======================================================================
# SECTION 3: Model 2 — Random Forest
# ======================================================================
print("=" * 70)
print("SECTION 3: MODEL 2 — RANDOM FOREST")
print("=" * 70)
print()
print("  Ensemble of decision trees. Strong on tabular data.")
print("  Non-linear, handles outliers, provides feature importance.")
print()

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_tr, y_tr)

rf_pred = rf_model.predict(X_te)
rf_prob = rf_model.predict_proba(X_te)[:, 1]

print(f"  Random Forest — Test Results:")
print(f"    Accuracy:  {accuracy_score(y_te, rf_pred):.4f}")
print(f"    Precision: {precision_score(y_te, rf_pred):.4f}")
print(f"    Recall:    {recall_score(y_te, rf_pred):.4f}")
print(f"    F1 Score:  {f1_score(y_te, rf_pred):.4f}")
print(f"    ROC-AUC:   {roc_auc_score(y_te, rf_prob):.4f}")
print()

# Feature importance
rf_importances = rf_model.feature_importances_
top5_idx = rf_importances.argsort()[::-1][:5]
print("  Top 5 most important features (Random Forest):")
for i in top5_idx:
    print(f"    {feature_names[i]:35s}: {rf_importances[i]:.4f}")
print()


# ======================================================================
# SECTION 4: Model 3 — Deep Neural Network
# ======================================================================
print("=" * 70)
print("SECTION 4: MODEL 3 — DEEP NEURAL NETWORK")
print("=" * 70)
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} — building DNN...")
    print()
except ImportError:
    print("  TensorFlow not installed — using sklearn MLP as substitute.")
    from sklearn.neural_network import MLPClassifier as SklearnMLP
    print()

if TF_AVAILABLE:
    dnn_model = keras.Sequential([
        layers.Dense(128, activation="relu", input_shape=(30,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(1,  activation="sigmoid"),
    ], name="DNN_BreastCancer")

    dnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    dnn_model.summary()
    print()

    callbacks_dnn = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=0
        ),
    ]

    history_dnn = dnn_model.fit(
        X_tr, y_tr,
        epochs=200, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks_dnn, verbose=1
    )
    print()

    dnn_prob = dnn_model.predict(X_te, verbose=0).flatten()
    dnn_pred = (dnn_prob > 0.5).astype(int)

else:
    # Fallback: sklearn MLP
    sk_mlp = SklearnMLP(hidden_layer_sizes=(128, 64, 32), max_iter=500,
                         learning_rate_init=0.001, random_state=42, early_stopping=True)
    sk_mlp.fit(X_tr, y_tr)
    dnn_prob = sk_mlp.predict_proba(X_te)[:, 1]
    dnn_pred = sk_mlp.predict(X_te)
    history_dnn = None

print(f"  Deep Neural Network — Test Results:")
print(f"    Accuracy:  {accuracy_score(y_te, dnn_pred):.4f}")
print(f"    Precision: {precision_score(y_te, dnn_pred):.4f}")
print(f"    Recall:    {recall_score(y_te, dnn_pred):.4f}")
print(f"    F1 Score:  {f1_score(y_te, dnn_pred):.4f}")
print(f"    ROC-AUC:   {roc_auc_score(y_te, dnn_prob):.4f}")
print()


# ======================================================================
# SECTION 5: Model Comparison Summary
# ======================================================================
print("=" * 70)
print("SECTION 5: MODEL COMPARISON SUMMARY")
print("=" * 70)
print()

models_results = {
    "Logistic Regression": {
        "pred": lr_pred, "prob": lr_prob,
        "color": "steelblue", "marker": "o"
    },
    "Random Forest": {
        "pred": rf_pred, "prob": rf_prob,
        "color": "darkorange", "marker": "s"
    },
    "Deep Neural Network": {
        "pred": dnn_pred, "prob": dnn_prob,
        "color": "green", "marker": "^"
    },
}

metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
print(f"  {'Model':25} {'Accuracy':10} {'Precision':10} {'Recall':10} {'F1':10} {'AUC':10}")
print("  " + "-" * 75)

all_metrics = {}
for name, d in models_results.items():
    acc = accuracy_score(y_te, d["pred"])
    pre = precision_score(y_te, d["pred"])
    rec = recall_score(y_te, d["pred"])
    f1  = f1_score(y_te, d["pred"])
    auc = roc_auc_score(y_te, d["prob"])
    all_metrics[name] = [acc, pre, rec, f1, auc]
    print(f"  {name:25} {acc:10.3f} {pre:10.3f} {rec:10.3f} {f1:10.3f} {auc:10.3f}")

print()
print("  VERDICT for this dataset (569 samples, 30 features):")
print("  -> Random Forest typically wins or ties on small tabular data")
print("  -> DNN is competitive but needs careful tuning + more data to shine")
print("  -> Logistic Regression is surprisingly strong when data is linearly separable")
print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Feature correlation heatmap ---
print("Generating: Feature correlation heatmap...")

fig, ax = plt.subplots(figsize=(14, 12))
corr = np.corrcoef(X.T)
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(30)); ax.set_yticks(range(30))
short_names = [n.replace("(", "").replace(")", "")[:18] for n in feature_names]
ax.set_xticklabels(short_names, rotation=90, fontsize=7)
ax.set_yticklabels(short_names, fontsize=7)
ax.set_title("Feature Correlation Heatmap (30 features)", fontsize=13, fontweight="bold")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "feature_correlation.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: feature_correlation.png")


# --- PLOT 2: Model comparison bar chart ---
print("Generating: Model comparison bar chart...")

x = np.arange(len(metrics))
width = 0.25
colors_bar = ["steelblue", "darkorange", "green"]

fig, ax = plt.subplots(figsize=(13, 6))
for i, (name, vals) in enumerate(all_metrics.items()):
    bars = ax.bar(x + i * width, vals, width, label=name,
                  color=colors_bar[i], alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x + width); ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0.8, 1.05)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison: Logistic Regression vs Random Forest vs DNN",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "model_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: model_comparison.png")


# --- PLOT 3: ROC curves ---
print("Generating: ROC curves...")

fig, ax = plt.subplots(figsize=(8, 7))
for name, d in models_results.items():
    fpr, tpr, _ = roc_curve(y_te, d["prob"])
    auc = roc_auc_score(y_te, d["prob"])
    ax.plot(fpr, tpr, color=d["color"], linewidth=2.5,
            label=f"{name} (AUC={auc:.3f})", marker=d["marker"],
            markevery=10, markersize=7)

ax.plot([0, 1], [0, 1], "gray", linestyle="--", linewidth=1.5, label="Random (AUC=0.5)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
ax.set_title("ROC Curves: All 3 Models", fontsize=13, fontweight="bold")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "roc_curves.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: roc_curves.png")


# --- PLOT 4: DNN training curves (if TF available) ---
if TF_AVAILABLE and history_dnn is not None:
    print("Generating: DNN training curves...")

    hist = history_dnn.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("DNN Training on Breast Cancer Dataset", fontsize=13, fontweight="bold")

    axes[0].plot(hist["loss"],     "steelblue",  linewidth=2, label="Train")
    axes[0].plot(hist["val_loss"], "darkorange", linewidth=2, linestyle="--", label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["accuracy"],     "steelblue",  linewidth=2, label="Train")
    axes[1].plot(hist["val_accuracy"], "darkorange", linewidth=2, linestyle="--", label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    axes[1].set_ylim(0.8, 1.0); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "dnn_training.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: dnn_training.png")


print()
print("=" * 70)
print("PROJECT 2: TABULAR DEEP LEARNING COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  On small tabular datasets (<10k rows): Random Forest usually wins")
print("  DNN shines with: large data, complex interactions, unstructured input")
print("  Always compare against baseline (LogReg / RF) before using DNN")
print("  ROC-AUC = best single metric for imbalanced classification")
print("  Batch Normalization + Dropout makes DNN competitive on tabular data")
print()
print("=" * 70)
print("PART 3: DEEP NEURAL NETWORKS — ALL 11 MODULES COMPLETE!")
print("=" * 70)
print()
print("You have covered:")
print("  Math Foundations (5 modules):")
print("    01. Neurons & Activations")
print("    02. Forward Propagation")
print("    03. Backpropagation")
print("    04. Loss Functions & Optimizers")
print("    05. Regularization")
print()
print("  Algorithms (4 modules):")
print("    Perceptron from Scratch")
print("    Multi-Layer Perceptron from Scratch")
print("    MLP with Keras")
print("    Hyperparameter Tuning")
print()
print("  Projects (2 modules):")
print("    MNIST Digit Classifier")
print("    Tabular Deep Learning")
print()
print("Next Part: CNNs — Convolutional Neural Networks for images!")
