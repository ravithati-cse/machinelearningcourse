"""
🤖 DEEP NEURAL NETWORKS — Algorithm 3: MLP with TensorFlow / Keras
====================================================================

Learning Objectives:
  1. Understand why Keras exists (auto-differentiation, GPU, clean API)
  2. Build a Sequential model with Dense layers
  3. Use model.compile() to set optimizer, loss, and metrics
  4. Use model.fit() with callbacks (EarlyStopping, ReduceLROnPlateau)
  5. Evaluate with model.evaluate() and model.predict()
  6. Plot training history (loss and accuracy curves)
  7. Save and reload a trained model

YouTube Resources:
  ⭐ TensorFlow Official - Keras intro https://www.tensorflow.org/tutorials
  ⭐ StatQuest - TensorFlow/Keras https://www.youtube.com/watch?v=tpCFfeUEGs8
  📚 Tech with Tim - Keras tutorial https://www.youtube.com/watch?v=qFJeN9V1ZsI

Time Estimate: 50-60 minutes
Difficulty: Beginner-Intermediate
Prerequisites: Algorithm 2 (MLP from scratch), TensorFlow installed
Key Concepts: Sequential model, Dense layer, compile, fit, callbacks, history
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "mlp_keras")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 3: MLP WITH TENSORFLOW / KERAS")
print("=" * 70)
print()
print("In Algorithm 2 we built an MLP from scratch (~200 lines).")
print("Keras does the same thing in ~10 lines AND adds:")
print("  ✅ Automatic differentiation (no manual backprop!)")
print("  ✅ GPU acceleration")
print("  ✅ Callbacks (early stopping, LR scheduling, checkpointing)")
print("  ✅ Dozens of optimizers, loss functions, layers built-in")
print()

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    tf.random.set_seed(42)
    TF_AVAILABLE = True
except ImportError:
    print("  TensorFlow not installed!")
    print("  Install it: pip install tensorflow")
    print()
    print("  This module will show the CODE but cannot run it.")
    print("  Follow along visually, then install TensorFlow to run it.")
    TF_AVAILABLE = False

print()


# ======================================================================
# SECTION 1: Data Preparation
# ======================================================================
print("=" * 70)
print("SECTION 1: DATA PREPARATION")
print("=" * 70)
print()

from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=600, noise=0.1, factor=0.4, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42)

print(f"  Dataset: make_circles (non-linearly separable, 2 classes)")
print(f"  Train:      {X_train.shape[0]} samples")
print(f"  Validation: {X_val.shape[0]} samples")
print(f"  Test:       {X_test.shape[0]} samples")
print(f"  Features:   {X_train.shape[1]}")
print()


# ======================================================================
# SECTION 2: Building the Model
# ======================================================================
print("=" * 70)
print("SECTION 2: BUILDING THE KERAS MODEL")
print("=" * 70)
print()
print("The Keras Sequential API:")
print()
print("  model = keras.Sequential([")
print("      layers.Dense(32, activation='relu', input_shape=(2,)),")
print("      layers.Dropout(0.2),")
print("      layers.Dense(16, activation='relu'),")
print("      layers.Dense(1, activation='sigmoid')   # binary output")
print("  ])")
print()
print("Each Dense layer:")
print("  units        = number of neurons")
print("  activation   = activation function (relu/sigmoid/softmax/tanh)")
print("  input_shape  = only needed for the FIRST layer")
print()

if TF_AVAILABLE:
    model = keras.Sequential([
        layers.Dense(32, activation="relu", input_shape=(2,)),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1,  activation="sigmoid"),   # binary classification
    ], name="MLP_Circles")

    model.summary()
    total_params = model.count_params()
    print(f"\n  Total trainable parameters: {total_params:,}")
    print(f"  (Our from-scratch version had manual W and b matrices)")
    print()
else:
    print("  [TensorFlow not installed — code shown for reference]")
    print()
    print("  Layer structure:")
    print("  Input(2) -> Dense(32, relu) -> Dropout(0.2) -> Dense(16, relu)")
    print("            -> Dropout(0.2) -> Dense(1, sigmoid)")
    print()


# ======================================================================
# SECTION 3: Compiling the Model
# ======================================================================
print("=" * 70)
print("SECTION 3: COMPILING THE MODEL")
print("=" * 70)
print()
print("model.compile() sets 3 things:")
print()
print("  optimizer = how to update weights")
print("    'adam'    → Adam (best default)")
print("    'sgd'     → vanilla SGD")
print("    keras.optimizers.Adam(learning_rate=0.001)  → with custom lr")
print()
print("  loss = what to minimize")
print("    'binary_crossentropy'      → binary classification (sigmoid out)")
print("    'categorical_crossentropy' → multi-class (softmax + one-hot labels)")
print("    'sparse_categorical_crossentropy' → multi-class (integer labels)")
print("    'mse'                      → regression")
print()
print("  metrics = what to MONITOR (not minimized, just tracked)")
print("    'accuracy', 'AUC', 'precision', 'recall'")
print()

if TF_AVAILABLE:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    print("  model.compile() done!")
print()


# ======================================================================
# SECTION 4: Training with Callbacks
# ======================================================================
print("=" * 70)
print("SECTION 4: TRAINING WITH CALLBACKS")
print("=" * 70)
print()
print("Callbacks run automatically during training.")
print()
print("  EarlyStopping:")
print("    monitor='val_loss'  → watch validation loss")
print("    patience=10         → stop if no improvement for 10 epochs")
print("    restore_best_weights=True  → revert to best checkpoint")
print()
print("  ReduceLROnPlateau:")
print("    Reduce lr by 'factor' when val_loss stagnates")
print("    Good for escaping local minima")
print()
print("  ModelCheckpoint:")
print("    Save the best model to disk automatically")
print()

if TF_AVAILABLE:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=1
        ),
    ]

    print("  Training...")
    print()
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    print()


# ======================================================================
# SECTION 5: Evaluation
# ======================================================================
print("=" * 70)
print("SECTION 5: EVALUATION")
print("=" * 70)
print()

if TF_AVAILABLE:
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print()

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred      = (y_pred_prob > 0.5).astype(int)

    from sklearn.metrics import classification_report, confusion_matrix
    print("  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))


# ======================================================================
# SECTION 6: Save and Reload
# ======================================================================
print("=" * 70)
print("SECTION 6: SAVE AND RELOAD THE MODEL")
print("=" * 70)
print()
print("Saving a Keras model:")
print()
print("  model.save('my_model.keras')                  # modern format")
print("  model.save('my_model.h5')                     # legacy HDF5")
print("  model.save_weights('my_weights.h5')           # weights only")
print()
print("Loading:")
print("  loaded = keras.models.load_model('my_model.keras')")
print("  loaded.predict(new_data)")
print()

if TF_AVAILABLE:
    save_path = os.path.join(VIS_DIR, "circles_model.keras")
    model.save(save_path)
    print(f"  Model saved to: {save_path}")

    loaded_model = keras.models.load_model(save_path)
    reload_acc = (
        (loaded_model.predict(X_test, verbose=0).flatten() > 0.5).astype(int)
        == y_test
    ).mean()
    print(f"  Reloaded model accuracy: {reload_acc:.4f} (same as before!)  ")
    print()


# ======================================================================
# SECTION 7: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)
print()

if TF_AVAILABLE:
    hist = history.history

    # --- PLOT 1: Training history ---
    print("Generating: Training history curves...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Keras Model Training History", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(hist["loss"],     "steelblue",  linewidth=2, label="Train Loss")
    ax.plot(hist["val_loss"], "darkorange", linewidth=2, linestyle="--", label="Val Loss")
    ax.set_title("Loss Over Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(hist["accuracy"],     "steelblue",  linewidth=2, label="Train Acc")
    ax.plot(hist["val_accuracy"], "darkorange", linewidth=2, linestyle="--", label="Val Acc")
    ax.set_title("Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "training_history.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: training_history.png")


    # --- PLOT 2: Decision boundary ---
    print("Generating: Keras model decision boundary...")

    xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid, verbose=0).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Keras MLP — Learned Decision Boundary  (Test Acc: {test_acc:.1%})",
                 fontsize=13, fontweight="bold")

    for ax, (X_plot, y_plot, split_name) in zip(axes, [
        (X_train, y_train, "Train set"),
        (X_test,  y_test,  "Test set"),
    ]):
        ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
        ax.contour(xx, yy, Z, levels=[0.5], colors=["black"], linewidths=2)
        ax.scatter(X_plot[y_plot==0, 0], X_plot[y_plot==0, 1],
                   c="tomato", s=25, alpha=0.7, edgecolors="none")
        ax.scatter(X_plot[y_plot==1, 0], X_plot[y_plot==1, 1],
                   c="royalblue", s=25, alpha=0.7, edgecolors="none")
        ax.set_title(f"{split_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "decision_boundary.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: decision_boundary.png")

else:
    print("  [Skipping visualizations — TensorFlow not installed]")
    print("  Install TensorFlow and re-run to see training curves and decision boundary.")


# ======================================================================
# SECTION 8: Summary Cheat Sheet
# ======================================================================
print()
print("=" * 70)
print("SECTION 8: KERAS CHEAT SHEET")
print("=" * 70)
print()
print("BINARY CLASSIFICATION:")
print("  output_layer = Dense(1, activation='sigmoid')")
print("  loss = 'binary_crossentropy'")
print()
print("MULTI-CLASS CLASSIFICATION:")
print("  output_layer = Dense(n_classes, activation='softmax')")
print("  loss = 'sparse_categorical_crossentropy'  (integer labels)")
print("  loss = 'categorical_crossentropy'          (one-hot labels)")
print()
print("REGRESSION:")
print("  output_layer = Dense(1)  (no activation = linear)")
print("  loss = 'mse'  or  'mae'")
print()
print("SAFE DEFAULTS:")
print("  optimizer = Adam(lr=0.001)")
print("  hidden activation = relu")
print("  He initialization (Keras uses this by default with ReLU)")
print("  callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]")
print()

# ============= CONCEPTUAL DIAGRAM =============
print("Generating: Keras Sequential API concept diagram...")
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes:
    ax.set_facecolor('#0f0f1a')

# --- LEFT panel: Keras API building blocks ---
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Keras Sequential API\n— Building Blocks", fontsize=13,
             fontweight='bold', color='white', pad=12)

api_steps = [
    ("model = Sequential()", '#3a7bd5'),
    ("model.add(Dense(64, activation='relu'))", '#2ecc71'),
    ("model.add(Dense(64, activation='relu'))", '#2ecc71'),
    ("model.add(Dense(1, activation='sigmoid'))", '#e67e22'),
    ("model.compile(optimizer='adam',\n  loss='binary_crossentropy')", '#9b59b6'),
    ("model.fit(X_train, y_train,\n  epochs=100)", '#e74c3c'),
]

box_h = 1.1
gap = 0.25
total_h = len(api_steps) * box_h + (len(api_steps) - 1) * gap
start_y = (10 - total_h) / 2 + total_h

for idx, (label, color) in enumerate(api_steps):
    y_top = start_y - idx * (box_h + gap)
    y_center = y_top - box_h / 2
    rect = FancyBboxPatch((0.5, y_top - box_h), 9.0, box_h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white',
                          linewidth=1.2, alpha=0.88)
    ax.add_patch(rect)
    ax.text(5.0, y_center, label, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold',
            fontfamily='monospace')
    if idx < len(api_steps) - 1:
        arrow_y = y_top - box_h
        ax.annotate('', xy=(5.0, arrow_y - gap + 0.04),
                    xytext=(5.0, arrow_y - 0.04),
                    arrowprops=dict(arrowstyle='->', color='#aaaacc',
                                   lw=2.0))

# --- RIGHT panel: resulting network architecture ---
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title("Resulting Network Architecture\n— Layer by Layer", fontsize=13,
             fontweight='bold', color='white', pad=12)

arch_layers = [
    ("Input Layer", "(2 features)", '#546e7a'),
    ("Dense 64", "activation = ReLU", '#1565c0'),
    ("Dense 64", "activation = ReLU", '#1565c0'),
    ("Dense 1", "activation = Sigmoid", '#6a1b9a'),
    ("Output", "P(class=1)", '#b71c1c'),
]

arch_box_h = 1.0
arch_gap = 0.35
arch_total = len(arch_layers) * arch_box_h + (len(arch_layers) - 1) * arch_gap
arch_start_y = (10 - arch_total) / 2 + arch_total

for idx, (layer_name, detail, color) in enumerate(arch_layers):
    y_top = arch_start_y - idx * (arch_box_h + arch_gap)
    y_center = y_top - arch_box_h / 2
    rect = FancyBboxPatch((1.0, y_top - arch_box_h), 8.0, arch_box_h,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white',
                          linewidth=1.5, alpha=0.90)
    ax.add_patch(rect)
    ax.text(5.0, y_center + 0.18, layer_name, ha='center', va='center',
            fontsize=10, color='white', fontweight='bold')
    ax.text(5.0, y_center - 0.22, detail, ha='center', va='center',
            fontsize=8.5, color='#ddddff', style='italic')
    if idx < len(arch_layers) - 1:
        arrow_y = y_top - arch_box_h
        ax.annotate('', xy=(5.0, arrow_y - arch_gap + 0.06),
                    xytext=(5.0, arrow_y - 0.06),
                    arrowprops=dict(arrowstyle='->', color='#aaaacc',
                                   lw=2.0))

# Formula at the bottom
ax.text(5.0, 0.3, "Layer output = activation(W · input + b)",
        ha='center', va='center', fontsize=9, color='#99aaff',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                  edgecolor='#4455aa', linewidth=1.2))

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(VIS_DIR, '03_keras_api_diagram.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 03_keras_api_diagram.png")
# ============= END CONCEPTUAL DIAGRAM =============

print("=" * 70)
print("ALGORITHM 3: MLP WITH KERAS COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  Keras = from-scratch power with 10x less code")
print("  model.compile() = set optimizer + loss + metrics")
print("  model.fit() = train with optional validation and callbacks")
print("  EarlyStopping = automatic overfitting prevention")
print("  model.save() = persist trained model for later use")
print()
print("Next: Algorithm 4 -> Hyperparameter Tuning!")
