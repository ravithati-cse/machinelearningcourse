"""
🏆 DEEP NEURAL NETWORKS — Project 1: MNIST Digit Classifier
=============================================================

Learning Objectives:
  1. Load and explore the famous MNIST handwritten digit dataset
  2. Preprocess images: normalize, flatten, and one-hot encode
  3. Build an MLP classifier for 10-class image recognition
  4. Train with Keras and visualize training progress
  5. Evaluate with confusion matrix and classification report
  6. Visualize correct AND incorrect predictions
  7. Understand where and why the model makes mistakes

YouTube Resources:
  ⭐ 3Blue1Brown - But what is a neural network? https://www.youtube.com/watch?v=aircAruvnKk
  ⭐ Sentdex - MNIST with TensorFlow https://www.youtube.com/watch?v=wQ8BIBpya2k
  📚 TensorFlow - MNIST tutorial https://www.tensorflow.org/tutorials/quickstart/beginner

Time Estimate: 60-90 minutes
Difficulty: Intermediate
Prerequisites: All 5 math foundation modules + algorithms 1-4
Key Concepts: image classification, MNIST, 10-class softmax, confusion matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("../visuals/mnist_classifier", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🏆 PROJECT 1: MNIST DIGIT CLASSIFIER")
print("=" * 70)
print()
print("MNIST = Modified National Institute of Standards and Technology")
print("THE benchmark dataset of deep learning — 70,000 handwritten digits")
print()
print("Task: look at a 28x28 pixel grayscale image -> predict digit 0-9")
print()
print("Historical context:")
print("  1998: Yann LeCun uses MNIST to demonstrate CNNs")
print("  2012: Deep learning on MNIST reaches 99%+ accuracy")
print("  Today: MNIST is the 'Hello World' of every DL framework")
print()


# ======================================================================
# SECTION 1: Load and Explore Data
# ======================================================================
print("=" * 70)
print("SECTION 1: LOAD AND EXPLORE THE MNIST DATASET")
print("=" * 70)
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} detected")
    print()

    (X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.mnist.load_data()

    print(f"  Training images: {X_train_raw.shape} (60,000 x 28 x 28 pixels)")
    print(f"  Test images:     {X_test_raw.shape} (10,000 x 28 x 28 pixels)")
    print(f"  Labels (train):  {y_train.shape}  values: {np.unique(y_train)}")
    print()
    print(f"  Pixel value range: {X_train_raw.min()} to {X_train_raw.max()}")
    print(f"  (0 = black, 255 = white)")
    print()

    # Class distribution
    print("  Class distribution (training set):")
    for digit in range(10):
        count = (y_train == digit).sum()
        bar = "#" * (count // 300)
        print(f"    Digit {digit}: {count:5d}  {bar}")
    print()

except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print("  This script will demonstrate the full pipeline conceptually.")
    print()
    TF_AVAILABLE = False


if not TF_AVAILABLE:
    # Generate placeholder MNIST-like data for demo
    X_train_raw = (np.random.rand(60000, 28, 28) * 255).astype(np.uint8)
    y_train     = np.random.randint(0, 10, 60000)
    X_test_raw  = (np.random.rand(10000, 28, 28) * 255).astype(np.uint8)
    y_test      = np.random.randint(0, 10, 10000)
    print("  Using synthetic MNIST-like data for demonstration.")
    print()


# ======================================================================
# SECTION 2: Preprocessing
# ======================================================================
print("=" * 70)
print("SECTION 2: PREPROCESSING")
print("=" * 70)
print()
print("Step 1: Normalize pixels from [0, 255] to [0.0, 1.0]")
print("  Why: neural networks work better with small input values")
print()
print("Step 2: Flatten images from (28, 28) to (784,)")
print("  Why: Dense layers expect 1D input per sample")
print("  28 x 28 = 784 pixels = 784 input features")
print()
print("Step 3: Validation split (last 10k of training set)")
print()

X_train = X_train_raw.astype("float32") / 255.0   # normalize
X_test  = X_test_raw.astype("float32") / 255.0

X_train_flat = X_train.reshape(-1, 784)   # flatten
X_test_flat  = X_test.reshape(-1, 784)

X_tr, X_val = X_train_flat[:-10000], X_train_flat[-10000:]
y_tr, y_val = y_train[:-10000],      y_train[-10000:]

print(f"  After preprocessing:")
print(f"    Train: X={X_tr.shape}, y={y_tr.shape}")
print(f"    Val:   X={X_val.shape}, y={y_val.shape}")
print(f"    Test:  X={X_test_flat.shape}, y={y_test.shape}")
print(f"    Pixel range: {X_tr.min():.1f} to {X_tr.max():.1f} (normalized)")
print()


# ======================================================================
# SECTION 3: Build Model
# ======================================================================
print("=" * 70)
print("SECTION 3: BUILDING THE MLP CLASSIFIER")
print("=" * 70)
print()
print("Architecture:")
print("  Input:    784 neurons (one per pixel)")
print("  Hidden 1: 256 neurons (ReLU) + Dropout(0.3)")
print("  Hidden 2: 128 neurons (ReLU) + Dropout(0.3)")
print("  Output:   10 neurons (Softmax) — one per digit")
print()
print("  Total: 784->256->128->10")
print()

if TF_AVAILABLE:
    model = keras.Sequential([
        layers.Dense(256, activation="relu", input_shape=(784,)),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10,  activation="softmax"),
    ], name="MNIST_MLP")

    model.summary()
    print()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


# ======================================================================
# SECTION 4: Train
# ======================================================================
print("=" * 70)
print("SECTION 4: TRAINING")
print("=" * 70)
print()

if TF_AVAILABLE:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5,
            restore_best_weights=True, verbose=1
        ),
    ]

    print("  Training MLP on 50,000 MNIST images...")
    history = model.fit(
        X_tr, y_tr,
        epochs=20,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    print()

    test_loss, test_acc = model.evaluate(X_test_flat, y_test, verbose=0)
    print(f"  Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Final Test Loss:     {test_loss:.4f}")
    print()

    y_pred_prob = model.predict(X_test_flat, verbose=0)
    y_pred      = y_pred_prob.argmax(axis=1)


# ======================================================================
# SECTION 5: Evaluation
# ======================================================================
print("=" * 70)
print("SECTION 5: EVALUATION")
print("=" * 70)
print()

if TF_AVAILABLE:
    from sklearn.metrics import classification_report, confusion_matrix

    print("  Per-digit performance:")
    report = classification_report(y_test, y_pred,
                                   target_names=[str(i) for i in range(10)])
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("  Most common mistakes (top 5):")
    mistakes = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                mistakes.append((cm[i, j], i, j))
    mistakes.sort(reverse=True)
    for count, true, pred in mistakes[:5]:
        print(f"    True={true} predicted as {pred}: {count} times")
    print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Sample digits ---
print("Generating: Sample MNIST digits...")

fig, axes = plt.subplots(5, 10, figsize=(14, 7))
fig.suptitle("Sample MNIST Digits (5 per class)", fontsize=14, fontweight="bold")

for digit in range(10):
    indices = np.where(y_train == digit)[0][:5]
    for row, idx in enumerate(indices):
        ax = axes[row, digit]
        ax.imshow(X_train_raw[idx], cmap="gray")
        ax.axis("off")
        if row == 0:
            ax.set_title(str(digit), fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/mnist_classifier/sample_digits.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: sample_digits.png")


if TF_AVAILABLE:
    # --- PLOT 2: Training history ---
    print("Generating: Training history...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("MNIST Training Progress", fontsize=14, fontweight="bold")

    hist = history.history
    axes[0].plot(hist["loss"],     "steelblue",  linewidth=2, label="Train")
    axes[0].plot(hist["val_loss"], "darkorange", linewidth=2, linestyle="--", label="Val")
    axes[0].set_title("Loss Over Epochs"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["accuracy"],     "steelblue",  linewidth=2, label="Train")
    axes[1].plot(hist["val_accuracy"], "darkorange", linewidth=2, linestyle="--", label="Val")
    axes[1].set_title(f"Accuracy (Final Test: {test_acc:.1%})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.8, 1.0); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../visuals/mnist_classifier/training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: training_history.png")


    # --- PLOT 3: Confusion matrix ---
    print("Generating: Confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(range(10)); ax.set_yticklabels(range(10))
    ax.set_xlabel("Predicted Digit", fontsize=12)
    ax.set_ylabel("True Digit", fontsize=12)
    ax.set_title(f"Confusion Matrix — MNIST MLP (Accuracy: {test_acc:.1%})",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            color = "white" if cm[i,j] > cm.max()*0.5 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig("../visuals/mnist_classifier/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: confusion_matrix.png")


    # --- PLOT 4: Prediction examples ---
    print("Generating: Prediction examples (correct + incorrect)...")

    correct_idx   = np.where(y_pred == y_test)[0]
    incorrect_idx = np.where(y_pred != y_test)[0]

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("Predictions: Green=Correct  Red=Wrong  (confidence shown)",
                 fontsize=13, fontweight="bold")

    for col, idx in enumerate(correct_idx[:16]):
        ax = axes[col // 8, col % 8]
        ax.imshow(X_test_raw[idx], cmap="gray")
        conf = y_pred_prob[idx].max()
        ax.set_title(f"P:{y_pred[idx]} T:{y_test[idx]}\n{conf:.2f}",
                     fontsize=8, color="darkgreen")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("green"); spine.set_linewidth(3)

    for col, idx in enumerate(incorrect_idx[:16]):
        ax = axes[2 + col // 8, col % 8]
        ax.imshow(X_test_raw[idx], cmap="gray")
        conf = y_pred_prob[idx].max()
        ax.set_title(f"P:{y_pred[idx]} T:{y_test[idx]}\n{conf:.2f}",
                     fontsize=8, color="darkred")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("red"); spine.set_linewidth(3)

    axes[0, 0].set_ylabel("CORRECT", fontsize=11, color="green", fontweight="bold")
    axes[2, 0].set_ylabel("WRONG", fontsize=11, color="red", fontweight="bold")

    plt.tight_layout()
    plt.savefig("../visuals/mnist_classifier/predictions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: predictions.png")

else:
    print("  [Skipping model-based plots — TensorFlow not installed]")
    print("  Sample digits plot was saved.")


print()
print("=" * 70)
print("PROJECT 1: MNIST DIGIT CLASSIFIER COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  MNIST = 70k grayscale 28x28 digit images, 10 classes")
print("  MLP achieves ~98% accuracy — great for a fully-connected network!")
print("  Confusion matrix: most errors happen between similar-looking digits")
print("  (e.g., 4 vs 9, 3 vs 8, 5 vs 6)")
print("  CNN (Part 4) will push this to 99%+ by exploiting image structure")
print()
print("Next: Project 2 -> Tabular Deep Learning (DNN vs traditional ML)")
