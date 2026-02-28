"""
🤖 CONVOLUTIONAL NEURAL NETWORKS — Algorithm 2: CNN with Keras
==============================================================

Learning Objectives:
  1. Build a CNN using Keras Conv2D, MaxPool2D, BatchNormalization layers
  2. Train on a real image dataset (CIFAR-10 via keras.datasets)
  3. Use data augmentation to improve generalization
  4. Monitor training with EarlyStopping and ModelCheckpoint
  5. Evaluate with accuracy, confusion matrix, and per-class report
  6. Visualize feature maps from intermediate layers (layer inspection)
  7. Save and reload the trained model

YouTube Resources:
  ⭐ TensorFlow CNN tutorial https://www.tensorflow.org/tutorials/images/cnn
  ⭐ Sentdex - CNNs in Keras https://www.youtube.com/watch?v=WvoLTXIjBYU
  📚 CS231n - Training CNNs https://www.youtube.com/watch?v=wEoyxE0GP2M

Time Estimate: 60-75 minutes
Difficulty: Intermediate
Prerequisites: Algorithm 1 (Conv from scratch), Part 3 MLP with Keras
Key Concepts: Conv2D, BatchNorm, data augmentation, feature map visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../visuals/cnn_keras", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 2: CNN WITH KERAS")
print("=" * 70)
print()
print("We'll train a CNN on CIFAR-10: 60,000 color images, 10 classes.")
print("  airplane, automobile, bird, cat, deer,")
print("  dog, frog, horse, ship, truck")
print()
print("Keras makes this MUCH simpler than from-scratch:")
print("  model.compile() → model.fit() → model.evaluate()")
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} ready!")
except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print("  Following along will show the full code and architecture.")
print()


# ======================================================================
# SECTION 1: Load and Explore CIFAR-10
# ======================================================================
print("=" * 70)
print("SECTION 1: LOADING CIFAR-10")
print("=" * 70)
print()

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

if TF_AVAILABLE:
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    print(f"  Training images: {X_train.shape}  (50,000 × 32×32 × 3 channels)")
    print(f"  Test images:     {X_test.shape}   (10,000 × 32×32 × 3 channels)")
    print(f"  Pixel range:     {X_train.min()} – {X_train.max()}")
    print()
    print("  Class distribution (training):")
    for i, name in enumerate(CLASS_NAMES):
        count = (y_train == i).sum()
        print(f"    {i}: {name:12s} → {count:,} images")
    print()

    # Normalize
    X_train_n = X_train.astype("float32") / 255.0
    X_test_n  = X_test.astype("float32") / 255.0

    # Validation split
    X_tr = X_train_n[:-5000]
    y_tr = y_train[:-5000]
    X_val = X_train_n[-5000:]
    y_val = y_train[-5000:]

    print(f"  After split — Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test_n.shape[0]}")
else:
    print("  [TF not installed — using synthetic placeholder data]")
    X_train = np.random.randint(0, 256, (50000, 32, 32, 3), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 50000)
    X_test  = np.random.randint(0, 256, (10000, 32, 32, 3), dtype=np.uint8)
    y_test  = np.random.randint(0, 10, 10000)
    X_tr = X_train[:45000].astype("float32") / 255.0
    X_val = X_train[45000:].astype("float32") / 255.0
    y_tr = y_train[:45000]
    y_val = y_train[45000:]
    X_test_n = X_test.astype("float32") / 255.0
print()


# ======================================================================
# SECTION 2: Build the CNN
# ======================================================================
print("=" * 70)
print("SECTION 2: BUILDING THE CNN MODEL")
print("=" * 70)
print()
print("Architecture — 3 Conv blocks + classification head:")
print()
print("  Block 1: Conv(32, 3x3) → BN → ReLU → Conv(32, 3x3) → BN → ReLU → MaxPool → Dropout")
print("  Block 2: Conv(64, 3x3) → BN → ReLU → Conv(64, 3x3) → BN → ReLU → MaxPool → Dropout")
print("  Block 3: Conv(128,3x3) → BN → ReLU → Conv(128,3x3) → BN → ReLU → MaxPool → Dropout")
print("  Head:    Flatten → Dense(256) → BN → ReLU → Dropout → Dense(10, softmax)")
print()
print("  BatchNormalization after each Conv: normalizes activations → faster, stabler training")
print("  Dropout (0.25 after pool, 0.5 before output): prevents overfitting")
print()

if TF_AVAILABLE:
    def build_cnn(input_shape=(32, 32, 3), n_classes=10):
        inputs = keras.Input(shape=input_shape)

        # Block 1
        x = layers.Conv2D(32, (3,3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 2
        x = layers.Conv2D(64, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        # Block 3
        x = layers.Conv2D(128, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPool2D((2,2))(x)
        x = layers.Dropout(0.25)(x)

        # Classification head
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(n_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs, name="CIFAR10_CNN")

    model = build_cnn()
    model.summary()
    print()
    print(f"  Total parameters: {model.count_params():,}")
    print()


# ======================================================================
# SECTION 3: Data Augmentation
# ======================================================================
print("=" * 70)
print("SECTION 3: DATA AUGMENTATION")
print("=" * 70)
print()
print("Data augmentation = randomly transform training images to create")
print("more variety. The model sees 'new' images every epoch.")
print()
print("  Why it helps:")
print("    - More effective training data without collecting more images")
print("    - Model learns to be invariant to flips, shifts, rotations")
print("    - Reduces overfitting significantly")
print()
print("  Common augmentations for images:")
print("    Random horizontal flip (good for most objects)")
print("    Random crop / translation")
print("    Random rotation (+/- 15 degrees)")
print("    Random brightness / contrast adjustment")
print("    Random zoom")
print()

if TF_AVAILABLE:
    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ], name="augmentation")
    print("  Keras augmentation pipeline ready!")
    print("  (Applied to training data only — test data is NOT augmented)")
    print()


# ======================================================================
# SECTION 4: Compile and Train
# ======================================================================
print("=" * 70)
print("SECTION 4: COMPILE AND TRAIN")
print("=" * 70)
print()

if TF_AVAILABLE:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5,
            min_lr=1e-6, verbose=1
        ),
    ]

    # Build augmented dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
    train_ds = train_ds.shuffle(10000).batch(64).map(
        lambda x, y: (augment(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)

    print("  Training CNN on CIFAR-10 with data augmentation...")
    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=val_ds,
        callbacks=callbacks_list,
        verbose=1
    )
    print()

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test_n, y_test, verbose=0)
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss:     {test_loss:.4f}")
    print()

    y_pred     = model.predict(X_test_n, verbose=0).argmax(axis=1)
    y_pred_prob = model.predict(X_test_n, verbose=0)


# ======================================================================
# SECTION 5: Feature Map Inspection
# ======================================================================
print("=" * 70)
print("SECTION 5: FEATURE MAP INSPECTION")
print("=" * 70)
print()
print("We can peek inside the CNN to see what each layer 'sees'.")
print("Use keras.Model(inputs, intermediate_layer.output) to extract features.")
print()

if TF_AVAILABLE:
    # Create intermediate output models
    layer_outputs = [layer.output for layer in model.layers
                     if isinstance(layer, (layers.Conv2D, layers.MaxPool2D))]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs[:4])

    # Get a single image
    sample_img = X_test_n[0:1]   # shape (1, 32, 32, 3)
    activations = activation_model.predict(sample_img, verbose=0)

    print(f"  Sample image shape: {sample_img.shape}")
    print(f"  Extracted {len(activations)} intermediate activation tensors:")
    for i, act in enumerate(activations):
        print(f"    Layer {i+1}: {act.shape}")
    print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Sample CIFAR-10 images ---
print("📊 Generating: Sample CIFAR-10 images...")

fig, axes = plt.subplots(4, 10, figsize=(15, 6))
fig.suptitle("CIFAR-10: Sample Images from All 10 Classes",
             fontsize=13, fontweight="bold")

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    idxs = np.where(y_train == cls_idx)[0][:4]
    for row, img_idx in enumerate(idxs):
        ax = axes[row, cls_idx]
        ax.imshow(X_train[img_idx])
        ax.axis("off")
        if row == 0:
            ax.set_title(cls_name, fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/cnn_keras/cifar10_samples.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: cifar10_samples.png")


if TF_AVAILABLE:
    # --- PLOT 2: Training history ---
    print("📊 Generating: Training history...")

    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CNN Training on CIFAR-10", fontsize=14, fontweight="bold")

    axes[0].plot(hist["loss"],     "steelblue",  linewidth=2, label="Train Loss")
    axes[0].plot(hist["val_loss"], "darkorange", linewidth=2, linestyle="--", label="Val Loss")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["accuracy"],     "steelblue",  linewidth=2, label="Train Acc")
    axes[1].plot(hist["val_accuracy"], "darkorange", linewidth=2, linestyle="--", label="Val Acc")
    axes[1].axhline(test_acc, color="green", linestyle=":", linewidth=2,
                    label=f"Test Acc = {test_acc:.3f}")
    axes[1].set_title(f"Accuracy (Test: {test_acc:.1%})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("../visuals/cnn_keras/training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: training_history.png")


    # --- PLOT 3: Confusion matrix ---
    print("📊 Generating: Confusion matrix...")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — CIFAR-10 CNN (Accuracy: {test_acc:.1%})",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            c = "white" if cm[i,j] > cm.max()*0.5 else "black"
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", color=c, fontsize=8)
    plt.tight_layout()
    plt.savefig("../visuals/cnn_keras/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: confusion_matrix.png")


    # --- PLOT 4: Feature maps ---
    print("📊 Generating: Feature maps from first conv layer...")

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle(f"Feature Maps: What the CNN Sees ('{CLASS_NAMES[y_test[0]]}' input image)",
                 fontsize=13, fontweight="bold")

    # Row 0: input image
    for j in range(8):
        axes[0, j].axis("off")
    axes[0, 0].imshow(X_test[0])
    axes[0, 0].set_title(f"Input:\n{CLASS_NAMES[y_test[0]]}", fontsize=9, fontweight="bold")

    # Rows 1-3: feature maps from first 3 conv activations (24 channels)
    for layer_idx, act in enumerate(activations[:3]):
        n_show = min(8, act.shape[-1])
        for ch in range(n_show):
            ax = axes[layer_idx + 1, ch]
            ax.imshow(act[0, :, :, ch], cmap="viridis")
            ax.set_title(f"L{layer_idx+1} ch{ch+1}", fontsize=7)
            ax.axis("off")
        # hide unused
        for ch in range(n_show, 8):
            axes[layer_idx + 1, ch].axis("off")

    plt.tight_layout()
    plt.savefig("../visuals/cnn_keras/feature_maps.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: feature_maps.png")

else:
    print("  [Skipping training-based plots — install TensorFlow to generate them]")


print()
print("=" * 70)
print("✅ ALGORITHM 2: CNN WITH KERAS COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🏗️  3 conv blocks (Conv → BN → ReLU → Pool → Dropout) is a solid baseline")
print("  🔄  Data augmentation: flip + rotate + zoom → huge generalization boost")
print("  📊  CIFAR-10 target: ~75-85% with a simple CNN (humans = ~94%)")
print("  🔭  Feature map inspection: see what the CNN actually learns!")
print()
print("Next: Algorithm 3 → Classic Architectures (LeNet, AlexNet, VGG, ResNet)")
