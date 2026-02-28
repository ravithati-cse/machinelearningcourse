"""
🏆 CONVOLUTIONAL NEURAL NETWORKS — Project 1: CIFAR-10 Classifier
=================================================================

Learning Objectives:
  1. Work with CIFAR-10: 60,000 color images across 10 real-world classes
  2. Understand why MLP fails on color images but CNN succeeds
  3. Build a production-grade CNN with data augmentation and BatchNorm
  4. Apply transfer learning (MobileNetV2) and compare to from-scratch CNN
  5. Analyze per-class accuracy and identify failure modes
  6. Visualize learned filters and feature maps from a trained model
  7. Build confidence that you understand end-to-end CNN development

YouTube Resources:
  ⭐ 3Blue1Brown - Convolutional neural networks https://www.youtube.com/watch?v=KuXjwB4LzSA
  ⭐ Sentdex - CIFAR-10 CNN https://www.youtube.com/watch?v=WvoLTXIjBYU
  📚 TensorFlow CIFAR-10 tutorial https://www.tensorflow.org/tutorials/images/cnn

Time Estimate: 75-90 minutes
Difficulty: Intermediate
Prerequisites: All 3 math foundation modules + all 4 algorithm modules
Key Concepts: color images, data augmentation, BatchNorm, per-class analysis, feature maps
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("../visuals/cifar10_classifier", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🏆 PROJECT 1: CIFAR-10 CLASSIFIER")
print("=" * 70)
print()
print("CIFAR-10 = Canadian Institute for Advanced Research, 10 classes")
print("60,000 color images (32x32 RGB) across 10 natural categories")
print()
print("Classes:")
CIFAR_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]
for i, cls in enumerate(CIFAR_CLASSES):
    print(f"  {i}: {cls}")
print()
print("Why CIFAR-10 is harder than MNIST:")
print("  • Color images (3 channels vs 1)")
print("  • 10 visually diverse real-world classes (not just digits)")
print("  • 32x32 is low resolution — small inter-class variation")
print("  • Intra-class variation: cats look very different from each other")
print()
print("MNIST MLP gets ~98% | CIFAR-10 requires CNN to get ~90%")
print()


# ======================================================================
# SECTION 1: Load and Explore
# ======================================================================
print("=" * 70)
print("SECTION 1: LOAD AND EXPLORE CIFAR-10")
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

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = keras.datasets.cifar10.load_data()

    y_train = y_train_raw.flatten()
    y_test  = y_test_raw.flatten()

    print(f"  Training images: {X_train_raw.shape}  → 50,000 RGB 32x32 images")
    print(f"  Test images:     {X_test_raw.shape}  → 10,000 RGB 32x32 images")
    print(f"  Labels shape:    {y_train.shape}")
    print(f"  Pixel range:     {X_train_raw.min()} – {X_train_raw.max()}")
    print()

    print("  Class distribution (training set):")
    for i, cls in enumerate(CIFAR_CLASSES):
        count = (y_train == i).sum()
        bar = "█" * (count // 500)
        print(f"    {i} {cls:<12}: {count:5d}  {bar}")
    print()
    print("  Perfectly balanced — 5,000 images per class")
    print()

except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print("  Generating synthetic data for demonstration...")
    print()
    X_train_raw = (np.random.rand(50000, 32, 32, 3) * 255).astype(np.uint8)
    y_train     = np.random.randint(0, 10, 50000)
    X_test_raw  = (np.random.rand(10000, 32, 32, 3) * 255).astype(np.uint8)
    y_test      = np.random.randint(0, 10, 10000)
    y_train_raw = y_train.reshape(-1, 1)
    y_test_raw  = y_test.reshape(-1, 1)


# ======================================================================
# SECTION 2: Preprocessing
# ======================================================================
print("=" * 70)
print("SECTION 2: PREPROCESSING")
print("=" * 70)
print()
print("Step 1: Normalize pixel values [0, 255] → [0.0, 1.0]")
print("Step 2: Validation split — reserve last 5k training images")
print()

X_train = X_train_raw.astype("float32") / 255.0
X_test  = X_test_raw.astype("float32")  / 255.0

X_tr, X_val = X_train[:-5000], X_train[-5000:]
y_tr, y_val = y_train[:-5000], y_train[-5000:]

print(f"  Train:      {X_tr.shape}  → {len(X_tr):,} images")
print(f"  Validation: {X_val.shape}  →  {len(X_val):,} images")
print(f"  Test:       {X_test.shape}  → {len(X_test):,} images")
print()

if TF_AVAILABLE:
    AUTOTUNE = tf.data.AUTOTUNE
    BATCH    = 64

    train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)) \
        .shuffle(10000).batch(BATCH).prefetch(AUTOTUNE)
    val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)) \
        .batch(BATCH).prefetch(AUTOTUNE)
    test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)) \
        .batch(BATCH).prefetch(AUTOTUNE)

    print(f"  Using batch size: {BATCH}")
    print(f"  Train batches: {len(list(train_ds))}, Val batches: {len(list(val_ds))}")
    print()


# ======================================================================
# SECTION 3: Why MLP Struggles on Color Images
# ======================================================================
print("=" * 70)
print("SECTION 3: WHY MLP STRUGGLES — THE CASE FOR CNN")
print("=" * 70)
print()
print("CIFAR-10 image: 32 x 32 x 3 = 3,072 inputs (vs MNIST's 784)")
print()
print("MLP approach (Dense layers):")
print("  Input: 3,072 neurons")
print("  Hidden: 512 neurons")
print("  Parameters in first layer: 3,072 × 512 + 512 = 1,573,376")
print()
print("  Problems:")
print("  1. TOO MANY PARAMETERS for only 50k training images → overfitting")
print("  2. No spatial awareness — pixel at (0,0) treated same as (15,15)")
print("  3. Not translation invariant — cat at left ≠ cat at right")
print("  4. Loses all 2D structure by flattening")
print()
print("CNN approach:")
print("  First Conv2D(32, 3x3):")
print("  Parameters: 3 × 3 × 3 × 32 + 32 = 896  (vs 1.5M for Dense!)")
print()
print("  Advantages:")
print("  1. LOCAL connectivity — only 3×3 region per neuron")
print("  2. WEIGHT SHARING — same filter scanned across entire image")
print("  3. SPATIAL hierarchy — builds from edges → shapes → objects")
print("  4. TRANSLATION invariance — pooling handles position shifts")
print()

if TF_AVAILABLE:
    # Quick MLP baseline to show the gap
    mlp_model = keras.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(10, activation="softmax"),
    ], name="MLP_Baseline")

    mlp_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("  Training MLP baseline (5 epochs for comparison)...")
    history_mlp = mlp_model.fit(
        X_tr, y_tr,
        epochs=5,
        batch_size=128,
        validation_data=(X_val, y_val),
        verbose=1,
    )
    mlp_val_acc = max(history_mlp.history["val_accuracy"])
    mlp_params  = mlp_model.count_params()
    print(f"\n  MLP params:     {mlp_params:,}")
    print(f"  MLP val acc:    {mlp_val_acc:.4f} ({mlp_val_acc*100:.1f}%) after 5 epochs")
    print(f"  (Typically plateaus around 48-55% with full training)")
    print()
else:
    mlp_val_acc = 0.50  # placeholder for plotting
    print("  MLP on CIFAR-10 typically plateaus around 48-55%")
    print("  CNN approaches 90%+ — this is the 'CNN advantage'")
    print()


# ======================================================================
# SECTION 4: Build CNN Model
# ======================================================================
print("=" * 70)
print("SECTION 4: BUILD THE CNN CLASSIFIER")
print("=" * 70)
print()
print("Architecture: 3-block CNN with BatchNorm + data augmentation")
print()
print("  Block 1:  Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.2)")
print("  Block 2:  Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.3)")
print("  Block 3:  Conv(128)→ BN → ReLU → Conv(128)→ BN → ReLU → MaxPool → Dropout(0.4)")
print("  Head:     GlobalAvgPool → Dense(256, ReLU) → BN → Dropout(0.5) → Dense(10, softmax)")
print()
print("Why GlobalAveragePooling instead of Flatten?")
print("  Flatten: 128 × 4 × 4 = 2,048 values → still large")
print("  GAP:     128 channels → average each → 128 values (much smaller!)")
print("  GAP reduces overfitting and makes the model size-agnostic")
print()

if TF_AVAILABLE:
    # Data augmentation layers
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")

    def build_cnn(num_classes=10):
        """3-block CNN for CIFAR-10."""
        inputs = keras.Input(shape=(32, 32, 3), name="input")

        # Augmentation (only applied during training)
        x = data_augmentation(inputs)

        # --- Block 1 ---
        x = layers.Conv2D(32, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(32, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.2)(x)

        # --- Block 2 ---
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.3)(x)

        # --- Block 3 ---
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.4)(x)

        # --- Head ---
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs, name="CIFAR10_CNN")

    cnn_model = build_cnn()
    cnn_model.summary()

    cnn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    print()
    print(f"  Model parameters: {cnn_model.count_params():,}")
    print()


# ======================================================================
# SECTION 5: Train
# ======================================================================
print("=" * 70)
print("SECTION 5: TRAINING")
print("=" * 70)
print()

if TF_AVAILABLE:
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10,
            restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1
        ),
    ]

    print("  Training CNN on CIFAR-10...")
    print("  (Expect ~85-90% val accuracy in 30-50 epochs)")
    print()

    history_cnn = cnn_model.fit(
        X_tr, y_tr,
        epochs=60,
        batch_size=BATCH,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    print()

    cnn_val_acc = max(history_cnn.history["val_accuracy"])
    test_loss, test_acc = cnn_model.evaluate(X_test, y_test, verbose=0)
    print(f"  Best val accuracy:  {cnn_val_acc:.4f} ({cnn_val_acc*100:.1f}%)")
    print(f"  Final test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  Final test loss:     {test_loss:.4f}")
    print()

    y_pred_prob = cnn_model.predict(X_test, verbose=0)
    y_pred      = y_pred_prob.argmax(axis=1)

else:
    print("  [TF not installed — skipping training]")
    print()
    print("  Expected results:")
    print("    MLP baseline:     ~50%")
    print("    3-block CNN:      ~85-88%")
    print("    CNN + TL:         ~90-93%")
    print()
    cnn_val_acc = 0.86  # placeholder
    test_acc    = 0.86


# ======================================================================
# SECTION 6: Per-Class Evaluation
# ======================================================================
print("=" * 70)
print("SECTION 6: PER-CLASS EVALUATION")
print("=" * 70)
print()

if TF_AVAILABLE:
    from sklearn.metrics import classification_report, confusion_matrix

    print("  Per-class performance:")
    report = classification_report(y_test, y_pred, target_names=CIFAR_CLASSES)
    print(report)

    cm = confusion_matrix(y_test, y_pred)

    print("  Easiest classes (highest recall):")
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    sorted_classes   = sorted(zip(per_class_recall, CIFAR_CLASSES), reverse=True)
    for recall, cls in sorted_classes[:3]:
        print(f"    {cls:<12}: {recall:.1%}")

    print()
    print("  Hardest classes (lowest recall):")
    for recall, cls in sorted_classes[-3:]:
        print(f"    {cls:<12}: {recall:.1%}")

    print()
    print("  Most common mistakes:")
    mistakes = []
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                mistakes.append((cm[i, j], CIFAR_CLASSES[i], CIFAR_CLASSES[j]))
    mistakes.sort(reverse=True)
    for count, true, pred in mistakes[:8]:
        print(f"    {true:<12} → {pred:<12}: {count} times")
    print()

else:
    print("  Known CIFAR-10 confusion patterns (even for well-trained models):")
    print()
    print("  Easiest classes:    ship, airplane, automobile (high contrast, uniform)")
    print("  Hardest classes:    cat, dog, deer (visual similarity, pose variation)")
    print()
    print("  Most common mistakes:")
    print("    cat → dog:        animal body shape confusion")
    print("    dog → cat:        same")
    print("    automobile → truck: both are vehicles with similar shape")
    print("    deer → horse:     four-legged animals")
    print("    airplane → bird:  both 'fly', similar wing shapes")
    print()


# ======================================================================
# SECTION 7: Transfer Learning Comparison (MobileNetV2)
# ======================================================================
print("=" * 70)
print("SECTION 7: TRANSFER LEARNING COMPARISON (MOBILENETV2)")
print("=" * 70)
print()
print("CIFAR-10 has small 32×32 images.")
print("MobileNetV2 expects 96×96 minimum, ideally 224×224.")
print("We'll upsample images using UpSampling2D (bilinear interpolation).")
print()

if TF_AVAILABLE:
    from tensorflow.keras.applications import MobileNetV2

    IMG_SIZE_TL = 96   # Minimum safe size for MobileNetV2

    # Resize within the model using Lambda/Resizing layer
    preprocess_mobilenet = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE_TL, IMG_SIZE_TL, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    tl_inputs  = keras.Input(shape=(32, 32, 3))
    x_tl       = layers.Resizing(IMG_SIZE_TL, IMG_SIZE_TL)(tl_inputs)
    x_tl       = preprocess_mobilenet(x_tl)
    x_tl       = base_model(x_tl, training=False)
    x_tl       = layers.GlobalAveragePooling2D()(x_tl)
    x_tl       = layers.Dropout(0.3)(x_tl)
    outputs_tl = layers.Dense(10, activation="softmax")(x_tl)
    tl_model   = keras.Model(tl_inputs, outputs_tl, name="MobileNetV2_CIFAR10")

    tl_model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"  MobileNetV2 feature extractor on CIFAR-10 (10 epochs)...")
    history_tl = tl_model.fit(
        X_tr, y_tr,
        epochs=10,
        batch_size=BATCH,
        validation_data=(X_val, y_val),
        verbose=1,
    )
    tl_val_acc  = max(history_tl.history["val_accuracy"])
    _, tl_test_acc = tl_model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  MobileNetV2 val acc: {tl_val_acc:.4f} ({tl_val_acc*100:.1f}%)")
    print(f"  MobileNetV2 test acc: {tl_test_acc:.4f} ({tl_test_acc*100:.1f}%)")
    print()
    print("  Note: TL advantage is smaller here than on flowers because:")
    print("    → CIFAR-10 has 50k images (not tiny)")
    print("    → 32x32 upsampled to 96x96 loses information")
    print("    → Custom CNN optimized for 32x32 may be competitive")
    print()

else:
    tl_test_acc = 0.88
    print("  MobileNetV2 feature extractor on CIFAR-10:")
    print("  → Images must be upsampled: 32×32 → 96×96")
    print("  → Expected test accuracy: ~86-90%")
    print("  → Custom 3-block CNN may be comparable or better here")
    print("  → TL really shines when you have <5k images")
    print()


# ======================================================================
# SECTION 8: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 8: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Sample images grid ---
print("Generating: Sample CIFAR-10 images...")

fig, axes = plt.subplots(5, 10, figsize=(16, 8))
fig.suptitle("CIFAR-10 Dataset — 5 Samples per Class", fontsize=14, fontweight="bold")

for cls_idx in range(10):
    indices = np.where(y_train == cls_idx)[0][:5]
    for row, idx in enumerate(indices):
        ax = axes[row, cls_idx]
        ax.imshow(X_train_raw[idx])
        ax.axis("off")
        if row == 0:
            ax.set_title(CIFAR_CLASSES[cls_idx], fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/cifar10_classifier/sample_images.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: sample_images.png")


# --- PLOT 2: Data augmentation examples ---
print("Generating: Data augmentation examples...")

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
fig.suptitle("Data Augmentation: Same Image — Different Views",
             fontsize=14, fontweight="bold")

sample_img = X_train_raw[0]   # pick a single image

# Row 0: Original (repeated)
for col in range(8):
    axes[0, col].imshow(sample_img)
    axes[0, col].axis("off")
    if col == 0:
        axes[0, col].set_ylabel("Original", fontsize=10, rotation=90, va="center")

# Row 1: Horizontal flips + crops (manual augmentation for demo)
aug_imgs = []
for _ in range(8):
    img = sample_img.copy().astype(np.float32)
    # Random horizontal flip
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :]
    # Random brightness
    img = np.clip(img * np.random.uniform(0.7, 1.3), 0, 255)
    aug_imgs.append(img.astype(np.uint8))

for col, img in enumerate(aug_imgs):
    axes[1, col].imshow(img)
    axes[1, col].axis("off")
    if col == 0:
        axes[1, col].set_ylabel("Flip+Brightness", fontsize=9, rotation=90, va="center")

# Row 2: Random crops (padding then crop)
crop_imgs = []
for _ in range(8):
    img = sample_img.copy()
    # Pad by 4 pixels
    pad = 4
    padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    # Random 32x32 crop
    y0 = np.random.randint(0, 2 * pad)
    x0 = np.random.randint(0, 2 * pad)
    crop_imgs.append(padded[y0:y0+32, x0:x0+32])

for col, img in enumerate(crop_imgs):
    axes[2, col].imshow(img)
    axes[2, col].axis("off")
    if col == 0:
        axes[2, col].set_ylabel("Random Crop", fontsize=9, rotation=90, va="center")

plt.tight_layout()
plt.savefig("../visuals/cifar10_classifier/data_augmentation.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: data_augmentation.png")


if TF_AVAILABLE:
    # --- PLOT 3: Training history ---
    print("Generating: Training history...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("CIFAR-10 CNN Training Progress", fontsize=14, fontweight="bold")

    hist = history_cnn.history
    epochs = range(1, len(hist["accuracy"]) + 1)

    axes[0].plot(epochs, hist["loss"],     color="steelblue",  linewidth=2, label="Train Loss")
    axes[0].plot(epochs, hist["val_loss"], color="darkorange", linewidth=2, linestyle="--", label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss over Epochs"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, hist["accuracy"],     color="steelblue",  linewidth=2, label="Train Acc")
    axes[1].plot(epochs, hist["val_accuracy"], color="darkorange", linewidth=2, linestyle="--", label="Val Acc")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy (Test: {test_acc:.1%})")
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Mark LR reductions
    if "lr" in hist:
        lrs = hist["lr"]
        changes = [i for i in range(1, len(lrs)) if lrs[i] < lrs[i-1]]
        for ep in changes:
            axes[1].axvline(x=ep + 1, color="red", alpha=0.4, linestyle=":")
        if changes:
            axes[1].text(changes[0] + 0.5, 0.55, "LR reduced", fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig("../visuals/cifar10_classifier/training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: training_history.png")


    # --- PLOT 4: Confusion matrix ---
    print("Generating: Confusion matrix...")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CIFAR_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    ax.set_title(f"Confusion Matrix — CIFAR-10 CNN (Test Acc: {test_acc:.1%})",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax)

    for i in range(10):
        for j in range(10):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig("../visuals/cifar10_classifier/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: confusion_matrix.png")


    # --- PLOT 5: Per-class accuracy bar chart ---
    print("Generating: Per-class accuracy chart...")

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors_bar = plt.cm.RdYlGn(per_class_acc)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(CIFAR_CLASSES, per_class_acc * 100, color=colors_bar, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Per-Class Accuracy — CIFAR-10 CNN (Overall: {test_acc:.1%})",
                 fontsize=13, fontweight="bold")
    ax.axhline(y=test_acc * 100, color="navy", linestyle="--", linewidth=1.5,
               label=f"Overall: {test_acc:.1%}")
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=10)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("../visuals/cifar10_classifier/per_class_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: per_class_accuracy.png")


    # --- PLOT 6: Model comparison ---
    print("Generating: Model comparison chart...")

    methods   = ["MLP\nBaseline", "CNN\nFrom Scratch", "MobileNetV2\nFeature Extract"]
    accs      = [mlp_val_acc, test_acc, tl_test_acc]
    params_k  = [mlp_model.count_params() / 1000,
                 cnn_model.count_params() / 1000,
                 tl_model.count_params() / 1000]
    colors_m  = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Model Comparison on CIFAR-10", fontsize=14, fontweight="bold")

    bars1 = axes[0].bar(methods, [a * 100 for a in accs],
                        color=colors_m, edgecolor="white", linewidth=2, width=0.5)
    axes[0].set_ylim(0, 110)
    axes[0].set_ylabel("Test Accuracy (%)", fontsize=12)
    axes[0].set_title("Accuracy Comparison")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, acc in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{acc:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    bars2 = axes[1].bar(methods, params_k, color=colors_m, edgecolor="white", linewidth=2, width=0.5)
    axes[1].set_ylabel("Parameters (thousands)", fontsize=12)
    axes[1].set_title("Parameter Count")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, p in zip(bars2, params_k):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{p:.0f}K", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("../visuals/cifar10_classifier/model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: model_comparison.png")


    # --- PLOT 7: Sample predictions ---
    print("Generating: Sample predictions (correct + incorrect)...")

    correct_idx   = np.where(y_pred == y_test)[0]
    incorrect_idx = np.where(y_pred != y_test)[0]

    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("CIFAR-10 CNN Predictions\nTop rows: Correct (green)  |  Bottom rows: Wrong (red)",
                 fontsize=13, fontweight="bold")

    for col, idx in enumerate(correct_idx[:16]):
        ax   = axes[col // 8, col % 8]
        conf = y_pred_prob[idx].max()
        ax.imshow(X_test_raw[idx])
        ax.set_title(f"P:{CIFAR_CLASSES[y_pred[idx]][:4]}\nT:{CIFAR_CLASSES[y_test[idx]][:4]}\n{conf:.2f}",
                     fontsize=7, color="darkgreen")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("green"); spine.set_linewidth(3)

    for col, idx in enumerate(incorrect_idx[:16]):
        ax   = axes[2 + col // 8, col % 8]
        conf = y_pred_prob[idx].max()
        ax.imshow(X_test_raw[idx])
        ax.set_title(f"P:{CIFAR_CLASSES[y_pred[idx]][:4]}\nT:{CIFAR_CLASSES[y_test[idx]][:4]}\n{conf:.2f}",
                     fontsize=7, color="darkred")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("red"); spine.set_linewidth(3)

    axes[0, 0].set_ylabel("CORRECT", fontsize=10, color="green", fontweight="bold")
    axes[2, 0].set_ylabel("WRONG",   fontsize=10, color="red",   fontweight="bold")

    plt.tight_layout()
    plt.savefig("../visuals/cifar10_classifier/predictions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: predictions.png")

else:
    print("  [Skipping model-dependent plots — TensorFlow not installed]")
    print("  Sample images and augmentation plots were saved.")


print()
print("=" * 70)
print("PROJECT 1: CIFAR-10 CLASSIFIER COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  CIFAR-10 = 60,000 color 32x32 images, 10 classes, perfectly balanced")
print("  MLP baseline: ~50% — spatial flattening loses too much info")
print("  3-block CNN: ~85-88% — local features + weight sharing works!")
print("  Transfer learning: competitive at ~88-90%, especially low-data regime")
print("  Hardest classes: cat/dog (look similar), deer/horse (similar body)")
print("  Data augmentation: essential — it's 50k images, not 500k")
print()
print("Visualizations saved to: ../visuals/cifar10_classifier/")
print("  1. sample_images.png          — 5 examples per class")
print("  2. data_augmentation.png      — augmentation pipeline demo")
print("  3. training_history.png       — loss and accuracy curves")
print("  4. confusion_matrix.png       — full 10x10 confusion matrix")
print("  5. per_class_accuracy.png     — class-level accuracy bar chart")
print("  6. model_comparison.png       — MLP vs CNN vs TL comparison")
print("  7. predictions.png            — correct and wrong predictions")
print()
print("Next: Project 2 → Custom Image Classifier (your own dataset!)")
