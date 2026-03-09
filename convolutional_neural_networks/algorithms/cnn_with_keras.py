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

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "cnn_keras")
os.makedirs(VIS_DIR, exist_ok=True)
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
plt.savefig(f"{VIS_DIR}/cifar10_samples.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{VIS_DIR}/training_history.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{VIS_DIR}/confusion_matrix.png", dpi=300, bbox_inches="tight")
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
    plt.savefig(f"{VIS_DIR}/feature_maps.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: feature_maps.png")

else:
    print("  [Skipping training-based plots — install TensorFlow to generate them]")



# ============= CONCEPTUAL DIAGRAM =============
print("📊 Generating: CNN Architecture concept diagram...")

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_title('CNN Architecture for Image Classification',
             color='white', fontsize=16, fontweight='bold', pad=18)

# Helper: draw a 3D-ish stacked box block
def draw_3d_block(ax, x, y, w, h, depth, face_color, edge_color, label_top, label_bot, fontsize=8):
    # Shadow / depth offset layers (back to front)
    for di in range(int(depth), 0, -1):
        offset = di * 0.07
        shade = FancyBboxPatch(
            (x + offset, y - offset), w, h,
            boxstyle="round,pad=0.05",
            facecolor=face_color, edgecolor=edge_color,
            linewidth=0.8, alpha=0.35, zorder=2
        )
        ax.add_patch(shade)
    # Front face
    front = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        facecolor=face_color, edgecolor=edge_color,
        linewidth=1.8, alpha=0.95, zorder=3
    )
    ax.add_patch(front)
    # Top label
    ax.text(x + w / 2, y + h + 0.18, label_top,
            color='white', fontsize=fontsize, ha='center', va='bottom',
            fontweight='bold', zorder=5)
    # Bottom dim label
    ax.text(x + w / 2, y - 0.22, label_bot,
            color='#aaaaaa', fontsize=fontsize - 0.5, ha='center', va='top',
            zorder=5)

# Helper: draw arrow between two x-positions at a fixed y center
def draw_arrow(ax, x_start, x_end, y_center, color='#888888'):
    ax.annotate('', xy=(x_end, y_center), xytext=(x_start, y_center),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                zorder=6)

# ---- Stage definitions ----
# (x_left, y_bottom, width, height, depth, face_color, edge_color, label_top, label_bottom)
center_y = 3.5

stages = [
    # Input image
    dict(x=0.4,  y=center_y - 1.0, w=1.2, h=2.0, depth=1,
         fc='#1a5276', ec='#5dade2',
         lt='Input\nImage', lb='32×32×3'),
    # Conv2D + ReLU block 1
    dict(x=2.3,  y=center_y - 1.3, w=0.8, h=2.6, depth=5,
         fc='#1e8449', ec='#58d68d',
         lt='Conv2D+ReLU\n(32 filters)', lb='32×32×32'),
    # MaxPool 1
    dict(x=4.1,  y=center_y - 0.9, w=0.7, h=1.8, depth=5,
         fc='#117a65', ec='#48c9b0',
         lt='MaxPool2D\n(2×2)', lb='16×16×32'),
    # Conv2D + ReLU block 2
    dict(x=5.9,  y=center_y - 1.5, w=0.8, h=3.0, depth=7,
         fc='#7d6608', ec='#f4d03f',
         lt='Conv2D+ReLU\n(64 filters)', lb='16×16×64'),
    # MaxPool 2
    dict(x=7.7,  y=center_y - 1.0, w=0.7, h=2.0, depth=7,
         fc='#6e2f1a', ec='#e59866',
         lt='MaxPool2D\n(2×2)', lb='8×8×64'),
    # Flatten
    dict(x=9.5,  y=center_y - 0.15, w=1.4, h=0.3, depth=1,
         fc='#512e5f', ec='#c39bd3',
         lt='Flatten', lb='4096\nneurons'),
    # Dense + Softmax
    dict(x=11.9, y=center_y - 0.5, w=1.2, h=1.0, depth=1,
         fc='#922b21', ec='#f1948a',
         lt='Dense +\nSoftmax', lb='10 classes'),
]

for s in stages:
    draw_3d_block(ax, s['x'], s['y'], s['w'], s['h'], s['depth'],
                  s['fc'], s['ec'], s['lt'], s['lb'])

# Arrows between stages
arrow_pairs = [
    (0.4 + 1.2,        2.3,           center_y),   # Input → Conv1
    (2.3 + 0.8 + 0.35, 4.1,           center_y),   # Conv1 → Pool1
    (4.1 + 0.7,        5.9,           center_y),   # Pool1 → Conv2
    (5.9 + 0.8 + 0.35, 7.7,           center_y),   # Conv2 → Pool2
    (7.7 + 0.7,        9.5,           center_y),   # Pool2 → Flatten
    (9.5 + 1.4,        11.9,          center_y),   # Flatten → Dense
]
arrow_colors = ['#5dade2', '#58d68d', '#48c9b0', '#f4d03f', '#e59866', '#c39bd3']
for (xs, xe, yc), col in zip(arrow_pairs, arrow_colors):
    draw_arrow(ax, xs, xe, yc, col)

# ---- Operation labels below arrows ----
op_labels = [
    (1.85,  'detect\npatterns'),
    (3.65,  'reduce\nsize ×2'),
    (5.35,  'detect\ncomplex\nfeatures'),
    (7.15,  'reduce\nsize ×2'),
    (9.15,  'vectorize'),
    (11.2,  'classify'),
]
for xm, lbl in op_labels:
    ax.text(xm, center_y - 2.2, lbl, color='#cccccc', fontsize=7,
            ha='center', va='top', fontstyle='italic', zorder=5)

# ---- Two key insight rows ----
ax.text(8.0, 1.05,
        'Convolution extracts local features — edges, textures, shapes',
        color='#5dade2', fontsize=10, ha='center', va='center',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1b2a', edgecolor='#5dade2', linewidth=1.5),
        zorder=6)

ax.text(8.0, 0.35,
        'Pooling progressively reduces spatial size — keeps the most important signal',
        color='#58d68d', fontsize=10, ha='center', va='center',
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0d1b2a', edgecolor='#58d68d', linewidth=1.5),
        zorder=6)

# ---- Legend strip across top ----
legend_items = [
    ('#1e8449', '#58d68d', 'Conv2D + ReLU'),
    ('#117a65', '#48c9b0', 'MaxPool2D'),
    ('#512e5f', '#c39bd3', 'Flatten'),
    ('#922b21', '#f1948a', 'Dense + Softmax'),
]
lx = 1.5
for fc, ec, label in legend_items:
    rect = FancyBboxPatch((lx, 8.1), 0.35, 0.5,
                          boxstyle='round,pad=0.05',
                          facecolor=fc, edgecolor=ec, linewidth=1.5, zorder=5)
    ax.add_patch(rect)
    ax.text(lx + 0.5, 8.35, label, color='white', fontsize=8,
            va='center', zorder=5)
    lx += 3.2

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '05_cnn_architecture_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 05_cnn_architecture_concept.png")
# ============= END CONCEPTUAL DIAGRAM =============


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
