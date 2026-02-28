"""
🤖 CONVOLUTIONAL NEURAL NETWORKS — Algorithm 3: Classic Architectures
======================================================================

Learning Objectives:
  1. Understand the historical progression: LeNet → AlexNet → VGG → ResNet
  2. Implement simplified LeNet-5 in Keras (1998, the original CNN)
  3. Understand AlexNet innovations: ReLU, Dropout, deep stacking (2012)
  4. Know VGG's philosophy: depth with small (3x3) filters only
  5. Understand the vanishing gradient problem in deep networks
  6. Master ResNet's skip connections (residual blocks) — the key insight
  7. Build and compare architectures on CIFAR-10

YouTube Resources:
  ⭐ Yannic Kilcher - ResNet explained https://www.youtube.com/watch?v=GWt6Fu05voI
  ⭐ StatQuest - ResNets https://www.youtube.com/watch?v=ZILIbUvp5lk
  📚 CS231n - CNN Architectures https://www.youtube.com/watch?v=DAOcjicFr1Y

Time Estimate: 65-80 minutes
Difficulty: Intermediate-Advanced
Prerequisites: Algorithm 2 (CNN with Keras), Module 3 (Pooling & Depth)
Key Concepts: LeNet, AlexNet, VGG, ResNet, skip connections, residual learning
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "classic_architectures")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 3: CLASSIC CNN ARCHITECTURES")
print("=" * 70)
print()
print("The history of CNNs is a story of going DEEPER and SMARTER.")
print()
print("  1998: LeNet-5     — 7 layers, handwritten digits, ~60K params")
print("  2012: AlexNet     — 8 layers, ImageNet winner, ~60M params")
print("  2014: VGGNet      — 16-19 layers, very deep, ~138M params")
print("  2015: ResNet-50   — 50 layers, skip connections, ~25M params")
print("  2017: MobileNet   — lightweight, mobile-friendly")
print("  2019: EfficientNet— best accuracy/params trade-off (current SOTA)")
print()
print("Each one solved a problem the previous one had.")
print("Let's build them and understand WHY they work!")
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
    print("  TensorFlow not installed. Architectures shown for reference.")
print()

# Load CIFAR-10 if available
if TF_AVAILABLE:
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    y_train = y_train.flatten(); y_test = y_test.flatten()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32") / 255.0
    X_tr, X_val = X_train[:45000], X_train[45000:]
    y_tr, y_val = y_train[:45000], y_train[45000:]


# ======================================================================
# SECTION 1: LeNet-5 (1998)
# ======================================================================
print("=" * 70)
print("SECTION 1: LeNet-5 (1998) — THE ORIGINAL CNN")
print("=" * 70)
print()
print("Yann LeCun, 1998. Built to recognize handwritten digits.")
print("First practical CNN — used by banks to read checks!")
print()
print("Architecture:")
print("  Input (32x32x1)")
print("  Conv(6, 5x5, tanh) → AvgPool(2x2)")
print("  Conv(16, 5x5, tanh) → AvgPool(2x2)")
print("  Flatten")
print("  Dense(120, tanh) → Dense(84, tanh) → Dense(10, softmax)")
print()
print("Key innovations at the time:")
print("  - Learnable convolutional filters (vs handcrafted)")
print("  - Subsampling (pooling) for spatial reduction")
print("  - End-to-end training via backprop")
print()

if TF_AVAILABLE:
    def build_lenet5(input_shape=(32, 32, 3), n_classes=10):
        """LeNet-5 adapted for CIFAR-10 (32x32 RGB instead of 28x28 grayscale)."""
        return keras.Sequential([
            layers.Conv2D(6,  (5,5), activation="tanh", padding="same",
                          input_shape=input_shape),
            layers.AveragePooling2D((2,2)),
            layers.Conv2D(16, (5,5), activation="tanh"),
            layers.AveragePooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(120, activation="tanh"),
            layers.Dense(84,  activation="tanh"),
            layers.Dense(n_classes, activation="softmax"),
        ], name="LeNet5")

    lenet = build_lenet5()
    lenet.summary()
    print(f"\n  Parameters: {lenet.count_params():,}")
    print()


# ======================================================================
# SECTION 2: AlexNet (2012)
# ======================================================================
print("=" * 70)
print("SECTION 2: AlexNet (2012) — THE DEEP LEARNING REVOLUTION")
print("=" * 70)
print()
print("Krizhevsky, Sutskever, Hinton, 2012.")
print("Won ImageNet ILSVRC 2012 with 15.3% top-5 error (vs 26% runner-up).")
print("This moment STARTED the deep learning revolution.")
print()
print("AlexNet innovations:")
print("  1. ReLU instead of tanh/sigmoid → faster training, no vanishing gradient")
print("  2. Dropout (0.5) → regularization")
print("  3. Data augmentation (flips, crops, color jitter)")
print("  4. GPU training (2 GTX 580s with 3GB each!)")
print("  5. Local Response Normalization (now replaced by BatchNorm)")
print()
print("Architecture (simplified for CIFAR-10):")
print("  Conv(64,11x11) → MaxPool → Conv(192,5x5) → MaxPool →")
print("  Conv(384,3x3) → Conv(256,3x3) → Conv(256,3x3) → MaxPool →")
print("  Flatten → Dense(4096) → Dense(4096) → Dense(10)")
print()

if TF_AVAILABLE:
    def build_alexnet_mini(input_shape=(32, 32, 3), n_classes=10):
        """Simplified AlexNet for CIFAR-10 (original was for 224x224)."""
        return keras.Sequential([
            # Block 1
            layers.Conv2D(64, (3,3), activation="relu", padding="same",
                          input_shape=input_shape),
            layers.MaxPool2D((2,2)),
            # Block 2
            layers.Conv2D(192, (3,3), activation="relu", padding="same"),
            layers.MaxPool2D((2,2)),
            # Block 3-5
            layers.Conv2D(384, (3,3), activation="relu", padding="same"),
            layers.Conv2D(256, (3,3), activation="relu", padding="same"),
            layers.Conv2D(256, (3,3), activation="relu", padding="same"),
            layers.MaxPool2D((2,2)),
            # Head
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(n_classes, activation="softmax"),
        ], name="AlexNet_Mini")

    alexnet = build_alexnet_mini()
    alexnet.summary()
    print(f"\n  Parameters: {alexnet.count_params():,}")
    print()


# ======================================================================
# SECTION 3: VGG (2014)
# ======================================================================
print("=" * 70)
print("SECTION 3: VGGNet (2014) — SIMPLICITY AND DEPTH")
print("=" * 70)
print()
print("Simonyan & Zisserman, Oxford, 2014.")
print()
print("VGG's philosophy: just use 3x3 conv, go VERY deep.")
print()
print("Why 3x3 only?")
print("  Two 3x3 convs = receptive field of one 5x5 conv")
print("  Three 3x3 convs = receptive field of one 7x7 conv")
print("  BUT fewer parameters AND more non-linearity (ReLU after each)!")
print()
print("  5x5 filter: 25 weights")
print("  Two 3x3:    2 × 9 = 18 weights  (28% fewer!)")
print()
print("VGG-16 Architecture:")
print("  [Conv64 × 2] → Pool → [Conv128 × 2] → Pool →")
print("  [Conv256 × 3] → Pool → [Conv512 × 3] → Pool →")
print("  [Conv512 × 3] → Pool → FC4096 → FC4096 → FC1000")
print()

if TF_AVAILABLE:
    def build_vgg_mini(input_shape=(32, 32, 3), n_classes=10):
        """VGG-style architecture scaled for CIFAR-10."""
        model_input = keras.Input(shape=input_shape)

        def vgg_block(x, filters, n_convs):
            for _ in range(n_convs):
                x = layers.Conv2D(filters, (3,3), padding="same")(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
            x = layers.MaxPool2D((2,2))(x)
            return x

        x = vgg_block(model_input, 32, 2)
        x = vgg_block(x, 64, 2)
        x = vgg_block(x, 128, 2)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(n_classes, activation="softmax")(x)
        return keras.Model(model_input, out, name="VGG_Mini")

    vgg = build_vgg_mini()
    vgg.summary()
    print(f"\n  Parameters: {vgg.count_params():,}")
    print()


# ======================================================================
# SECTION 4: The Vanishing Gradient Problem
# ======================================================================
print("=" * 70)
print("SECTION 4: THE VANISHING GRADIENT — WHY DEEP NETWORKS FAIL")
print("=" * 70)
print()
print("As we go deeper (VGG-16, 19 layers+), training gets harder.")
print()
print("The vanishing gradient problem:")
print("  Backprop multiplies gradients at each layer")
print("  If activations saturate (sigmoid/tanh → derivative ≈ 0):")
print("    gradient × 0.1 × 0.1 × 0.1 × ... = ≈ 0 at early layers")
print("  Early layers stop learning!")
print()
print("  ReLU helped a lot (gradient = 1 for z>0)")
print("  But even with ReLU, very deep networks degrade")
print()
print("Surprising finding (ResNet paper):")
print("  Adding MORE layers makes accuracy WORSE (training accuracy!)")
print("  This shouldn't happen — extra layers could at least learn identity")
print("  Conclusion: optimization problem, not overfitting")
print()
print("ResNet's solution: skip connections (residual learning)")
print()

# Simulate vanishing gradient
depths = np.arange(1, 25)
sigmoid_grad_decay = 0.25 ** depths        # sigmoid max derivative = 0.25
relu_grad_decay    = 1.0 ** depths         # ReLU = 1 (no decay)
# But in practice, other factors cause degradation even with ReLU
practical_decay    = 0.95 ** depths        # practical degradation

print(f"  Gradient magnitude after N layers (sigmoid, starting=1.0):")
for d in [1, 5, 10, 15, 20]:
    print(f"    {d:2d} layers: {sigmoid_grad_decay[d-1]:.2e}")
print()
print("  After 15 layers: gradient ≈ 0 → zero learning in early layers!")
print()


# ======================================================================
# SECTION 5: ResNet — Skip Connections
# ======================================================================
print("=" * 70)
print("SECTION 5: ResNet (2015) — RESIDUAL LEARNING")
print("=" * 70)
print()
print("He et al., Microsoft Research, 2015.")
print("Won ILSVRC 2015 with 3.57% top-5 error (better than humans!)")
print()
print("The key idea — SKIP CONNECTIONS (residual blocks):")
print()
print("  Standard block:")
print("    x → Conv → BN → ReLU → Conv → BN → output")
print("    output = F(x)")
print()
print("  Residual block:")
print("    x → Conv → BN → ReLU → Conv → BN")
print("          ↓                          ↓")
print("          └───────────── x ──────────┘")
print("    output = F(x) + x")
print()
print("  The shortcut adds the INPUT directly to the block output.")
print("  The block only needs to learn the RESIDUAL (difference from input).")
print()
print("  If the block does nothing useful: F(x) = 0 → output = x (identity)")
print("  This means extra layers can't HURT performance — worst case = identity!")
print()
print("  Gradient benefit: gradient can skip directly through the shortcut")
print("  → No vanishing gradient, even with 100+ layers!")
print()

if TF_AVAILABLE:
    def residual_block(x, filters, downsample=False):
        """Basic ResNet residual block."""
        stride = 2 if downsample else 1
        shortcut = x

        # Main path
        x = layers.Conv2D(filters, (3,3), strides=stride, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)

        # Shortcut path: match dimensions if needed
        if downsample or shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1,1), strides=stride)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation("relu")(x)
        return x

    def build_resnet_mini(input_shape=(32, 32, 3), n_classes=10):
        """ResNet-style for CIFAR-10."""
        inputs = keras.Input(shape=input_shape)

        # Initial conv
        x = layers.Conv2D(32, (3,3), padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Residual blocks
        x = residual_block(x, 32)
        x = residual_block(x, 32)
        x = residual_block(x, 64, downsample=True)
        x = residual_block(x, 64)
        x = residual_block(x, 128, downsample=True)
        x = residual_block(x, 128)

        # Global Average Pooling (modern alternative to Flatten)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(n_classes, activation="softmax")(x)

        return keras.Model(inputs, x, name="ResNet_Mini")

    resnet = build_resnet_mini()
    resnet.summary()
    print(f"\n  Parameters: {resnet.count_params():,}")
    print()

    # Quick comparison: compile all and check sizes
    print("  Architecture parameter comparison:")
    print(f"  {'Model':15} {'Parameters':>12}")
    print("  " + "-" * 30)
    for name, m in [("LeNet-5", lenet), ("AlexNet-Mini", alexnet),
                     ("VGG-Mini", vgg), ("ResNet-Mini", resnet)]:
        print(f"  {name:15} {m.count_params():>12,}")
    print()


# ======================================================================
# SECTION 6: Train and Compare (ResNet vs Simple CNN)
# ======================================================================
print("=" * 70)
print("SECTION 6: TRAINING RESNET ON CIFAR-10")
print("=" * 70)
print()

if TF_AVAILABLE:
    resnet.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    cb = [keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                         restore_best_weights=True, verbose=0),
          keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0)]

    augment = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    train_ds = (tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
                .shuffle(10000).batch(128)
                .map(lambda x, y: (augment(x, training=True), y))
                .prefetch(tf.data.AUTOTUNE))

    print("  Training ResNet-Mini on CIFAR-10...")
    history_resnet = resnet.fit(
        train_ds,
        epochs=40,
        validation_data=(X_val, y_val),
        callbacks=cb,
        verbose=1
    )
    loss, acc = resnet.evaluate(X_test, y_test, verbose=0)
    print(f"\n  ResNet-Mini Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print()


# ======================================================================
# SECTION 7: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Architecture timeline ---
print("📊 Generating: Architecture timeline...")

fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(1996, 2022); ax.set_ylim(0, 10)
ax.set_facecolor("#f8f9fa"); fig.patch.set_facecolor("#f8f9fa")
ax.set_title("CNN Architecture Timeline: Deeper and Smarter",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Depth (# layers)", fontsize=12)

archs = [
    (1998, "LeNet-5",     7,   "#4CAF50",  "First practical CNN\n60K params"),
    (2012, "AlexNet",     8,   "#2196F3",  "Started DL revolution\n60M params"),
    (2014, "VGGNet",      16,  "#9C27B0",  "Deep + simple 3x3\n138M params"),
    (2015, "ResNet-50",   50,  "#F44336",  "Skip connections!\n25M params"),
    (2017, "MobileNet",   28,  "#FF9800",  "Mobile-friendly\n4M params"),
    (2019, "EfficientNet",18,  "#00BCD4",  "Best efficiency\n5M params"),
]

for yr, name, depth, color, note in archs:
    ax.scatter(yr, depth, s=300, color=color, zorder=5, edgecolors="black")
    ax.annotate(f"{name}\n({depth}L)", (yr, depth),
                textcoords="offset points", xytext=(0, 15),
                ha="center", fontsize=9, fontweight="bold", color=color)
    ax.text(yr, depth - 2.5, note, ha="center", fontsize=7, color="gray")

ax.plot([a[0] for a in archs], [a[2] for a in archs],
        "gray", linewidth=1.5, linestyle="--", alpha=0.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/architecture_timeline.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: architecture_timeline.png")


# --- PLOT 2: Residual block diagram ---
print("📊 Generating: Residual block diagram...")

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Standard Block vs Residual Block — The ResNet Innovation",
             fontsize=13, fontweight="bold")

for ax, (title, has_skip) in zip(axes, [
    ("Standard Block\n(no skip)", False),
    ("Residual Block\n(with skip connection)", True),
]):
    ax.set_xlim(0, 6); ax.set_ylim(0, 10)
    ax.axis("off"); ax.set_facecolor("#fafafa")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    boxes = [
        (3, 8.5, "Input x", "#4CAF50"),
        (3, 7.0, "Conv 3x3\nBatchNorm\nReLU", "#2196F3"),
        (3, 5.0, "Conv 3x3\nBatchNorm", "#1565C0"),
        (3, 3.0, "Output", "#E91E63"),
    ]
    for (x, y, label, color) in boxes:
        ax.add_patch(plt.Rectangle((x-1, y-0.6), 2, 1.0,
                                    facecolor=color, alpha=0.85, edgecolor="black"))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")

    # Vertical arrows
    for y_from, y_to in [(7.9, 7.6), (6.4, 5.6), (4.4, 3.6)]:
        ax.annotate("", xy=(3, y_to), xytext=(3, y_from),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

    if has_skip:
        # Skip connection
        ax.annotate("", xy=(4.5, 3.0), xytext=(4.5, 8.5),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2.5,
                                    connectionstyle="arc3,rad=-0.3"))
        ax.text(5.2, 5.75, "+ x\n(skip)", ha="center", fontsize=10,
                color="red", fontweight="bold")
        ax.text(3, 2.0, "F(x) + x", ha="center", fontsize=11,
                color="darkred", fontweight="bold")
    else:
        ax.text(3, 2.0, "F(x)", ha="center", fontsize=11,
                color="darkblue", fontweight="bold")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/residual_block.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: residual_block.png")


if TF_AVAILABLE:
    # --- PLOT 3: Training history ---
    print("📊 Generating: ResNet training history...")

    hist = history_resnet.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"ResNet-Mini on CIFAR-10 (Test Acc: {acc:.1%})",
                 fontsize=13, fontweight="bold")

    axes[0].plot(hist["loss"],     "steelblue",  2, label="Train")
    axes[0].plot(hist["val_loss"], "darkorange", 2, linestyle="--", label="Val")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist["accuracy"],     "steelblue",  2, label="Train")
    axes[1].plot(hist["val_accuracy"], "darkorange", 2, linestyle="--", label="Val")
    axes[1].set_title(f"Accuracy"); axes[1].legend()
    axes[1].set_ylim(0, 1); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/resnet_training.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("   ✅ Saved: resnet_training.png")


print()
print("=" * 70)
print("✅ ALGORITHM 3: CLASSIC ARCHITECTURES COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  📜 LeNet (1998): proved CNNs work for vision")
print("  ⚡ AlexNet (2012): ReLU + Dropout + GPU = deep learning revolution")
print("  🔢 VGG (2014): 3x3 filters everywhere — simplicity + depth")
print("  ♻️  ResNet (2015): skip connections solve vanishing gradient → 152 layers!")
print("  🎯 Rule: start with ResNet or EfficientNet for new projects")
print()
print("Next: Algorithm 4 → Transfer Learning (use pretrained ResNet on your data!)")
