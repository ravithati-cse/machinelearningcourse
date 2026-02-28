"""
🤖 CONVOLUTIONAL NEURAL NETWORKS — Algorithm 1: Conv Layer from Scratch
========================================================================

Learning Objectives:
  1. Build a Conv2D layer class in numpy with forward pass
  2. Implement Max Pooling and ReLU as standalone layers
  3. Chain layers into a mini CNN and run a forward pass
  4. Understand the weight sharing that makes CNNs efficient
  5. Compare outputs with TensorFlow's Conv2D for validation
  6. Visualize learned filter weights and their feature maps
  7. Build intuition for what "training a CNN" actually optimizes

YouTube Resources:
  ⭐ Andrej Karpathy - Building micrograd (backprop insight) https://www.youtube.com/watch?v=VMj-3S1tku0
  ⭐ CS231n - CNN forward pass https://www.youtube.com/watch?v=bNb2fEVKeEo
  📚 deeplizard - Convolutional layers explained https://www.youtube.com/watch?v=YRhxdVk_sIs

Time Estimate: 60-70 minutes
Difficulty: Intermediate
Prerequisites: All 3 math foundation modules, Part 3 MLP from scratch
Key Concepts: Conv2D, weight sharing, forward pass, filter banks, feature maps
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../visuals/conv_layer_from_scratch", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 1: CONV LAYER FROM SCRATCH")
print("=" * 70)
print()
print("We've seen the math. Now let's build it.")
print("We'll implement Conv2D, ReLU, MaxPool as Python classes,")
print("chain them into a mini CNN, and verify against TensorFlow.")
print()


# ======================================================================
# SECTION 1: Layer Classes
# ======================================================================
print("=" * 70)
print("SECTION 1: LAYER CLASSES — CONV2D, RELU, MAXPOOL")
print("=" * 70)
print()

class Conv2D:
    """
    Convolutional layer (forward pass only).
    Implements: Z = conv(X, W) + b, with padding and stride.
    """

    def __init__(self, n_filters, kernel_size, stride=1, padding="same",
                 activation=None):
        self.n_filters = n_filters
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.W = None
        self.b = None

    def build(self, input_shape):
        """Initialize weights once we know the input shape."""
        _, _, C = input_shape
        kH, kW = self.kernel_size
        # He initialization
        fan_in = kH * kW * C
        self.W = np.random.randn(kH, kW, C, self.n_filters) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(self.n_filters)
        n_params = self.W.size + self.b.size
        print(f"    Conv2D built: W={self.W.shape}, b={self.b.shape}, params={n_params:,}")

    def _pad(self, X):
        if self.padding == "same":
            kH, kW = self.kernel_size
            pH = (kH - 1) // 2
            pW = (kW - 1) // 2
            return np.pad(X, ((0,0),(pH,pH),(pW,pW),(0,0)), mode="constant")
        return X

    def forward(self, X):
        """
        X: (N, H, W, C)  batch of images
        Returns: (N, H', W', n_filters)
        """
        N, H, W, C = X.shape
        if self.W is None:
            self.build((N, H, W, C)[1:])

        X_pad = self._pad(X)
        _, H_pad, W_pad, _ = X_pad.shape
        kH, kW = self.kernel_size
        s = self.stride

        out_H = (H_pad - kH) // s + 1
        out_W = (W_pad - kW) // s + 1
        Z = np.zeros((N, out_H, out_W, self.n_filters))

        for n in range(self.n_filters):
            for i in range(out_H):
                for j in range(out_W):
                    patch = X_pad[:, i*s:i*s+kH, j*s:j*s+kW, :]
                    Z[:, i, j, n] = np.sum(patch * self.W[:,:,:,n], axis=(1,2,3)) + self.b[n]

        if self.activation == "relu":
            return np.maximum(0, Z)
        return Z

    @property
    def n_params(self):
        if self.W is None: return 0
        return self.W.size + self.b.size


class MaxPool2D:
    """Max pooling layer."""

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        """X: (N, H, W, C)"""
        N, H, W, C = X.shape
        p, s = self.pool_size, self.stride
        out_H = (H - p) // s + 1
        out_W = (W - p) // s + 1
        out = np.zeros((N, out_H, out_W, C))
        for i in range(out_H):
            for j in range(out_W):
                patch = X[:, i*s:i*s+p, j*s:j*s+p, :]
                out[:, i, j, :] = patch.max(axis=(1, 2))
        return out


class Flatten:
    """Flatten spatial dimensions."""
    def forward(self, X):
        N = X.shape[0]
        return X.reshape(N, -1)


class Dense:
    """Fully-connected layer."""
    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.W = None
        self.b = None

    def build(self, n_in):
        self.W = np.random.randn(n_in, self.units) * np.sqrt(2.0 / n_in)
        self.b = np.zeros(self.units)

    def forward(self, X):
        if self.W is None:
            self.build(X.shape[1])
        Z = X @ self.W + self.b
        if self.activation == "relu":
            return np.maximum(0, Z)
        if self.activation == "softmax":
            e = np.exp(Z - Z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        return Z


print("  Layer classes defined: Conv2D, MaxPool2D, Flatten, Dense")
print()


# ======================================================================
# SECTION 2: Build a Mini CNN
# ======================================================================
print("=" * 70)
print("SECTION 2: BUILD A MINI CNN")
print("=" * 70)
print()
print("Architecture: (8,8,1) → Conv(8,3x3) → MaxPool → Conv(16,3x3) → Flatten → Dense(10)")
print()

class MiniCNN:
    """
    A small CNN for demonstration (forward pass only).
    Architecture:
      Input:     (N, 8, 8, 1)
      Conv2D(8): (N, 8, 8, 8)   [ReLU, same padding]
      MaxPool:   (N, 4, 4, 8)
      Conv2D(16):(N, 4, 4, 16)  [ReLU, same padding]
      MaxPool:   (N, 2, 2, 16)
      Flatten:   (N, 64)
      Dense(10): (N, 10)        [Softmax]
    """

    def __init__(self):
        self.layers = [
            Conv2D(n_filters=8,  kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(n_filters=16, kernel_size=3, padding="same", activation="relu"),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(units=10, activation="softmax"),
        ]
        self.layer_names = [
            "Conv2D(8, 3x3, relu)",
            "MaxPool2D(2x2)",
            "Conv2D(16, 3x3, relu)",
            "MaxPool2D(2x2)",
            "Flatten",
            "Dense(10, softmax)",
        ]

    def forward(self, X, verbose=True):
        if verbose:
            print(f"  Input shape: {X.shape}")
        out = X
        for layer, name in zip(self.layers, self.layer_names):
            out = layer.forward(out)
            if verbose:
                print(f"  After {name:35s}: {out.shape}")
        return out

    def summary(self):
        total = 0
        for layer, name in zip(self.layers, self.layer_names):
            if hasattr(layer, "n_params"):
                p = layer.n_params
                total += p
                print(f"  {name:40s} params: {p:>8,}")
            else:
                print(f"  {name:40s} params:        -")
        print(f"  {'Total':40s} params: {total:>8,}")


print("Building MiniCNN...")
cnn = MiniCNN()

# Batch of 4 grayscale 8x8 images
batch = np.random.rand(4, 8, 8, 1).astype(np.float32)
print()
predictions = cnn.forward(batch, verbose=True)
print()
print("  Predictions (softmax probabilities):")
print(predictions.round(3))
print(f"  Predicted classes: {predictions.argmax(axis=1)}")
print()


# ======================================================================
# SECTION 3: Weight Sharing Proof
# ======================================================================
print("=" * 70)
print("SECTION 3: WEIGHT SHARING — ONE FILTER, WHOLE IMAGE")
print("=" * 70)
print()
print("The SAME filter weights are used at EVERY position in the image.")
print("This is what 'weight sharing' means — and why CNNs are so efficient.")
print()

conv_layer = cnn.layers[0]
print(f"  Conv2D layer: {conv_layer.n_filters} filters, each {conv_layer.kernel_size}")
print(f"  Filter 0 weights (3x3x1 = 9 values):")
print(conv_layer.W[:, :, 0, 0].round(3))
print()
print(f"  This ONE set of 9 weights scans the ENTIRE 8x8 image.")
print(f"  It produces the ENTIRE first feature map (8x8 = 64 outputs).")
print(f"  Without sharing: 64 outputs × 9 inputs = 576 params per filter")
print(f"  With sharing:                              {9} params per filter")
print(f"  Saving: {576 - 9} parameters (×{576//9} reduction) — PER FILTER!")
print()


# ======================================================================
# SECTION 4: TensorFlow Comparison
# ======================================================================
print("=" * 70)
print("SECTION 4: TENSORFLOW COMPARISON")
print("=" * 70)
print()

try:
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(42)
    print("  Building equivalent Keras model...")

    tf_model = keras.Sequential([
        keras.layers.Conv2D(8, (3,3), activation="relu", padding="same",
                            input_shape=(8, 8, 1)),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ])

    tf_model.summary()
    print()
    print("  Keras CNN ready! In a real project you'd now call model.compile()")
    print("  and model.fit() to train it on labeled image data.")

    # Get prediction shape
    tf_pred = tf_model.predict(batch, verbose=0)
    print(f"  Keras output shape: {tf_pred.shape}  (same as our from-scratch: {predictions.shape})")

except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print()
    print("  Equivalent Keras code (for reference):")
    print("""
    model = keras.Sequential([
        keras.layers.Conv2D(8,  (3,3), activation='relu', padding='same'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPool2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation='softmax'),
    ])
    """)
print()


# ======================================================================
# SECTION 5: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 5: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Filter bank visualization ---
print("📊 Generating: Filter bank visualization...")

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle("🔍 Conv2D Filter Bank: 8 Filters of 3×3×1\n(Before training — random initialization)",
             fontsize=13, fontweight="bold")

conv1 = cnn.layers[0]

for f_idx in range(8):
    filt = conv1.W[:, :, 0, f_idx]

    # Filter weights
    ax = axes[0, f_idx]
    im = ax.imshow(filt, cmap="coolwarm",
                   vmin=-abs(conv1.W).max(), vmax=abs(conv1.W).max())
    ax.set_title(f"Filter {f_idx+1}", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Feature map (apply to a test image)
    test_img_np = np.zeros((1, 8, 8, 1))
    test_img_np[0, 2:6, 2:6, 0] = 1.0  # white square

    # Manual conv for this single filter
    padded = np.pad(test_img_np[0,:,:,0], 1, mode="constant")
    feat = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            feat[i,j] = np.sum(padded[i:i+3, j:j+3] * filt)

    ax = axes[1, f_idx]
    ax.imshow(feat, cmap="viridis")
    ax.set_title(f"Feature {f_idx+1}", fontsize=9)
    ax.axis("off")

axes[0, 0].set_ylabel("Filters", fontsize=10, fontweight="bold")
axes[1, 0].set_ylabel("Features", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/conv_layer_from_scratch/filter_bank.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: filter_bank.png")


# --- PLOT 2: Forward pass shape trace ---
print("📊 Generating: Forward pass shape trace...")

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
ax.set_title("📐 Mini CNN: Shape at Each Layer", fontsize=14, fontweight="bold")

shape_trace = [
    ("Input", "(4,8,8,1)",  "#4CAF50"),
    ("Conv2D(8)", "(4,8,8,8)",  "#2196F3"),
    ("MaxPool", "(4,4,4,8)",  "#1565C0"),
    ("Conv2D(16)", "(4,4,4,16)", "#9C27B0"),
    ("MaxPool", "(4,2,2,16)", "#6A1B9A"),
    ("Flatten", "(4,64)",     "#E91E63"),
    ("Dense(10)", "(4,10)",     "#4CAF50"),
]

xs = np.linspace(0.08, 0.92, len(shape_trace))
for i, (name, shape, color) in enumerate(shape_trace):
    ax.text(xs[i], 0.7, name, ha="center", va="center", fontsize=10,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9))
    ax.text(xs[i], 0.4, shape, ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)
    if i < len(shape_trace) - 1:
        ax.annotate("", xy=(xs[i+1] - 0.04, 0.55),
                    xytext=(xs[i] + 0.04, 0.55),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

plt.tight_layout()
plt.savefig("../visuals/conv_layer_from_scratch/shape_trace.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: shape_trace.png")


# --- PLOT 3: Single image through first conv layer ---
print("📊 Generating: Single image through conv layer...")

test_img_viz = np.zeros((8, 8))
test_img_viz[1:7, 1:7] = 0.8
test_img_viz[3:5, 3:5] = 1.0
test_img_viz += np.random.rand(8, 8) * 0.1

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("🖼️ One Image → 8 Feature Maps (After Conv2D)",
             fontsize=13, fontweight="bold")

axes[0, 0].imshow(test_img_viz, cmap="gray")
axes[0, 0].set_title("Input Image\n(8×8×1)", fontsize=10, fontweight="bold")
axes[0, 0].axis("off")

# Clear unused
for ax in axes[1, :]:
    ax.axis("off")
axes[1, 0].axis("off")

# Show 8 feature maps
for f in range(8):
    padded = np.pad(test_img_viz, 1, mode="constant")
    feat = np.zeros((8, 8))
    filt = conv1.W[:, :, 0, f]
    for i in range(8):
        for j in range(8):
            feat[i, j] = max(0, np.sum(padded[i:i+3, j:j+3] * filt))

    row, col = divmod(f, 4)
    if row == 0:
        ax = axes[0, col + 1]
    else:
        ax = axes[1, col]
    ax.imshow(feat, cmap="viridis")
    ax.set_title(f"Feature {f+1}", fontsize=9, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("../visuals/conv_layer_from_scratch/image_to_features.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: image_to_features.png")


print()
print("=" * 70)
print("✅ ALGORITHM 1: CONV LAYER FROM SCRATCH COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🏗️  Conv2D: slide filter → dot product → feature map")
print("  ♻️  Weight sharing: same 9 weights → 64 outputs (huge efficiency!)")
print("  🏊 MaxPool: reduces spatial size by 2x, keeps strongest activations")
print("  📐 Shape flows: (N,H,W,C) → Conv → Pool → ... → Flatten → Dense")
print()
print("Next: Algorithm 2 → CNN with Keras (full training on real data!)")
