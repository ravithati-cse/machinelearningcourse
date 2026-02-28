"""
🖼️ CONVOLUTIONAL NEURAL NETWORKS — Module 3: Pooling & Depth
=============================================================

Learning Objectives:
  1. Understand Max Pooling and Average Pooling — how and why
  2. Know the full CNN layer stack: Conv → BatchNorm → ReLU → Pool
  3. Understand how depth builds hierarchical feature representations
  4. Compute the shape at every layer of a CNN
  5. Understand receptive fields and why they grow with depth
  6. Connect the final feature maps to a classification head (Dense layers)
  7. Implement a mini CNN forward pass in numpy end to end

YouTube Resources:
  ⭐ StatQuest - Pooling explained https://www.youtube.com/watch?v=ZjM_XQa5s6s
  ⭐ CS231n - CNN Architectures https://www.youtube.com/watch?v=DAOcjicFr1Y
  📚 3Blue1Brown - Neural network layers https://www.youtube.com/watch?v=aircAruvnKk

Time Estimate: 50-60 minutes
Difficulty: Intermediate
Prerequisites: Modules 1-2 (Image Basics, Convolution)
Key Concepts: pooling, receptive field, depth, feature hierarchy, flatten, global average pooling
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "03_pooling_and_depth")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🖼️  MODULE 3: POOLING & DEPTH — BUILDING THE FULL CNN PIPELINE")
print("=" * 70)
print()
print("After convolution we have feature maps full of activation values.")
print("Pooling SHRINKS them — keeping the most important information.")
print()
print("Why shrink?")
print("  1. Reduces computation (fewer numbers to process)")
print("  2. Adds translation invariance (small shifts don't change pool output)")
print("  3. Increases receptive field (each later neuron 'sees' more image)")
print()


# ======================================================================
# SECTION 1: Max Pooling
# ======================================================================
print("=" * 70)
print("SECTION 1: MAX POOLING")
print("=" * 70)
print()
print("Max Pooling: take the MAXIMUM value in each pool window.")
print()
print("  Window size: typically 2x2")
print("  Stride:      typically 2 (so windows don't overlap)")
print("  Effect:      output = input // 2 in each spatial dimension")
print()

feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 7, 8],
    [3, 2, 1, 0],
    [1, 2, 3, 4],
], dtype=float)

print("Feature map (4x4):")
print(feature_map.astype(int))
print()

def max_pool2d(x, pool_size=2, stride=2):
    H, W = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = patch.max()
    return out

def avg_pool2d(x, pool_size=2, stride=2):
    H, W = x.shape
    out_H = (H - pool_size) // stride + 1
    out_W = (W - pool_size) // stride + 1
    out = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            patch = x[i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
            out[i, j] = patch.mean()
    return out

mp = max_pool2d(feature_map, pool_size=2, stride=2)
ap = avg_pool2d(feature_map, pool_size=2, stride=2)

print("Max Pool (2x2, stride=2) → 2x2 output:")
print()
print("  Top-left     [1,3,5,6] → max = 6   Top-right    [2,4,7,8] → max = 8")
print("  Bottom-left  [3,2,1,2] → max = 3   Bottom-right [1,0,3,4] → max = 4")
print()
print(mp.astype(int))
print()

print("Average Pool (2x2, stride=2) → 2x2 output:")
print(ap.round(2))
print()
print("MAX pool: keeps the strongest activation (most common in CNNs)")
print("AVG pool: smoother, used in global average pooling at end of network")
print()


# ======================================================================
# SECTION 2: Global Average Pooling (GAP)
# ======================================================================
print("=" * 70)
print("SECTION 2: GLOBAL AVERAGE POOLING (GAP)")
print("=" * 70)
print()
print("A modern alternative to Flatten → Dense at the end of the network.")
print()
print("  Flatten:  (7, 7, 512) → 25,088 values → Dense(1024) = 25M params!")
print("  GAP:      (7, 7, 512) → average each 7x7 map → 512 values only")
print()
print("  GAP = average over ALL spatial positions, per channel")
print()

# Example
final_feature_maps = np.random.rand(7, 7, 512)
gap_output = final_feature_maps.mean(axis=(0, 1))   # average over H and W
print(f"  Feature maps shape: {final_feature_maps.shape}")
print(f"  After GAP:          {gap_output.shape}  (one value per channel)")
print()
print("  Benefits of GAP:")
print("    - Far fewer parameters (512 vs 25,088)")
print("    - Works with any input image size")
print("    - Less overfitting (simpler head)")
print("    - Used by MobileNet, ResNet, EfficientNet")
print()


# ======================================================================
# SECTION 3: Layer-by-Layer CNN Shape Tracking
# ======================================================================
print("=" * 70)
print("SECTION 3: CNN SHAPE TRACKING — FOLLOW THE DATA")
print("=" * 70)
print()
print("Let's trace the shape of data through a full CNN (CIFAR-10 style).")
print("Input: (32, 32, 3) — 32x32 RGB image")
print()

layers_spec = [
    ("Input",           (32, 32,  3), ""),
    ("Conv2D(32, 3x3, same) + ReLU",  (32, 32, 32), "32 filters, same padding → same spatial size"),
    ("MaxPool(2x2)",    (16, 16, 32), "halves spatial dims"),
    ("Conv2D(64, 3x3, same) + ReLU",  (16, 16, 64), "64 filters, depth grows"),
    ("MaxPool(2x2)",    ( 8,  8, 64), "halves again"),
    ("Conv2D(128, 3x3, same) + ReLU", ( 8,  8,128), "128 filters"),
    ("MaxPool(2x2)",    ( 4,  4,128), "4x4 spatial"),
    ("Flatten",         (4*4*128,),   f"= {4*4*128} values"),
    ("Dense(256) + ReLU", (256,),     "classification head"),
    ("Dense(10) + Softmax", (10,),    "10 classes"),
]

print(f"  {'Layer':40} {'Output Shape':20} {'Notes'}")
print("  " + "-" * 90)
for layer_name, shape, note in layers_spec:
    shape_str = str(shape)
    print(f"  {layer_name:40} {shape_str:20} {note}")

total_params_cnn = (
    (3*3*3*32 + 32) +       # Conv1
    (3*3*32*64 + 64) +      # Conv2
    (3*3*64*128 + 128) +    # Conv3
    (4*4*128*256 + 256) +   # Dense1
    (256*10 + 10)            # Dense2
)
print()
print(f"  Total parameters: ~{total_params_cnn:,}")
print()


# ======================================================================
# SECTION 4: Receptive Field — What Each Neuron Sees
# ======================================================================
print("=" * 70)
print("SECTION 4: RECEPTIVE FIELD — WHAT EACH NEURON SEES")
print("=" * 70)
print()
print("Receptive field = how much of the INPUT image a single neuron 'sees'")
print()
print("With 3x3 filters and stride=1:")
print()
print("  Layer 1 neuron sees: 3x3 patch of input image")
print("  Layer 2 neuron sees: 5x5 (each 3x3 of layer 1 came from 3x3 of input)")
print("  Layer 3 neuron sees: 7x7")
print("  Layer N neuron sees: (2N+1) x (2N+1)")
print()
print("  Add MaxPool(2x2) after each layer → receptive field grows MUCH faster:")
print()

# Compute receptive fields with pooling
rf = 1
for i in range(1, 7):
    rf = rf * 2 + 2 if i % 2 == 0 else rf + 2  # rough estimate with pooling
    print(f"  After layer {i}: receptive field ≈ {rf}x{rf} pixels")
print()
print("  → Deep layers see LARGE regions: high-level features (faces, objects)")
print("  → Early layers see SMALL regions: low-level features (edges, colors)")
print()


# ======================================================================
# SECTION 5: Feature Hierarchy — What Each Layer Learns
# ======================================================================
print("=" * 70)
print("SECTION 5: FEATURE HIERARCHY — WHAT EACH LAYER LEARNS")
print("=" * 70)
print()
print("This was empirically discovered by visualizing CNN filters:")
print()
print("  Layer 1 (early):  edges, colors, gradients, blobs")
print("                    → simple, interpretable filters")
print()
print("  Layer 2-3 (mid):  textures, corners, curves")
print("                    → combinations of Layer 1 features")
print()
print("  Layer 4-5 (deep): object parts (eyes, wheels, legs)")
print("                    → combinations of textures and shapes")
print()
print("  Last layers:      whole objects (faces, cars, cats)")
print("                    → highly abstract representations")
print()
print("This hierarchy is WHY deep learning works for vision!")
print("Each layer builds on the previous layer's representations.")
print()


# ======================================================================
# SECTION 6: End-to-End Mini CNN in Numpy
# ======================================================================
print("=" * 70)
print("SECTION 6: MINI CNN FORWARD PASS IN NUMPY")
print("=" * 70)
print()

def relu(x): return np.maximum(0, x)

def conv2d_np(image, filters, bias, stride=1, padding=1):
    """Multi-filter 2D convolution. image: (H,W,C), filters: (kH,kW,C,N_filt)"""
    H, W, C = image.shape
    kH, kW, _, N = filters.shape

    if padding > 0:
        image = np.pad(image, ((padding,padding),(padding,padding),(0,0)), 'constant')

    out_H = (image.shape[0] - kH) // stride + 1
    out_W = (image.shape[1] - kW) // stride + 1
    out = np.zeros((out_H, out_W, N))

    for n in range(N):
        for i in range(out_H):
            for j in range(out_W):
                patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
                out[i, j, n] = np.sum(patch * filters[:,:,:,n]) + bias[n]
    return out

# Tiny example: (8,8,1) image → Conv(4 filters, 3x3) → Pool → Flatten
np.random.seed(0)
tiny_img = np.random.rand(8, 8, 1)
conv_filters = np.random.randn(3, 3, 1, 4) * 0.1
conv_bias = np.zeros(4)

conv_out = relu(conv2d_np(tiny_img, conv_filters, conv_bias, stride=1, padding=1))
print(f"  Input:           {tiny_img.shape}")
print(f"  Conv(4, 3x3):    {conv_out.shape}  (same padding → same spatial size)")

# Max pool per channel
pool_out = np.zeros((4, 4, 4))
for c in range(4):
    pool_out[:, :, c] = max_pool2d(conv_out[:, :, c], pool_size=2, stride=2)
print(f"  MaxPool(2x2):    {pool_out.shape}")

# Flatten + Dense (just show shapes)
flat = pool_out.flatten()
print(f"  Flatten:         {flat.shape}  ({4*4*4} values)")

W_dense = np.random.randn(flat.shape[0], 3) * 0.1
logits = flat @ W_dense
probs = np.exp(logits) / np.exp(logits).sum()
print(f"  Dense(3 classes): {probs.shape}  probabilities = {probs.round(3)}")
print(f"  Predicted class:  {probs.argmax()}")
print()
print("  Full forward pass complete! This is what Keras does behind the scenes.")
print()


# ======================================================================
# SECTION 7: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Pooling comparison ---
print("📊 Generating: Pooling comparison...")

# Use a structured feature map to show pooling clearly
fm = np.array([
    [1, 3, 5, 7, 2, 4, 6, 8],
    [9, 2, 8, 1, 3, 9, 1, 5],
    [4, 6, 3, 2, 7, 1, 4, 3],
    [8, 1, 5, 9, 2, 6, 8, 2],
    [3, 7, 2, 4, 5, 3, 7, 1],
    [6, 2, 8, 1, 9, 2, 4, 6],
    [1, 5, 4, 7, 3, 8, 2, 9],
    [9, 3, 6, 2, 1, 5, 7, 4],
], dtype=float)

mp8 = max_pool2d(fm, 2, 2)
ap8 = avg_pool2d(fm, 2, 2)
mp4 = max_pool2d(mp8, 2, 2)

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("🏊 Pooling: Shrinking Feature Maps While Keeping Key Info",
             fontsize=13, fontweight="bold")

for ax, data, title, cmap in zip(axes,
    [fm, mp8, ap8, mp4],
    ["Feature Map\n(8×8)", "Max Pool 2×2\n→ 4×4", "Avg Pool 2×2\n→ 4×4",
     "Max Pool again\n→ 2×2"],
    ["Blues", "Reds", "Greens", "Purples"]
):
    im = ax.imshow(data, cmap=cmap, interpolation="nearest")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{int(data[i,j])}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/pooling_comparison.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: pooling_comparison.png")


# --- PLOT 2: CNN architecture pipeline ---
print("📊 Generating: CNN architecture pipeline...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16); ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
ax.set_title("🧠 CNN Pipeline: How an Image Becomes a Prediction",
             fontsize=14, fontweight="bold", pad=12)

stages = [
    (1.0, "Input\n32×32×3", "#4CAF50", 3.0, 3.0),
    (3.5, "Conv + ReLU\n32×32×32", "#2196F3", 3.0, 3.0),
    (5.5, "MaxPool\n16×16×32", "#1565C0", 2.0, 2.0),
    (7.5, "Conv + ReLU\n16×16×64", "#9C27B0", 2.0, 2.0),
    (9.5, "MaxPool\n8×8×64", "#6A1B9A", 1.5, 1.5),
    (11.2, "Conv + ReLU\n8×8×128", "#E91E63", 1.5, 1.5),
    (12.8, "Flatten\n8192", "#FF5722", 0.4, 2.5),
    (14.2, "Dense\n256", "#FF9800", 0.4, 1.8),
    (15.3, "Softmax\n10", "#4CAF50", 0.4, 1.2),
]

for i, (x, label, color, w, h) in enumerate(stages):
    rect = plt.Rectangle((x - w/2, 4 - h/2), w, h,
                          facecolor=color, alpha=0.85, edgecolor="black",
                          linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x, 4, label, ha="center", va="center",
            fontsize=8, color="white", fontweight="bold", zorder=4)
    if i < len(stages) - 1:
        next_x = stages[i+1][0]
        ax.annotate("", xy=(next_x - stages[i+1][2]/2, 4),
                    xytext=(x + w/2, 4),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

# Labels
ax.text(4.5, 1.5, "Feature Extraction\n(learns WHAT to look for)",
        ha="center", fontsize=10, color="#1565C0", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#E3F2FD", edgecolor="#1565C0"))
ax.text(13.8, 1.5, "Classification\n(makes decision)",
        ha="center", fontsize=10, color="#E65100", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#FFF3E0", edgecolor="#E65100"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/cnn_pipeline.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: cnn_pipeline.png")


# --- PLOT 3: Feature hierarchy ---
print("📊 Generating: Feature hierarchy visualization...")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("📚 Feature Hierarchy: What Each CNN Layer Learns",
             fontsize=13, fontweight="bold")

np.random.seed(1)
examples = [
    # Layer 1: random edge-like filters
    np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=float),
    # Layer 2: texture-like pattern
    np.tile(np.array([[1,-1],[-1,1]], dtype=float), (4, 4)) if False
    else (np.sin(np.linspace(0,4*np.pi,16).reshape(4,4)) * np.cos(np.linspace(0,4*np.pi,16).reshape(4,4).T)),
    # Layer 3: part-like
    np.zeros((8, 8)),
    # Layer 4: object-like
    np.zeros((16, 16)),
]

# Generate synthetic "what neurons see"
layer_names = ["Layer 1\n(Edges & Blobs)", "Layer 2\n(Textures)", "Layer 3\n(Parts)", "Layer 4\n(Objects)"]
colors_list = ["Blues", "Greens", "Oranges", "Purples"]
np.random.seed(42)

for ax, name, cmap_name in zip(axes, layer_names, colors_list):
    # Simulate increasingly complex patterns
    size = 8
    if "1" in name:
        arr = np.zeros((size, size))
        arr[:, size//2-1:size//2+1] = 1   # vertical edge
    elif "2" in name:
        x = np.linspace(0, 4*np.pi, size)
        arr = np.outer(np.sin(x), np.cos(x))
    elif "3" in name:
        arr = np.zeros((size, size))
        # Simulate a "face part" — circle
        for i in range(size):
            for j in range(size):
                if (i-3)**2 + (j-3)**2 < 5:
                    arr[i, j] = 1
                if (i-3)**2 + (j-6)**2 < 5:
                    arr[i, j] = 1
    else:
        # Simulate "face"
        arr = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if (i-4)**2 + (j-4)**2 < 10:
                    arr[i, j] = 0.5
                    if (i-2)**2 + (j-2)**2 < 2 or (i-2)**2 + (j-6)**2 < 2:
                        arr[i, j] = 1
                    if i == 5 and 2 <= j <= 6:
                        arr[i, j] = 1

    ax.imshow(arr, cmap=cmap_name, interpolation="bilinear")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/feature_hierarchy.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: feature_hierarchy.png")


print()
print("=" * 70)
print("✅ MODULE 3 COMPLETE! — Math Foundations Section DONE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🏊 Max Pool: keeps maximum → fast downsampling, translation invariance")
print("  📐 Output after Pool 2x2: spatial dims halved, channels preserved")
print("  🌍 Global Average Pooling: modern, fewer params, works at any size")
print("  🔭 Receptive field grows with depth → later = bigger picture")
print("  📚 Feature hierarchy: edges → textures → parts → objects")
print()
print("Next: Conv Layer from Scratch (put it all together in code!)")
