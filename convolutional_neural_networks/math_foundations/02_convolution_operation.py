"""
🖼️ CONVOLUTIONAL NEURAL NETWORKS — Module 2: The Convolution Operation
========================================================================

Learning Objectives:
  1. Understand what a convolution is: sliding a filter over an image
  2. Implement 2D convolution from scratch in numpy
  3. Know stride and padding and their effect on output size
  4. Understand multiple filters → multiple feature maps
  5. See classic filters: edge detection, blur, sharpen
  6. Connect math convolution to CNN Conv2D layers
  7. Compute output size formula: (W - F + 2P) / S + 1

YouTube Resources:
  ⭐ 3Blue1Brown - But what is a convolution? https://www.youtube.com/watch?v=KuXjwB4LzSA
  ⭐ StatQuest - CNNs explained https://www.youtube.com/watch?v=HGwBXDKFk9I
  📚 CS231n Lecture 5 - Convolutional Networks https://www.youtube.com/watch?v=bNb2fEVKeEo

Time Estimate: 55-65 minutes
Difficulty: Intermediate
Prerequisites: Module 1 (Image Basics), numpy matrix operations
Key Concepts: convolution, filter, kernel, stride, padding, feature map
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "02_convolution_operation")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🖼️  MODULE 2: THE CONVOLUTION OPERATION")
print("=" * 70)
print()
print("A convolution = sliding a small FILTER (kernel) across an image")
print("and computing a dot product at each position.")
print()
print("The output is called a FEATURE MAP — it highlights where the")
print("pattern captured by the filter appears in the image.")
print()
print("Think of it like a flashlight sliding across the image,")
print("lighting up wherever it finds its pattern!")
print()


# ======================================================================
# SECTION 1: The Convolution Step by Step
# ======================================================================
print("=" * 70)
print("SECTION 1: CONVOLUTION STEP BY STEP")
print("=" * 70)
print()
print("Given:")
print("  Image:  5x5 grayscale")
print("  Filter: 3x3 (also called kernel)")
print("  Stride: 1 (move filter 1 pixel at a time)")
print("  Padding: 0 (no padding → output is smaller)")
print()

# Simple 5x5 image
img_5x5 = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
], dtype=float)

# Vertical edge detector filter
edge_filter = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=float)

print("Image (5x5):")
print(img_5x5.astype(int))
print()
print("Filter (3x3) — vertical edge detector:")
print(edge_filter.astype(int))
print()
print("Output size formula:")
print("  output_size = (input_size - filter_size + 2*padding) / stride + 1")
print(f"  = (5 - 3 + 2*0) / 1 + 1 = {(5 - 3 + 0) // 1 + 1}")
print("  → Output is 3x3")
print()
print("Computing element by element:")
print()

output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        patch = img_5x5[i:i+3, j:j+3]
        output[i, j] = np.sum(patch * edge_filter)
        if i < 2 and j < 2:
            print(f"  Position [{i},{j}]: patch * filter = {int(np.sum(patch * edge_filter))}")
            print(f"    Patch:  {patch[0].astype(int)}")
            print(f"            {patch[1].astype(int)}")
            print(f"            {patch[2].astype(int)}")

print()
print("Feature map (output):")
print(output.astype(int))
print()
print("→ Positive values: left→right edge transition")
print("→ Negative values: right→left edge transition")
print("→ Near-zero: no edge in that region")
print()


# ======================================================================
# SECTION 2: Convolution from Scratch
# ======================================================================
print("=" * 70)
print("SECTION 2: CONVOLUTION FROM SCRATCH")
print("=" * 70)
print()

def conv2d(image, kernel, stride=1, padding=0):
    """
    2D convolution (cross-correlation as used in CNNs).
    image:   (H, W) or (H, W, C)
    kernel:  (kH, kW) or (kH, kW, C)
    Returns feature map.
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if kernel.ndim == 2:
        kernel = kernel[:, :, np.newaxis]

    H, W, C = image.shape
    kH, kW, kC = kernel.shape

    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)),
                       mode="constant", constant_values=0)
        H_pad, W_pad = image.shape[:2]
    else:
        H_pad, W_pad = H, W

    out_H = (H_pad - kH) // stride + 1
    out_W = (W_pad - kW) // stride + 1
    output = np.zeros((out_H, out_W))

    for i in range(0, out_H):
        for j in range(0, out_W):
            patch = image[i*stride:i*stride+kH, j*stride:j*stride+kW, :]
            output[i, j] = np.sum(patch * kernel)

    return output


def apply_filters(image_gray, filters_dict):
    """Apply multiple named filters to a grayscale image."""
    results = {}
    for name, filt in filters_dict.items():
        results[name] = conv2d(image_gray, filt, stride=1, padding=0)
    return results


# Create a test image with clear edges
test_img = np.zeros((32, 32), dtype=float)
test_img[8:24, 8:24] = 1.0     # white square in center
test_img = test_img + np.random.randn(32, 32) * 0.05

# Classic image processing filters
filters = {
    "Vertical Edge\n[-1,0,1]": np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=float),
    "Horizontal Edge\n[-1,-1,-1]": np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=float),
    "Diagonal Edge": np.array([[0,1,2],[-1,0,1],[-2,-1,0]], dtype=float),
    "Blur (Mean)": np.ones((3,3), dtype=float) / 9,
    "Sharpen": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=float),
    "Emboss": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=float),
}

print("Applying 6 classic filters to a test image:")
for name, filt in filters.items():
    result = conv2d(test_img, filt)
    print(f"  {name.replace(chr(10), ' '):30s}: output shape {result.shape}")
print()


# ======================================================================
# SECTION 3: Stride and Padding
# ======================================================================
print("=" * 70)
print("SECTION 3: STRIDE AND PADDING")
print("=" * 70)
print()

print("STRIDE: how many pixels the filter moves at each step")
print()
print("  Stride=1: move 1 pixel → output is large (default)")
print("  Stride=2: move 2 pixels → output is half the size (downsampling)")
print()

H, F, P = 32, 3, 0
for stride in [1, 2, 3]:
    out = (H - F + 2*P) // stride + 1
    print(f"  Input=32, Filter=3, Padding=0, Stride={stride} → output = {out}")
print()

print("PADDING: adding zeros around the image border")
print()
print("  padding='valid' (P=0): no padding → output SHRINKS")
print("  padding='same'  (P=1 for 3x3): output is SAME SIZE as input")
print()

for padding in [0, 1, 2]:
    out = (32 - 3 + 2*padding) // 1 + 1
    label = "'valid'" if padding == 0 else f"padding={padding}"
    print(f"  Input=32, Filter=3, {label:12s}, Stride=1 → output = {out}")
print()
print("  Most CNNs use padding='same' in hidden layers to preserve spatial size.")
print()

# Demonstrate padding visually
img_small = np.ones((5, 5), dtype=float) * 0.5
img_small[2, 2] = 1.0
padded = np.pad(img_small, 1, mode="constant", constant_values=0)
print(f"  Original (5x5):  shape {img_small.shape}")
print(f"  Padded (P=1):    shape {padded.shape}")
print()
print("  Padded image:")
print(padded.round(1))
print()


# ======================================================================
# SECTION 4: Multiple Filters → Multiple Feature Maps
# ======================================================================
print("=" * 70)
print("SECTION 4: MULTIPLE FILTERS → MULTIPLE FEATURE MAPS")
print("=" * 70)
print()
print("In a real CNN layer, we use MANY filters simultaneously.")
print("Each filter learns to detect a DIFFERENT pattern.")
print()
print("  N_filters filters → N_filters feature maps stacked depth-wise")
print()
print("  Input:  (H, W, C_in)                    e.g. (32, 32, 3)")
print("  Filter: (kH, kW, C_in) × N_filters      e.g. (3, 3, 3) × 32")
print("  Output: (H', W', N_filters)              e.g. (30, 30, 32)")
print()
print("  Think of it as: 32 different 'detectors' scanning the image")
print("  Each one learns to recognize a different low-level feature")
print("  (edges, corners, blobs, colors, gradients...)")
print()

n_filters = 8
kH, kW, C_in = 3, 3, 1
weights_per_filter = kH * kW * C_in + 1   # +1 for bias
total_params = weights_per_filter * n_filters
print(f"  Example Conv2D layer: {n_filters} filters of size {kH}x{kW}x{C_in}")
print(f"  Parameters per filter: {kH}x{kW}x{C_in} + 1 (bias) = {weights_per_filter}")
print(f"  Total parameters: {weights_per_filter} x {n_filters} = {total_params}")
print()
print("  (Compare to MLP: 784 → 128 = 100,480 params for just one layer!)")
print()


# ======================================================================
# SECTION 5: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 5: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Classic filters and their results ---
print("📊 Generating: Classic filters and feature maps...")

filter_results = apply_filters(test_img, filters)

fig, axes = plt.subplots(2, 7, figsize=(18, 6))
fig.suptitle("🔍 Convolution: Filters Detect Different Image Features",
             fontsize=14, fontweight="bold")

# Original image (top-left and bottom-left)
for row in range(2):
    axes[row, 0].imshow(test_img, cmap="gray")
    axes[row, 0].set_title("Original\nImage", fontsize=10, fontweight="bold")
    axes[row, 0].axis("off")

# Filter visualizations (top row) and results (bottom row)
for col, (name, filt) in enumerate(filters.items(), start=1):
    # Filter heatmap
    im = axes[0, col].imshow(filt, cmap="coolwarm",
                              vmin=-filt.max(), vmax=filt.max())
    axes[0, col].set_title(f"Filter:\n{name}", fontsize=8, fontweight="bold")
    axes[0, col].axis("off")

    # Feature map
    feat = filter_results[name]
    axes[1, col].imshow(feat, cmap="gray")
    axes[1, col].set_title(f"Feature Map\n({feat.shape[0]}x{feat.shape[1]})",
                            fontsize=8)
    axes[1, col].axis("off")

axes[0, 0].set_ylabel("Filters", fontsize=11, fontweight="bold")
axes[1, 0].set_ylabel("Feature Maps", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/filters_and_features.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: filters_and_features.png")


# --- PLOT 2: Stride comparison ---
print("📊 Generating: Stride effect visualization...")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("📐 Stride Effect: Larger Stride = Smaller Output",
             fontsize=13, fontweight="bold")

v_filt = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], dtype=float)
for ax, stride in zip(axes, [1, 2, 3, 4]):
    out = conv2d(test_img, v_filt, stride=stride, padding=0)
    ax.imshow(out, cmap="RdBu", interpolation="nearest")
    ax.set_title(f"Stride={stride}\nOutput: {out.shape[0]}×{out.shape[1]}",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/stride_effect.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: stride_effect.png")


# --- PLOT 3: Step-by-step convolution diagram ---
print("📊 Generating: Convolution step-by-step diagram...")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("🔢 Convolution Step-by-Step: Filter Slides Across Image",
             fontsize=13, fontweight="bold")

pad_img = np.pad(img_5x5, 0, mode="constant")
positions = [(0,0), (0,1), (1,0), (1,1)]
labels = ["Step 1\n(0,0)", "Step 2\n(0,1)", "Step 3\n(1,0)", "Step 4\n(1,1)"]

for ax, (r, c), label in zip(axes, positions, labels):
    display = pad_img.copy()
    highlight = np.zeros_like(pad_img)
    highlight[r:r+3, c:c+3] = 1
    val = int(np.sum(pad_img[r:r+3, c:c+3] * edge_filter))

    ax.imshow(display, cmap="Blues", alpha=0.6, vmin=0, vmax=1)
    ax.imshow(highlight, cmap="Reds", alpha=0.4, vmin=0, vmax=1)

    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(int(display[i, j])), ha="center", va="center",
                    fontsize=12, color="black", fontweight="bold")

    ax.set_title(f"{label}\nOutput = {val}", fontsize=10, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/conv_steps.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: conv_steps.png")


print()
print("=" * 70)
print("✅ MODULE 2 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🔍 Convolution = sliding filter × dot product at each position")
print("  📐 Output size = (W - F + 2P) / S + 1")
print("  🔄 Stride: 1=full size, 2=half size (downsampling)")
print("  🟦 Padding='same': output same size as input (P = F//2)")
print("  📚 N filters → N feature maps stacked in depth")
print()
print("Next: Module 3 → Pooling & Depth (reducing feature maps efficiently!)")
