"""
🖼️ CONVOLUTIONAL NEURAL NETWORKS — Module 1: Image Basics
===========================================================

Learning Objectives:
  1. Understand how digital images are stored as numpy arrays
  2. Know the difference between grayscale (H x W) and RGB (H x W x 3) images
  3. Understand pixel values, normalization, and color channels
  4. Learn how images are batched for neural networks: (N, H, W, C)
  5. Apply basic image operations: crop, resize, flip, brightness
  6. Understand why CNNs beat MLPs on images (translation invariance, locality)
  7. Visualize RGB channels separately and recombine them

YouTube Resources:
  ⭐ 3Blue1Brown - But what is a convolution? https://www.youtube.com/watch?v=KuXjwB4LzSA
  ⭐ StatQuest - Image classification https://www.youtube.com/watch?v=HGwBXDKFk9I
  📚 CS231n Lecture 5 - CNNs https://www.youtube.com/watch?v=bNb2fEVKeEo

Time Estimate: 40-50 minutes
Difficulty: Beginner
Prerequisites: numpy basics, Part 3 Module 1 (neurons)
Key Concepts: pixel, channel, RGB, grayscale, image tensor, batch
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "01_image_basics")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🖼️  MODULE 1: IMAGE BASICS — HOW COMPUTERS SEE IMAGES")
print("=" * 70)
print()
print("Before CNNs, let's understand what an image actually IS to a computer.")
print()
print("An image = a grid of numbers.")
print("  Grayscale: each pixel is ONE number (0=black, 255=white)")
print("  Color RGB: each pixel is THREE numbers (Red, Green, Blue channels)")
print()


# ======================================================================
# SECTION 1: Grayscale Images
# ======================================================================
print("=" * 70)
print("SECTION 1: GRAYSCALE IMAGES")
print("=" * 70)
print()
print("A grayscale image is a 2D array: shape = (Height, Width)")
print("Each value: 0 (black) to 255 (white)")
print()

# Create a simple synthetic 8x8 grayscale image (like a mini MNIST digit)
digit_like = np.array([
    [0,   0,   0,  50, 200, 255, 200,  50],
    [0,   0,  50, 200, 255, 255, 255, 200],
    [0,  50, 200, 255,  50,   0,  50, 255],
    [0,  50, 200, 255,   0,   0,   0, 200],
    [0,  50, 200, 255,   0,   0,   0, 200],
    [0,  50, 200, 255,  50,   0,  50, 255],
    [0,   0,  50, 200, 255, 255, 255, 200],
    [0,   0,   0,  50, 200, 255, 200,  50],
], dtype=np.uint8)

print(f"  Image shape: {digit_like.shape}   (Height=8, Width=8)")
print(f"  Pixel dtype: {digit_like.dtype}      (unsigned 8-bit integer)")
print(f"  Value range: {digit_like.min()} to {digit_like.max()}")
print()
print("  Raw pixel values (the actual 8x8 number grid):")
print(digit_like)
print()
print("  Normalized version (divide by 255 for neural network input):")
print((digit_like / 255.0).round(2))
print()

# Key operations
print("  Common image operations:")
flipped_h = np.fliplr(digit_like)
flipped_v = np.flipud(digit_like)
rotated   = np.rot90(digit_like)
brighter  = np.clip(digit_like.astype(int) + 80, 0, 255).astype(np.uint8)

print(f"  Horizontal flip: np.fliplr(img)  → shape {flipped_h.shape}")
print(f"  Vertical flip:   np.flipud(img)  → shape {flipped_v.shape}")
print(f"  Rotate 90°:      np.rot90(img)   → shape {rotated.shape}")
print(f"  Brighter:        clip(img+80)    → range {brighter.min()}-{brighter.max()}")
print()


# ======================================================================
# SECTION 2: RGB Color Images
# ======================================================================
print("=" * 70)
print("SECTION 2: RGB COLOR IMAGES")
print("=" * 70)
print()
print("A color image has 3 channels: Red, Green, Blue")
print("Shape = (Height, Width, Channels) = (H, W, 3)")
print()
print("Each channel is a full grayscale image — one per color.")
print("Combining them gives color!")
print()

# Create a synthetic 16x16 RGB image
H, W = 16, 16
rgb_img = np.zeros((H, W, 3), dtype=np.uint8)

# Red diagonal stripe
for i in range(H):
    for j in range(W):
        if abs(i - j) < 3:
            rgb_img[i, j, 0] = 220   # Red channel
        if i > H//2:
            rgb_img[i, j, 2] = 180   # Blue in bottom half
        if j > W//2 and i < H//2:
            rgb_img[i, j, 1] = 180   # Green in top-right

print(f"  RGB image shape: {rgb_img.shape}  (H=16, W=16, C=3)")
print(f"  Total values: {rgb_img.size}  (16 × 16 × 3 = {16*16*3})")
print()
print("  A single pixel (row=4, col=4):")
px = rgb_img[4, 4]
print(f"    rgb_img[4, 4] = [{px[0]}, {px[1]}, {px[2]}]")
print(f"    R={px[0]}, G={px[1]}, B={px[2]}")
print()

# Separating channels
R = rgb_img[:, :, 0]
G = rgb_img[:, :, 1]
B = rgb_img[:, :, 2]
print("  Extracting channels:")
print(f"    Red channel:   img[:, :, 0]  → shape {R.shape}")
print(f"    Green channel: img[:, :, 1]  → shape {G.shape}")
print(f"    Blue channel:  img[:, :, 2]  → shape {B.shape}")
print()

# Common formats
print("  Image format conventions (important for libraries!):")
print("  ┌────────────┬──────────────────┬────────────────────────────┐")
print("  │ Framework  │ Format           │ Example (batch of 32)      │")
print("  ├────────────┼──────────────────┼────────────────────────────┤")
print("  │ TensorFlow │ (N, H, W, C)     │ (32, 224, 224, 3)          │")
print("  │ PyTorch    │ (N, C, H, W)     │ (32, 3, 224, 224)          │")
print("  │ OpenCV     │ (H, W, C) BGR    │ (224, 224, 3) — note BGR!  │")
print("  │ Matplotlib │ (H, W, C) RGB    │ (224, 224, 3)              │")
print("  └────────────┴──────────────────┴────────────────────────────┘")
print()


# ======================================================================
# SECTION 3: Image Batches for Neural Networks
# ======================================================================
print("=" * 70)
print("SECTION 3: IMAGE BATCHES FOR NEURAL NETWORKS")
print("=" * 70)
print()
print("Neural networks process images in BATCHES (multiple at once).")
print()
print("  Single image: (H, W, C)        e.g. (28, 28, 1) for MNIST")
print("  Batch of N:   (N, H, W, C)     e.g. (32, 28, 28, 1)")
print()

# Simulate a batch
batch = np.random.randint(0, 256, size=(8, 16, 16, 3), dtype=np.uint8)
print(f"  Batch of 8 images (16x16 RGB): shape = {batch.shape}")
print(f"  First image: batch[0].shape = {batch[0].shape}")
print(f"  Red channel of first image: batch[0, :, :, 0].shape = {batch[0,:,:,0].shape}")
print()

# Normalizing
batch_norm = batch.astype(np.float32) / 255.0
print(f"  After normalization (/ 255): dtype={batch_norm.dtype}, range=[{batch_norm.min():.2f}, {batch_norm.max():.2f}]")
print()


# ======================================================================
# SECTION 4: Why CNNs Beat MLPs on Images
# ======================================================================
print("=" * 70)
print("SECTION 4: WHY CNNs BEAT MLPs ON IMAGES")
print("=" * 70)
print()
print("Problem with MLP on images:")
print()
print("  A 224x224 RGB image = 224 × 224 × 3 = 150,528 inputs")
print("  MLP first hidden layer with 1024 neurons:")
print(f"  Parameters = 150,528 × 1,024 + 1,024 = {150528*1024+1024:,}")
print("  → 154 MILLION parameters just in the first layer!")
print("  → Insane memory, slow, easy to overfit")
print()
print("Two more problems:")
print()
print("  1. MLPs have NO SPATIAL AWARENESS")
print("     Each pixel is an independent feature — the MLP doesn't know")
print("     that pixel (10,10) is next to pixel (10,11)!")
print()
print("  2. MLPs have NO TRANSLATION INVARIANCE")
print("     A cat in the top-left vs bottom-right = completely different input")
print("     The MLP must relearn 'cat' for every possible position!")
print()
print("CNNs solve BOTH problems with two key ideas:")
print()
print("  1. LOCAL CONNECTIVITY: each neuron only looks at a small patch (3x3, 5x5)")
print("     → Captures local patterns: edges, textures")
print()
print("  2. WEIGHT SHARING: the SAME filter is applied everywhere")
print("     → Learns 'edge detector' once, applies it to the whole image")
print("     → 99% fewer parameters than MLP")
print("     → Automatically translation invariant!")
print()

# Parameter comparison
img_h, img_w, img_c = 28, 28, 1
n_inputs = img_h * img_w * img_c

mlp_params = n_inputs * 128 + 128 + 128 * 10 + 10
cnn_filter_size = 3 * 3 * 1   # 3x3 filter, 1 input channel
cnn_n_filters = 32
cnn_params = cnn_filter_size * cnn_n_filters + cnn_n_filters  # Conv layer only
print(f"  Parameter comparison (MNIST 28x28x1 → 10 classes):")
print(f"    MLP (784→128→10):  {mlp_params:>10,} parameters")
print(f"    1 Conv layer only: {cnn_params:>10,} parameters  (Conv2D: 32 filters of 3x3)")
print(f"    Ratio: MLP has {mlp_params // cnn_params}× more params!")
print()


# ======================================================================
# SECTION 5: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 5: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Grayscale image and operations ---
print("📊 Generating: Grayscale image and augmentations...")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("🖼️ Grayscale Image Basics: Pixel Values & Augmentations",
             fontsize=13, fontweight="bold")

images = [digit_like, flipped_h, flipped_v, rotated, brighter,
          (digit_like / 255.0 * digit_like / 255.0 * 255).astype(np.uint8),
          np.clip(digit_like.astype(int) - 50, 0, 255).astype(np.uint8),
          np.clip(digit_like * 1.5, 0, 255).astype(np.uint8)]
titles = ["Original", "Flip Horizontal", "Flip Vertical", "Rotate 90°",
          "Brighter (+80)", "Gamma (²)", "Darker (-50)", "Contrast (×1.5)"]

for ax, img, title in zip(axes.flatten(), images, titles):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/grayscale_ops.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: grayscale_ops.png")


# --- PLOT 2: RGB channels ---
print("📊 Generating: RGB channel decomposition...")

# Create a more colorful synthetic image
img_color = np.zeros((64, 64, 3), dtype=np.uint8)
for i in range(64):
    for j in range(64):
        img_color[i, j, 0] = int(255 * i / 63)           # Red gradient top→bottom
        img_color[i, j, 1] = int(255 * j / 63)           # Green gradient left→right
        img_color[i, j, 2] = int(255 * (1 - i / 63))     # Blue gradient bottom→top

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
fig.suptitle("🎨 RGB Image: Full Color and Individual Channels",
             fontsize=13, fontweight="bold")

axes[0].imshow(img_color)
axes[0].set_title("Full RGB Image", fontsize=11, fontweight="bold")
axes[0].axis("off")

channel_info = [(0, "Red Channel",   "Reds"),
                (1, "Green Channel", "Greens"),
                (2, "Blue Channel",  "Blues")]
for ax, (ch, title, cmap) in zip(axes[1:4], channel_info):
    ch_img = np.zeros_like(img_color)
    ch_img[:, :, ch] = img_color[:, :, ch]
    ax.imshow(ch_img)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axis("off")

# Grayscale version
gray = (0.299 * img_color[:,:,0] + 0.587 * img_color[:,:,1]
        + 0.114 * img_color[:,:,2]).astype(np.uint8)
axes[4].imshow(gray, cmap="gray")
axes[4].set_title("Grayscale\n(0.299R+0.587G+0.114B)", fontsize=11, fontweight="bold")
axes[4].axis("off")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/rgb_channels.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: rgb_channels.png")


# --- PLOT 3: MLP vs CNN parameter comparison ---
print("📊 Generating: MLP vs CNN parameter comparison...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Why CNNs? Drastically Fewer Parameters Than MLPs",
             fontsize=13, fontweight="bold")

# Bar chart
model_names = ["MLP\n(784→128→10)", "CNN\n(1 Conv layer only)", "Full CNN\n(industry standard)"]
param_counts = [mlp_params, cnn_params, 60000]  # rough LeNet-like count
colors_bar = ["#E57373", "#81C784", "#64B5F6"]

ax = axes[0]
bars = ax.bar(model_names, param_counts, color=colors_bar, edgecolor="black", width=0.5)
ax.set_ylabel("Number of Parameters", fontsize=11)
ax.set_title("Parameter Count Comparison\n(MNIST-sized input)", fontsize=11, fontweight="bold")
ax.set_yscale("log")
for bar, count in zip(bars, param_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
            f"{count:,}", ha="center", fontsize=9, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")

# Receptive field diagram
ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
ax.set_title("CNN: Local Receptive Field\n(3×3 filter sees 9 pixels, not all 784)",
             fontsize=11, fontweight="bold")

# Draw pixel grid
for i in range(7):
    for j in range(7):
        color = "#BBDEFB" if (1 <= i <= 3 and 1 <= j <= 3) else "#EEEEEE"
        rect = plt.Rectangle((j*0.9+0.5, i*0.8+1.0), 0.85, 0.75,
                              facecolor=color, edgecolor="gray", linewidth=0.5)
        ax.add_patch(rect)

ax.add_patch(plt.Rectangle((1.4+0.5, 1.8+1.0), 2.7, 2.4,
                             facecolor="none", edgecolor="red", linewidth=3))
ax.text(5.5, 4.5, "3×3 filter\n(9 weights shared\nacross whole image!)",
        fontsize=10, ha="center",
        bbox=dict(boxstyle="round", facecolor="#FFCDD2", edgecolor="red"))
ax.text(3.0, 0.5, "Input image (7×7 shown)", ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/mlp_vs_cnn.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: mlp_vs_cnn.png")



# ============= CONCEPTUAL DIAGRAM =============
print("📊 Generating: Image data structure concept diagram...")
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes:
    ax.set_facecolor('#0f0f1a')
fig.suptitle("Image as a Tensor — How Computers See Images",
             fontsize=15, fontweight='bold', color='white', y=1.01)

# ---- LEFT panel: Grayscale 5×5 grid ----
ax = axes[0]
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(-1.2, 6.0)
ax.axis('off')
ax.set_title("Grayscale Image\n2D: Height × Width\n(1 channel)",
             fontsize=11, color='#aabbcc', fontweight='bold')

pixel_vals = [
    [12,  45,  80, 120, 200],
    [30,  90, 150, 210, 240],
    [60, 130, 180, 230, 250],
    [20,  70, 130, 190, 220],
    [ 8,  35,  75, 110, 170],
]
for row in range(5):
    for col in range(5):
        val = pixel_vals[row][col]
        grey = val / 255.0
        facecolor = (grey, grey, grey)
        rect = FancyBboxPatch((col, 4 - row), 1.0, 1.0,
                              boxstyle="square,pad=0.02",
                              facecolor=facecolor,
                              edgecolor='#555588', linewidth=1.0)
        ax.add_patch(rect)
        text_color = 'white' if grey < 0.55 else 'black'
        ax.text(col + 0.5, 4 - row + 0.5, str(val),
                ha='center', va='center',
                fontsize=7.5, color=text_color, fontweight='bold')

ax.text(2.5, -0.6, "Each cell = 1 pixel value (0–255)",
        ha='center', va='center', fontsize=8.5, color='#99aacc',
        style='italic')

# ---- CENTER panel: RGB 3-channel stack ----
ax = axes[1]
ax.set_xlim(-0.5, 6.5)
ax.set_ylim(-1.2, 7.5)
ax.axis('off')
ax.set_title("RGB Color Image\n3D: Height × Width × 3 channels",
             fontsize=11, color='#aabbcc', fontweight='bold')

channel_configs = [
    ('R', '#cc2222', 0.0, 4.8),
    ('G', '#22aa44', 0.5, 3.8),
    ('B', '#2255cc', 1.0, 2.8),
]

grid_size = 4

for ch_label, ch_color, x_off, y_off in channel_configs:
    r, g, b = (
        (0.7, 0.1, 0.1) if ch_label == 'R' else
        (0.1, 0.6, 0.2) if ch_label == 'G' else
        (0.1, 0.2, 0.8)
    )
    for row in range(grid_size):
        for col in range(grid_size):
            intensity = 0.3 + 0.5 * (col + row) / (2 * (grid_size - 1))
            fc = (r * intensity, g * intensity, b * intensity)
            rect = FancyBboxPatch(
                (x_off + col, y_off + (grid_size - 1 - row)),
                0.9, 0.9,
                boxstyle="square,pad=0.02",
                facecolor=fc,
                edgecolor='#334466', linewidth=0.8, alpha=0.92
            )
            ax.add_patch(rect)
    ax.text(x_off - 0.35, y_off + grid_size / 2, ch_label,
            ha='center', va='center', fontsize=11,
            color=ch_color, fontweight='bold')

ax.text(3.0, -0.6, "3 full grids stacked → one color image",
        ha='center', va='center', fontsize=8.5, color='#99aacc',
        style='italic')

ax.annotate('', xy=(4.8, 6.5), xytext=(4.1, 5.5),
            arrowprops=dict(arrowstyle='->', color='#ff6644', lw=1.5))
ax.text(5.1, 6.6, 'R', fontsize=9, color='#ff4422', fontweight='bold')
ax.annotate('', xy=(5.3, 5.5), xytext=(4.6, 4.5),
            arrowprops=dict(arrowstyle='->', color='#44cc44', lw=1.5))
ax.text(5.6, 5.6, 'G', fontsize=9, color='#44cc44', fontweight='bold')
ax.annotate('', xy=(5.8, 4.5), xytext=(5.1, 3.5),
            arrowprops=dict(arrowstyle='->', color='#4488ff', lw=1.5))
ax.text(6.1, 4.6, 'B', fontsize=9, color='#4488ff', fontweight='bold')

# ---- RIGHT panel: Batch of 4 RGB "cubes" ----
ax = axes[2]
ax.set_xlim(-0.5, 7.0)
ax.set_ylim(-1.5, 8.5)
ax.axis('off')
ax.set_title("Batch of Images\n4D tensor: Batch × H × W × C\n(how PyTorch/TensorFlow work)",
             fontsize=11, color='#aabbcc', fontweight='bold')

batch_colors = [
    ('#1a3a6a', '#2255aa', '#3377cc'),
    ('#1a4a2a', '#227733', '#33aa55'),
    ('#4a2a1a', '#883311', '#cc5533'),
    ('#3a1a4a', '#662288', '#9933bb'),
]

cube_w, cube_h = 2.5, 2.0
x_gap, y_gap = 0.6, 0.5

positions = [
    (0.0, 5.2),
    (3.0, 5.2),
    (0.0, 2.2),
    (3.0, 2.2),
]

for b_idx, ((bx, by), (c_dark, c_mid, c_light)) in enumerate(
        zip(positions, batch_colors)):
    # Draw 3 layers to simulate depth
    for depth, fc in enumerate([c_dark, c_mid, c_light]):
        off = depth * 0.18
        rect = FancyBboxPatch(
            (bx + off, by - off),
            cube_w, cube_h,
            boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor='white',
            linewidth=0.8, alpha=0.85
        )
        ax.add_patch(rect)
    ax.text(bx + cube_w / 2, by + cube_h / 2,
            f"Image {b_idx + 1}\nH × W × C",
            ha='center', va='center', fontsize=8.5,
            color='white', fontweight='bold')

ax.text(3.1, -0.7,
        "N=4 images stacked into one 4D tensor\n"
        "Shape: (4, H, W, C)",
        ha='center', va='center', fontsize=8.5, color='#99aacc',
        style='italic')

ax.text(3.1, 1.3,
        "batch[0] → Image 1    batch[1] → Image 2\n"
        "batch[2] → Image 3    batch[3] → Image 4",
        ha='center', va='center', fontsize=7.5, color='#ccddee')

plt.tight_layout(pad=1.8)
plt.savefig(os.path.join(VIS_DIR, '04_image_data_structure_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   ✅ Saved: 04_image_data_structure_concept.png")
# ============= END CONCEPTUAL DIAGRAM =============

print()
print("=" * 70)
print("✅ MODULE 1 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🖼️  Grayscale image = 2D array (H, W); Color = 3D array (H, W, 3)")
print("  📦  Batch format for TF/Keras: (N, H, W, C)")
print("  🔢  Always normalize pixels: img / 255.0  →  [0.0, 1.0]")
print("  ⚡  CNNs: local filters + weight sharing = 99% fewer params than MLP")
print("  🔄  Same filter applied everywhere = translation invariance!")
print()
print("Next: Module 2 → The Convolution Operation (how filters find features!)")
