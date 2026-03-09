"""
🧠 DEEP NEURAL NETWORKS — Module 4: Loss Functions & Optimizers
================================================================

Learning Objectives:
  1. Understand MSE, Binary Cross-Entropy, and Categorical Cross-Entropy
  2. Compute each loss by hand and understand when to use which
  3. Know vanilla SGD and its limitations
  4. Understand Momentum, RMSprop, and Adam optimizers
  5. See the effect of learning rate (too high / too low / just right)
  6. Visualize optimizer paths on a loss landscape
  7. Know the Adam optimizer as the safe default choice

YouTube Resources:
  ⭐ StatQuest - Gradient Descent https://www.youtube.com/watch?v=sDv4f4s2SB8
  ⭐ StatQuest - Adam Optimizer https://www.youtube.com/watch?v=JXQT_vxqwIs
  📚 3Blue1Brown - Gradient descent, how neural networks learn https://www.youtube.com/watch?v=IHZwWFHWa-w

Time Estimate: 50-65 minutes
Difficulty: Intermediate
Prerequisites: Module 3 (Backpropagation), basic calculus
Key Concepts: loss function, gradient descent, SGD, Adam, learning rate
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "04_loss_and_optimizers")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🧠 MODULE 4: LOSS FUNCTIONS & OPTIMIZERS")
print("=" * 70)
print()
print("The loss function measures HOW WRONG your model is.")
print("The optimizer decides HOW to update weights to fix it.")
print()
print("These two work together every training step:")
print("  1. Forward pass → compute prediction")
print("  2. Loss function → measure error")
print("  3. Backprop → compute gradients")
print("  4. Optimizer → update weights")
print()


# ======================================================================
# SECTION 1: Loss Functions
# ======================================================================
print("=" * 70)
print("SECTION 1: LOSS FUNCTIONS")
print("=" * 70)
print()

# --- MSE ---
print("1. MEAN SQUARED ERROR (MSE) — for REGRESSION")
print("   Formula: MSE = (1/n) Σ (y_true - y_pred)²")
print()

y_true_reg = np.array([3.0, 5.0, 2.5, 7.0, 4.0])
y_pred_reg = np.array([2.8, 5.5, 2.0, 6.5, 4.2])

mse = np.mean((y_true_reg - y_pred_reg) ** 2)
rmse = np.sqrt(mse)

print(f"   y_true: {y_true_reg}")
print(f"   y_pred: {y_pred_reg}")
print(f"   errors: {(y_true_reg - y_pred_reg).round(2)}")
print(f"   squared errors: {((y_true_reg - y_pred_reg)**2).round(3)}")
print(f"   MSE  = {mse:.4f}")
print(f"   RMSE = {rmse:.4f}  (same units as target — easier to interpret)")
print()
print("   Gradient: dMSE/dŷ = -2/n * (y_true - y_pred)  →  ŷ - y_true (simplified)")
print()

# --- Binary Cross-Entropy ---
print("2. BINARY CROSS-ENTROPY (BCE) — for BINARY CLASSIFICATION")
print("   Formula: BCE = -(1/n) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]")
print()

y_true_bin = np.array([1, 0, 1, 1, 0])
y_pred_bin = np.array([0.9, 0.1, 0.8, 0.3, 0.2])  # probabilities

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

bce = binary_cross_entropy(y_true_bin, y_pred_bin)
print(f"   y_true:  {y_true_bin}")
print(f"   y_pred:  {y_pred_bin}  (probabilities, 0-1)")
print()
for i in range(5):
    y, yhat = y_true_bin[i], y_pred_bin[i]
    contrib = -(y * np.log(yhat + 1e-15) + (1-y) * np.log(1-yhat + 1e-15))
    print(f"   Sample {i+1}: y={y}, ŷ={yhat} → loss contribution = {contrib:.4f}")
print(f"   BCE = {bce:.4f}")
print()
print("   Key insight: BCE penalizes CONFIDENT wrong predictions VERY harshly!")
print("   (if y=1 and ŷ=0.001 → loss = -log(0.001) = 6.9!)")
print()

# --- Categorical Cross-Entropy ---
print("3. CATEGORICAL CROSS-ENTROPY (CCE) — for MULTI-CLASS CLASSIFICATION")
print("   Formula: CCE = -(1/n) Σᵢ Σₖ y_ik · log(ŷ_ik)")
print()

# One-hot encoded labels and softmax predictions
y_true_cat = np.array([
    [1, 0, 0],  # class 0
    [0, 1, 0],  # class 1
    [0, 0, 1],  # class 2
])
y_pred_cat = np.array([
    [0.85, 0.10, 0.05],
    [0.05, 0.90, 0.05],
    [0.10, 0.20, 0.70],
])

cce = -np.mean(np.sum(y_true_cat * np.log(y_pred_cat + 1e-15), axis=1))
print(f"   3-class problem (cat, dog, bird)")
print(f"   y_true (one-hot): class 0, class 1, class 2")
print(f"   Predictions (softmax):")
for i, (yt, yp) in enumerate(zip(y_true_cat, y_pred_cat)):
    true_cls = np.argmax(yt)
    pred_cls = np.argmax(yp)
    correct = "✅" if true_cls == pred_cls else "❌"
    print(f"     Sample {i+1}: {yp} → pred={pred_cls}, true={true_cls} {correct}")
print(f"   CCE = {cce:.4f}")
print()

print("   QUICK REFERENCE:")
print("   ┌────────────────────────┬─────────────────────┬─────────────┐")
print("   │ Loss Function          │ Task                │ Output Act. │")
print("   ├────────────────────────┼─────────────────────┼─────────────┤")
print("   │ MSE / RMSE             │ Regression          │ Linear      │")
print("   │ Binary Cross-Entropy   │ Binary classif.     │ Sigmoid     │")
print("   │ Categorical Cross-Ent. │ Multi-class classif.│ Softmax     │")
print("   └────────────────────────┴─────────────────────┴─────────────┘")
print()


# ======================================================================
# SECTION 2: Optimizers
# ======================================================================
print("=" * 70)
print("SECTION 2: OPTIMIZERS — FROM SGD TO ADAM")
print("=" * 70)
print()
print("Goal: minimize loss by moving weights in the direction of steepest descent.")
print()

# Simple 1D loss landscape for demonstration
def loss_fn(w):
    """A bumpy loss landscape to show optimizer behavior."""
    return (w - 3)**2 + 2 * np.sin(2 * w) + 0.1 * w**2

def loss_grad(w):
    return 2 * (w - 3) + 4 * np.cos(2 * w) + 0.2 * w


# --- Vanilla SGD ---
print("1. VANILLA SGD (Stochastic Gradient Descent)")
print("   w = w - lr * gradient")
print("   Simple. Can oscillate in narrow valleys or get stuck.")
print()

def sgd(w_init, lr, steps):
    w, history = w_init, [w_init]
    for _ in range(steps):
        g = loss_grad(w)
        w = w - lr * g
        history.append(w)
    return np.array(history)


# --- SGD with Momentum ---
print("2. SGD WITH MOMENTUM")
print("   v = β·v + (1-β)·gradient")
print("   w = w - lr·v")
print("   Accumulates past gradients → faster through valleys, overshoots less.")
print("   β (momentum) typically = 0.9")
print()

def sgd_momentum(w_init, lr, beta, steps):
    w, v, history = w_init, 0, [w_init]
    for _ in range(steps):
        g = loss_grad(w)
        v = beta * v + (1 - beta) * g
        w = w - lr * v
        history.append(w)
    return np.array(history)


# --- RMSprop ---
print("3. RMSprop")
print("   s = β·s + (1-β)·gradient²")
print("   w = w - lr / (√s + ε) * gradient")
print("   Adaptive learning rate per weight. Handles sparse gradients well.")
print()

def rmsprop(w_init, lr, beta, steps, eps=1e-8):
    w, s, history = w_init, 0, [w_init]
    for _ in range(steps):
        g = loss_grad(w)
        s = beta * s + (1 - beta) * g**2
        w = w - lr / (np.sqrt(s) + eps) * g
        history.append(w)
    return np.array(history)


# --- Adam ---
print("4. ADAM (Adaptive Moment Estimation)")
print("   m = β₁·m + (1-β₁)·g          ← momentum (1st moment)")
print("   v = β₂·v + (1-β₂)·g²         ← squared grad (2nd moment)")
print("   m̂ = m/(1-β₁ᵗ)                ← bias correction")
print("   v̂ = v/(1-β₂ᵗ)                ← bias correction")
print("   w = w - lr·m̂ / (√v̂ + ε)")
print()
print("   Adam = Momentum + RMSprop + bias correction")
print("   Default: β₁=0.9, β₂=0.999, ε=1e-8")
print("   → THE standard optimizer for neural networks")
print()

def adam(w_init, lr, beta1, beta2, steps, eps=1e-8):
    w, m, v, history = w_init, 0, 0, [w_init]
    for t in range(1, steps + 1):
        g = loss_grad(w)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(w)
    return np.array(history)


# Run all optimizers
STEPS = 60
w0 = -3.0  # bad starting point

hist_sgd = sgd(w0, lr=0.05, steps=STEPS)
hist_mom = sgd_momentum(w0, lr=0.05, beta=0.9, steps=STEPS)
hist_rms = rmsprop(w0, lr=0.1, beta=0.9, steps=STEPS)
hist_adam = adam(w0, lr=0.3, beta1=0.9, beta2=0.999, steps=STEPS)

w_range = np.linspace(-4, 6, 400)
loss_vals = loss_fn(w_range)
optimal_w = w_range[loss_vals.argmin()]
optimal_loss = loss_vals.min()

print(f"Starting point: w = {w0}")
print(f"True minimum:   w ≈ {optimal_w:.2f}, loss ≈ {optimal_loss:.4f}")
print()
print(f"After {STEPS} steps:")
for name, hist in [("SGD", hist_sgd), ("Momentum", hist_mom),
                   ("RMSprop", hist_rms), ("Adam", hist_adam)]:
    final_loss = loss_fn(hist[-1])
    print(f"  {name:10s}: w = {hist[-1]:6.3f}, loss = {final_loss:.4f}")
print()

# --- Learning Rate Demo ---
print("=" * 70)
print("SECTION 3: LEARNING RATE — THE MOST IMPORTANT HYPERPARAMETER")
print("=" * 70)
print()
print("Too LOW:  training is painfully slow")
print("Too HIGH: loss oscillates or diverges (explodes)")
print("Just right: converges smoothly")
print()

hist_low  = sgd(-3.0, lr=0.005, steps=STEPS)
hist_good = sgd(-3.0, lr=0.05,  steps=STEPS)
hist_high = sgd(-3.0, lr=0.45,  steps=STEPS)

for name, hist, lr in [("Too low  (lr=0.005)", hist_low, 0.005),
                        ("Good     (lr=0.050)", hist_good, 0.05),
                        ("Too high (lr=0.450)", hist_high, 0.45)]:
    losses = [loss_fn(w) for w in hist]
    print(f"  {name}: final loss = {losses[-1]:.4f}")
print()


# ======================================================================
# SECTION 4: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 4: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Loss function shapes ---
print("📊 Generating: Loss function shapes...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("📉 Loss Functions: Measuring How Wrong We Are",
             fontsize=14, fontweight="bold")

# MSE shape
ax = axes[0]
y_true_val = 0.0
y_pred_range = np.linspace(-3, 3, 300)
mse_vals = (y_true_val - y_pred_range)**2
ax.plot(y_pred_range, mse_vals, "steelblue", linewidth=2.5)
ax.axvline(0, color="green", linestyle="--", linewidth=2, label="y_true = 0")
ax.set_title("MSE (Regression)\n(y_true - ŷ)²", fontsize=12, fontweight="bold")
ax.set_xlabel("ŷ (prediction)"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)
ax.fill_between(y_pred_range, mse_vals, alpha=0.15, color="steelblue")

# BCE shape
ax = axes[1]
p = np.linspace(0.001, 0.999, 300)
bce_y1 = -np.log(p)         # when y_true=1
bce_y0 = -np.log(1 - p)     # when y_true=0
ax.plot(p, bce_y1, "darkorange", linewidth=2.5, label="y_true = 1")
ax.plot(p, bce_y0, "purple", linewidth=2.5, label="y_true = 0")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_ylim(0, 6)
ax.set_title("Binary Cross-Entropy\n−[y·log(ŷ) + (1−y)·log(1−ŷ)]",
             fontsize=12, fontweight="bold")
ax.set_xlabel("ŷ (predicted probability)"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)
ax.text(0.05, 5.5, "Confident\nwrong = big loss!", color="red",
        fontsize=9, ha="left")

# Loss landscape
ax = axes[2]
ax.plot(w_range, loss_vals, "black", linewidth=2.5)
ax.axvline(optimal_w, color="green", linestyle="--", linewidth=2,
           label=f"Minimum: w≈{optimal_w:.1f}")
ax.fill_between(w_range, loss_vals, loss_vals.min() - 0.5, alpha=0.1, color="blue")
ax.set_title("Loss Landscape (1D example)\nWhat the optimizer navigates",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Weight w"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True, alpha=0.3)
ax.scatter([w0], [loss_fn(w0)], color="red", s=150, zorder=5, label="Start")

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "loss_functions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: loss_functions.png")


# --- PLOT 2: Optimizer paths ---
print("📊 Generating: Optimizer paths on loss landscape...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("🏃 Optimizer Paths: How Each Optimizer Navigates the Loss",
             fontsize=13, fontweight="bold")

colors_opt = {"SGD": "blue", "Momentum": "darkorange", "RMSprop": "green", "Adam": "red"}
hists = {"SGD": hist_sgd, "Momentum": hist_mom, "RMSprop": hist_rms, "Adam": hist_adam}

# Path plot
ax = axes[0]
ax.plot(w_range, loss_vals, "black", linewidth=2, alpha=0.6, label="Loss landscape")
ax.axvline(optimal_w, color="gray", linestyle="--", linewidth=1, alpha=0.5)

for name, hist in hists.items():
    losses = [loss_fn(w) for w in hist]
    ax.plot(hist, losses, "o-", color=colors_opt[name], markersize=3,
            linewidth=1.5, alpha=0.8, label=name)
    ax.scatter(hist[0], losses[0], color=colors_opt[name], s=100, zorder=5, marker="^")
    ax.scatter(hist[-1], losses[-1], color=colors_opt[name], s=100, zorder=5, marker="*")

ax.set_title("Paths on Loss Landscape", fontsize=12, fontweight="bold")
ax.set_xlabel("Weight w"); ax.set_ylabel("Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Loss over time
ax = axes[1]
for name, hist in hists.items():
    losses = [loss_fn(w) for w in hist]
    ax.plot(losses, color=colors_opt[name], linewidth=2, label=name)

ax.axhline(optimal_loss, color="gray", linestyle="--", linewidth=1.5,
           label=f"Optimal loss ≈ {optimal_loss:.2f}")
ax.set_title("Loss Over Iterations", fontsize=12, fontweight="bold")
ax.set_xlabel("Iteration"); ax.set_ylabel("Loss")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "optimizer_paths.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: optimizer_paths.png")


# --- PLOT 3: Learning rate effect ---
print("📊 Generating: Learning rate effect on convergence...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("📐 Learning Rate: Too Low / Just Right / Too High",
             fontsize=13, fontweight="bold")

for name, hist, color, style in [
    ("Too Low  (lr=0.005)", hist_low,  "blue",   "-"),
    ("Good     (lr=0.050)", hist_good, "green",  "-"),
    ("Too High (lr=0.450)", hist_high, "red",    "-"),
]:
    losses = [loss_fn(w) for w in hist]
    ax.plot(losses, color=color, linewidth=2.5, linestyle=style, label=name)

ax.axhline(optimal_loss, color="black", linestyle="--", linewidth=1.5,
           label=f"Optimal ≈ {optimal_loss:.2f}", alpha=0.6)
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-1, 30)

ax.text(40, 22, "Too high: diverges / oscillates", color="red", fontsize=10)
ax.text(40, 5.5, "Too low: converges slowly", color="blue", fontsize=10)
ax.text(15, optimal_loss + 0.5, "Just right ✅", color="green", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "learning_rate_effect.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: learning_rate_effect.png")



# ======================================================================
# SECTION 5: CONCEPTUAL ARCHITECTURE DIAGRAM — Optimizer Comparison
# ======================================================================
# Left panel: loss landscape (ellipse contours) with 3 optimizer paths
#   SGD (red zigzag), SGD+Momentum (orange smoother), Adam (green direct)
# Right panel: learning rate schedule with warm-up, decay, final phases
# ======================================================================
print("📊 Generating: Optimizer comparison architecture diagram...")

from matplotlib.patches import FancyBboxPatch, Ellipse
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')

# ---------- LEFT PANEL: Loss Landscape & Optimizer Paths ----------
ax = axes[0]
ax.set_facecolor('#0f0f1a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Loss Landscape & Optimizer Paths', color='#e0e0e0',
             fontsize=12, fontweight='bold', pad=10)

# Draw contour-like ellipses representing the loss surface (light to dark = high to low)
ellipse_specs = [
    (7.0, 3.8, 11.5, 5.8, '#1a2a4a', 0.95),   # outermost (high loss)
    (7.0, 3.8, 9.0,  4.8, '#1a3a5a', 0.92),
    (7.0, 3.8, 6.5,  3.8, '#1a4a6a', 0.88),
    (7.0, 3.8, 4.2,  2.8, '#1a5a7a', 0.80),
    (7.0, 3.8, 2.2,  1.6, '#1a6a8a', 0.70),   # innermost (low loss)
]
for cx, cy, w, h, fc, alpha in ellipse_specs:
    ell = Ellipse((cx, cy), width=w, height=h, angle=18,
                  facecolor=fc, edgecolor='#3a7aaa', linewidth=0.8,
                  alpha=alpha, zorder=2)
    ax.add_patch(ell)

# Minimum marker (gold star)
min_x, min_y = 7.0, 3.8
ax.scatter([min_x], [min_y], marker='*', s=320, color='#ffd700',
           zorder=6, linewidths=0.8, edgecolors='white')
ax.text(min_x + 0.35, min_y + 0.35, 'minimum', color='#ffd700',
        fontsize=8, fontweight='bold', zorder=7)

# Common starting point (top-left of landscape)
start_x, start_y = 2.2, 6.8

# SGD path: zigzag (oscillating) in red
sgd_path_x = [start_x, 2.8, 2.4, 3.5, 3.0, 4.2, 3.7, 5.0, 4.6, 5.8,
               5.5, 6.3, 6.1, 6.7, 6.5, 7.0]
sgd_path_y = [start_y, 5.9, 5.2, 5.4, 4.7, 4.9, 4.3, 4.5, 4.1, 4.3,
               4.0, 4.1, 3.95, 4.05, 3.88, 3.8]
ax.plot(sgd_path_x, sgd_path_y, '-o', color='#ff4444', linewidth=1.8,
        markersize=3, alpha=0.9, zorder=4, label='SGD (zigzag)')

# SGD+Momentum path: smoother arc in orange
mom_path_x = [start_x, 3.3, 4.5, 5.4, 6.2, 6.8, 7.0]
mom_path_y = [start_y, 5.6, 4.8, 4.3, 4.0, 3.82, 3.8]
ax.plot(mom_path_x, mom_path_y, '-o', color='#ff9900', linewidth=2.2,
        markersize=3.5, alpha=0.9, zorder=4, label='SGD+Momentum (smooth)')

# Adam path: most direct route in green
adam_path_x = [start_x, 3.8, 5.5, 6.5, 7.0]
adam_path_y = [start_y, 5.2, 4.4, 3.9, 3.8]
ax.plot(adam_path_x, adam_path_y, '-o', color='#44cc66', linewidth=2.5,
        markersize=4, alpha=0.95, zorder=5, label='Adam (direct)')

# Starting point marker
ax.scatter([start_x], [start_y], marker='o', s=140, color='white',
           zorder=6, edgecolors='#aaaaaa', linewidths=1.2)
ax.text(start_x - 0.15, start_y + 0.3, 'start', color='#cccccc',
        fontsize=8, fontweight='bold', ha='center', zorder=7)

# Legend
legend_items = [
    Line2D([0], [0], color='#ff4444', linewidth=2, label='SGD'),
    Line2D([0], [0], color='#ff9900', linewidth=2, label='SGD + Momentum'),
    Line2D([0], [0], color='#44cc66', linewidth=2.5, label='Adam'),
]
legend = ax.legend(handles=legend_items, loc='lower left',
                   fontsize=9, framealpha=0.25,
                   labelcolor='white', facecolor='#111122',
                   edgecolor='#444466')
for text in legend.get_texts():
    text.set_color('white')

# Annotation for zigzag
ax.annotate('oscillates\nacross valley', xy=(3.5, 5.4), xytext=(1.2, 7.1),
            fontsize=7.5, color='#ff6666',
            arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.2))

# ---------- RIGHT PANEL: Learning Rate Schedule ----------
ax = axes[1]
ax.set_facecolor('#0f0f1a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title('Learning Rate Schedule', color='#e0e0e0',
             fontsize=12, fontweight='bold', pad=10)

# Phase boxes
phase_specs = [
    (1.0, 5.2, 3.2, 1.6, '#1a3a1a', '#66cc66', 'Warm-up\n(lr increases)'),
    (5.0, 5.2, 3.2, 1.6, '#3a2a0a', '#ffaa33', 'Decay Phase\n(lr decreases)'),
    (9.0, 5.2, 3.2, 1.6, '#2a1a3a', '#aa66ff', 'Final\n(small lr)'),
]
for bx, by, bw, bh, fc, ec, lbl in phase_specs:
    box = FancyBboxPatch((bx, by), bw, bh,
                         boxstyle='round,pad=0.15', facecolor=fc,
                         edgecolor=ec, linewidth=1.8, alpha=0.92, zorder=3)
    ax.add_patch(box)
    ax.text(bx + bw / 2, by + bh / 2, lbl, ha='center', va='center',
            fontsize=9, color=ec, fontweight='bold', zorder=4)

# Arrows connecting the phase boxes
for x_start, x_end, color in [(4.2, 5.0, '#aaaaaa'), (8.2, 9.0, '#aaaaaa')]:
    ax.annotate('', xy=(x_end, 6.0), xytext=(x_start, 6.0),
                arrowprops=dict(arrowstyle='->', color=color, lw=2.0), zorder=3)

# lr curve drawn as a polyline below the boxes
#   x: 0..14 mapped to a timeline; y: curve shape
curve_x = [1.0, 1.5, 2.3, 3.3, 4.2,   # warm-up (rising)
           5.0, 6.0, 7.0, 8.2,          # decay (falling)
           9.0, 10.5, 12.2]             # final (near-flat low)
curve_y = [1.4, 1.75, 2.6, 3.5, 4.0,
           4.0, 3.4, 2.5, 1.6,
           1.6, 1.35, 1.2]

ax.plot(curve_x, curve_y, color='#66aaff', linewidth=2.8, zorder=5)
ax.fill_between(curve_x, [1.1] * len(curve_x), curve_y,
                color='#66aaff', alpha=0.18, zorder=4)

# Phase region shading under curve
# Warm-up region
ax.fill_between([1.0, 4.2],
                [1.1, 1.1], [4.0, 4.0],
                color='#66cc66', alpha=0.07)
# Decay region
ax.fill_between([5.0, 8.2],
                [1.1, 1.1], [4.0, 1.6],
                color='#ffaa33', alpha=0.07)
# Final region
ax.fill_between([9.0, 12.2],
                [1.1, 1.1], [1.6, 1.2],
                color='#aa66ff', alpha=0.10)

# Axis labels for the curve
ax.text(6.6, 0.75, 'Training Steps  ->', ha='center', fontsize=9,
        color='#888888')
ax.text(0.4, 2.7, 'lr', ha='center', fontsize=10,
        color='#66aaff', fontweight='bold', rotation=90)

# Horizontal dashed line at peak lr
ax.plot([1.0, 12.2], [4.0, 4.0], '--', color='#888888',
        linewidth=0.9, alpha=0.5)
ax.text(12.4, 4.0, 'peak lr', ha='left', va='center',
        fontsize=7.5, color='#aaaaaa')

# Horizontal dashed line at minimum lr
ax.plot([1.0, 12.2], [1.2, 1.2], '--', color='#888888',
        linewidth=0.9, alpha=0.5)
ax.text(12.4, 1.2, 'min lr', ha='left', va='center',
        fontsize=7.5, color='#aaaaaa')

# Tip box at bottom right panel
tip_box = FancyBboxPatch((0.8, 0.15), 12.5, 0.85,
                         boxstyle='round,pad=0.1', facecolor='#0f1f0f',
                         edgecolor='#44cc66', linewidth=1.2, alpha=0.90, zorder=3)
ax.add_patch(tip_box)
ax.text(7.05, 0.58,
        'Tip: Adam handles lr internally per-parameter — '
        'often no schedule needed',
        ha='center', va='center', fontsize=8.2, color='#88dd88',
        fontweight='bold', zorder=4)

plt.savefig(os.path.join(VIS_DIR, '04_optimizer_comparison_diagram.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 04_optimizer_comparison_diagram.png")
print()

print("=" * 70)
print("✅ MODULE 4 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  📉 MSE for regression, BCE for binary, CCE for multi-class")
print("  🏃 Adam = best default optimizer (β₁=0.9, β₂=0.999, lr=0.001)")
print("  📐 Learning rate is the most critical hyperparameter")
print("  ⚠️  Too high lr → diverges, too low → painfully slow")
print()
print("Next: Module 5 → Regularization (preventing overfitting!)")
