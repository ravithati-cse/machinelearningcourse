"""
🧠 DEEP NEURAL NETWORKS — Module 5: Regularization
====================================================

Learning Objectives:
  1. Understand overfitting: why models fail on new data
  2. Grasp the bias-variance tradeoff
  3. Apply L2 regularization (weight decay) and understand its effect
  4. Apply L1 regularization (Lasso) and understand sparsity
  5. Implement Dropout in numpy and understand why it works
  6. Understand Batch Normalization conceptually
  7. Use Early Stopping to prevent overfitting automatically

YouTube Resources:
  ⭐ StatQuest - Regularization Part 1 https://www.youtube.com/watch?v=Q81RR3yKn30
  ⭐ StatQuest - Dropout https://www.youtube.com/watch?v=D8PJAL-MZv8
  📚 3Blue1Brown - Neural network regularization https://www.youtube.com/watch?v=Uyf4IQZS8fc

Time Estimate: 50-60 minutes
Difficulty: Intermediate
Prerequisites: Module 3 (Backprop), Module 4 (Loss & Optimizers)
Key Concepts: overfitting, L1/L2 regularization, dropout, batch normalization, early stopping
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "05_regularization")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🧠 MODULE 5: REGULARIZATION — FIGHTING OVERFITTING")
print("=" * 70)
print()
print("Overfitting: model learns the TRAINING data TOO well")
print("  → It memorizes noise instead of learning true patterns")
print("  → Great on training data, terrible on new (test) data")
print()
print("Regularization: add CONSTRAINTS to prevent overfitting")
print()


# ======================================================================
# SECTION 1: Demonstrating Overfitting
# ======================================================================
print("=" * 70)
print("SECTION 1: DEMONSTRATING OVERFITTING")
print("=" * 70)
print()

def generate_data(n=60, noise=0.3):
    X = np.sort(np.random.uniform(-3, 3, n))
    y = np.sin(X) + noise * np.random.randn(n)
    return X, y

X_all, y_all = generate_data(n=60)
split = 40
X_train, y_train = X_all[:split], y_all[:split]
X_test,  y_test  = X_all[split:], y_all[split:]

def fit_polynomial(X, y, degree):
    """Fit polynomial of given degree."""
    coeffs = np.polyfit(X, y, degree)
    return coeffs

def poly_predict(X, coeffs):
    return np.polyval(coeffs, X)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

print("Fitting polynomials of increasing complexity to the same data:")
print()
print(f"{'Degree':8} {'Train MSE':12} {'Test MSE':12} {'Status':20}")
print("-" * 55)

X_plot = np.linspace(-3, 3, 300)
results = {}
for deg in [1, 3, 5, 10, 15]:
    c = fit_polynomial(X_train, y_train, deg)
    train_mse = mse(y_train, poly_predict(X_train, c))
    test_mse  = mse(y_test,  poly_predict(X_test, c))
    results[deg] = (c, train_mse, test_mse)
    if train_mse < 0.01 and test_mse > 0.5:
        status = "⚠️  OVERFIT"
    elif train_mse > 0.3:
        status = "📉 Underfit"
    else:
        status = "✅ Good fit"
    print(f"  deg={deg:2d}  {train_mse:10.4f}  {test_mse:10.4f}  {status}")

print()
print("Key observation:")
print("  Higher degree → better train error BUT worse test error = overfitting!")
print()


# ======================================================================
# SECTION 2: Bias-Variance Tradeoff
# ======================================================================
print("=" * 70)
print("SECTION 2: THE BIAS-VARIANCE TRADEOFF")
print("=" * 70)
print()
print("BIAS: error from wrong assumptions (too simple model)")
print("  → High bias = underfitting (model can't capture patterns)")
print()
print("VARIANCE: sensitivity to fluctuations in training data")
print("  → High variance = overfitting (model captures noise)")
print()
print("Total Error = Bias² + Variance + Irreducible Noise")
print()
print("Goal: find the sweet spot — low bias AND low variance")
print()
print("  Simple model:  High Bias,  Low Variance  → underfits")
print("  Complex model: Low Bias,   High Variance → overfits")
print("  GOAL:          Low Bias,   Low Variance  ← regularization helps!")
print()


# ======================================================================
# SECTION 3: L2 Regularization (Weight Decay)
# ======================================================================
print("=" * 70)
print("SECTION 3: L2 REGULARIZATION (WEIGHT DECAY)")
print("=" * 70)
print()
print("L2 adds the sum of SQUARED weights to the loss:")
print()
print("  L2_loss = original_loss + λ · Σ wᵢ²")
print()
print("Effect on gradient:")
print("  dL2_loss/dW = dOriginal_loss/dW + 2λW")
print("  W = W - lr * (grad + 2λW)")
print("  W = W * (1 - 2λ·lr) - lr * grad   ← weights DECAY every step!")
print()
print("  λ (lambda): regularization strength")
print("    λ=0.0:   no regularization")
print("    λ=0.001: light regularization (good default)")
print("    λ=0.1:   strong regularization (may underfit)")
print()
print("Why it works: penalizes large weights → forces the model to")
print("distribute information across many small weights → generalization")
print()

# Demonstrate L2 effect on weights
weights_no_reg  = np.array([5.2, -3.8, 0.1, 4.5, -2.9])
lr, lam = 0.01, 0.01

# Simulate 100 gradient steps with/without L2
W_noreg = weights_no_reg.copy()
W_l2    = weights_no_reg.copy()

for _ in range(100):
    grad = np.random.randn(5) * 0.1  # small gradient
    W_noreg -= lr * grad
    W_l2    -= lr * (grad + 2 * lam * W_l2)

print(f"After 100 gradient steps (same gradients, λ={lam}):")
print(f"  Without L2: {W_noreg.round(3)}, norm = {np.linalg.norm(W_noreg):.3f}")
print(f"  With L2:    {W_l2.round(3)},    norm = {np.linalg.norm(W_l2):.3f}")
print(f"  → L2 shrinks weights toward zero ✅")
print()


# ======================================================================
# SECTION 4: L1 Regularization (Lasso)
# ======================================================================
print("=" * 70)
print("SECTION 4: L1 REGULARIZATION (LASSO)")
print("=" * 70)
print()
print("L1 adds the sum of ABSOLUTE weights to the loss:")
print()
print("  L1_loss = original_loss + λ · Σ |wᵢ|")
print()
print("Gradient: dL1/dW = sign(W)  (either +1 or -1)")
print()
print("Key difference from L2:")
print("  L2: smoothly shrinks all weights toward zero")
print("  L1: drives many weights EXACTLY to zero (sparse solution!)")
print()
print("L1 creates SPARSE networks — useful for feature selection")
print("(irrelevant features get weight exactly 0)")
print()

W_l1 = weights_no_reg.copy()
for _ in range(100):
    grad = np.random.randn(5) * 0.1
    W_l1 -= lr * (grad + lam * np.sign(W_l1))

print(f"After 100 steps with L1 (λ={lam}):")
print(f"  Without reg:  {W_noreg.round(3)}")
print(f"  With L1:      {W_l1.round(3)}")
print(f"  With L2:      {W_l2.round(3)}")
print(f"  → L1 pushes some weights to near-zero (sparsity) ✅")
print()


# ======================================================================
# SECTION 5: Dropout
# ======================================================================
print("=" * 70)
print("SECTION 5: DROPOUT")
print("=" * 70)
print()
print("Dropout: randomly set a fraction of neurons to ZERO during each")
print("training step.")
print()
print("  During training: each neuron is kept with probability p")
print("    (typical p = 0.8 for input, 0.5 for hidden layers)")
print("  During inference: all neurons active, weights scaled by p")
print()
print("WHY it works:")
print("  1. Forces the network to not rely on any single neuron")
print("  2. Trains many different 'sub-networks' simultaneously")
print("  3. Acts like ensemble learning — averaging many models")
print()

def dropout(A, keep_prob, training=True):
    """Apply dropout to activations."""
    if not training:
        return A
    mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
    return A * mask

# Demo
neurons = np.array([[0.8, 1.2, -0.5, 0.3, 1.5, -0.9, 0.7, 0.4]])
kept = 0.5

print(f"Layer activations (before dropout): {neurons[0].round(2)}")
for trial in range(4):
    dropped = dropout(neurons, keep_prob=kept)
    zeros = (dropped == 0).sum()
    print(f"  Trial {trial+1} (keep_prob={kept}): {dropped[0].round(2)}  [{zeros} neurons dropped]")
print()
print("Note: non-zero values are SCALED UP (÷p) so the expected sum stays the same!")
print()


# ======================================================================
# SECTION 6: Batch Normalization
# ======================================================================
print("=" * 70)
print("SECTION 6: BATCH NORMALIZATION")
print("=" * 70)
print()
print("Problem: as data passes through layers, the distribution of")
print("activations can shift and grow — making training unstable.")
print("(called 'internal covariate shift')")
print()
print("Batch Norm: normalize the inputs of EACH LAYER to")
print("mean=0, std=1 (then optionally scale/shift with learned γ, β)")
print()
print("  x̂ = (x - μ_batch) / (σ_batch + ε)")
print("  output = γ · x̂ + β    (γ, β are learned)")
print()

def batch_norm(Z, gamma=1.0, beta=0.0, eps=1e-8):
    mu = Z.mean(axis=0)
    sigma = Z.std(axis=0)
    Z_norm = (Z - mu) / (sigma + eps)
    return gamma * Z_norm + beta, mu, sigma

# Demo
Z_raw = np.random.randn(4, 3) * 5 + 10  # skewed activations
Z_normed, mu, sig = batch_norm(Z_raw)

print(f"Before Batch Norm:")
print(f"  Mean per feature: {Z_raw.mean(axis=0).round(2)}")
print(f"  Std per feature:  {Z_raw.std(axis=0).round(2)}")
print(f"After Batch Norm:")
print(f"  Mean per feature: {Z_normed.mean(axis=0).round(2)}")
print(f"  Std per feature:  {Z_normed.std(axis=0).round(2)}")
print()
print("Benefits:")
print("  ✅ Faster training (can use larger learning rates)")
print("  ✅ Less sensitive to weight initialization")
print("  ✅ Slight regularization effect (adds noise via batch statistics)")
print()


# ======================================================================
# SECTION 7: Early Stopping
# ======================================================================
print("=" * 70)
print("SECTION 7: EARLY STOPPING")
print("=" * 70)
print()
print("Idea: monitor VALIDATION loss during training.")
print("Stop when validation loss stops improving.")
print()
print("  Train until validation loss plateaus or starts INCREASING")
print("  Save the model at the best validation checkpoint")
print()

# Simulate training curves
epochs = np.arange(1, 101)
train_loss = 1.5 * np.exp(-0.04 * epochs) + 0.05 + 0.01 * np.random.randn(100)
val_loss   = 1.5 * np.exp(-0.03 * epochs) + 0.15 + 0.0003 * epochs + 0.02 * np.random.randn(100)

best_epoch = np.argmin(val_loss) + 1
print(f"Simulated training over 100 epochs:")
print(f"  Best validation loss at epoch: {best_epoch}")
print(f"  → Early stopping would stop here (with patience buffer)")
print()
print("Patience: how many epochs to wait after no improvement")
print("  patience=5: stop if val loss doesn't improve for 5 epochs")
print("  patience=10: more lenient (good for noisy training)")
print()


# ======================================================================
# SECTION 8: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 8: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Overfitting curves ---
print("📊 Generating: Overfitting visualization...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("📈 Polynomial Fitting: Underfitting → Good Fit → Overfitting",
             fontsize=13, fontweight="bold")

for ax, (deg, label, color) in zip(axes, [(1, "Degree 1\n(Underfitting)", "blue"),
                                           (5, "Degree 5\n(Good Fit)", "green"),
                                           (15, "Degree 15\n(Overfitting)", "red")]):
    c, tr_mse, te_mse = results[deg]
    y_plot = np.polyval(c, X_plot)
    y_plot_clipped = np.clip(y_plot, -3, 3)
    ax.scatter(X_train, y_train, color="navy", s=40, zorder=5, label="Train")
    ax.scatter(X_test, y_test, color="orange", s=40, zorder=5, marker="^", label="Test")
    ax.plot(X_plot, y_plot_clipped, color=color, linewidth=2.5)
    ax.set_title(f"{label}\nTrain MSE={tr_mse:.3f}  Test MSE={te_mse:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(-2.5, 2.5); ax.grid(True, alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "overfitting_demo.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: overfitting_demo.png")


# --- PLOT 2: L1 vs L2 weight distributions ---
print("📊 Generating: L1 vs L2 weight distributions...")

np.random.seed(0)
n_weights = 500

W_init  = np.random.randn(n_weights) * 2
W_l2_dist = W_init * 0.3       # L2: Gaussian shrinkage
W_l1_dist = np.sign(W_init) * np.maximum(0, np.abs(W_init) - 0.8)  # L1: soft-thresholding → sparsity

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("⚖️ L1 vs L2 Regularization: Effect on Weight Distribution",
             fontsize=13, fontweight="bold")

for ax, (W, title, color) in zip(axes, [
    (W_init, "No Regularization\n(large weights allowed)", "steelblue"),
    (W_l2_dist, "L2 Regularization\n(Gaussian — many small weights)", "green"),
    (W_l1_dist, "L1 Regularization\n(Sparse — many ZERO weights)", "darkorange"),
]):
    ax.hist(W, bins=40, color=color, edgecolor="white", alpha=0.85)
    zeros = (np.abs(W) < 0.01).sum()
    ax.set_title(f"{title}\n{zeros} weights ≈ 0  ({zeros/n_weights:.0%})", fontsize=11)
    ax.set_xlabel("Weight value"); ax.set_ylabel("Count")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "l1_vs_l2_weights.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: l1_vs_l2_weights.png")


# --- PLOT 3: Early stopping curves ---
print("📊 Generating: Early stopping visualization...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, train_loss, "steelblue", linewidth=2.5, label="Training Loss")
ax.plot(epochs, val_loss,   "darkorange", linewidth=2.5, label="Validation Loss")
ax.axvline(best_epoch, color="green", linestyle="--", linewidth=2.5,
           label=f"Best epoch: {best_epoch} (early stopping here)")
ax.fill_betweenx([0, 2], best_epoch, 100, alpha=0.1, color="red",
                 label="Continue → overfitting zone")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("🛑 Early Stopping: Stop When Validation Loss Stops Improving",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.8)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "early_stopping.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: early_stopping.png")


# ============= CONCEPTUAL DIAGRAM =============
# 04_regularization_techniques_concept.png — 4-panel comparison of regularization techniques
print("📊 Generating: Regularization techniques concept diagram...")

from matplotlib.patches import FancyBboxPatch, Circle as MplCircle2

fig_reg, axes_reg = plt.subplots(1, 4, figsize=(20, 6))
fig_reg.patch.set_facecolor('#0f0f1a')
fig_reg.suptitle('Regularization Techniques — Side-by-Side Comparison',
                 fontsize=15, fontweight='bold', color='white', y=1.01)

for ax in axes_reg:
    ax.set_facecolor('#0f0f1a')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

# ------------------------------------------------------------------
# Panel 1: L2 Regularization — Gaussian bell curve (prefer small weights)
# ------------------------------------------------------------------
ax1 = axes_reg[0]
ax1.set_title('L2 Regularization', fontsize=12, fontweight='bold', color='#55ee88', pad=8)
w_vals = np.linspace(-4, 4, 300)
gaussian = np.exp(-0.5 * w_vals**2) / np.sqrt(2 * np.pi)
ax1.fill_between(w_vals, gaussian, color='#55ee88', alpha=0.35)
ax1.plot(w_vals, gaussian, color='#55ee88', linewidth=2.5)
ax1.axvline(0, color='white', linewidth=1.0, linestyle='--', alpha=0.5)
ax1.set_xlabel('Weight value', color='#aaaacc', fontsize=9)
ax1.set_ylabel('Prior density', color='#aaaacc', fontsize=9)
ax1.tick_params(colors='#888899')
ax1.text(0, 0.18, 'Prefer\nsmall weights', ha='center', va='center',
         fontsize=10, fontweight='bold', color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a2a1a', edgecolor='#55ee88'))
ax1.set_ylim(0, 0.48)
ax1.text(0, -0.07, 'Shrinks all weights\ntoward zero (Gaussian prior)',
         ha='center', va='top', fontsize=8, color='#aaaacc',
         transform=ax1.get_xaxis_transform())

# ------------------------------------------------------------------
# Panel 2: L1 Regularization — sparse bar chart
# ------------------------------------------------------------------
ax2 = axes_reg[1]
ax2.set_title('L1 Regularization', fontsize=12, fontweight='bold', color='#ff8844', pad=8)
np.random.seed(7)
bar_weights = np.random.randn(12) * 2
# Apply soft-thresholding to simulate L1 sparsity
threshold = 1.5
bar_weights_l1 = np.sign(bar_weights) * np.maximum(0, np.abs(bar_weights) - threshold)
bar_colors = ['#ff8844' if w != 0 else '#333344' for w in bar_weights_l1]
ax2.bar(range(12), bar_weights_l1, color=bar_colors, edgecolor='#555566', linewidth=0.8)
ax2.axhline(0, color='white', linewidth=1.0, alpha=0.6)
ax2.set_xlabel('Weight index', color='#aaaacc', fontsize=9)
ax2.set_ylabel('Weight value', color='#aaaacc', fontsize=9)
ax2.tick_params(colors='#888899')
n_zeros = (bar_weights_l1 == 0).sum()
ax2.text(5.5, max(bar_weights_l1) * 0.75 if max(bar_weights_l1) > 0 else 0.5,
         f'Creates sparsity\n({n_zeros}/12 weights = 0)',
         ha='center', va='center', fontsize=9, fontweight='bold', color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#2a1a0a', edgecolor='#ff8844'))

# ------------------------------------------------------------------
# Panel 3: Dropout — network with neurons X'd out
# ------------------------------------------------------------------
ax3 = axes_reg[2]
ax3.set_title('Dropout', fontsize=12, fontweight='bold', color='#44aaff', pad=8)
ax3.set_xlim(0, 10); ax3.set_ylim(0, 10); ax3.axis('off')

layer_xs   = [1.5, 4.5, 7.5, 9.5]
layer_sizes = [3, 4, 4, 2]
# Which neurons in hidden layers are dropped (50% dropout)
dropped = {1: {1, 3}, 2: {0, 2}}   # layer index → set of dropped node indices

node_positions = {}
for li, (lx, n_nodes) in enumerate(zip(layer_xs, layer_sizes)):
    ys = np.linspace(2.5, 7.5, n_nodes)
    for ni, ny in enumerate(ys):
        node_positions[(li, ni)] = (lx, ny)
        is_dropped = li in dropped and ni in dropped[li]
        color = '#223344' if is_dropped else '#44aaff'
        alpha = 0.3 if is_dropped else 0.9
        circ = MplCircle2((lx, ny), 0.45, color=color, alpha=alpha, zorder=4)
        ax3.add_patch(circ)
        if is_dropped:
            # Draw an X
            ax3.text(lx, ny, 'X', ha='center', va='center', fontsize=14,
                     fontweight='bold', color='#ff4444', zorder=5)

# Draw connections (only between non-dropped neurons)
for li in range(len(layer_xs) - 1):
    for ni in range(layer_sizes[li]):
        if li in dropped and ni in dropped[li]:
            continue
        for nj in range(layer_sizes[li + 1]):
            if (li + 1) in dropped and nj in dropped[li + 1]:
                continue
            x0, y0 = node_positions[(li, ni)]
            x1, y1 = node_positions[(li + 1, nj)]
            ax3.plot([x0 + 0.45, x1 - 0.45], [y0, y1],
                     color='#44aaff', alpha=0.35, linewidth=0.8, zorder=2)

ax3.text(5.5, 1.4, 'Randomly disable 50% of neurons\neach training batch',
         ha='center', va='center', fontsize=8.5, color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a1a2a', edgecolor='#44aaff'))

# ------------------------------------------------------------------
# Panel 4: Batch Normalization — before / after distribution
# ------------------------------------------------------------------
ax4 = axes_reg[3]
ax4.set_title('Batch Normalization', fontsize=12, fontweight='bold', color='#cc44ff', pad=8)

# Before: skewed/spread distribution
np.random.seed(3)
before = np.random.exponential(scale=3, size=600) + 5
after  = (before - before.mean()) / (before.std() + 1e-8)

ax4.hist(before / before.std() * 0.6 + 0.5, bins=35, color='#884422',
         alpha=0.7, label='Before BatchNorm', density=True)
ax4.hist(after, bins=35, color='#cc44ff',
         alpha=0.7, label='After BatchNorm', density=True)
ax4.axvline(0, color='white', linewidth=1.2, linestyle='--', alpha=0.7, label='mean=0')
ax4.set_xlabel('Activation value', color='#aaaacc', fontsize=9)
ax4.set_ylabel('Density', color='#aaaacc', fontsize=9)
ax4.tick_params(colors='#888899')
ax4.legend(fontsize=7.5, facecolor='#1a1a2a', edgecolor='#444466',
           labelcolor='white', loc='upper right')
ax4.text(0, ax4.get_ylim()[1] * 0.55 if ax4.get_ylim()[1] > 0 else 0.3,
         'Normalize each\nlayer\'s inputs\n(mean=0, std=1)',
         ha='center', va='center', fontsize=8.5, fontweight='bold', color='white',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a0a2a', edgecolor='#cc44ff'))

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '04_regularization_techniques_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig_reg.get_facecolor())
plt.close()
print("   Saved: 04_regularization_techniques_concept.png")
# ============= END CONCEPTUAL DIAGRAM =============

print()
print("=" * 70)
print("✅ MODULE 5 COMPLETE! — Math Foundations Section DONE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  📉 Overfitting = memorizing training data, failing on new data")
print("  ⚖️  L2: shrink weights → Gaussian distribution (default choice)")
print("  🕳️  L1: zero out weights → sparse network (feature selection)")
print("  🎲 Dropout: randomly disable neurons → ensemble-like effect")
print("  📊 Batch Norm: normalize layer inputs → faster, stabler training")
print("  🛑 Early Stopping: monitor val loss, stop before it gets worse")
print()
print("Next: Algorithm modules → Perceptron & MLP from scratch!")
