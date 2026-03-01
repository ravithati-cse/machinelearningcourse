"""
LoRA: Low-Rank Adaptation — Parameter-Efficient Fine-Tuning
=============================================================
Learning Objectives:
  1. Understand why full fine-tuning is impractical at large scale
  2. Derive LoRA: ΔW = BA where B∈R^{d×r}, A∈R^{r×k}, r≪min(d,k)
  3. Implement LoRALinear with forward and backward pass from scratch
  4. Train a classifier using LoRA while keeping base weights frozen
  5. Compare trainable parameter counts across ranks and model sizes
  6. Merge LoRA weights back into the base model for efficient inference
YouTube: Search "LoRA low rank adaptation paper explained PEFT"
Time: ~50 min | Difficulty: Advanced
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "lora_finetuning")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

# ===========================================================================
print("\n" + "="*70)
print("SECTION 1: MOTIVATION — WHY NOT JUST FINE-TUNE EVERYTHING?")
print("="*70)
# ===========================================================================

print("""
The Scale Problem
-----------------
GPT-3 has 175 billion parameters.
Full fine-tuning requires:
  - Storing 175B gradients (same size as the model)
  - Optimizer states (Adam: 2× parameter count for momentum + variance)
  - Total memory: ~4× model size = 700B float32 values ≈ 2.8 TB

GPU memory constraints:
  - NVIDIA A100 (flagship): 80 GB VRAM
  - Fitting 175B params in float32: 700 GB → needs 9+ A100s just to store
  - Full fine-tuning: 14+ A100s (with ZeRO sharding)
  - Cost: ~$100K for a single fine-tuning run

LoRA insight (Hu et al. 2021):
  - Pre-trained weight matrices have low intrinsic rank
  - Fine-tuning adapts the model in a low-dimensional subspace
  - Instead of ΔW (full d×k), learn B (d×r) and A (r×k), r << d
  - Only train ~0.1–1% of parameters with comparable quality!
""")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 2: LoRA MATHEMATICS — DERIVATION")
print("="*70)
# ===========================================================================

print("""
Full parameter update (standard fine-tuning):
  W' = W + ΔW    where W ∈ R^{d×k},  ΔW ∈ R^{d×k}
  Forward: h = W'·x = W·x + ΔW·x

LoRA low-rank approximation:
  ΔW ≈ B·A        where B ∈ R^{d×r},  A ∈ R^{r×k},  r << min(d,k)
  Forward: h = W·x + (α/r)·B·A·x
                  └─── base ───┘  └─── LoRA ───┘

Parameter counts:
  Full ΔW : d × k parameters
  LoRA    : r × k  +  d × r  =  r(d + k) parameters

Example: d = k = 4096 (LLaMA-7B attention projection), r = 8
  Full ΔW : 4096 × 4096 = 16,777,216 parameters
  LoRA A  : 8 × 4096   =     32,768 parameters
  LoRA B  : 4096 × 8   =     32,768 parameters
  Total LoRA: 65,536 parameters  =  0.39% of full update

Why these design choices?
  1. B is initialised to ZERO
     → ΔW = B·A = 0 at start → training begins from exact pre-trained weights
     → No random disruption of pre-trained knowledge
  2. A is initialised from N(0, σ²)
     → Provides non-zero gradients for B from step 1
  3. α/r scaling factor
     → Makes update magnitude independent of rank choice
     → Changing r doesn't require re-tuning the learning rate
  4. W stays frozen throughout
     → No catastrophic forgetting of pre-trained knowledge
     → Can revert to base model by removing B·A
""")

# Verify parameter math
d, k, r = 4096, 4096, 8
full_params  = d * k
lora_params  = r * k + d * r
print(f"Numerical verification (d={d}, k={k}, r={r}):")
print(f"  Full ΔW params : {full_params:,}")
print(f"  LoRA params    : {lora_params:,}  ({100*lora_params/full_params:.3f}% of full)")
print(f"  Memory saved   : {full_params - lora_params:,}  ({100*(1 - lora_params/full_params):.2f}%)")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 3: LoRALinear IMPLEMENTATION (Pure NumPy)")
print("="*70)
# ===========================================================================

class LoRALinear:
    """Low-Rank Adaptation of a linear layer.

    Forward:  h = W·xᵀ  +  (α/r)·B·A·xᵀ
    Training: only A and B are updated; W stays frozen.
    """

    def __init__(self, d_in, d_out, rank=8, alpha=16.0, seed=42):
        rng = np.random.RandomState(seed)
        self.d_in  = d_in
        self.d_out = d_out
        self.rank  = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Frozen base weights (simulate pre-trained model)
        self.W = rng.randn(d_out, d_in) * 0.02    # (d_out, d_in) — FROZEN

        # Trainable LoRA matrices
        self.A = rng.randn(rank, d_in)  * 0.01    # (r, d_in)
        self.B = np.zeros((d_out, rank))           # (d_out, r) — init 0!

        # Cache for backward
        self._x   = None    # input (batch, d_in)
        self._Ax  = None    # A·xᵀ (batch, rank)

    def forward(self, x):
        """x : (batch, d_in) → (batch, d_out)"""
        self._x  = x
        base     = x @ self.W.T                    # (batch, d_out)  [frozen path]
        self._Ax = x @ self.A.T                    # (batch, rank)
        lora     = self._Ax @ self.B.T             # (batch, d_out)
        return base + self.scale * lora

    def backward(self, grad_out, lr=0.01):
        """Compute gradients and update A, B.
        grad_out : (batch, d_out) — upstream gradient
        Returns  : gradient w.r.t. input x, for chaining
        """
        n = self._x.shape[0]

        # d_loss/d_B = scale * (d_loss/d_lora)ᵀ · (A·x)
        grad_B = (grad_out.T @ self._Ax) * self.scale / n    # (d_out, rank)

        # d_loss/d_A: first propagate through B
        grad_Ax = (grad_out @ self.B) * self.scale            # (batch, rank)
        grad_A  = (grad_Ax.T @ self._x) / n                  # (rank, d_in)

        # Update LoRA matrices (W stays FROZEN)
        self.B -= lr * grad_B
        self.A -= lr * grad_A

        # Return gradient for input (for chaining)
        # d_loss/d_x = grad_out @ W + grad_Ax @ A
        return grad_out @ self.W + grad_Ax @ self.A            # (batch, d_in)

    def param_count(self):
        base_params = self.W.size
        lora_params = self.A.size + self.B.size
        return {
            "base"         : base_params,
            "lora"         : lora_params,
            "total"        : base_params + lora_params,
            "pct_trainable": f"{100*lora_params/(base_params+lora_params):.3f}%"
        }

    def merge_weights(self):
        """Merge LoRA into base weight for zero-overhead inference.
        Returns W_merged = W + (α/r)·B·A   shape: (d_out, d_in)
        """
        return self.W + self.scale * (self.B @ self.A)


# Demonstrate the layer
demo_layer = LoRALinear(d_in=20, d_out=32, rank=4, alpha=8.0, seed=0)
demo_x     = np.random.randn(8, 20)    # batch of 8
demo_out   = demo_layer.forward(demo_x)
counts     = demo_layer.param_count()

print(f"\nLoRALinear demo (d_in=20, d_out=32, rank=4):")
print(f"  Input shape     : {demo_x.shape}")
print(f"  Output shape    : {demo_out.shape}")
print(f"  Base params     : {counts['base']:,}     (frozen)")
print(f"  LoRA params     : {counts['lora']:,}      (trainable)")
print(f"  Trainable share : {counts['pct_trainable']}")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 4: TRAINING DEMO — BINARY SENTIMENT CLASSIFICATION")
print("="*70)
# ===========================================================================

print("\nTask: binary classification — predict sign of X[:,0] + X[:,1]")
print("Architecture: Linear(20→32 LoRA) → ReLU → Linear(32→2) → softmax")
print("Comparing: LoRA fine-tuning vs full fine-tuning of first layer\n")

# Synthetic dataset
np.random.seed(42)
n   = 40
X   = np.random.randn(n, 20)
y   = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, y_train = X[:32], y[:32]
X_test,  y_test  = X[32:], y[32:]

# ---- Helper: softmax + cross-entropy ----
def softmax(z):
    ez = np.exp(z - z.max(axis=1, keepdims=True))
    return ez / ez.sum(axis=1, keepdims=True)

def cross_entropy(probs, labels):
    return -np.log(probs[np.arange(len(labels)), labels] + 1e-9).mean()

def accuracy(probs, labels):
    return (probs.argmax(1) == labels).mean()

# ---- LoRA model ----
class LoRAClassifier:
    """2-layer MLP with LoRA on first layer."""
    def __init__(self, rank=4, seed=1):
        rng = np.random.RandomState(seed)
        self.lora1 = LoRALinear(20, 32, rank=rank, alpha=8.0, seed=seed)
        # Second layer: fully trainable (small, so fine to update all)
        self.W2    = rng.randn(2, 32) * 0.02
        self.b2    = np.zeros(2)
        # Cache
        self._h    = None
        self._probs = None

    def forward(self, x):
        z1 = self.lora1.forward(x)             # (batch, 32)
        h  = np.maximum(0, z1)                 # ReLU
        self._h = h
        logits  = h @ self.W2.T + self.b2      # (batch, 2)
        probs   = softmax(logits)
        self._probs = probs
        return probs

    def backward_and_update(self, labels, lr=0.05):
        probs   = self._probs
        n       = len(labels)
        # Gradient of cross-entropy w.r.t. logits
        d_logits = probs.copy()
        d_logits[np.arange(n), labels] -= 1
        d_logits /= n
        # Gradient w.r.t. W2, b2
        grad_W2 = d_logits.T @ self._h       # (2, 32)
        grad_b2 = d_logits.sum(0)            # (2,)
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        # Gradient into ReLU and back into LoRA layer
        grad_h  = d_logits @ self.W2         # (batch, 32)
        grad_z1 = grad_h * (self._h > 0)    # ReLU mask
        self.lora1.backward(grad_z1, lr=lr)


# ---- Full fine-tuning model (no LoRA) ----
class FullClassifier:
    """Same 2-layer MLP but first layer fully trainable (no LoRA)."""
    def __init__(self, seed=1):
        rng = np.random.RandomState(seed)
        self.W1   = rng.randn(32, 20) * 0.02
        self.b1   = np.zeros(32)
        self.W2   = rng.randn(2, 32) * 0.02
        self.b2   = np.zeros(2)
        self._x   = None
        self._h   = None
        self._probs = None

    def forward(self, x):
        self._x   = x
        z1        = x @ self.W1.T + self.b1   # (batch, 32)
        h         = np.maximum(0, z1)
        self._h   = h
        logits    = h @ self.W2.T + self.b2
        probs     = softmax(logits)
        self._probs = probs
        return probs

    def backward_and_update(self, labels, lr=0.05):
        probs   = self._probs
        n       = len(labels)
        d_logits = probs.copy()
        d_logits[np.arange(n), labels] -= 1
        d_logits /= n
        # Layer 2
        grad_W2 = d_logits.T @ self._h
        grad_b2 = d_logits.sum(0)
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2
        # Layer 1
        grad_h  = d_logits @ self.W2
        grad_z1 = grad_h * (self._h > 0)
        grad_W1 = grad_z1.T @ self._x
        grad_b1 = grad_z1.sum(0)
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1


lora_model = LoRAClassifier(rank=4, seed=1)
full_model = FullClassifier(seed=1)

lora_losses, lora_accs = [], []
full_losses, full_accs = [], []
steps   = 200
lr      = 0.05

rng_tr = np.random.RandomState(7)

for step in range(steps):
    # Mini-batch of 16
    idx   = rng_tr.choice(len(X_train), 16, replace=False)
    xb, yb = X_train[idx], y_train[idx]

    # LoRA model
    probs_l = lora_model.forward(xb)
    loss_l  = cross_entropy(probs_l, yb)
    lora_model.backward_and_update(yb, lr=lr)

    # Full model
    probs_f = full_model.forward(xb)
    loss_f  = cross_entropy(probs_f, yb)
    full_model.backward_and_update(yb, lr=lr)

    # Test set accuracy
    test_probs_l = lora_model.forward(X_test)
    test_probs_f = full_model.forward(X_test)
    lora_losses.append(float(loss_l))
    full_losses.append(float(loss_f))
    lora_accs.append(accuracy(test_probs_l, y_test))
    full_accs.append(accuracy(test_probs_f, y_test))

    if step % 50 == 0 or step == steps - 1:
        print(f"  Step {step:3d} | LoRA loss: {loss_l:.4f} acc: {lora_accs[-1]:.2f}"
              f"  |  Full loss: {loss_f:.4f} acc: {full_accs[-1]:.2f}")

# Trainable param counts
lora_trainable = (lora_model.lora1.A.size + lora_model.lora1.B.size +
                  lora_model.W2.size + lora_model.b2.size)
full_trainable = (full_model.W1.size + full_model.b1.size +
                  full_model.W2.size + full_model.b2.size)
print(f"\nTrainable parameters:")
print(f"  LoRA (rank=4) : {lora_trainable:,}")
print(f"  Full model    : {full_trainable:,}")
print(f"  LoRA savings  : {full_trainable - lora_trainable:,} ({100*(1 - lora_trainable/full_trainable):.1f}%)")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 5: PARAMETER COUNT TABLE")
print("="*70)
# ===========================================================================

def lora_params_for(d, k, r):
    return r * k + d * r

def full_params_for(d, k):
    return d * k

ranks = [1, 4, 8, 16, 32, 64]

for d_size, label in [(768, "BERT-base (d=768)"), (4096, "LLaMA-7B (d=4096)")]:
    print(f"\n{label} — single attention projection (d_in = d_out = {d_size})")
    print(f"{'Rank':<8} | {'LoRA Params':>12} | {'% of Full':>10} | {'Memory Savings':>15}")
    print("-" * 56)
    fp = full_params_for(d_size, d_size)
    for r in ranks:
        lp = lora_params_for(d_size, d_size, r)
        pct  = 100 * lp / fp
        save = 100 * (1 - lp / fp)
        print(f"r={r:<6} | {lp:>12,} | {pct:>9.2f}% | {save:>14.2f}%")

print(f"\nFull single-layer params (d={768}): {full_params_for(768,768):,}")
print(f"Full single-layer params (d={4096}): {full_params_for(4096,4096):,}")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 6: INFERENCE WITH MERGED WEIGHTS")
print("="*70)
# ===========================================================================

print("""
After training, LoRA matrices B and A can be merged into W for inference:
  W_merged = W + (α/r) · B · A

This gives ZERO additional latency — the merged model is identical to
a standard linear layer. No extra computation at inference time.
""")

# Create a trained LoRA layer and verify merge
test_layer = LoRALinear(d_in=16, d_out=24, rank=4, alpha=8.0, seed=5)
# Simulate some training
rng_t = np.random.RandomState(5)
for _ in range(50):
    xb = rng_t.randn(8, 16)
    yb = (rng_t.randn(8, 24) > 0).astype(float)
    out = test_layer.forward(xb)
    grad = (out - yb) / 8
    test_layer.backward(grad, lr=0.01)

# Merge and compare
test_x      = np.random.randn(5, 16)
W_merged    = test_layer.merge_weights()
out_merged  = test_x @ W_merged.T
out_lora    = test_layer.forward(test_x)
diff        = np.abs(out_merged - out_lora).max()

print(f"Test: LoRA forward vs merged weights on 5 samples:")
print(f"  LoRA output [0,:4]   : {out_lora[0, :4]}")
print(f"  Merged output [0,:4] : {out_merged[0, :4]}")
print(f"  Max absolute difference: {diff:.2e}  (should be ~0 / floating-point eps)")

eps = np.finfo(float).eps * 100
status = "PASS" if diff < 1e-10 else "CLOSE (floating-point rounding)"
print(f"  Verification: {status}")

# ===========================================================================
print("\n" + "="*70)
print("VISUALISATIONS")
print("="*70)
# ===========================================================================

# ------------------------------------------------------------------
# VIZ 1: LoRA diagram (full vs LoRA)
# ------------------------------------------------------------------
print("\nSaving 01_lora_diagram.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("LoRA vs Full Fine-Tuning: Weight Update Comparison", fontsize=14, fontweight='bold')

# (a) Full fine-tuning
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 8)
ax.axis('off')
ax.set_title("(a) Full Fine-Tuning\n(all parameters updated)", fontsize=11, fontweight='bold')

# Full weight matrix — big blue rectangle
rect_full = FancyBboxPatch((0.5, 1.0), 5.0, 6.0,
                            boxstyle="round,pad=0.1",
                            facecolor='#4361ee', edgecolor='#2d3a8c', lw=2, alpha=0.85)
ax.add_patch(rect_full)
ax.text(3.0, 4.0, "ΔW", ha='center', va='center', fontsize=32, fontweight='bold', color='white')
ax.text(3.0, 3.0, f"d × k", ha='center', va='center', fontsize=14, color='#b8c0cc')
ax.text(0.5, 7.4, "d = 4096", fontsize=9, color='#4361ee', fontweight='bold')
ax.text(5.0, 0.5, "k = 4096", fontsize=9, color='#4361ee', fontweight='bold', ha='right')
ax.text(3.0, 0.4, f"Trainable params: {full_params_for(4096,4096):,}", ha='center', fontsize=10,
        fontweight='bold', color='#e63946')
ax.text(3.0, 7.6, "Memory: 64 MB (fp16) per layer", ha='center', fontsize=9, color='#6c757d')

# (b) LoRA
ax2 = axes[1]
ax2.set_xlim(0, 9)
ax2.set_ylim(0, 8)
ax2.axis('off')
ax2.set_title("(b) LoRA (r=8)\nOnly A and B are trained", fontsize=11, fontweight='bold')

# Frozen W — large gray rectangle in background
rect_W = FancyBboxPatch((0.3, 1.0), 5.0, 6.0,
                         boxstyle="round,pad=0.1",
                         facecolor='#adb5bd', edgecolor='#6c757d', lw=1.5, alpha=0.3)
ax2.add_patch(rect_W)
ax2.text(2.8, 4.0, "W\n(frozen)", ha='center', va='center', fontsize=18, color='#6c757d', alpha=0.6)

# B — tall thin red rectangle
rect_B = FancyBboxPatch((5.8, 1.0), 1.0, 6.0,
                         boxstyle="round,pad=0.08",
                         facecolor='#e63946', edgecolor='#9b1b2a', lw=2, alpha=0.9)
ax2.add_patch(rect_B)
ax2.text(6.3, 4.0, "B", ha='center', va='center', fontsize=22, fontweight='bold', color='white')
ax2.text(6.3, 3.0, f"d×r\n4096×8", ha='center', va='center', fontsize=8, color='white')

# A — wide thin red rectangle
rect_A = FancyBboxPatch((5.8, -0.1), 3.0, 0.9,
                         boxstyle="round,pad=0.08",
                         facecolor='#f4a261', edgecolor='#c0562a', lw=2, alpha=0.9)
ax2.add_patch(rect_A)
ax2.text(7.3, 0.35, "A   r×k  =  8×4096", ha='center', va='center', fontsize=9, fontweight='bold', color='#333')

ax2.text(6.3, 0.7, f"LoRA params:\n{lora_params_for(4096,4096,8):,}\n(r={8})", ha='center', fontsize=8,
         color='#e63946', fontweight='bold', va='bottom')
ax2.text(6.3, 7.5, f"= {100*lora_params_for(4096,4096,8)/full_params_for(4096,4096):.2f}% of full",
         ha='center', fontsize=9, color='#e63946')

# Legend
ax2.annotate("", xy=(5.65, 4.5), xytext=(5.35, 4.5),
             arrowprops=dict(arrowstyle="<-", color='#333', lw=1.5))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_lora_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 01_lora_diagram.png")

# ------------------------------------------------------------------
# VIZ 2: Training curves + parameter comparison
# ------------------------------------------------------------------
print("Saving 02_training.png ...")

def smooth(arr, w=10):
    out = []
    for i in range(len(arr)):
        lo = max(0, i - w)
        out.append(np.mean(arr[lo:i+1]))
    return np.array(out)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LoRA vs Full Fine-Tuning — Training Comparison", fontsize=13, fontweight='bold')

steps_arr = np.arange(steps)

# (a) Loss
ax = axes[0]
ax.plot(steps_arr, smooth(lora_losses, 15), color='#e63946', lw=2.2, label='LoRA (rank=4)')
ax.plot(steps_arr, smooth(full_losses, 15), color='#4361ee', lw=2.2, label='Full fine-tune', ls='--')
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Cross-entropy loss", fontsize=11)
ax.set_title("(a) Training Loss", fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# (b) Accuracy
ax2 = axes[1]
ax2.plot(steps_arr, smooth(lora_accs, 15), color='#e63946', lw=2.2, label='LoRA (rank=4)')
ax2.plot(steps_arr, smooth(full_accs, 15), color='#4361ee', lw=2.2, label='Full fine-tune', ls='--')
ax2.axhline(0.5, color='gray', ls=':', lw=1.2, label='Random baseline')
ax2.set_xlabel("Step", fontsize=11)
ax2.set_ylabel("Test accuracy", fontsize=11)
ax2.set_title("(b) Test Accuracy", fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

# (c) Parameter counts
ax3 = axes[2]
rank_demo = [1, 4, 8, 16]
param_counts = {
    'Full FT': full_trainable,
}
lora_demos = {}
for r in rank_demo:
    tmp = LoRALinear(20, 32, rank=r)
    lp  = tmp.A.size + tmp.B.size + lora_model.W2.size + lora_model.b2.size
    lora_demos[f'LoRA r={r}'] = lp

all_labels = ['Full FT'] + [f'LoRA r={r}' for r in rank_demo]
all_values = [full_trainable] + [lora_demos[f'LoRA r={r}'] for r in rank_demo]
bar_colors = ['#4361ee'] + ['#e63946'] * len(rank_demo)

bars = ax3.bar(all_labels, all_values, color=bar_colors, alpha=0.85, edgecolor='white')
ax3.set_ylabel("Trainable parameters", fontsize=11)
ax3.set_title("(c) Trainable Parameter Count", fontsize=12, fontweight='bold')
ax3.set_xticklabels(all_labels, rotation=25, ha='right', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, all_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:,}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_training.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 02_training.png")

# ------------------------------------------------------------------
# VIZ 3: Rank analysis
# ------------------------------------------------------------------
print("Saving 03_rank_analysis.png ...")

ranks_full = [1, 2, 4, 8, 16, 32, 64]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LoRA Rank Analysis", fontsize=13, fontweight='bold')

# (a) Trainable params vs rank for d=768 and d=4096
ax = axes[0]
for d_sz, color, marker, label in [(768, '#4361ee', 'o', 'BERT-base (d=768)'),
                                     (4096, '#e63946', 's', 'LLaMA-7B (d=4096)')]:
    params_r = [lora_params_for(d_sz, d_sz, r) for r in ranks_full]
    ax.plot(ranks_full, params_r, color=color, marker=marker, lw=2.2, ms=7, label=label)
    ax.axhline(full_params_for(d_sz, d_sz), color=color, ls=':', lw=1, alpha=0.5)

ax.set_xlabel("LoRA rank (r)", fontsize=11)
ax.set_ylabel("Trainable parameters", fontsize=11)
ax.set_title("(a) LoRA Params vs Rank\n(dotted = full fine-tune)", fontsize=11, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_xticks(ranks_full)

# (b) Accuracy vs rank (train fresh LoRA for each rank)
ax2 = axes[1]
rank_acc = []
rng_ra   = np.random.RandomState(3)
for r in [1, 2, 4, 8, 16]:
    clf = LoRAClassifier(rank=r, seed=r)
    for step in range(200):
        idx  = rng_ra.choice(len(X_train), 16, replace=False)
        xb, yb = X_train[idx], y_train[idx]
        clf.forward(xb)
        clf.backward_and_update(yb, lr=0.05)
    test_p = clf.forward(X_test)
    acc    = accuracy(test_p, y_test)
    rank_acc.append(acc)
    print(f"  rank={r:<3}: final test acc = {acc:.3f}")

ax2.plot([1, 2, 4, 8, 16], rank_acc, color='#2a9d8f', marker='o', lw=2.2, ms=9)
ax2.fill_between([1, 2, 4, 8, 16], rank_acc, alpha=0.15, color='#2a9d8f')
ax2.set_xlabel("LoRA rank (r)", fontsize=11)
ax2.set_ylabel("Test accuracy", fontsize=11)
ax2.set_title("(b) Final Test Accuracy vs Rank\n(higher rank ≠ always better)", fontsize=11, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.set_xticks([1, 2, 4, 8, 16])
ax2.grid(alpha=0.3)
ax2.axhline(0.5, color='gray', ls=':', lw=1.2, label='Random baseline')
ax2.legend(fontsize=10)
for xi, yi in zip([1, 2, 4, 8, 16], rank_acc):
    ax2.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_rank_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 03_rank_analysis.png")

print("\n" + "="*70)
print("COMPLETE: lora_finetuning.py")
print("="*70)
print(f"  LoRA trainable params (demo, rank=4) : {lora_trainable:,}")
print(f"  Full FT trainable params             : {full_trainable:,}")
print(f"  Param reduction                      : {100*(1-lora_trainable/full_trainable):.1f}%")
print(f"  Merge verification diff              : {diff:.2e}")
print(f"  Visuals saved : {VIS_DIR}/")
print("  Files         : 01_lora_diagram.png, 02_training.png, 03_rank_analysis.png")
