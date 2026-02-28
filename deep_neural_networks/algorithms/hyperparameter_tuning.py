"""
🤖 DEEP NEURAL NETWORKS — Algorithm 4: Hyperparameter Tuning
=============================================================

Learning Objectives:
  1. Know the difference: hyperparameters (you set) vs parameters (model learns)
  2. Understand 7 key hyperparameters: lr, layers, neurons, batch, epochs, dropout, activation
  3. See the effect of learning rate empirically (the most critical HP)
  4. Compare depth vs width trade-offs with visualizations
  5. Run a manual grid search and read a results table
  6. Understand learning rate schedules (step decay, exponential, cosine)
  7. Use validation strategy correctly (holdout vs cross-validation)

YouTube Resources:
  ⭐ StatQuest - Hyperparameters https://www.youtube.com/watch?v=IN2XmBhILt4
  ⭐ Andrew Ng - Tuning process https://www.youtube.com/watch?v=1waHlpKiNyY
  📚 TensorFlow - KerasTuner https://www.tensorflow.org/tutorials/keras/keras_tuner

Time Estimate: 55-70 minutes
Difficulty: Intermediate
Prerequisites: Algorithms 1-3 (Perceptron, MLP from scratch, Keras MLP)
Key Concepts: hyperparameter, grid search, learning rate schedule, validation strategy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "hyperparameter_tuning")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 4: HYPERPARAMETER TUNING")
print("=" * 70)
print()
print("PARAMETERS    (model learns these from data):")
print("  weights W and biases b in every layer")
print()
print("HYPERPARAMETERS (YOU set these before training):")
print("  learning rate, # layers, # neurons, batch size, epochs,")
print("  dropout rate, activation function, optimizer choice...")
print()
print("Good hyperparameters can make the difference between")
print("70% accuracy and 95% accuracy on the SAME dataset!")
print()


# ======================================================================
# SECTION 1: Dataset
# ======================================================================
print("=" * 70)
print("SECTION 1: DATASET")
print("=" * 70)
print()

X, y = make_classification(n_samples=800, n_features=10, n_informative=6,
                            n_redundant=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)

print(f"  Dataset: make_classification (10 features, 2 classes)")
print(f"  Train: {X_tr.shape[0]}, Val: {X_val.shape[0]}, Test: {X_te.shape[0]}")
print()


# ======================================================================
# SECTION 2: Simple MLP for experiments
# ======================================================================

def relu(z):      return np.maximum(0, z)
def relu_g(z):    return (z > 0).astype(float)
def sigmoid(z):   return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def bce(y, p):
    p = np.clip(p, 1e-15, 1-1e-15)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

class SimpleMLP:
    """Lightweight MLP for hyperparameter experiments."""

    def __init__(self, hidden=(32,), lr=0.001, epochs=100,
                 batch=32, dropout=0.0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        self.dropout = dropout
        self.train_loss, self.val_loss = [], []
        self.train_acc,  self.val_acc  = [], []
        self.W, self.b = [], []

    def _init(self, n_in, n_out):
        sizes = [n_in] + list(self.hidden) + [n_out]
        self.W = [np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2/sizes[i])
                  for i in range(len(sizes)-1)]
        self.b = [np.zeros((1, sizes[i+1])) for i in range(len(sizes)-1)]

    def _fwd(self, X, training=False):
        cache, A = [], X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Z = A @ W + b
            A_next = relu(Z) if i < len(self.W)-1 else sigmoid(Z)
            if training and i < len(self.W)-1 and self.dropout > 0:
                mask = (np.random.rand(*A_next.shape) > self.dropout)
                A_next = A_next * mask / (1 - self.dropout + 1e-8)
            cache.append((A, Z)); A = A_next
        return A, cache

    def _bwd(self, cache, y, y_pred):
        m, grads_W, grads_b = len(y), [], []
        dA = (y_pred.flatten() - y) / m
        dA = dA.reshape(-1, 1)
        for i in range(len(self.W)-1, -1, -1):
            A_prev, Z = cache[i]
            if i < len(self.W)-1:
                dZ = dA * relu_g(Z)
            else:
                dZ = dA  # sigmoid + BCE simplification
            grads_W.insert(0, A_prev.T @ dZ)
            grads_b.insert(0, dZ.sum(axis=0, keepdims=True))
            dA = dZ @ self.W[i].T
        return grads_W, grads_b

    def fit(self, X, y, X_val=None, y_val=None):
        self._init(X.shape[1], 1)
        # Adam state
        mW=[np.zeros_like(w) for w in self.W]; vW=[np.zeros_like(w) for w in self.W]
        mb=[np.zeros_like(b) for b in self.b]; vb=[np.zeros_like(b) for b in self.b]
        t = 0
        for ep in range(self.epochs):
            idx = np.random.permutation(len(y))
            for s in range(0, len(y), self.batch):
                Xb, yb = X[idx[s:s+self.batch]], y[idx[s:s+self.batch]]
                pred, cache = self._fwd(Xb, training=True)
                gW, gb = self._bwd(cache, yb, pred)
                t += 1
                b1, b2, eps = 0.9, 0.999, 1e-8
                for i in range(len(self.W)):
                    mW[i]=b1*mW[i]+(1-b1)*gW[i]; vW[i]=b2*vW[i]+(1-b2)*gW[i]**2
                    mb[i]=b1*mb[i]+(1-b1)*gb[i]; vb[i]=b2*vb[i]+(1-b2)*gb[i]**2
                    mhW=mW[i]/(1-b1**t); vhW=vW[i]/(1-b2**t)
                    mhb=mb[i]/(1-b1**t); vhb=vb[i]/(1-b2**t)
                    self.W[i] -= self.lr * mhW/(np.sqrt(vhW)+eps)
                    self.b[i] -= self.lr * mhb/(np.sqrt(vhb)+eps)
            # Record
            p_tr, _ = self._fwd(X)
            self.train_loss.append(bce(y, p_tr.flatten()))
            self.train_acc.append(((p_tr.flatten()>0.5)==y).mean())
            if X_val is not None:
                p_v, _ = self._fwd(X_val)
                self.val_loss.append(bce(y_val, p_v.flatten()))
                self.val_acc.append(((p_v.flatten()>0.5)==y_val).mean())

    def score(self, X, y):
        p, _ = self._fwd(X)
        return ((p.flatten() > 0.5) == y).mean()


# ======================================================================
# SECTION 3: Learning Rate — The Most Critical Hyperparameter
# ======================================================================
print("=" * 70)
print("SECTION 3: LEARNING RATE — THE MOST CRITICAL HYPERPARAMETER")
print("=" * 70)
print()
print("Learning rate (lr) controls HOW BIG each weight update step is.")
print()
print("  lr too LOW  (1e-5): converges, but painfully slowly")
print("  lr just right (1e-3): converges smoothly to a good minimum")
print("  lr too HIGH (0.5):   oscillates or diverges — loss EXPLODES")
print()
print("  Rule of thumb: start with lr=0.001 and adjust from there.")
print()

lr_experiments = [
    ("lr=1e-5  (too low)",   1e-5,  "steelblue"),
    ("lr=1e-3  (good)",      1e-3,  "green"),
    ("lr=1e-2  (slightly high)", 1e-2, "orange"),
    ("lr=0.5   (too high)",  0.5,   "red"),
]

print("  Training with 4 learning rates (60 epochs each)...")
lr_models = {}
for name, lr, color in lr_experiments:
    m = SimpleMLP(hidden=(32, 16), lr=lr, epochs=60, batch=32)
    m.fit(X_tr, y_tr, X_val, y_val)
    lr_models[name] = (m, color)
    fin_acc = m.score(X_val, y_val)
    print(f"  {name:25s} → val accuracy: {fin_acc:.3f}")
print()


# ======================================================================
# SECTION 4: Depth vs Width
# ======================================================================
print("=" * 70)
print("SECTION 4: DEPTH VS WIDTH")
print("=" * 70)
print()
print("DEPTH  = number of hidden layers (making network 'taller')")
print("WIDTH  = number of neurons per layer (making network 'wider')")
print()
print("Depth vs Width rules of thumb:")
print("  • More depth: better at learning hierarchical features")
print("    (e.g., image: pixels -> edges -> shapes -> objects)")
print("  • More width: more capacity at each level of abstraction")
print("  • Both increase total parameters — watch for overfitting!")
print()

arch_experiments = [
    ("Shallow-Narrow [16]",     (16,)),
    ("Shallow-Wide   [128]",    (128,)),
    ("Deep-Narrow    [16,16,16]", (16, 16, 16)),
    ("Deep-Wide      [64,64,64]", (64, 64, 64)),
    ("Deep-Tapered   [128,64,32]", (128, 64, 32)),
]

print(f"  {'Architecture':30} {'#Params':10} {'Val Acc':10}")
print("  " + "-" * 52)
arch_results = []
for name, hidden in arch_experiments:
    m = SimpleMLP(hidden=hidden, lr=0.001, epochs=80, batch=32)
    m.fit(X_tr, y_tr, X_val, y_val)
    n_params = sum(w.size + b.size for w, b in zip(m.W, m.b))
    val_acc = m.score(X_val, y_val)
    print(f"  {name:30s} {n_params:10,d} {val_acc:10.3f}")
    arch_results.append((name, n_params, val_acc, m))
print()


# ======================================================================
# SECTION 5: Manual Grid Search
# ======================================================================
print("=" * 70)
print("SECTION 5: MANUAL GRID SEARCH")
print("=" * 70)
print()
print("Grid search: try all combinations of a set of hyperparameters.")
print("  Choose the combination with the best VALIDATION performance.")
print()

grid = {
    "lr":      [0.01, 0.001],
    "hidden":  [(32,), (64, 32)],
    "dropout": [0.0, 0.3],
}

print(f"  {'lr':8} {'hidden':15} {'dropout':10} {'Val Acc':10}")
print("  " + "-" * 46)

best_acc, best_cfg = 0, None
for lr in grid["lr"]:
    for hidden in grid["hidden"]:
        for dropout in grid["dropout"]:
            m = SimpleMLP(hidden=hidden, lr=lr, epochs=80,
                           batch=32, dropout=dropout)
            m.fit(X_tr, y_tr, X_val, y_val)
            val_acc = m.score(X_val, y_val)
            marker = " <-- BEST" if val_acc > best_acc else ""
            print(f"  {lr:8} {str(hidden):15} {dropout:10} {val_acc:10.3f}{marker}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_cfg = {"lr": lr, "hidden": hidden, "dropout": dropout}
                best_model = m

print()
print(f"  Best config: {best_cfg}")
print(f"  Best val accuracy: {best_acc:.3f}")
print(f"  Test accuracy with best config: {best_model.score(X_te, y_te):.3f}")
print()
print("  IMPORTANT: Only check test accuracy ONCE at the very end.")
print("  Using test accuracy to guide tuning = data leakage!")
print()


# ======================================================================
# SECTION 6: Learning Rate Schedules
# ======================================================================
print("=" * 70)
print("SECTION 6: LEARNING RATE SCHEDULES")
print("=" * 70)
print()
print("Instead of a fixed lr, decay it during training:")
print()
print("  Step decay:     halve lr every N epochs")
print("  Exponential:    lr = lr0 * decay^epoch")
print("  Cosine annealing: lr follows a cosine curve (smooth, popular)")
print("  Warm-up + decay: start small, grow, then decay (used in Transformers)")
print()

epochs_range = np.arange(0, 100)
lr0 = 0.01

step_decay    = lr0 * (0.5 ** (epochs_range // 20))
exp_decay     = lr0 * (0.95 ** epochs_range)
cosine_anneal = lr0 * 0.5 * (1 + np.cos(np.pi * epochs_range / 100))
warmup_decay  = np.where(epochs_range < 10,
                          lr0 * epochs_range / 10,
                          lr0 * 0.5 * (1 + np.cos(np.pi * (epochs_range - 10) / 90)))

print("  Schedules computed. See visualization for comparison.")
print()


# ======================================================================
# SECTION 7: Validation Strategy
# ======================================================================
print("=" * 70)
print("SECTION 7: VALIDATION STRATEGY")
print("=" * 70)
print()
print("HOLDOUT VALIDATION (what we've been doing):")
print("  train / val / test split — simple and fast")
print("  Risk: val set too small → noisy estimate")
print()
print("K-FOLD CROSS-VALIDATION:")
print("  Split data into K folds, train K times (each fold as val once)")
print("  Average performance = more reliable estimate")
print("  Downside: K× more compute")
print()
print("  For neural networks: holdout is usually fine if val set > ~500 samples")
print("  K-fold: use when dataset is small (<1000 samples)")
print()


# ======================================================================
# SECTION 8: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 8: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Learning rate effect ---
print("Generating: Learning rate effect...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Learning Rate: Too Low / Good / Too High",
             fontsize=14, fontweight="bold")

for name, (m, color) in lr_models.items():
    axes[0].plot(m.train_loss, color=color, linewidth=2, label=name)
    axes[1].plot(m.val_acc,   color=color, linewidth=2, label=name)

for ax, ylabel, title in zip(axes,
    ["Training Loss", "Validation Accuracy"],
    ["Loss Over Epochs", "Accuracy Over Epochs"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
axes[0].set_ylim(0, 1.5)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "learning_rate_effect.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: learning_rate_effect.png")


# --- PLOT 2: LR schedules ---
print("Generating: Learning rate schedules...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs_range, step_decay,     "steelblue",  linewidth=2.5, label="Step Decay (halve every 20 epochs)")
ax.plot(epochs_range, exp_decay,      "darkorange",  linewidth=2.5, label="Exponential Decay (0.95^epoch)")
ax.plot(epochs_range, cosine_anneal,  "green",       linewidth=2.5, label="Cosine Annealing")
ax.plot(epochs_range, warmup_decay,   "red",         linewidth=2.5, label="Warm-Up + Cosine Decay")
ax.set_title("Learning Rate Schedules: Decay Strategies",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "lr_schedules.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: lr_schedules.png")


# --- PLOT 3: Depth vs Width results ---
print("Generating: Architecture comparison...")

names   = [r[0] for r in arch_results]
params  = [r[1] for r in arch_results]
val_accs= [r[2] for r in arch_results]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Architecture Trade-offs: Depth vs Width",
             fontsize=13, fontweight="bold")

colors = ["steelblue", "darkorange", "green", "red", "purple"]
bar = axes[0].bar(range(len(names)), val_accs, color=colors, alpha=0.8, edgecolor="black")
axes[0].set_xticks(range(len(names)))
axes[0].set_xticklabels([n.split()[0] for n in names], rotation=20, ha="right")
axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Accuracy by Architecture", fontsize=12, fontweight="bold")
axes[0].set_ylim(0.5, 1.0); axes[0].grid(True, alpha=0.3, axis="y")
for i, (v, p) in enumerate(zip(val_accs, params)):
    axes[0].text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

axes[1].scatter(params, val_accs, c=colors, s=150, zorder=5, edgecolors="black")
for i, (p, a, n) in enumerate(zip(params, val_accs, names)):
    axes[1].annotate(n.split("[")[0].strip(), (p, a),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
axes[1].set_xlabel("Number of Parameters")
axes[1].set_ylabel("Validation Accuracy")
axes[1].set_title("Parameters vs Accuracy\n(More params != always better)",
                  fontsize=12, fontweight="bold")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "architecture_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: architecture_comparison.png")


print()
print("=" * 70)
print("ALGORITHM 4: HYPERPARAMETER TUNING COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  lr=0.001 is the safe starting point for Adam")
print("  Grid search: try combos, pick best on VAL (never test!)")
print("  Deep-Tapered [wide -> narrow] often works best")
print("  LR schedules: cosine annealing is popular and effective")
print("  Holdout validation is fine; use K-fold for small datasets")
print()
print("Next: Projects! -> MNIST Digit Classifier")
