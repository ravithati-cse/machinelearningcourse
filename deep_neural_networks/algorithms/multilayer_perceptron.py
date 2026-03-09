"""
🤖 DEEP NEURAL NETWORKS — Algorithm 2: Multi-Layer Perceptron from Scratch
===========================================================================

Learning Objectives:
  1. Build a full MLPClassifier class in numpy (forward + backward pass)
  2. Implement mini-batch gradient descent with Adam optimizer
  3. Solve the XOR problem that defeated the single Perceptron
  4. Apply the MLP to a real 2D classification problem
  5. Visualize how more layers create more complex decision boundaries
  6. Compare from-scratch MLP with sklearn's MLPClassifier
  7. Understand how to tune depth and width

YouTube Resources:
  ⭐ 3Blue1Brown - Neural networks series https://www.youtube.com/watch?v=aircAruvnKk
  ⭐ StatQuest - Neural Networks Pt.3 https://www.youtube.com/watch?v=83LYR-1IcjA
  📚 Andrej Karpathy - micrograd https://www.youtube.com/watch?v=VMj-3S1tku0

Time Estimate: 60-75 minutes
Difficulty: Intermediate
Prerequisites: Modules 1-5 (all math foundations), Algorithm 1 (Perceptron)
Key Concepts: MLP, hidden layers, mini-batch, backprop, Adam, architecture
"""

import numpy as np
import matplotlib.pyplot as plt
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "multilayer_perceptron")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 2: MULTI-LAYER PERCEPTRON FROM SCRATCH")
print("=" * 70)
print()
print("The Perceptron failed at XOR. The MLP fixes this by adding")
print("one or more HIDDEN LAYERS between input and output.")
print()
print("Architecture: Input -> [Hidden layers] -> Output")
print("Each hidden layer transforms the data into a new representation")
print("that makes the final classification easier.")
print()


# ======================================================================
# SECTION 1: Helper Functions
# ======================================================================
print("=" * 70)
print("SECTION 1: HELPER FUNCTIONS")
print("=" * 70)
print()

def relu(z):      return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """Multi-class cross-entropy. y_true = integer labels."""
    m = y_true.shape[0]
    p = y_pred[range(m), y_true]
    return -np.mean(np.log(p + 1e-15))

def accuracy(y_true, y_pred_probs):
    return (y_pred_probs.argmax(axis=1) == y_true).mean()

print("  relu, relu_grad, sigmoid, softmax, cross_entropy_loss, accuracy defined.")
print()


# ======================================================================
# SECTION 2: Full MLP Class with Adam Optimizer
# ======================================================================
print("=" * 70)
print("SECTION 2: MLP CLASS WITH BACKPROP + ADAM")
print("=" * 70)
print()

class MLPClassifier:
    """
    Multi-Layer Perceptron Classifier from scratch.
    - Arbitrary depth and width
    - ReLU hidden activations, Softmax output
    - Adam optimizer
    - Mini-batch gradient descent
    """

    def __init__(self, hidden_sizes=(64, 32), lr=0.001,
                 epochs=200, batch_size=32, verbose=True):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.weights = []    # list of W matrices
        self.biases  = []    # list of b vectors
        self.loss_history = []
        self.acc_history  = []

    def _init_weights(self, input_size, output_size):
        """He initialization for all layers."""
        layer_sizes = [input_size] + list(self.hidden_sizes) + [output_size]
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _forward(self, X):
        """Forward pass. Returns list of (Z, A) for each layer."""
        cache = []
        A = X
        n_hidden = len(self.hidden_sizes)

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            if i < n_hidden:       # hidden layers: ReLU
                A_next = relu(Z)
            else:                  # output layer: Softmax
                A_next = softmax(Z)
            cache.append((A, Z, A_next))
            A = A_next

        return cache

    def _backward(self, cache, y):
        """Backprop. Returns gradients for all W and b."""
        m = y.shape[0]
        n_layers = len(self.weights)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers

        # Output layer gradient (softmax + cross-entropy)
        A_prev, Z_out, A_out = cache[-1]
        dZ = A_out.copy()
        dZ[range(m), y] -= 1       # dLoss/dZ_out = softmax_out - one_hot(y)
        dZ /= m
        grads_W[-1] = A_prev.T @ dZ
        grads_b[-1] = dZ.sum(axis=0, keepdims=True)

        # Hidden layer gradients (chain rule backwards)
        dA = dZ @ self.weights[-1].T
        for i in range(n_layers - 2, -1, -1):
            A_prev_i, Z_i, A_i = cache[i]
            dZ = dA * relu_grad(Z_i)
            grads_W[i] = A_prev_i.T @ dZ
            grads_b[i] = dZ.sum(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.weights[i].T

        return grads_W, grads_b

    def _adam_step(self, grads_W, grads_b, t,
                   m_W, v_W, m_b, v_b,
                   beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam optimizer update."""
        for i in range(len(self.weights)):
            m_W[i] = beta1 * m_W[i] + (1 - beta1) * grads_W[i]
            v_W[i] = beta2 * v_W[i] + (1 - beta2) * grads_W[i]**2
            m_b[i] = beta1 * m_b[i] + (1 - beta1) * grads_b[i]
            v_b[i] = beta2 * v_b[i] + (1 - beta2) * grads_b[i]**2

            m_hat_W = m_W[i] / (1 - beta1**t)
            v_hat_W = v_W[i] / (1 - beta2**t)
            m_hat_b = m_b[i] / (1 - beta1**t)
            v_hat_b = v_b[i] / (1 - beta2**t)

            self.weights[i] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + eps)
            self.biases[i]  -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        self._init_weights(n_features, n_classes)

        # Adam moment accumulators
        m_W = [np.zeros_like(W) for W in self.weights]
        v_W = [np.zeros_like(W) for W in self.weights]
        m_b = [np.zeros_like(b) for b in self.biases]
        v_b = [np.zeros_like(b) for b in self.biases]

        if self.verbose:
            arch = [n_features] + list(self.hidden_sizes) + [n_classes]
            print(f"  Architecture: {' -> '.join(map(str, arch))}")
            print(f"  Params: lr={self.lr}, epochs={self.epochs}, batch={self.batch_size}")
            print()

        t = 0  # Adam step counter
        for epoch in range(self.epochs):
            # Shuffle data
            idx = np.random.permutation(n_samples)
            X_sh, y_sh = X[idx], y[idx]

            # Mini-batches
            for start in range(0, n_samples, self.batch_size):
                Xb = X_sh[start:start + self.batch_size]
                yb = y_sh[start:start + self.batch_size]
                t += 1
                cache = self._forward(Xb)
                grads_W, grads_b = self._backward(cache, yb)
                self._adam_step(grads_W, grads_b, t, m_W, v_W, m_b, v_b)

            # Track metrics
            cache_full = self._forward(X)
            y_pred_probs = cache_full[-1][2]
            loss = cross_entropy_loss(y, y_pred_probs)
            acc  = accuracy(y, y_pred_probs)
            self.loss_history.append(loss)
            self.acc_history.append(acc)

            if self.verbose and (epoch % 50 == 0 or epoch == self.epochs - 1):
                print(f"  Epoch {epoch:4d}/{self.epochs} | Loss: {loss:.4f} | Acc: {acc:.3f}")

    def predict_proba(self, X):
        cache = self._forward(X)
        return cache[-1][2]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# ======================================================================
# SECTION 3: Solve XOR
# ======================================================================
print("=" * 70)
print("SECTION 3: SOLVING XOR — THE MLP'S FIRST VICTORY")
print("=" * 70)
print()
print("The XOR problem that defeated the Perceptron:")
print()

X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

# Scale up XOR for training stability
X_xor_big = np.tile(X_xor, (25, 1)) + np.random.randn(100, 2) * 0.05
y_xor_big = np.tile(y_xor, 25)

mlp_xor = MLPClassifier(hidden_sizes=(8,), lr=0.01, epochs=300,
                         batch_size=20, verbose=False)
mlp_xor.fit(X_xor_big, y_xor_big)

preds_xor = mlp_xor.predict(X_xor)
print(f"  XOR truth:       {y_xor}")
print(f"  MLP predictions: {preds_xor}")
correct = (preds_xor == y_xor).all()
print(f"  All correct: {correct}  <- {'MLP solves XOR!' if correct else 'train longer'}")
print()


# ======================================================================
# SECTION 4: Real Classification Problem
# ======================================================================
print("=" * 70)
print("SECTION 4: REAL CLASSIFICATION — CIRCLES DATASET")
print("=" * 70)
print()

from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_circ, y_circ = make_circles(n_samples=400, noise=0.1, factor=0.4, random_state=42)
scaler = StandardScaler()
X_circ = scaler.fit_transform(X_circ)
X_tr, X_te, y_tr, y_te = train_test_split(X_circ, y_circ, test_size=0.25, random_state=42)

print(f"  Dataset: make_circles (non-linearly separable)")
print(f"  Train: {X_tr.shape[0]} samples, Test: {X_te.shape[0]} samples")
print()

mlp1 = MLPClassifier(hidden_sizes=(4,),    lr=0.005, epochs=300, batch_size=32, verbose=False)
mlp2 = MLPClassifier(hidden_sizes=(16, 8), lr=0.005, epochs=300, batch_size=32, verbose=False)
mlp3 = MLPClassifier(hidden_sizes=(32, 16, 8), lr=0.005, epochs=300, batch_size=32, verbose=False)

print("  Training 3 architectures for comparison...")
mlp1.fit(X_tr, y_tr)
mlp2.fit(X_tr, y_tr)
mlp3.fit(X_tr, y_tr)

for name, model in [("Shallow [4]", mlp1),
                     ("Medium  [16,8]", mlp2),
                     ("Deep    [32,16,8]", mlp3)]:
    tr_acc = model.score(X_tr, y_tr)
    te_acc = model.score(X_te, y_te)
    print(f"  {name}: Train={tr_acc:.3f}, Test={te_acc:.3f}")
print()


# ======================================================================
# SECTION 5: sklearn Comparison
# ======================================================================
print("=" * 70)
print("SECTION 5: SKLEARN COMPARISON")
print("=" * 70)
print()

try:
    from sklearn.neural_network import MLPClassifier as SklearnMLP
    sk_mlp = SklearnMLP(hidden_layer_sizes=(16, 8), max_iter=300,
                         learning_rate_init=0.005, random_state=42)
    sk_mlp.fit(X_tr, y_tr)
    print(f"  sklearn MLP — Train: {sk_mlp.score(X_tr, y_tr):.3f}, "
          f"Test: {sk_mlp.score(X_te, y_te):.3f}")
    print(f"  My MLP     — Train: {mlp2.score(X_tr, y_tr):.3f}, "
          f"Test: {mlp2.score(X_te, y_te):.3f}")
    print()
    print("  Both achieve similar results on this dataset!")
except ImportError:
    print("  sklearn not available for comparison.")
print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

print("Generating: Decision boundaries comparison...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("MLP Depth vs Decision Boundary Complexity",
             fontsize=14, fontweight="bold")

xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, (name, model) in zip(axes, [
    ("Shallow: [4]", mlp1),
    ("Medium: [16,8]", mlp2),
    ("Deep: [32,16,8]", mlp3),
]):
    Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors=["black"], linewidths=2)
    ax.scatter(X_tr[y_tr==0, 0], X_tr[y_tr==0, 1], c="tomato",
               s=20, alpha=0.6, edgecolors="none")
    ax.scatter(X_tr[y_tr==1, 0], X_tr[y_tr==1, 1], c="royalblue",
               s=20, alpha=0.6, edgecolors="none")
    ax.scatter(X_te[y_te==0, 0], X_te[y_te==0, 1], c="tomato",
               s=40, marker="^", edgecolors="black", linewidth=0.7, alpha=0.9)
    ax.scatter(X_te[y_te==1, 0], X_te[y_te==1, 1], c="royalblue",
               s=40, marker="^", edgecolors="black", linewidth=0.7, alpha=0.9)
    te_acc = model.score(X_te, y_te)
    ax.set_title(f"{name}\nTest Acc: {te_acc:.1%}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "decision_boundaries.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: decision_boundaries.png")


print("Generating: Training loss curves...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Training Progress: Loss and Accuracy Over Epochs",
             fontsize=13, fontweight="bold")

for name, model, color in [
    ("Shallow [4]", mlp1, "steelblue"),
    ("Medium [16,8]", mlp2, "darkorange"),
    ("Deep [32,16,8]", mlp3, "green"),
]:
    axes[0].plot(model.loss_history, color=color, linewidth=2, label=name)
    axes[1].plot(model.acc_history,  color=color, linewidth=2, label=name)

for ax, ylabel, title in zip(axes,
    ["Cross-Entropy Loss", "Accuracy"],
    ["Loss Over Training", "Accuracy Over Training"]):
    ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "training_curves.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: training_curves.png")


print("Generating: Weight heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Layer 1 Weights — What the Network Learned",
             fontsize=13, fontweight="bold")

for ax, model, title in zip(axes,
    [mlp2, mlp3],
    ["Medium [16, 8]", "Deep [32, 16, 8]"]
):
    im = ax.imshow(model.weights[0].T, cmap="coolwarm", aspect="auto")
    ax.set_title(f"{title}\nW1 shape: {model.weights[0].shape}", fontsize=11)
    ax.set_xlabel("Input features (2)"); ax.set_ylabel("Hidden neurons")
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "weight_heatmap.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: weight_heatmap.png")


# ============= CONCEPTUAL DIAGRAM =============
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')
ax.set_facecolor('#0f0f1a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Network architecture: 4-8-8-3
layer_sizes = [4, 8, 8, 3]
layer_labels = [
    "Input Layer\n(4 features)",
    "Hidden Layer 1\n(8 neurons, ReLU)",
    "Hidden Layer 2\n(8 neurons, ReLU)",
    "Output Layer\n(3 classes, Softmax)",
]
layer_colors = ['#3a7bd5', '#27ae60', '#27ae60', '#e07b39']
layer_xs = [1.5, 4.5, 8.5, 12.5]
neuron_radius = 0.28

# Pre-compute neuron y-positions for each layer
all_neuron_ys = []
for n in layer_sizes:
    total_span = (n - 1) * 0.9
    center = 4.5
    ys = [center - total_span / 2 + i * 0.9 for i in range(n)]
    all_neuron_ys.append(ys)

# Draw connections (lines behind neurons)
for li in range(len(layer_sizes) - 1):
    for yi in all_neuron_ys[li]:
        for yj in all_neuron_ys[li + 1]:
            ax.plot([layer_xs[li], layer_xs[li + 1]], [yi, yj],
                    color='#2a2a4a', lw=0.5, zorder=1, alpha=0.7)

# Draw neurons
for li, (lx, color, ys) in enumerate(zip(layer_xs, layer_colors, all_neuron_ys)):
    for yi in ys:
        circ = Circle((lx, yi), neuron_radius, color=color,
                      zorder=3, linewidth=1.2, edgecolor='white')
        ax.add_patch(circ)

# Layer label boxes below neurons
for li, (lx, color, label) in enumerate(zip(layer_xs, layer_colors, layer_labels)):
    ax.text(lx, 0.9, label, ha='center', va='top', color='white',
            fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                      edgecolor='white', lw=1.0, alpha=0.85))

# Right-pointing arrows between layers (at mid-height)
for li in range(len(layer_sizes) - 1):
    mid_x_start = layer_xs[li] + neuron_radius + 0.05
    mid_x_end = layer_xs[li + 1] - neuron_radius - 0.05
    mid_y = 4.5
    ax.annotate('', xy=(mid_x_end, mid_y), xytext=(mid_x_start, mid_y),
                arrowprops=dict(arrowstyle='->', color='#aaaacc', lw=1.5))

# Key insight text
ax.text(7.0, 8.55, "Each layer learns increasingly abstract features",
        ha='center', va='center', color='#f0c040', fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a1a2e',
                  edgecolor='#f0c040', lw=1.5))

# Forward pass formula at the bottom
fwd = "x  →  z₁=W₁x+b₁  →  a₁=ReLU(z₁)  →  z₂=W₂a₁+b₂  →  a₂=ReLU(z₂)  →  ŷ=softmax(W₃a₂+b₃)"
ax.text(7.0, 0.22, fwd, ha='center', va='center', color='#aaddff',
        fontsize=8.5, family='monospace')

# Color legend
for lx, color, lbl in [(2.5, '#3a7bd5', 'Input'),
                        (5.5, '#27ae60', 'Hidden (ReLU)'),
                        (9.5, '#e07b39', 'Output (Softmax)')]:
    ax.add_patch(Circle((lx - 0.25, 8.0), 0.2, color=color, zorder=5))
    ax.text(lx + 0.1, 8.0, lbl, color='white', fontsize=9, va='center')

fig.suptitle("Multi-Layer Perceptron Architecture  [4 → 8 → 8 → 3]",
             color='white', fontsize=15, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(os.path.join(VIS_DIR, '04_mlp_architecture_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 04_mlp_architecture_concept.png")
# ============= END CONCEPTUAL DIAGRAM =============

print()
print("=" * 70)
print("ALGORITHM 2: MULTI-LAYER PERCEPTRON COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  MLP = stacked layers with non-linear activations")
print("  More layers = more complex decision boundaries")
print("  Adam optimizer: the best default (beta1=0.9, beta2=0.999)")
print("  Mini-batch: each update uses a random subset of data")
print("  Backprop automates gradient computation through all layers")
print()
print("Next: Algorithm 3 -> MLP with Keras (same power, less code!)")
