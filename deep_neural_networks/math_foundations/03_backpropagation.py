"""
🧠 DEEP NEURAL NETWORKS — Module 3: Backpropagation
=====================================================

Learning Objectives:
  1. Understand backprop as "blame assignment" — which weights caused the error?
  2. Grasp the chain rule of calculus through an intuitive analogy
  3. Derive gradients for the output layer step by step
  4. Derive gradients for hidden layers using the chain rule
  5. Implement a complete forward + backward pass in numpy
  6. Watch loss decrease over training epochs
  7. Visualize gradient flow and weight updates

YouTube Resources:
  ⭐ 3Blue1Brown - Backpropagation calculus https://www.youtube.com/watch?v=tIeHLnjs5U8
  ⭐ StatQuest - Backpropagation! https://www.youtube.com/watch?v=IN2XmBhILt4
  📚 Andrej Karpathy - micrograd walkthrough https://www.youtube.com/watch?v=VMj-3S1tku0

Time Estimate: 60-75 minutes
Difficulty: Intermediate
Prerequisites: Module 1 (Activations), Module 2 (Forward Prop), basic calculus (derivatives)
Key Concepts: gradient, chain rule, backpropagation, weight update, learning rate
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../visuals/03_backpropagation", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🧠 MODULE 3: BACKPROPAGATION — HOW NEURAL NETWORKS LEARN")
print("=" * 70)
print()
print("Forward prop: data flows INPUT → OUTPUT (make a prediction)")
print("Backprop:     error flows OUTPUT → INPUT (update weights to improve)")
print()
print("Key question: 'How much did each weight CONTRIBUTE to the error?'")
print("Key tool:     The CHAIN RULE of calculus")
print()


# ======================================================================
# SECTION 1: The Chain Rule — Intuition First
# ======================================================================
print("=" * 70)
print("SECTION 1: THE CHAIN RULE — INTUITION FIRST")
print("=" * 70)
print()
print("The chain rule answers: if A affects B, and B affects C,")
print("how much does A affect C?")
print()
print("  dC/dA = (dC/dB) × (dB/dA)")
print()
print("Real-world analogy:")
print("  Your mood → your productivity → your boss's happiness")
print("  How much does mood affect boss's happiness?")
print("  = (how much productivity affects boss) × (how much mood affects productivity)")
print()
print("In a neural network:")
print("  weights → Z (pre-activation) → A (post-activation) → Loss")
print()
print("  dLoss/dW = (dLoss/dA) × (dA/dZ) × (dZ/dW)")
print("              ↑            ↑           ↑")
print("           from output   activation  = X (the input!)")
print("           layer         derivative")
print()


# ======================================================================
# SECTION 2: Gradient of the Loss
# ======================================================================
print("=" * 70)
print("SECTION 2: GRADIENT DERIVATIONS")
print("=" * 70)
print()
print("Let's use Binary Cross-Entropy Loss and Sigmoid output:")
print()
print("  Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]")
print()
print("Gradients we need (derived once, used forever):")
print()
print("  OUTPUT LAYER:")
print("  dLoss/dZ_out = ŷ - y   (remarkably clean!)")
print()
print("  HIDDEN LAYER (chain rule):")
print("  dLoss/dZ_h = (dLoss/dA_h) * ReLU'(Z_h)")
print("  dLoss/dA_h = dLoss/dZ_out @ W_out.T")
print()
print("  WEIGHT GRADIENTS:")
print("  dLoss/dW = X.T @ dLoss/dZ")
print("  dLoss/db = mean(dLoss/dZ, axis=0)")
print()
print("  WEIGHT UPDATE:")
print("  W = W - learning_rate * dLoss/dW")
print()


# ======================================================================
# SECTION 3: Full Implementation from Scratch
# ======================================================================
print("=" * 70)
print("SECTION 3: FULL IMPLEMENTATION — FORWARD + BACKWARD")
print("=" * 70)
print()

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class NeuralNetworkFromScratch:
    """
    2-layer neural network with full backpropagation.
    Architecture: n_inputs → n_hidden → 1 (binary classification)
    """

    def __init__(self, n_inputs, n_hidden, learning_rate=0.1):
        self.lr = learning_rate
        # He initialization
        self.W1 = np.random.randn(n_inputs, n_hidden) * np.sqrt(2.0 / n_inputs)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, 1) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros((1, 1))
        self.loss_history = []

    def forward(self, X):
        """Pass data forward through the network."""
        # Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        # Layer 2 (output)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def backward(self, X, y):
        """Compute gradients and update weights."""
        m = X.shape[0]  # batch size

        # --- OUTPUT LAYER GRADIENTS ---
        # dLoss/dZ2 = ŷ - y  (for sigmoid + binary cross-entropy)
        dZ2 = self.A2 - y.reshape(-1, 1)           # (m, 1)
        dW2 = self.A1.T @ dZ2 / m                  # (n_hidden, 1)
        db2 = dZ2.mean(axis=0, keepdims=True)       # (1, 1)

        # --- HIDDEN LAYER GRADIENTS (chain rule) ---
        dA1 = dZ2 @ self.W2.T                       # (m, n_hidden)
        dZ1 = dA1 * relu_derivative(self.Z1)        # (m, n_hidden)
        dW1 = X.T @ dZ1 / m                        # (n_inputs, n_hidden)
        db1 = dZ1.mean(axis=0, keepdims=True)       # (1, n_hidden)

        # --- WEIGHT UPDATES ---
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        return dW1, dW2

    def train(self, X, y, epochs=1000, print_every=100):
        """Full training loop."""
        print(f"  Training: {epochs} epochs, lr={self.lr}")
        print(f"  Network: {X.shape[1]} → {self.W1.shape[1]} → 1")
        print()

        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred.flatten())
            self.loss_history.append(loss)
            self.backward(X, y)

            if epoch % print_every == 0 or epoch == epochs - 1:
                acc = ((y_pred.flatten() > 0.5) == y).mean()
                print(f"  Epoch {epoch:4d}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc:.3f}")

    def predict(self, X):
        return (self.forward(X).flatten() > 0.5).astype(int)

    def score(self, X, y):
        return (self.predict(X) == y).mean()


# Generate classification data (two circles — non-linearly separable)
from sklearn.datasets import make_circles
X_data, y_data = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)

# Normalize
X_data = (X_data - X_data.mean(axis=0)) / X_data.std(axis=0)

print("Dataset: make_circles (two concentric rings — not linearly separable)")
print(f"  Samples: {X_data.shape[0]}, Features: {X_data.shape[1]}, Classes: 2")
print(f"  Class balance: {y_data.mean():.1%} positive")
print()

model = NeuralNetworkFromScratch(n_inputs=2, n_hidden=8, learning_rate=0.5)

# Show weights BEFORE training
W1_before = model.W1.copy()
W2_before = model.W2.copy()

model.train(X_data, y_data, epochs=1000, print_every=200)

print()
print(f"  Final accuracy: {model.score(X_data, y_data):.3f}")
print()

# Show how much weights changed
W1_change = np.abs(model.W1 - W1_before).mean()
W2_change = np.abs(model.W2 - W2_before).mean()
print(f"  Avg weight change — Layer 1: {W1_change:.4f}, Layer 2: {W2_change:.4f}")
print(f"  (Large changes = the network was learning hard!)")
print()


# ======================================================================
# SECTION 4: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 4: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Chain rule computation graph ---
print("📊 Generating: Chain rule computation graph...")

fig, ax = plt.subplots(figsize=(13, 6))
ax.axis("off")
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.set_facecolor("#fafafa")
fig.patch.set_facecolor("#fafafa")
ax.set_title("🔗 Chain Rule: How Gradient Flows Backward Through the Network",
             fontsize=13, fontweight="bold", pad=12)

nodes = [
    (1.0, 3.0, "W, b\n(weights)", "#4CAF50"),
    (3.5, 3.0, "Z = X@W+b\n(pre-act)", "#2196F3"),
    (6.0, 3.0, "A = ReLU(Z)\n(post-act)", "#9C27B0"),
    (8.5, 3.0, "ŷ = σ(Z₂)\n(output)", "#FF5722"),
    (11.0, 3.0, "Loss\n(error)", "#F44336"),
]

for (x, y, label, color) in nodes:
    ax.add_patch(plt.Circle((x, y), 0.7, color=color, zorder=3, alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=4)

# Forward arrows (top)
arrow_pairs = [(1.7, 3.3, 2.8, 3.3), (4.2, 3.3, 5.3, 3.3),
               (6.7, 3.3, 7.8, 3.3), (9.2, 3.3, 10.3, 3.3)]
for (x1, y1, x2, y2) in arrow_pairs:
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="navy", lw=2))
ax.text(6.0, 4.5, "► FORWARD PASS (prediction)", ha="center", fontsize=11,
        color="navy", fontweight="bold")

# Backward arrows (bottom)
grad_labels = ["dL/dW = X.T @ dZ", "dL/dZ = dL/dA * ReLU'(Z)",
               "dL/dA = dL/dZ @ W.T", "dL/dZ₂ = ŷ - y"]
back_pairs = [(2.8, 2.7, 1.7, 2.7), (5.3, 2.7, 4.2, 2.7),
              (7.8, 2.7, 6.7, 2.7), (10.3, 2.7, 9.2, 2.7)]
for i, ((x1, y1, x2, y2), label) in enumerate(zip(back_pairs, grad_labels)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="darkred", lw=2))
    mid_x = (x1 + x2) / 2
    ax.text(mid_x, 1.8 - (i % 2) * 0.5, label, ha="center", fontsize=7.5,
            color="darkred",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", alpha=0.9))
ax.text(6.0, 1.1, "◄ BACKWARD PASS (learning)", ha="center", fontsize=11,
        color="darkred", fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/03_backpropagation/chain_rule_graph.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: chain_rule_graph.png")


# --- PLOT 2: Loss curve ---
print("📊 Generating: Loss curve over training...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("📉 Training Progress: Loss Decreasing = Network Learning",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(model.loss_history, color="#E91E63", linewidth=2)
ax.fill_between(range(len(model.loss_history)), model.loss_history, alpha=0.15, color="#E91E63")
ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
ax.set_title("Loss Over Training", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.text(0.98, 0.95, f"Start: {model.loss_history[0]:.3f}\nEnd:   {model.loss_history[-1]:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightyellow"))

# Decision boundary
ax = axes[1]
xx, yy = np.meshgrid(np.linspace(-2.5, 2.5, 200), np.linspace(-2.5, 2.5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.forward(grid).reshape(xx.shape)
ax.contourf(xx, yy, Z, levels=50, cmap="RdBu", alpha=0.7)
ax.contour(xx, yy, Z, levels=[0.5], colors=["black"], linewidths=2)
scatter = ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data,
                     cmap="RdBu", edgecolors="black", linewidth=0.5, s=50, zorder=5)
ax.set_title(f"Learned Decision Boundary\nAccuracy: {model.score(X_data, y_data):.1%}",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("../visuals/03_backpropagation/loss_and_boundary.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: loss_and_boundary.png")


# --- PLOT 3: Weight values before vs after ---
print("📊 Generating: Weight changes visualization...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("⚖️ Weight Changes During Training (Backprop at Work!)",
             fontsize=13, fontweight="bold")

ax = axes[0]
im = ax.imshow(W1_before, cmap="coolwarm", aspect="auto")
ax.set_title("Layer 1 Weights — BEFORE Training\n(random initialization)", fontsize=11)
ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Input Features")
plt.colorbar(im, ax=ax)

ax = axes[1]
im = ax.imshow(model.W1, cmap="coolwarm", aspect="auto")
ax.set_title("Layer 1 Weights — AFTER Training\n(structured by backprop)", fontsize=11)
ax.set_xlabel("Hidden Neurons"); ax.set_ylabel("Input Features")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("../visuals/03_backpropagation/weight_changes.png", dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: weight_changes.png")


print()
print("=" * 70)
print("✅ MODULE 3 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🔗 Backprop = chain rule applied layer by layer, OUTPUT → INPUT")
print("  📐 dLoss/dZ_out = ŷ - y  (the simplest gradient you'll ever see)")
print("  🔄 Weight update: W = W - lr * dLoss/dW")
print("  📉 Loss decreasing over epochs = the network is learning!")
print()
print("Next: Module 4 → Loss Functions & Optimizers (Adam, SGD, and beyond!)")
