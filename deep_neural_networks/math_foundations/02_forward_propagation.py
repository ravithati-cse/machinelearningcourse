"""
🧠 DEEP NEURAL NETWORKS — Module 2: Forward Propagation
=========================================================

Learning Objectives:
  1. Understand what a "layer" is and how layers connect
  2. Compute a full forward pass through a 2-layer network by hand (numpy)
  3. Master the matrix equation: Z = X @ W + b, A = activation(Z)
  4. Track tensor shapes through each layer (critical debugging skill)
  5. Build intuition for why depth helps (each layer = higher abstraction)
  6. Visualize information flow through a network
  7. Solve the XOR problem — proof that depth enables non-linearity

YouTube Resources:
  ⭐ 3Blue1Brown - But what is a neural network? https://www.youtube.com/watch?v=aircAruvnKk
  ⭐ 3Blue1Brown - Gradient descent, how neural networks learn https://www.youtube.com/watch?v=IHZwWFHWa-w
  📚 StatQuest - Neural Networks Pt.2 https://www.youtube.com/watch?v=IN2XmBhILt4

Time Estimate: 50-60 minutes
Difficulty: Beginner-Intermediate
Prerequisites: Module 1 (Neurons & Activations), numpy basics, matrix multiplication
Key Concepts: layer, forward pass, matrix multiplication, shape tracking, depth
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "02_forward_propagation")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("🧠 MODULE 2: FORWARD PROPAGATION")
print("=" * 70)
print()
print("Forward propagation = passing data FORWARD through the network")
print("from input → hidden layers → output, one layer at a time.")
print()


# ======================================================================
# SECTION 1: What Is a Layer?
# ======================================================================
print("=" * 70)
print("SECTION 1: WHAT IS A LAYER?")
print("=" * 70)
print()
print("A layer is simply a GROUP of neurons that all receive the same input.")
print()
print("If we have:")
print("  • 3 input features  (x1, x2, x3)")
print("  • 4 neurons in the hidden layer")
print()
print("Then each of the 4 neurons has its own 3 weights + 1 bias.")
print("Total parameters in this layer = 4 × 3 (weights) + 4 (biases) = 16")
print()
print("We can compute ALL 4 neurons at once using MATRIX MULTIPLICATION!")
print()
print("  Single neuron:  z = w · x + b       (dot product)")
print("  Full layer:     Z = X @ W + b        (matrix multiply)")
print()
print("  X shape: (batch_size, n_inputs)")
print("  W shape: (n_inputs, n_neurons)")
print("  b shape: (n_neurons,)")
print("  Z shape: (batch_size, n_neurons)  ← one output per neuron per sample")
print()


# ======================================================================
# SECTION 2: Single Layer — Step by Step
# ======================================================================
print("=" * 70)
print("SECTION 2: SINGLE LAYER — STEP BY STEP")
print("=" * 70)
print()
print("Let's build a single dense layer from scratch.")
print()

class DenseLayer:
    """A single fully-connected (dense) layer."""

    def __init__(self, n_inputs, n_neurons, activation="relu"):
        # He initialization (good for ReLU)
        self.W = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((1, n_neurons))
        self.activation = activation

    def _activate(self, Z):
        if self.activation == "relu":
            return np.maximum(0, Z)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation == "tanh":
            return np.tanh(Z)
        elif self.activation == "softmax":
            e = np.exp(Z - Z.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        else:  # linear
            return Z

    def forward(self, X):
        self.X = X
        self.Z = X @ self.W + self.b
        self.A = self._activate(self.Z)
        return self.A


# Example: 5 samples, 3 features → 4-neuron hidden layer
X = np.array([
    [0.2, 0.8, 0.5],
    [0.9, 0.1, 0.3],
    [0.4, 0.6, 0.7],
    [0.1, 0.9, 0.2],
    [0.7, 0.3, 0.8],
])

layer1 = DenseLayer(n_inputs=3, n_neurons=4, activation="relu")
A1 = layer1.forward(X)

print(f"Input X shape:        {X.shape}     ← (5 samples, 3 features)")
print(f"Weights W shape:      {layer1.W.shape}  ← (3 inputs, 4 neurons)")
print(f"Bias b shape:         {layer1.b.shape}   ← (1, 4 neurons)")
print(f"Pre-activation Z:     {layer1.Z.shape}  ← (5 samples, 4 outputs)")
print(f"Post-activation A:    {A1.shape}  ← same shape, values ≥ 0 (ReLU)")
print()
print("Pre-activation Z (first 2 samples):")
print(layer1.Z[:2].round(4))
print()
print("Post-activation A = ReLU(Z) (first 2 samples):")
print(A1[:2].round(4))
print("  Notice: negative values become 0 ✅")
print()


# ======================================================================
# SECTION 3: Multi-Layer Forward Pass
# ======================================================================
print("=" * 70)
print("SECTION 3: MULTI-LAYER FORWARD PASS")
print("=" * 70)
print()
print("A deep network is just layers chained together:")
print("  Input → Layer 1 → Layer 2 → Output")
print("  (output of each layer = input of next layer)")
print()
print("Architecture: 3 → 4 → 3 → 2")
print("  Input:    3 features")
print("  Hidden 1: 4 neurons (ReLU)")
print("  Hidden 2: 3 neurons (ReLU)")
print("  Output:   2 neurons (Sigmoid — binary classification)")
print()

class SimpleNetwork:
    """A simple multi-layer neural network for forward propagation."""

    def __init__(self, layer_sizes, activations):
        """
        layer_sizes: list like [3, 4, 3, 2]
        activations: list like ['relu', 'relu', 'sigmoid']
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

    def forward(self, X):
        A = X
        print("  Shape trace:")
        print(f"    Input:   {A.shape}")
        for i, layer in enumerate(self.layers):
            A = layer.forward(A)
            print(f"    Layer {i+1}: {A.shape}  (activation: {layer.activation})")
        return A


net = SimpleNetwork(
    layer_sizes=[3, 4, 3, 2],
    activations=["relu", "relu", "sigmoid"]
)

print("Running forward pass on 5 samples...")
output = net.forward(X)
print()
print("Final output (probabilities for 2 classes):")
print(output.round(4))
print()
print("Predicted class (argmax):", output.argmax(axis=1))
print()


# ======================================================================
# SECTION 4: The XOR Problem — Why Depth Matters
# ======================================================================
print("=" * 70)
print("SECTION 4: THE XOR PROBLEM — WHY DEPTH MATTERS")
print("=" * 70)
print()
print("XOR truth table (0=False, 1=True):")
print("  x1  x2  →  x1 XOR x2")
print("   0   0  →     0")
print("   0   1  →     1")
print("   1   0  →     1")
print("   1   1  →     0")
print()
print("A single linear layer CANNOT separate XOR — no straight line can!")
print("But a 2-layer network CAN. Let's prove it with pre-set weights.")
print()

# XOR data
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

# Hand-crafted weights that solve XOR
# Layer 1: 2→2 (creates two intermediate features)
W1 = np.array([[1, 1], [1, 1]], dtype=float)
b1 = np.array([[0, -1]], dtype=float)

# Layer 2: 2→1 (combines intermediate features)
W2 = np.array([[1], [-2]], dtype=float)
b2 = np.array([[0]], dtype=float)

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z))

Z1 = X_xor @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = sigmoid(Z2)

print("Forward pass through XOR network:")
print(f"{'Input':10} {'Hidden A1':20} {'Output':10} {'Target':10} {'Correct?':10}")
print("-" * 60)
for i in range(4):
    pred = "1" if A2[i,0] > 0.5 else "0"
    correct = "✅" if pred == str(int(y_xor[i,0])) else "❌"
    print(f"{str(X_xor[i].astype(int)):10} {str(A1[i].round(2)):20} "
          f"{A2[i,0]:.3f}     {int(y_xor[i,0])}         {correct}")
print()
print("The 2-layer network SOLVES XOR perfectly! 🎉")
print("A single layer could never do this.")
print()


# ======================================================================
# SECTION 5: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 5: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Network architecture diagram ---
print("📊 Generating: Network architecture diagram...")

def draw_network(ax, layer_sizes, title):
    ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
    max_nodes = max(layer_sizes)
    ax.set_ylim(-0.5, max_nodes - 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    colors = ["#4CAF50", "#2196F3", "#9C27B0", "#FF5722"]
    node_positions = []

    for l_idx, n_nodes in enumerate(layer_sizes):
        positions = []
        offset = (max_nodes - n_nodes) / 2
        for n_idx in range(n_nodes):
            y = offset + n_idx
            x = l_idx
            color = colors[min(l_idx, len(colors)-1)]
            circ = plt.Circle((x, y), 0.25, color=color, zorder=3, alpha=0.85)
            ax.add_patch(circ)
            ax.text(x, y, str(n_idx+1), ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold", zorder=4)
            positions.append((x, y))
        node_positions.append(positions)

    # Draw edges
    for l_idx in range(len(node_positions) - 1):
        for (x1, y1) in node_positions[l_idx]:
            for (x2, y2) in node_positions[l_idx + 1]:
                ax.plot([x1 + 0.25, x2 - 0.25], [y1, y2],
                        color="gray", alpha=0.4, linewidth=0.8, zorder=1)

    # Layer labels
    layer_names = ["Input\n(3)"] + [f"Hidden {i+1}\n({s})" for i, s in enumerate(layer_sizes[1:-1])] + [f"Output\n({layer_sizes[-1]})"]
    for l_idx, name in enumerate(layer_names):
        ax.text(l_idx, -0.3, name, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"))


fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("🧠 Network Architectures", fontsize=15, fontweight="bold")

draw_network(axes[0], [3, 4, 2], "Shallow Network: 3 → 4 → 2\n(1 hidden layer)")
draw_network(axes[1], [3, 4, 3, 2], "Deep Network: 3 → 4 → 3 → 2\n(2 hidden layers)")

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "network_diagram.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: network_diagram.png")


# --- PLOT 2: Shape transformation trace ---
print("📊 Generating: Shape transformation trace...")

fig, ax = plt.subplots(figsize=(13, 5))
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
ax.set_title("📐 Tensor Shape Transformations During Forward Pass",
             fontsize=14, fontweight="bold", pad=12)

shapes = [
    ("Input X", "(5, 3)", "5 samples\n3 features", "#4CAF50"),
    ("Z¹=X@W¹+b¹", "(5, 4)", "5 samples\n4 neurons", "#2196F3"),
    ("A¹=ReLU(Z¹)", "(5, 4)", "5 samples\n4 activations", "#1565C0"),
    ("Z²=A¹@W²+b²", "(5, 3)", "5 samples\n3 neurons", "#9C27B0"),
    ("A²=ReLU(Z²)", "(5, 3)", "5 samples\n3 activations", "#6A1B9A"),
    ("Ŷ=σ(A²@W³+b³)", "(5, 2)", "5 samples\n2 probs", "#E91E63"),
]

x_positions = np.linspace(0.05, 0.95, len(shapes))
for i, (name, shape, desc, color) in enumerate(shapes):
    x = x_positions[i]
    ax.text(x, 0.75, name, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9))
    ax.text(x, 0.5, shape, ha="center", va="center", fontsize=14,
            fontweight="bold", color=color)
    ax.text(x, 0.25, desc, ha="center", va="center", fontsize=8.5,
            color="gray", style="italic")
    if i < len(shapes) - 1:
        ax.annotate("", xy=(x_positions[i+1] - 0.05, 0.5),
                    xytext=(x + 0.05, 0.5),
                    arrowprops=dict(arrowstyle="->", color="black", lw=2))

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "shape_trace.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: shape_trace.png")


# --- PLOT 3: XOR decision boundary ---
print("📊 Generating: XOR — linear vs non-linear decision boundary...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("🧩 XOR Problem: Why Depth Enables Non-Linearity",
             fontsize=14, fontweight="bold")

X_xor_plot = np.array([[0,0],[0,1],[1,0],[1,1]])
y_colors = ["red", "blue", "blue", "red"]
labels = ["0 (class 0)", "1 (class 1)", "1 (class 1)", "0 (class 0)"]

for ax_idx, ax in enumerate(axes):
    for (px, py), color, label in zip(X_xor_plot, y_colors, labels):
        ax.scatter(px, py, c=color, s=300, zorder=5, edgecolors="black", linewidth=1.5)
        ax.text(px + 0.05, py + 0.05, label, fontsize=9)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.grid(True, alpha=0.3)

    if ax_idx == 0:
        # Try to draw a straight line — impossible
        ax.plot([-0.5, 1.5], [1.0, 0.0], "gray", linestyle="--", linewidth=2, label="Any straight line fails")
        ax.set_title("❌ Single Layer: Cannot Separate XOR\n(No straight line works)", fontsize=11, color="darkred")
        ax.legend(fontsize=9)
    else:
        # Show the non-linear boundary (approximate)
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z1g = grid @ W1 + b1
        A1g = relu(Z1g)
        Z2g = A1g @ W2 + b2
        A2g = sigmoid(Z2g).reshape(xx.shape)
        ax.contourf(xx, yy, A2g, levels=[0, 0.5, 1], colors=["#FFCDD2", "#BBDEFB"], alpha=0.4)
        ax.contour(xx, yy, A2g, levels=[0.5], colors=["purple"], linewidths=2)
        ax.set_title("✅ 2-Layer Network: Perfectly Separates XOR\n(Non-linear boundary)", fontsize=11, color="darkgreen")

red_patch = mpatches.Patch(color="red", label="Class 0")
blue_patch = mpatches.Patch(color="blue", label="Class 1")
axes[1].legend(handles=[red_patch, blue_patch], fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "xor_boundary.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: xor_boundary.png")


# ======================================================================
# SECTION 6: Key Rules to Remember
# ======================================================================
print()
print("=" * 70)
print("SECTION 6: KEY RULES TO REMEMBER")
print("=" * 70)
print()
print("📐 Shape Rules (memorize these!):")
print("  • X:  (batch_size, n_inputs)")
print("  • W:  (n_inputs, n_outputs)")
print("  • b:  (1, n_outputs)   or  (n_outputs,)")
print("  • Z = X @ W + b  →  shape (batch_size, n_outputs)")
print()
print("  Rule of thumb: inner dimensions must match for @")
print("  (batch, n_in) @ (n_in, n_out) = (batch, n_out)  ✅")
print()
print("🔗 Chaining Layers:")
print("  n_outputs of layer i  ==  n_inputs of layer i+1")
print("  [3 → 4 → 3 → 2]: 3→4 ✅, 4→3 ✅, 3→2 ✅")
print()
print("🧠 Why Depth Helps:")
print("  Layer 1: learns edges/patterns in raw data")
print("  Layer 2: combines edges → shapes/concepts")
print("  Layer 3+: combines concepts → high-level understanding")
print("  (This is how your visual cortex actually works!)")
print()

print("=" * 70)
print("✅ MODULE 2 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  📐 Z = X @ W + b  (matrix multiply for the whole layer at once)")
print("  🔁 Forward pass: chain layer outputs as the next layer's inputs")
print("  🧩 XOR proved that 2 layers > 1 layer for non-linear problems")
print("  📊 Always track shapes: (batch, n_inputs) × (n_inputs, n_outputs)")
print()
print("Next: Module 3 → Backpropagation (how networks learn from mistakes!)")
print()
print(f"Visualizations saved to: {VIS_DIR}/")
