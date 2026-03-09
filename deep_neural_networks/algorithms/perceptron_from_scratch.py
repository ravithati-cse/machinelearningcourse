"""
🤖 DEEP NEURAL NETWORKS — Algorithm 1: Perceptron from Scratch
===============================================================

Learning Objectives:
  1. Understand the historical context of the 1957 Rosenblatt Perceptron
  2. Implement the Perceptron learning rule from scratch in numpy
  3. Train on AND, OR logic gates (linearly separable)
  4. See why a single Perceptron FAILS on XOR (not linearly separable)
  5. Visualize the decision boundary moving during training
  6. Compare results with sklearn's Perceptron implementation
  7. Understand why multi-layer networks were invented

YouTube Resources:
  ⭐ StatQuest - Perceptrons https://www.youtube.com/watch?v=4Gac5I64LM4
  ⭐ 3Blue1Brown - But what is a neural network? https://www.youtube.com/watch?v=aircAruvnKk
  📚 ML Fundamentals - Perceptron Algorithm https://www.youtube.com/watch?v=4Gac5I64LM4

Time Estimate: 45-55 minutes
Difficulty: Beginner
Prerequisites: Module 1 (Neurons & Activations), basic Python
Key Concepts: perceptron, step function, linear separability, decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "perceptron")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🤖 ALGORITHM 1: THE PERCEPTRON — THE ORIGINAL NEURAL UNIT")
print("=" * 70)
print()
print("Year: 1957. Frank Rosenblatt creates the Perceptron.")
print("It was THE first machine learning algorithm — a huge deal!")
print()
print("The dream: a machine that learns from examples, like a brain.")
print("The reality: it only works for linearly separable problems.")
print("The legacy: it IS the building block of every modern neural network.")
print()


# ======================================================================
# SECTION 1: The Perceptron Algorithm
# ======================================================================
print("=" * 70)
print("SECTION 1: THE PERCEPTRON ALGORITHM")
print("=" * 70)
print()
print("The Perceptron has:")
print("  • n input weights (w1, w2, ..., wn) — one per feature")
print("  • 1 bias weight (b)")
print("  • A STEP activation function: output = 1 if z>=0, else 0")
print()
print("Forward pass (prediction):")
print("  z = w · x + b")
print("  y_hat = 1 if z >= 0  else  0")
print()
print("Learning rule (update on each MISCLASSIFIED example):")
print("  w = w + lr * (y - y_hat) * x")
print("  b = b + lr * (y - y_hat)")
print()
print("Key insight: ONLY update when the prediction is WRONG.")
print("  Correct:         (y - y_hat) = 0  -> no update")
print("  Wrong (0, true=1): (y - y_hat) = +1 -> push weights up")
print("  Wrong (1, true=0): (y - y_hat) = -1 -> push weights down")
print()


# ======================================================================
# SECTION 2: Perceptron Class from Scratch
# ======================================================================
print("=" * 70)
print("SECTION 2: PERCEPTRON CLASS FROM SCRATCH")
print("=" * 70)
print()

class Perceptron:
    """
    Classic Perceptron classifier (Rosenblatt 1957).
    Uses step activation and the Perceptron learning rule.
    """

    def __init__(self, learning_rate=0.1, max_epochs=100):
        self.lr = learning_rate
        self.max_epochs = max_epochs
        self.w = None
        self.b = None
        self.errors_per_epoch = []
        self.boundary_history = []

    def _step(self, z):
        return (z >= 0).astype(int)

    def fit(self, X, y, verbose=True):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        if verbose:
            print(f"  Training: lr={self.lr}, max_epochs={self.max_epochs}")
            print(f"  Data: {n_samples} samples, {n_features} features")
            print()

        for epoch in range(self.max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(self.w, xi) + self.b
                y_hat = self._step(np.array([z]))[0]
                update = self.lr * (yi - y_hat)
                self.w += update * xi
                self.b += update
                if update != 0:
                    errors += 1

            self.errors_per_epoch.append(errors)
            self.boundary_history.append((self.w.copy(), self.b))

            if verbose and (epoch % 10 == 0 or errors == 0):
                acc = self.score(X, y)
                print(f"  Epoch {epoch:3d}: errors={errors}, accuracy={acc:.3f}")

            if errors == 0:
                if verbose:
                    print(f"\n  Converged at epoch {epoch}!")
                break

    def predict(self, X):
        z = X @ self.w + self.b
        return self._step(z)

    def score(self, X, y):
        return (self.predict(X) == y).mean()

    def decision_boundary_y(self, x1_vals):
        if abs(self.w[1]) < 1e-10:
            return None
        return -(self.w[0] * x1_vals + self.b) / self.w[1]


# ======================================================================
# SECTION 3: Logic Gates
# ======================================================================
print("=" * 70)
print("SECTION 3: LOGIC GATES — AND, OR (LINEARLY SEPARABLE)")
print("=" * 70)
print()

X_and = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_and = np.array([0, 0, 0, 1])
X_or  = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_or  = np.array([0, 1, 1, 1])
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

print("--- AND GATE ---")
for row, label in zip(X_and, y_and):
    print(f"  {row.astype(int)} -> {label}")
print()
p_and = Perceptron(learning_rate=0.1, max_epochs=100)
p_and.fit(X_and, y_and)
print(f"\n  Result: {p_and.predict(X_and)} (truth: {y_and})")
print(f"  Accuracy: {p_and.score(X_and, y_and):.1%}")
print()

print("--- OR GATE ---")
p_or = Perceptron(learning_rate=0.1, max_epochs=100)
p_or.fit(X_or, y_or)
print(f"\n  Result: {p_or.predict(X_or)} (truth: {y_or})")
print(f"  Accuracy: {p_or.score(X_or, y_or):.1%}")
print()


# ======================================================================
# SECTION 4: XOR — The Perceptron's Limitation
# ======================================================================
print("=" * 70)
print("SECTION 4: XOR — THE PERCEPTRON'S LIMITATION")
print("=" * 70)
print()
print("XOR is NOT linearly separable. Watch it fail:")
print()
p_xor = Perceptron(learning_rate=0.1, max_epochs=100)
p_xor.fit(X_xor, y_xor, verbose=False)
print(f"  Predictions: {p_xor.predict(X_xor)}")
print(f"  Truth:       {y_xor}")
print(f"  Accuracy:    {p_xor.score(X_xor, y_xor):.1%}  <- can never reach 100%!")
print()
print("  This is the famous 'XOR problem' (1969, Minsky & Papert).")
print("  Solution: add MORE LAYERS -> Multi-Layer Perceptron (MLP)")
print()


# ======================================================================
# SECTION 5: Generate 2D data and compare with sklearn
# ======================================================================
print("=" * 70)
print("SECTION 5: 2D CLASSIFICATION + SKLEARN COMPARISON")
print("=" * 70)
print()

try:
    from sklearn.datasets import make_classification
    from sklearn.linear_model import Perceptron as SklearnPerceptron

    X_rand, y_rand = make_classification(
        n_samples=120, n_features=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )

    sk_p = SklearnPerceptron(eta0=0.1, max_iter=100, random_state=42)
    sk_p.fit(X_rand, y_rand)

    my_p = Perceptron(learning_rate=0.1, max_epochs=100)
    my_p.fit(X_rand, y_rand, verbose=False)

    print(f"  sklearn Perceptron accuracy: {sk_p.score(X_rand, y_rand):.3f}")
    print(f"  My Perceptron accuracy:      {my_p.score(X_rand, y_rand):.3f}")
    print()
    print("  Both find a good decision boundary on linearly separable data!")

except ImportError:
    print("  scikit-learn not installed. Run: pip install scikit-learn")
    from sklearn.datasets import make_classification
    X_rand, y_rand = make_classification(
        n_samples=120, n_features=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    my_p = Perceptron(learning_rate=0.1, max_epochs=100)
    my_p.fit(X_rand, y_rand, verbose=False)

print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: Logic gates ---
print("Generating: Logic gate decision boundaries...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Perceptron on Logic Gates: AND (OK) OR (OK) XOR (FAILS)",
             fontsize=13, fontweight="bold")

gate_data = [
    (X_and, y_and, p_and, "AND Gate  (Separable)", "green"),
    (X_or,  y_or,  p_or,  "OR Gate   (Separable)", "blue"),
    (X_xor, y_xor, p_xor, "XOR Gate  (NOT Separable)", "red"),
]

for ax, (X, y, model, title, color) in zip(axes, gate_data):
    for xi, yi in zip(X, y):
        ax.scatter(*xi, c="royalblue" if yi == 1 else "tomato",
                   s=300, zorder=5, edgecolors="black", linewidth=1.5)
        ax.text(xi[0]+0.05, xi[1]+0.07,
                f"({int(xi[0])},{int(xi[1])})->{yi}", fontsize=9)

    x1_r = np.array([-0.3, 1.3])
    db_y = model.decision_boundary_y(x1_r)
    if db_y is not None:
        ls = "-" if model.score(X, y) >= 1.0 else "--"
        ax.plot(x1_r, db_y, color=color, linewidth=2.5, linestyle=ls)

    acc = model.score(X, y)
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.4, 1.4)
    ax.set_title(f"{title}\nAccuracy: {acc:.0%}", fontsize=11, fontweight="bold",
                 color="darkgreen" if acc == 1.0 else "darkred")
    ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.grid(True, alpha=0.3)
    blue_p = mpatches.Patch(color="royalblue", label="Class 1")
    red_p  = mpatches.Patch(color="tomato",    label="Class 0")
    ax.legend(handles=[blue_p, red_p], fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "logic_gates.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: logic_gates.png")


# --- PLOT 2: Boundary evolution ---
print("Generating: Decision boundary evolution over epochs...")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Perceptron Learning: Boundary Moving Each Epoch",
             fontsize=13, fontweight="bold")

my_p2 = Perceptron(learning_rate=0.3, max_epochs=30)
my_p2.fit(X_rand, y_rand, verbose=False)

n_epochs_saved = len(my_p2.boundary_history)
snapshots = [0, max(1, n_epochs_saved//4), max(2, n_epochs_saved//2), n_epochs_saved-1]
x1_r2 = np.array([X_rand[:,0].min() - 0.5, X_rand[:,0].max() + 0.5])

for ax, ep in zip(axes, snapshots):
    w_s, b_s = my_p2.boundary_history[ep]
    ax.scatter(X_rand[y_rand==0, 0], X_rand[y_rand==0, 1],
               c="tomato", s=25, alpha=0.7, label="Class 0")
    ax.scatter(X_rand[y_rand==1, 0], X_rand[y_rand==1, 1],
               c="royalblue", s=25, alpha=0.7, label="Class 1")

    if abs(w_s[1]) > 1e-10:
        x2_line = -(w_s[0] * x1_r2 + b_s) / w_s[1]
        ax.plot(x1_r2, x2_line, "green", linewidth=2.5)

    errs = my_p2.errors_per_epoch[ep]
    ax.set_title(f"Epoch {ep+1}\nerrors={errs}", fontsize=11, fontweight="bold")
    ax.set_xlim(x1_r2); ax.grid(True, alpha=0.3)
    if ep == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "boundary_evolution.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: boundary_evolution.png")


# --- PLOT 3: Error convergence ---
print("Generating: Errors per epoch...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Perceptron Convergence: Errors Per Epoch", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(p_and.errors_per_epoch, "green", linewidth=2, label="AND gate")
ax.plot(p_or.errors_per_epoch,  "blue",  linewidth=2, label="OR gate")
ax.set_title("AND/OR: Converges to 0 errors", fontsize=11, color="darkgreen")
ax.set_xlabel("Epoch"); ax.set_ylabel("Errors")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(p_xor.errors_per_epoch, "red", linewidth=2, label="XOR gate")
ax.axhline(1, color="gray", linestyle="--", linewidth=1.5, label="Minimum possible")
ax.set_title("XOR: NEVER converges (linearly inseparable)", fontsize=11, color="darkred")
ax.set_xlabel("Epoch"); ax.set_ylabel("Errors")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "convergence.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: convergence.png")


# ============= CONCEPTUAL DIAGRAM =============
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('#0f0f1a')
for ax in axes.flat:
    ax.set_facecolor('#0f0f1a')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# --- LEFT PANEL: Perceptron Structure ---
ax_left = axes[0]
ax_left.set_title("The Perceptron Structure", color='white', fontsize=13,
                   fontweight='bold', pad=10)

# Input nodes: x1, x2, x3
input_labels = ['x₁', 'x₂', 'x₃']
weight_labels = ['w₁', 'w₂', 'w₃']
input_ys = [7.5, 5.0, 2.5]
input_x = 1.8

for i, (label, iy) in enumerate(zip(input_labels, input_ys)):
    circ = Circle((input_x, iy), 0.55, color='#3a7bd5', zorder=5)
    ax_left.add_patch(circ)
    ax_left.text(input_x, iy, label, ha='center', va='center',
                 color='white', fontsize=12, fontweight='bold', zorder=6)

# Step (activation) node
step_x, step_y = 5.5, 5.0
step_circ = Circle((step_x, step_y), 0.75, color='#e07b39', zorder=5)
ax_left.add_patch(step_circ)
ax_left.text(step_x, step_y + 0.05, 'STEP', ha='center', va='center',
             color='white', fontsize=10, fontweight='bold', zorder=6)

# Output node
out_x, out_y = 8.5, 5.0
out_circ = Circle((out_x, out_y), 0.55, color='#8e44ad', zorder=5)
ax_left.add_patch(out_circ)
ax_left.text(out_x, out_y, '0/1', ha='center', va='center',
             color='white', fontsize=11, fontweight='bold', zorder=6)

# Arrows from inputs to step node, labeled with weights
for i, (iy, wlabel) in enumerate(zip(input_ys, weight_labels)):
    ax_left.annotate('', xy=(step_x - 0.75, step_y + (iy - step_y) * 0.35),
                     xytext=(input_x + 0.55, iy),
                     arrowprops=dict(arrowstyle='->', color='#aaaacc',
                                     lw=1.8, connectionstyle='arc3,rad=0.0'))
    mid_x = (input_x + 0.55 + step_x - 0.75) / 2
    mid_y = (iy + step_y + (iy - step_y) * 0.35) / 2
    ax_left.text(mid_x - 0.1, mid_y + 0.25, wlabel, color='#f0c040',
                 fontsize=10, fontweight='bold', ha='center')

# Arrow from step to output
ax_left.annotate('', xy=(out_x - 0.55, out_y),
                 xytext=(step_x + 0.75, step_y),
                 arrowprops=dict(arrowstyle='->', color='#aaaacc', lw=2.0))

# Bias label
ax_left.text(step_x, step_y - 1.25, '+b (bias)', ha='center', va='center',
             color='#aaaacc', fontsize=9)

# Formula
ax_left.text(5.0, 0.7, 'z = Σ wᵢxᵢ + b', ha='center', va='center',
             color='#f0c040', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e', edgecolor='#f0c040', lw=1.5))

# Layer color legend
for cx, cy, col, lbl in [(1.2, 9.2, '#3a7bd5', 'Input'),
                          (4.8, 9.2, '#e07b39', 'Activation'),
                          (8.2, 9.2, '#8e44ad', 'Output')]:
    ax_left.add_patch(Circle((cx, cy), 0.25, color=col, zorder=5))
    ax_left.text(cx + 0.45, cy, lbl, color='white', fontsize=9, va='center')

# --- RIGHT PANEL: Historical Lineage ---
ax_right = axes[1]
ax_right.set_title("Neural Network Lineage", color='white', fontsize=13,
                    fontweight='bold', pad=10)

lineage = [
    ('[Perceptron — 1957]',    '#3a7bd5', 'Rosenblatt: first learning machine'),
    ('[MLP — 1985]',           '#27ae60', 'Rumelhart: backprop unlocks hidden layers'),
    ('[Deep Learning — 2006]', '#e07b39', 'Hinton: deep nets beat shallow methods'),
    ('[Transformers — 2017]',  '#8e44ad', 'Vaswani: attention is all you need'),
    ('[LLMs — 2020+]',         '#e74c3c', 'GPT/BERT: language understood at scale'),
]

box_x0, box_w, box_h = 1.0, 8.0, 1.0
start_y = 8.8
gap = 1.55

for i, (title, color, desc) in enumerate(lineage):
    by = start_y - i * gap
    rect = FancyBboxPatch((box_x0, by - box_h / 2), box_w, box_h,
                          boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='white',
                          linewidth=1.2, alpha=0.88, zorder=4)
    ax_right.add_patch(rect)
    ax_right.text(box_x0 + box_w / 2, by + 0.18, title,
                  ha='center', va='center', color='white',
                  fontsize=10, fontweight='bold', zorder=5)
    ax_right.text(box_x0 + box_w / 2, by - 0.22, desc,
                  ha='center', va='center', color='#eeeeee',
                  fontsize=8, style='italic', zorder=5)
    if i < len(lineage) - 1:
        arrow_y_top = by - box_h / 2
        arrow_y_bot = arrow_y_top - (gap - box_h)
        ax_right.annotate('', xy=(box_x0 + box_w / 2, arrow_y_bot),
                          xytext=(box_x0 + box_w / 2, arrow_y_top),
                          arrowprops=dict(arrowstyle='->', color='white', lw=2.0))

fig.suptitle("The Perceptron — The Ancestor of All Neural Networks",
             color='white', fontsize=15, fontweight='bold', y=0.97)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(VIS_DIR, '04_perceptron_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 04_perceptron_architecture.png")
# ============= END CONCEPTUAL DIAGRAM =============

print()
print("=" * 70)
print("ALGORITHM 1: PERCEPTRON COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  1957: Perceptron = first ML algorithm, ancestor of all neural nets")
print("  Works on AND, OR (linearly separable)")
print("  Fails on XOR  -> motivated invention of multi-layer networks")
print("  Update rule: w += lr * (y - y_hat) * x  (only on mistakes)")
print()
print("Next: Algorithm 2 -> Multi-Layer Perceptron from scratch (solves XOR!)")
