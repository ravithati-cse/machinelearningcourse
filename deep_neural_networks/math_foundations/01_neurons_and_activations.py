"""
🧠 DEEP NEURAL NETWORKS — Module 1: Neurons and Activation Functions
======================================================================

Learning Objectives:
  1. Understand the biological neuron → mathematical neuron analogy
  2. Master the neuron equation: z = w·x + b, output = activation(z)
  3. Know 6 activation functions: Step, Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax
  4. Understand WHY activation functions matter: non-linearity
  5. Recognize the vanishing gradient problem in sigmoid/tanh
  6. Know the dying ReLU problem and when to use Leaky ReLU
  7. Compute a neuron output by hand

YouTube Resources:
  ⭐ 3Blue1Brown - But what is a neural network? https://www.youtube.com/watch?v=aircAruvnKk
  ⭐ StatQuest - Neural Networks Pt.1 https://www.youtube.com/watch?v=CqOfi41LfDw
  📚 Andrej Karpathy - The spelled-out intro to neural networks https://www.youtube.com/watch?v=VMj-3S1tku0

Time Estimate: 45-60 minutes
Difficulty: Beginner
Prerequisites: Linear algebra basics (dot product), basic Python
Key Concepts: neuron, weight, bias, activation function, non-linearity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Create visuals directory
VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "01_neurons_activations")
os.makedirs(VIS_DIR, exist_ok=True)


print("=" * 70)
print("🧠 MODULE 1: NEURONS AND ACTIVATION FUNCTIONS")
print("=" * 70)
print()
print("Welcome to Deep Neural Networks!")
print("This module is about the SMALLEST unit: a single artificial neuron.")
print()


# ======================================================================
# SECTION 1: The Biological Inspiration
# ======================================================================
print("=" * 70)
print("SECTION 1: THE BIOLOGICAL INSPIRATION")
print("=" * 70)
print()
print("Your brain has ~86 BILLION neurons. Each one:")
print("  🔵 Receives electrical signals from other neurons (inputs)")
print("  🔵 Combines those signals (weighted sum)")
print("  🔵 Decides whether to 'fire' — send a signal forward (activation)")
print()
print("An artificial neuron copies this exact structure:")
print()
print("  Biological neuron    →    Artificial neuron")
print("  ─────────────────────────────────────────")
print("  Dendrites (inputs)   →    x1, x2, ..., xn")
print("  Synapse strength     →    weights w1, w2, ..., wn")
print("  Cell body (sum)      →    z = w·x + b")
print("  Axon (firing)        →    output = activation(z)")
print()


# ======================================================================
# SECTION 2: The Neuron Equation
# ======================================================================
print("=" * 70)
print("SECTION 2: THE NEURON EQUATION")
print("=" * 70)
print()
print("A neuron takes n inputs and produces 1 output:")
print()
print("  STEP 1 — Weighted sum:")
print("  z = w1*x1 + w2*x2 + ... + wn*xn + b")
print("  (This is a dot product: z = w · x + b)")
print()
print("  STEP 2 — Apply activation function:")
print("  output = activation(z)")
print()
print("The WEIGHTS (w) control how important each input is.")
print("The BIAS (b) shifts the activation threshold.")
print()

# Manual example
print("--- HAND CALCULATION EXAMPLE ---")
print()
print("Imagine a neuron deciding 'Will it rain today?'")
print()
x = np.array([0.8, 0.3, 0.9])   # humidity, wind speed, cloud cover (0-1 scale)
w = np.array([0.6, 0.2, 0.8])   # weights (importance)
b = -0.5                          # bias

feature_names = ["Humidity", "Wind Speed", "Cloud Cover"]
for i, (xi, wi, fname) in enumerate(zip(x, w, feature_names)):
    print(f"  {fname}: input={xi}, weight={wi} → contribution={xi*wi:.3f}")

z = np.dot(w, x) + b
print()
print(f"  Weighted sum: z = {w[0]}×{x[0]} + {w[1]}×{x[1]} + {w[2]}×{x[2]} + ({b})")
print(f"  z = {w[0]*x[0]:.3f} + {w[1]*x[1]:.3f} + {w[2]*x[2]:.3f} + ({b})")
print(f"  z = {z:.3f}")
print()
print(f"  Before activation: z = {z:.3f}")
print(f"  Sigmoid(z) = 1/(1+e^(-{z:.3f})) = {1/(1+np.exp(-z)):.3f}")
print(f"  → ~{1/(1+np.exp(-z))*100:.0f}% probability of rain")
print()


# ======================================================================
# SECTION 3: Why Activation Functions Matter
# ======================================================================
print("=" * 70)
print("SECTION 3: WHY ACTIVATION FUNCTIONS MATTER")
print("=" * 70)
print()
print("Without activation functions, neural networks can ONLY learn")
print("linear relationships — no matter how many layers you add!")
print()
print("  Linear: y = w2*(w1*x + b1) + b2 = (w1*w2)*x + (w2*b1+b2)")
print("  ↑ Still just a line! Extra layers do nothing.")
print()
print("Activation functions introduce NON-LINEARITY.")
print("Non-linearity lets networks learn curves, spirals, complex boundaries!")
print()
print("Think of it like this:")
print("  Linear model → can only draw a straight line through data")
print("  Neural net with activations → can draw ANY shape")
print()


# ======================================================================
# SECTION 4: The 6 Activation Functions
# ======================================================================
print("=" * 70)
print("SECTION 4: THE 6 ACTIVATION FUNCTIONS")
print("=" * 70)
print()

x_range = np.linspace(-5, 5, 500)

def step_function(z):
    return (z >= 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def tanh_fn(z):
    return np.tanh(z)

def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


# ---- Step Function ----
print("1. STEP FUNCTION (historical, rarely used now)")
print("   • Output: 0 if z<0, 1 if z≥0")
print("   • Problem: gradient is 0 everywhere → can't learn with backprop!")
print("   • Used in: the original 1957 perceptron")
print()

# ---- Sigmoid ----
print("2. SIGMOID σ(z) = 1 / (1 + e^(-z))")
print("   • Output range: (0, 1) → great for probabilities")
print("   • Problem: VANISHING GRADIENT near z=-5 or z=5")
print("     (gradient ≈ 0 at extremes → learning stops)")
print("   • Used in: output layer for binary classification")
print()

# ---- ReLU ----
print("3. ReLU — Rectified Linear Unit: max(0, z)")
print("   • Output: 0 if z<0, z if z≥0")
print("   • Fast to compute, no vanishing gradient for z>0")
print("   • Problem: DYING ReLU — if z<0 always, neuron never activates")
print("   • Used in: hidden layers (the DEFAULT choice)")
print()

# ---- Leaky ReLU ----
print("4. LEAKY ReLU: max(0.01z, z)")
print("   • Like ReLU but small slope for z<0 (fixes dying ReLU)")
print("   • Used in: hidden layers when dying ReLU is a problem")
print()

# ---- Tanh ----
print("5. TANH: (e^z - e^(-z)) / (e^z + e^(-z))")
print("   • Output range: (-1, 1) → zero-centered (helpful for training)")
print("   • Still has vanishing gradient at extremes (less severe than sigmoid)")
print("   • Used in: hidden layers (older networks), RNNs")
print()

# ---- Softmax ----
print("6. SOFTMAX: e^zi / Σ(e^zj)")
print("   • Converts a vector of scores → probability distribution (sums to 1)")
print("   • Used in: OUTPUT layer for multi-class classification")
scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(f"   • Example: scores [2.0, 1.0, 0.1] → probabilities {probs.round(3)}")
print(f"     Sum: {probs.sum():.3f} ✅")
print()


# ======================================================================
# SECTION 5: Derivatives (Why They Matter for Learning)
# ======================================================================
print("=" * 70)
print("SECTION 5: DERIVATIVES — WHY THEY MATTER FOR LEARNING")
print("=" * 70)
print()
print("During training, we need to compute GRADIENTS (derivatives)")
print("of the loss with respect to the weights.")
print()
print("The activation function's derivative appears in EVERY gradient calculation.")
print()
print("If the derivative is near ZERO → gradient is near ZERO → weights don't update")
print("This is the VANISHING GRADIENT PROBLEM.")
print()

# Compute derivatives
eps = 1e-5
sigmoid_vals = sigmoid(x_range)
sigmoid_deriv = sigmoid_vals * (1 - sigmoid_vals)   # analytical derivative
tanh_vals = tanh_fn(x_range)
tanh_deriv = 1 - tanh_vals**2

relu_deriv = (x_range > 0).astype(float)

print("Maximum derivatives:")
print(f"  Sigmoid: max derivative = {sigmoid_deriv.max():.3f} (at z=0)")
print(f"  Tanh:    max derivative = {tanh_deriv.max():.3f} (at z=0)")
print(f"  ReLU:    derivative = 1.0 for all z>0 ✅ (no vanishing!)")
print()
print("→ ReLU has the best gradient properties for deep networks")
print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()

# --- PLOT 1: All 6 activation functions ---
print("📊 Generating: All 6 activation functions...")

activations = {
    "Step Function": (step_function(x_range), "royalblue", "z >= 0 ? 1 : 0"),
    "Sigmoid σ(z)": (sigmoid(x_range), "darkorange", "1/(1+e^-z)"),
    "ReLU": (relu(x_range), "green", "max(0, z)"),
    "Leaky ReLU": (leaky_relu(x_range), "red", "max(0.01z, z)"),
    "Tanh": (tanh_fn(x_range), "purple", "(e^z - e^-z)/(e^z + e^-z)"),
    "Softmax*": (sigmoid(x_range), "brown", "e^z / Σe^z  (*single input)"),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("🧠 Activation Functions: The Non-Linear Magic", fontsize=16, fontweight="bold")
axes = axes.flatten()

for ax, (name, (vals, color, formula)) in zip(axes, activations.items()):
    ax.plot(x_range, vals, color=color, linewidth=2.5)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"{name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("z (pre-activation)", fontsize=10)
    ax.set_ylabel("output", fontsize=10)
    ax.text(0.05, 0.95, formula, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "activation_functions.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: activation_functions.png")


# --- PLOT 2: Neuron diagram ---
print("📊 Generating: Neuron diagram...")

fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

ax.set_title("🧠 The Artificial Neuron", fontsize=16, fontweight="bold", pad=15)

# Input nodes
inputs = [(1.5, 6.5), (1.5, 4.0), (1.5, 1.5)]
input_labels = ["x₁ = 0.8\n(Humidity)", "x₂ = 0.3\n(Wind)", "x₃ = 0.9\n(Clouds)"]
weights = ["w₁=0.6", "w₂=0.2", "w₃=0.8"]
colors_input = ["#4CAF50", "#2196F3", "#FF9800"]

for (ix, iy), label, color in zip(inputs, input_labels, colors_input):
    circ = plt.Circle((ix, iy), 0.5, color=color, zorder=3, alpha=0.8)
    ax.add_patch(circ)
    ax.text(ix, iy, label, ha="center", va="center", fontsize=7.5,
            fontweight="bold", color="white", zorder=4)

# Neuron (summation node)
neuron_x, neuron_y = 5.5, 4.0
circ = plt.Circle((neuron_x, neuron_y), 0.8, color="#9C27B0", zorder=3, alpha=0.9)
ax.add_patch(circ)
ax.text(neuron_x, neuron_y + 0.15, "Σ", ha="center", va="center",
        fontsize=20, color="white", zorder=4, fontweight="bold")
ax.text(neuron_x, neuron_y - 0.45, "z = w·x+b", ha="center", va="center",
        fontsize=7, color="white", zorder=4)

# Bias node
ax.add_patch(plt.Circle((3.5, 0.5), 0.4, color="#607D8B", zorder=3, alpha=0.8))
ax.text(3.5, 0.5, "b=-0.5\n(bias)", ha="center", va="center",
        fontsize=7, color="white", zorder=4, fontweight="bold")
ax.annotate("", xy=(neuron_x - 0.8, neuron_y - 0.3), xytext=(3.9, 0.7),
            arrowprops=dict(arrowstyle="->", color="#607D8B", lw=1.5))

# Activation node
act_x, act_y = 8.0, 4.0
ax.add_patch(plt.Circle((act_x, act_y), 0.7, color="#E91E63", zorder=3, alpha=0.9))
ax.text(act_x, act_y + 0.15, "σ", ha="center", va="center",
        fontsize=18, color="white", zorder=4, fontweight="bold")
ax.text(act_x, act_y - 0.4, "sigmoid", ha="center", va="center",
        fontsize=7, color="white", zorder=4)

# Arrows from inputs to neuron
for (ix, iy), weight, color in zip(inputs, weights, colors_input):
    ax.annotate("", xy=(neuron_x - 0.8, neuron_y), xytext=(ix + 0.5, iy),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.0))
    mid_x = (ix + neuron_x) / 2
    mid_y = (iy + neuron_y) / 2
    ax.text(mid_x, mid_y + 0.3, weight, ha="center", fontsize=9,
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9))

# Arrow neuron → activation
ax.annotate("", xy=(act_x - 0.7, act_y), xytext=(neuron_x + 0.8, neuron_y),
            arrowprops=dict(arrowstyle="->", color="purple", lw=2.5))
ax.text(6.75, 4.4, f"z = {z:.3f}", ha="center", fontsize=10,
        color="purple", fontweight="bold")

# Output arrow
ax.annotate("", xy=(9.8, act_y), xytext=(act_x + 0.7, act_y),
            arrowprops=dict(arrowstyle="->", color="#E91E63", lw=2.5))
output_val = sigmoid(np.array([z]))[0]
ax.text(9.5, 4.5, f"ŷ = {output_val:.3f}\n({output_val*100:.0f}% rain)",
        ha="center", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="#FCE4EC", edgecolor="#E91E63"))

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "neuron_diagram.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: neuron_diagram.png")


# --- PLOT 3: Derivatives comparison ---
print("📊 Generating: Activation function derivatives...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("📉 Activation Derivatives — The Vanishing Gradient Problem",
             fontsize=14, fontweight="bold")

# Sigmoid + derivative
ax = axes[0]
ax.plot(x_range, sigmoid(x_range), "darkorange", linewidth=2, label="Sigmoid")
ax.plot(x_range, sigmoid_deriv, "red", linewidth=2, linestyle="--", label="Derivative")
ax.fill_between(x_range, sigmoid_deriv, alpha=0.2, color="red")
ax.set_title("Sigmoid: max derivative = 0.25\n⚠️ Vanishing gradient at extremes",
             fontsize=11, color="darkred")
ax.legend(fontsize=10)
ax.set_xlim(-5, 5); ax.grid(True, alpha=0.3)
ax.set_xlabel("z")

# Tanh + derivative
ax = axes[1]
ax.plot(x_range, tanh_fn(x_range), "purple", linewidth=2, label="Tanh")
ax.plot(x_range, tanh_deriv, "magenta", linewidth=2, linestyle="--", label="Derivative")
ax.fill_between(x_range, tanh_deriv, alpha=0.2, color="magenta")
ax.set_title("Tanh: max derivative = 1.0\n⚠️ Still vanishes at extremes",
             fontsize=11, color="purple")
ax.legend(fontsize=10)
ax.set_xlim(-5, 5); ax.grid(True, alpha=0.3)
ax.set_xlabel("z")

# ReLU + derivative
ax = axes[2]
ax.plot(x_range, relu(x_range), "green", linewidth=2, label="ReLU")
ax.plot(x_range, relu_deriv, "limegreen", linewidth=2, linestyle="--", label="Derivative")
ax.set_title("ReLU: derivative = 1 for z>0\n✅ No vanishing gradient!",
             fontsize=11, color="darkgreen")
ax.legend(fontsize=10)
ax.set_xlim(-5, 5); ax.grid(True, alpha=0.3)
ax.set_xlabel("z")

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "activation_derivatives.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   ✅ Saved: activation_derivatives.png")


# ======================================================================
# SECTION 7: QUICK REFERENCE SUMMARY
# ======================================================================
print()
print("=" * 70)
print("SECTION 7: QUICK REFERENCE — WHICH ACTIVATION TO USE?")
print("=" * 70)
print()
print("┌─────────────┬──────────────┬─────────────────────────────────────┐")
print("│ Activation  │ Output Range │ When to Use                         │")
print("├─────────────┼──────────────┼─────────────────────────────────────┤")
print("│ ReLU        │ [0, ∞)       │ Hidden layers (DEFAULT choice)      │")
print("│ Leaky ReLU  │ (-∞, ∞)      │ Hidden layers (dying ReLU fix)      │")
print("│ Sigmoid     │ (0, 1)       │ Binary classification OUTPUT only   │")
print("│ Softmax     │ (0,1) sums=1 │ Multi-class classification OUTPUT   │")
print("│ Tanh        │ (-1, 1)      │ Hidden layers (zero-centered)       │")
print("│ Linear      │ (-∞, ∞)      │ Regression OUTPUT only              │")
print("└─────────────┴──────────────┴─────────────────────────────────────┘")
print()

# ============= CONCEPTUAL DIAGRAM =============
# 04_neuron_anatomy_concept.png — A Single Artificial Neuron: Anatomy
print("📊 Generating: Neuron anatomy concept diagram...")

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle as MplCircle
import matplotlib.patheffects as pe

fig_cd, ax_cd = plt.subplots(1, 1, figsize=(14, 8))
fig_cd.patch.set_facecolor('#0f0f1a')
ax_cd.set_facecolor('#0f0f1a')
ax_cd.set_xlim(0, 14)
ax_cd.set_ylim(0, 9)
ax_cd.axis('off')

ax_cd.text(7, 8.5, 'A Single Artificial Neuron — Anatomy',
           ha='center', va='center', fontsize=16, fontweight='bold',
           color='white')

# --- Input circles (x1–x4) ---
input_positions = [(1.2, 7.0), (1.2, 5.2), (1.2, 3.4), (1.2, 1.6)]
input_labels    = ['x\u2081', 'x\u2082', 'x\u2083', 'x\u2084']
weight_labels   = ['w\u2081', 'w\u2082', 'w\u2083', 'w\u2084']
input_color     = '#4488ff'   # blue
weight_color    = '#44dd88'   # green
neuron_color    = '#ff8844'   # orange
output_color    = '#bb44ff'   # purple

for (ix, iy), lbl in zip(input_positions, input_labels):
    circ_in = MplCircle((ix, iy), 0.42, color=input_color, zorder=4)
    ax_cd.add_patch(circ_in)
    ax_cd.text(ix, iy, lbl, ha='center', va='center', fontsize=13,
               fontweight='bold', color='white', zorder=5)

# --- Neuron body (large circle) ---
neuron_cx, neuron_cy = 6.5, 4.3
neuron_r = 1.5
neuron_circ = MplCircle((neuron_cx, neuron_cy), neuron_r,
                         color=neuron_color, alpha=0.92, zorder=4)
ax_cd.add_patch(neuron_circ)
ax_cd.text(neuron_cx, neuron_cy + 0.5, 'Neuron', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white', zorder=6)

# Inner boxes inside neuron
box1 = FancyBboxPatch((5.2, 3.75), 2.6, 0.5,
                       boxstyle='round,pad=0.08', linewidth=1.2,
                       edgecolor='white', facecolor='#3a2000', zorder=7)
ax_cd.add_patch(box1)
ax_cd.text(neuron_cx, 4.0, 'Weighted Sum  z = w\u00b7x + b',
           ha='center', va='center', fontsize=7.5, color='#ffe0b0', zorder=8)

box2 = FancyBboxPatch((5.2, 3.05), 2.6, 0.5,
                       boxstyle='round,pad=0.08', linewidth=1.2,
                       edgecolor='white', facecolor='#1a2a00', zorder=7)
ax_cd.add_patch(box2)
ax_cd.text(neuron_cx, 3.3, 'Activation  f(z)',
           ha='center', va='center', fontsize=7.5, color='#c8ffb0', zorder=8)

# --- Arrows: inputs → neuron ---
for (ix, iy), wlbl in zip(input_positions, weight_labels):
    ax_cd.annotate('',
        xy=(neuron_cx - neuron_r, neuron_cy),
        xytext=(ix + 0.42, iy),
        arrowprops=dict(arrowstyle='->', color=weight_color, lw=2.0))
    mid_x = (ix + 0.42 + neuron_cx - neuron_r) / 2
    mid_y = (iy + neuron_cy) / 2
    ax_cd.text(mid_x, mid_y + 0.22, wlbl, ha='center', va='center',
               fontsize=10, fontweight='bold', color=weight_color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='#0f0f1a',
                         edgecolor=weight_color, alpha=0.9))

# --- Output circle ---
out_x, out_y = 12.0, 4.3
out_circ = MplCircle((out_x, out_y), 0.55, color=output_color, zorder=4)
ax_cd.add_patch(out_circ)
ax_cd.text(out_x, out_y, 'Output', ha='center', va='center',
           fontsize=8, fontweight='bold', color='white', zorder=5)

# Arrow: neuron → output
ax_cd.annotate('',
    xy=(out_x - 0.55, out_y),
    xytext=(neuron_cx + neuron_r, neuron_cy),
    arrowprops=dict(arrowstyle='->', color=output_color, lw=2.5))

# --- Formula bar at the bottom ---
formula_box = FancyBboxPatch((0.5, 0.15), 13.0, 1.15,
                              boxstyle='round,pad=0.12', linewidth=1.5,
                              edgecolor='#555577', facecolor='#12122a', zorder=3)
ax_cd.add_patch(formula_box)
ax_cd.text(7.0, 0.88,
           'z  =  w\u2081x\u2081 + w\u2082x\u2082 + w\u2083x\u2083 + w\u2084x\u2084 + b',
           ha='center', va='center', fontsize=11, color='#ffe0b0',
           fontfamily='monospace')
ax_cd.text(7.0, 0.42,
           'output  =  ReLU(z)  =  max(0, z)',
           ha='center', va='center', fontsize=11, color='#c8ffb0',
           fontfamily='monospace')

# --- Legend ---
legend_items = [
    (input_color,  'Inputs (x)'),
    (weight_color, 'Weights (w)'),
    (neuron_color, 'Neuron body'),
    (output_color, 'Output'),
]
for i, (col, lbl) in enumerate(legend_items):
    lx = 0.9 + i * 3.3
    ly = 8.45
    ax_cd.add_patch(MplCircle((lx, ly), 0.15, color=col, zorder=5))
    ax_cd.text(lx + 0.3, ly, lbl, va='center', fontsize=9, color='#cccccc')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, '04_neuron_anatomy_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig_cd.get_facecolor())
plt.close()
print("   Saved: 04_neuron_anatomy_concept.png")
# ============= END CONCEPTUAL DIAGRAM =============

print("=" * 70)
print("✅ MODULE 1 COMPLETE!")
print("=" * 70)
print()
print("Key Takeaways:")
print("  🧠 A neuron = weighted sum + bias + activation function")
print("  🔑 Activation functions enable non-linear learning")
print("  ⚠️  Sigmoid/Tanh suffer from vanishing gradients in deep networks")
print("  ✅ ReLU is the default choice for hidden layers")
print("  📊 Softmax for multi-class output, Sigmoid for binary output")
print()
print("Next: Module 2 → Forward Propagation (connecting neurons into layers!)")
print()
print(f"Visualizations saved to: {VIS_DIR}/")
print("  • activation_functions.png")
print("  • neuron_diagram.png")
print("  • activation_derivatives.png")


if __name__ == "__main__":
    pass
