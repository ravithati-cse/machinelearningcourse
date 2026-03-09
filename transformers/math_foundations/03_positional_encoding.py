"""
Positional Encoding — Teaching Transformers Word Order
======================================================

Learning Objectives:
  1. Understand why Transformers lose word order without positional information
  2. Implement sinusoidal positional encoding from scratch (Vaswani 2017)
  3. Understand the mathematical properties: unique, smooth, relative-position-aware
  4. Compare sinusoidal vs learned positional encodings (BERT-style)
  5. Visualize the sinusoidal patterns and their frequency decomposition
  6. Understand relative position encodings used in modern LLMs (RoPE, ALiBi)

YouTube: Search "Positional Encoding Transformer Explained" for companion videos
Time: ~30 minutes | Difficulty: Intermediate | Prerequisites: 01_attention_mechanism
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "03_positional_encoding")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("POSITIONAL ENCODING — Teaching Transformers Word Order")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: The Problem — Attention Is Permutation Invariant
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: The Problem — Attention Is Permutation Invariant")
print("=" * 70)

print("""
Attention computes Q×K^T — a set of dot products.

  For "The cat sat" and "Sat cat the", the dot products are the same!
  Attention treats input as a SET, not a SEQUENCE.

  This is a HUGE problem:
    "Dog bites man"  ≠  "Man bites dog"  (same words, opposite meaning)
    "Not good"       ≠  "Good not"

  RNNs handle order naturally through sequential processing.
  Transformers must INJECT positional information explicitly.

Solution options:
  1. Sinusoidal encoding (Vaswani 2017) — fixed mathematical functions
  2. Learned positional embeddings (BERT, GPT-2) — trainable lookup table
  3. Relative position encodings (Transformer-XL, RoPE) — encode relative distance
  4. ALiBi (Press 2021) — add position-based bias to attention scores

The simplest and most widely studied: SINUSOIDAL ENCODING.
""")

# Demo: attention without position info is permutation invariant
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return np.exp(x) / np.exp(x).sum(axis=axis, keepdims=True)

np.random.seed(42)
d = 8
# Two orderings of the same 3 embeddings
e1 = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0],   # "The"
               [0, 1.0, 0, 0, 0, 0, 0, 0],   # "cat"
               [0, 0, 1.0, 0, 0, 0, 0, 0]])  # "sat"

e2 = np.array([[0, 0, 1.0, 0, 0, 0, 0, 0],   # "sat"
               [0, 1.0, 0, 0, 0, 0, 0, 0],   # "cat"
               [1.0, 0, 0, 0, 0, 0, 0, 0]])  # "The"

W = np.random.randn(d, d) * 0.5
Q1, K1 = e1 @ W, e1 @ W
Q2, K2 = e2 @ W, e2 @ W
attn1 = softmax(Q1 @ K1.T / np.sqrt(d))
attn2 = softmax(Q2 @ K2.T / np.sqrt(d))

print("Attention weights — 'The cat sat' ordering:")
print("  " + "  ".join(f"{v:.3f}" for v in attn1[0]))
print()
print("Attention weights — 'sat cat The' ordering:")
print("  " + "  ".join(f"{v:.3f}" for v in attn2[0]))
print()
print("→ The attention PATTERN is the same (permuted) — order is lost!")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Sinusoidal Positional Encoding — The Math
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Sinusoidal Positional Encoding")
print("=" * 70)

print("""
Vaswani et al. (2017) proposed encoding position with sine and cosine functions
of different frequencies:

  PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

  where:  pos  = position in sequence (0, 1, 2, ...)
          i    = dimension index (0, 1, ..., d_model/2 - 1)
          d_model = embedding dimension

Why sinusoidal?
  ✓ Unique encoding for each position (no two positions identical)
  ✓ Smooth — nearby positions have similar encodings
  ✓ The model can extrapolate to longer sequences than seen during training
  ✓ No extra parameters to learn
  ✓ Relative positions can be expressed as linear functions of PE(pos)
     (PE(pos+k) can be written as a linear function of PE(pos))

Intuition: Think of a clock with multiple hands at different speeds.
  Low frequency (large i): slow rotation — captures long-range position
  High frequency (small i): fast rotation — captures fine-grained position

  Like binary numbers: 1, 10, 100, 1000 columns tick at different rates.
""")


def sinusoidal_positional_encoding(max_seq_len, d_model):
    """
    Compute sinusoidal positional encoding matrix.

    Returns:
        PE: shape (max_seq_len, d_model)
            PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
            PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """
    PE = np.zeros((max_seq_len, d_model))
    positions = np.arange(max_seq_len)[:, np.newaxis]       # (max_seq_len, 1)
    dims = np.arange(0, d_model, 2)                          # [0, 2, 4, ..., d_model-2]

    # Frequencies: 1 / 10000^(2i/d_model)
    div_term = 1.0 / np.power(10000.0, dims / d_model)      # (d_model/2,)

    # Even dimensions: sine
    PE[:, 0::2] = np.sin(positions * div_term)
    # Odd dimensions: cosine
    PE[:, 1::2] = np.cos(positions * div_term)

    return PE


# Demonstrate
MAX_LEN = 50
D_MODEL = 64
PE = sinusoidal_positional_encoding(MAX_LEN, D_MODEL)

print(f"Positional encoding matrix: shape={PE.shape}")
print(f"  PE[0, :8]  (pos=0):  {PE[0, :8].round(3)}")
print(f"  PE[1, :8]  (pos=1):  {PE[1, :8].round(3)}")
print(f"  PE[10, :8] (pos=10): {PE[10, :8].round(3)}")
print()
print("Value range check (should be [-1, 1]):")
print(f"  min={PE.min():.4f}  max={PE.max():.4f}")
print()

# ── How PE is added to embeddings ────────────────────────────────────────
print("How PE is used in Transformer:")
print()
print("""
  embedding = token_embedding(x)       # shape: (seq, d_model)
  embedding = embedding + PE[:seq, :]  # ADD positional encoding

  Both embedding and PE have the same shape (seq, d_model).
  The sum gives each token a unique position-aware representation.
""")

# ── Relative position property ─────────────────────────────────────────
print("Relative position property:")
print()
print("  PE(pos + k) = PE(pos) × M_k  for some fixed matrix M_k")
print("  → The model can learn 'token at pos+1' from PE alone")
print()
# Verify numerically for one dimension
pos = 10
k = 5
freq = 1.0 / np.power(10000.0, 0 / D_MODEL)
pe_pos_k = np.sin((pos + k) * freq)
pe_pos   = np.sin(pos * freq)
# sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
pe_reconstr = pe_pos * np.cos(k * freq) + np.cos(pos * freq) * np.sin(k * freq)
print(f"  PE(pos={pos}, k={k}) = {pe_pos_k:.6f}")
print(f"  Reconstructed via rotation matrix: {pe_reconstr:.6f}")
print(f"  Match: {np.isclose(pe_pos_k, pe_reconstr)} ✓")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Learned vs Sinusoidal Positional Encodings
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Learned vs Sinusoidal Positional Encodings")
print("=" * 70)

print("""
Learned Positional Embeddings (BERT, GPT-2):
  → Standard embedding lookup table: E ∈ ℝ^(max_pos × d_model)
  → Each position gets its own trainable vector
  → Trained jointly with the rest of the model

  Pros:
    ✓ Can capture arbitrary patterns — not constrained to sinusoids
    ✓ Can outperform sinusoidal on tasks with fixed sequence lengths
  Cons:
    ✗ Cannot generalize beyond max_pos seen during training
    ✗ Adds max_pos × d_model extra parameters (e.g., 512 × 768 = 393K)

Sinusoidal (original Transformer):
  Pros:
    ✓ No extra parameters
    ✓ Can extrapolate to longer sequences
    ✓ Relative position linearly expressible
  Cons:
    ✗ Fixed — cannot adapt to specific dataset patterns

Modern alternatives (for very long sequences):
  Rotary Position Embedding (RoPE) — LLaMA, PaLM:
    → Encodes position as rotation of query/key vectors
    → Relative position naturally preserved in dot products

  ALiBi (Attention with Linear Biases) — BLOOM:
    → Adds position-based penalty to attention scores: -m|i-j|
    → No position vectors at all — pure distance penalty
    → Strong length extrapolation

Empirically (Vaswani 2017):
  Sinusoidal ≈ Learned (nearly identical results on translation)
  → Most modern models use Learned (simpler to implement)
""")


class LearnedPositionalEmbedding:
    """Learned positional embeddings (BERT-style)."""

    def __init__(self, max_seq_len, d_model):
        # Initialize close to zero — learned from data
        self.embeddings = np.random.randn(max_seq_len, d_model) * 0.02

    def __call__(self, seq_len):
        return self.embeddings[:seq_len]


def rope_rotate(x, sin_vals, cos_vals):
    """
    Rotary Position Embedding (RoPE) — simplified 1D demo.
    Rotates pairs of dimensions in x by position-dependent angles.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    # Rotation: [x_e*cos - x_o*sin,  x_e*sin + x_o*cos]
    rotated_even = x_even * cos_vals - x_odd * sin_vals
    rotated_odd = x_even * sin_vals + x_odd * cos_vals
    out = np.zeros_like(x)
    out[..., 0::2] = rotated_even
    out[..., 1::2] = rotated_odd
    return out


# Compute RoPE sin/cos values
seq_rope = 20
d_rope = 16
theta = 1.0 / np.power(10000.0, np.arange(0, d_rope, 2) / d_rope)
positions_rope = np.arange(seq_rope)[:, None]
sin_rope = np.sin(positions_rope * theta)
cos_rope = np.cos(positions_rope * theta)

print("RoPE: rotates Q and K vectors by position angle")
print(f"  sin/cos shapes: {sin_rope.shape}  (seq, d/2)")
print("  Key property: RoPE(pos, q) · RoPE(pos+k, k) depends only on |k| (relative distance)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Position Similarity Analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Position Similarity Analysis")
print("=" * 70)

print("""
A good positional encoding should have:
  1. Unique vectors for each position (no collisions)
  2. Smooth transitions (close positions → similar vectors)
  3. The model can infer relative distances from PE alone

We verify these properties with cosine similarity of PE vectors.
""")

# Cosine similarity between position pairs
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

pos_a = 10
for delta in [0, 1, 2, 5, 10, 20]:
    pos_b = pos_a + delta
    if pos_b < MAX_LEN:
        sim = cosine_sim(PE[pos_a], PE[pos_b])
        print(f"  cos_sim(pos={pos_a}, pos={pos_b}) [Δ={delta:2d}] = {sim:.4f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Generating Visualizations")
print("=" * 70)

# ── Visualization 1: Sinusoidal PE Heatmap + Individual Waves ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Sinusoidal Positional Encoding", fontsize=14, fontweight="bold")

# 1a: Full PE heatmap
ax = axes[0]
im = ax.imshow(PE, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_title("PE Matrix Heatmap\n(rows=positions, cols=dimensions)", fontsize=11, fontweight="bold")
ax.set_xlabel("Encoding Dimension →")
ax.set_ylabel("Position (token index) →")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 1b: Individual sinusoidal waves (first few dimensions)
ax = axes[1]
positions = np.arange(MAX_LEN)
wave_dims = [0, 2, 6, 14, 30, 62]  # dim indices (even = sine)
colors = plt.cm.tab10(np.linspace(0, 1, len(wave_dims)))
for dim_idx, col in zip(wave_dims, colors):
    ax.plot(positions, PE[:, dim_idx], color=col, linewidth=2,
            label=f"dim {dim_idx} (freq={1/np.power(10000,dim_idx/D_MODEL):.4f})")
ax.set_title("Sinusoidal Waves by Dimension\n(low dim=high freq, high dim=low freq)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Position")
ax.set_ylabel("PE Value")
ax.legend(fontsize=7, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1c: Position similarity heatmap (cosine)
ax = axes[2]
n_pos = 30
sim_matrix = np.zeros((n_pos, n_pos))
for i in range(n_pos):
    for j in range(n_pos):
        sim_matrix[i, j] = cosine_sim(PE[i], PE[j])
im2 = ax.imshow(sim_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_title("Position Cosine Similarity\n(smooth decay with distance)", fontsize=11, fontweight="bold")
ax.set_xlabel("Position j")
ax.set_ylabel("Position i")
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_sinusoidal_pe.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_sinusoidal_pe.png")

# ── Visualization 2: Sinusoidal vs Learned vs RoPE comparison ─────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Positional Encoding Comparison", fontsize=14, fontweight="bold")

# 2a: Sinusoidal PE
ax = axes[0]
PE_show = sinusoidal_positional_encoding(20, 32)
im = ax.imshow(PE_show, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_title("Sinusoidal (fixed)\nPE(pos, 2i) = sin(pos/10000^(2i/d))",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 2b: Learned PE (random init, would be trained)
ax = axes[1]
lpe = LearnedPositionalEmbedding(20, 32)
# Simulate trained by adding structured noise
learned_pe = lpe.embeddings + 0.1 * sinusoidal_positional_encoding(20, 32)
im2 = ax.imshow(learned_pe, cmap="RdBu", aspect="auto", vmin=-0.3, vmax=0.3)
ax.set_title("Learned (trainable)\nLookup table, max_pos × d_model params",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position")
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

# 2c: RoPE sin values
ax = axes[2]
rope_vis = sinusoidal_positional_encoding(20, 32)  # RoPE uses same formula differently
im3 = ax.imshow(sin_rope[:20], cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_title("RoPE sin component\n(rotates Q/K in d/2 pairs)",
             fontsize=10, fontweight="bold")
ax.set_xlabel("Dimension pair index")
ax.set_ylabel("Position")
plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_pe_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_pe_comparison.png")

# ── Visualization 3: PE added to embeddings + frequency decomposition ─
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Positional Encoding in Action", fontsize=14, fontweight="bold")

# 3a: Embedding + PE visualization
ax = axes[0]
np.random.seed(99)
seq_demo = 8
d_demo = 16
# Simulate word embeddings
token_embeds = np.random.randn(seq_demo, d_demo) * 0.5
pe_demo = sinusoidal_positional_encoding(seq_demo, d_demo)
combined = token_embeds + pe_demo

# Plot as stacked bar-like display
im_tok = ax.imshow(token_embeds, cmap="PuOr", aspect="auto", vmin=-1.5, vmax=1.5)
ax.set_title("Token Embeddings Only\n(no position info — all rows could be swapped)", fontsize=9, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position (token)")
ax.set_yticks(range(seq_demo))
ax.set_yticklabels([f"pos {i}" for i in range(seq_demo)], fontsize=8)
plt.colorbar(im_tok, ax=ax, fraction=0.046, pad=0.04)

# 3b: PE only
ax = axes[1]
im_pe = ax.imshow(pe_demo, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_title("Positional Encoding Only\n(unique, smooth pattern per position)", fontsize=9, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position (token)")
ax.set_yticks(range(seq_demo))
ax.set_yticklabels([f"pos {i}" for i in range(seq_demo)], fontsize=8)
plt.colorbar(im_pe, ax=ax, fraction=0.046, pad=0.04)

# 3c: Combined (what the Transformer actually sees)
ax = axes[2]
im_comb = ax.imshow(combined, cmap="RdBu", aspect="auto", vmin=-2, vmax=2)
ax.set_title("Token Embed + PE\n(position injected — permuting rows changes meaning)",
             fontsize=9, fontweight="bold")
ax.set_xlabel("Dimension")
ax.set_ylabel("Position (token)")
ax.set_yticks(range(seq_demo))
ax.set_yticklabels([f"pos {i}" for i in range(seq_demo)], fontsize=8)
plt.colorbar(im_comb, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_pe_in_action.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_pe_in_action.png")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Positional Encoding Concept Diagram
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Generating Positional Encoding Concept Diagram")
print("=" * 70)

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, (ax_l, ax_c, ax_r) = plt.subplots(1, 3, figsize=(14, 9),
                                        gridspec_kw={"width_ratios": [1, 1.1, 0.9]})
fig.patch.set_facecolor('#0f0f1a')
for ax in (ax_l, ax_c, ax_r):
    ax.set_facecolor('#0f0f1a')
    ax.axis('off')

fig.suptitle("Positional Encoding: Why It Matters and How It Works",
             fontsize=14, fontweight='bold', color='white', y=0.97)

# ── LEFT PANEL: "Why PE?" problem illustration ────────────────────────
ax_l.set_xlim(0, 4.5)
ax_l.set_ylim(0, 9)
ax_l.set_title('Why Positional Encoding?', color='white',
               fontsize=11, fontweight='bold', pad=8)

# --- WITHOUT PE section ---
ax_l.text(2.25, 8.55, 'WITHOUT PE', color='#FF6B6B', fontsize=9.5,
          fontweight='bold', ha='center')

# Three identical "cat" boxes (no position signal)
cat_xs_no = [0.75, 2.25, 3.75]
cat_y_no = 7.6
for cx in cat_xs_no:
    b = FancyBboxPatch((cx - 0.42, cat_y_no - 0.28), 0.84, 0.56,
                       boxstyle="round,pad=0.06",
                       facecolor='#3a2a2a', edgecolor='#FF6B6B',
                       linewidth=1.8, zorder=4)
    ax_l.add_patch(b)
    ax_l.text(cx, cat_y_no, '"cat"', color='#ffaaaa', fontsize=9,
              ha='center', va='center', fontweight='bold', zorder=5)

# Labels showing positions but identical vectors
for i, cx in enumerate(cat_xs_no):
    ax_l.text(cx, cat_y_no - 0.55, f'pos {i}', color='#888888',
              fontsize=7, ha='center', style='italic')

# Arrow and problem text
ax_l.annotate('', xy=(2.25, 6.25), xytext=(2.25, 7.32),
              arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=2))
prob_b = FancyBboxPatch((0.3, 5.5), 3.9, 0.72,
                        boxstyle="round,pad=0.08",
                        facecolor='#3a1111', edgecolor='#FF6B6B',
                        linewidth=1.5, zorder=4)
ax_l.add_patch(prob_b)
ax_l.text(2.25, 5.86, 'Attention cannot distinguish\npositions — all "cat" are identical!',
          color='#FF9999', fontsize=7.5, ha='center', va='center', zorder=5)

# Divider line
ax_l.plot([0.2, 4.3], [5.1, 5.1], color='#444466', linewidth=1.5, linestyle='--')

# --- WITH PE section ---
ax_l.text(2.25, 4.75, 'WITH PE', color='#6BFF9E', fontsize=9.5,
          fontweight='bold', ha='center')

cat_y_pe = 3.85
sine_colors = ['#FF8C00', '#00BFFF', '#DA70D6']
for i, (cx, sc) in enumerate(zip(cat_xs_no, sine_colors)):
    # Word box
    b = FancyBboxPatch((cx - 0.42, cat_y_pe - 0.28), 0.84, 0.56,
                       boxstyle="round,pad=0.06",
                       facecolor='#1a2e1a', edgecolor='#6BFF9E',
                       linewidth=1.8, zorder=4)
    ax_l.add_patch(b)
    ax_l.text(cx, cat_y_pe, '"cat"', color='#aaffcc', fontsize=9,
              ha='center', va='center', fontweight='bold', zorder=5)
    # Sine wave badge underneath the word box
    wave_x = np.linspace(cx - 0.38, cx + 0.38, 30)
    wave_y = 3.05 + 0.18 * np.sin(np.linspace(0, (i + 1) * 2 * np.pi, 30))
    ax_l.plot(wave_x, wave_y, color=sc, linewidth=2, zorder=5)
    ax_l.text(cx, 2.7, f'pos {i}', color=sc, fontsize=7.5,
              ha='center', fontweight='bold', style='italic')

# Arrow and success text
ax_l.annotate('', xy=(2.25, 2.0), xytext=(2.25, 2.42),
              arrowprops=dict(arrowstyle='->', color='#6BFF9E', lw=2))
succ_b = FancyBboxPatch((0.3, 1.25), 3.9, 0.72,
                        boxstyle="round,pad=0.08",
                        facecolor='#0d2b1a', edgecolor='#6BFF9E',
                        linewidth=1.5, zorder=4)
ax_l.add_patch(succ_b)
ax_l.text(2.25, 1.61, 'Each position gets a unique\nsinusoidal fingerprint!',
          color='#99ffcc', fontsize=7.5, ha='center', va='center', zorder=5)

ax_l.text(2.25, 0.5,
          'Sine waves at different frequencies\nencode unique positions without parameters',
          color='#888899', fontsize=7, ha='center', style='italic')

# ── CENTER PANEL: PE matrix heat-map using Rectangle patches ─────────
ax_c.set_xlim(0, 5.6)
ax_c.set_ylim(0, 9)
ax_c.set_title('PE Matrix: Each Row = Position Fingerprint',
               color='white', fontsize=11, fontweight='bold', pad=8)

n_rows, n_cols = 8, 8
pe_mini = sinusoidal_positional_encoding(n_rows, n_cols * 2)[:, :n_cols]

cell_w = 0.5
cell_h = 0.52
grid_left = 0.45
grid_bottom = 1.6

for row in range(n_rows):
    for col in range(n_cols):
        val = pe_mini[row, col]
        # Map [-1, 1] to color: blue (neg) → white (0) → red (pos)
        if val >= 0:
            r, g, b_ch = 1.0, 1.0 - val * 0.85, 1.0 - val * 0.85
        else:
            r, g, b_ch = 1.0 + val * 0.85, 1.0 + val * 0.85, 1.0
        rect = Rectangle((grid_left + col * cell_w,
                           grid_bottom + (n_rows - 1 - row) * cell_h),
                          cell_w - 0.03, cell_h - 0.03,
                          facecolor=(r, g, b_ch), edgecolor='#1a1a2e',
                          linewidth=0.8, zorder=4)
        ax_c.add_patch(rect)
        ax_c.text(grid_left + col * cell_w + cell_w / 2,
                  grid_bottom + (n_rows - 1 - row) * cell_h + cell_h / 2,
                  f'{val:.2f}', color='#111111', fontsize=5,
                  ha='center', va='center', zorder=5)

# Row labels (position index)
for row in range(n_rows):
    ax_c.text(grid_left - 0.08,
              grid_bottom + (n_rows - 1 - row) * cell_h + cell_h / 2,
              f'pos {row}', color='#aaaacc', fontsize=7,
              ha='right', va='center')

# Column labels (dimension index)
for col in range(n_cols):
    ax_c.text(grid_left + col * cell_w + cell_w / 2,
              grid_bottom + n_rows * cell_h + 0.08,
              f'd{col}', color='#aaaacc', fontsize=6.5,
              ha='center', va='bottom')

# Axis labels
ax_c.text(grid_left + n_cols * cell_w / 2, grid_bottom + n_rows * cell_h + 0.52,
          'Encoding Dimensions \u2192', color='#ccccdd', fontsize=7.5, ha='center')
ax_c.text(grid_left - 0.55, grid_bottom + n_rows * cell_h / 2,
          'Positions \u2191', color='#ccccdd', fontsize=7.5, ha='center',
          rotation=90, va='center')

# Color legend bar
legend_x = np.linspace(0.4, 5.2, 60)
for k, lx in enumerate(legend_x):
    frac = k / (len(legend_x) - 1)
    val = frac * 2 - 1
    if val >= 0:
        r, g, b_ch = 1.0, 1.0 - val * 0.85, 1.0 - val * 0.85
    else:
        r, g, b_ch = 1.0 + val * 0.85, 1.0 + val * 0.85, 1.0
    ax_c.add_patch(Rectangle((lx, 1.05), 0.09, 0.3,
                              facecolor=(r, g, b_ch), linewidth=0))
ax_c.text(0.4, 0.85, '-1.0 (min)', color='#8888bb', fontsize=6.5, ha='center')
ax_c.text(2.8, 0.85, '0', color='#8888bb', fontsize=6.5, ha='center')
ax_c.text(5.2, 0.85, '+1.0 (max)', color='#8888bb', fontsize=6.5, ha='center')
ax_c.text(2.8, 0.5,
          'Blue=negative   White=zero   Red=positive',
          color='#888899', fontsize=7, ha='center', style='italic')

ax_c.text(2.8, 8.55,
          'sin/cos at decreasing frequencies\n'
          'Low dims: high-freq (fast-changing)  |  High dims: low-freq (slow)',
          color='#aaaacc', fontsize=7, ha='center', style='italic')

# ── RIGHT PANEL: Embedding + PE = Final Transformer Input ────────────
ax_r.set_xlim(0, 3.6)
ax_r.set_ylim(0, 9)
ax_r.set_title('How PE is Applied', color='white',
               fontsize=11, fontweight='bold', pad=8)

# Helper: draw a tall vector bar
def draw_vector_bar(ax, x_center, y_bottom, height, color, label_top,
                    label_bottom, bar_width=0.72):
    b = FancyBboxPatch((x_center - bar_width / 2, y_bottom),
                       bar_width, height,
                       boxstyle="round,pad=0.06",
                       facecolor=color, edgecolor='white',
                       linewidth=1.8, zorder=4, alpha=0.88)
    ax.add_patch(b)
    # Mini horizontal stripes to suggest a vector
    for k in range(6):
        yk = y_bottom + (k + 1) * height / 7
        stripe_val = np.sin(k * 1.3 + x_center * 2.1)
        stripe_col = '#ffffff' if stripe_val > 0 else '#000000'
        ax.plot([x_center - bar_width / 2 + 0.05,
                 x_center + bar_width / 2 - 0.05],
                [yk, yk], color=stripe_col, linewidth=0.9,
                alpha=0.35, zorder=5)
    ax.text(x_center, y_bottom + height + 0.18, label_top,
            color='white', fontsize=8, ha='center', fontweight='bold', zorder=6)
    ax.text(x_center, y_bottom - 0.22, label_bottom,
            color='#aaaacc', fontsize=7, ha='center', style='italic', zorder=6)

vec_x = 1.8
bar_h = 2.1

# Token Embedding vector
draw_vector_bar(ax_r, vec_x, 6.05, bar_h, '#1a5276',
                'Word Embedding', '"great"  \u2192  d-dim vector')

# Plus symbol
ax_r.text(vec_x, 5.65, '+', color='white', fontsize=26,
          ha='center', va='center', fontweight='bold')

# Positional Encoding vector
draw_vector_bar(ax_r, vec_x, 3.2, bar_h, '#7d3c98',
                'Positional Encoding', 'sin/cos at position t')

# Equals symbol
ax_r.text(vec_x, 2.82, '=', color='white', fontsize=26,
          ha='center', va='center', fontweight='bold')

# Final combined vector
draw_vector_bar(ax_r, vec_x, 0.55, bar_h, '#1a6b3c',
                'Final Input to Transformer', 'position-aware embedding')

# Dimension annotation
ax_r.annotate('', xy=(vec_x + 0.6, 6.05), xytext=(vec_x + 0.6, 6.05 + bar_h),
              arrowprops=dict(arrowstyle='<->', color='#dddddd', lw=1.5))
ax_r.text(vec_x + 0.82, 6.05 + bar_h / 2, 'd_model\ndimensions',
          color='#cccccc', fontsize=6.5, va='center')

# Same-shape annotation for PE bar
ax_r.annotate('', xy=(vec_x + 0.6, 3.2), xytext=(vec_x + 0.6, 3.2 + bar_h),
              arrowprops=dict(arrowstyle='<->', color='#dddddd', lw=1.5))
ax_r.text(vec_x + 0.82, 3.2 + bar_h / 2, 'same\nshape',
          color='#cccccc', fontsize=6.5, va='center')

# Formula text
ax_r.text(vec_x, 8.7,
          'embed(x) + PE[:seq_len]',
          color='#F0E68C', fontsize=9, ha='center', fontweight='bold',
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#1c1c30',
                    edgecolor='#F0E68C', linewidth=1.5))

ax_r.text(vec_x, 8.2,
          'Element-wise addition\n(same shape required)',
          color='#888899', fontsize=7, ha='center', style='italic')

plt.savefig(os.path.join(VIS_DIR, '04_positional_encoding_concept.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  Saved: {VIS_DIR}/04_positional_encoding_concept.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Positional Encoding")
print("=" * 70)
print("""
What we covered:
  ✓ Attention is permutation invariant — order must be injected explicitly
  ✓ Sinusoidal PE: PE(pos, 2i) = sin(pos / 10000^(2i/d))  — no parameters
  ✓ Properties: unique per position, smooth, can express relative positions
  ✓ Learned PE (BERT/GPT) — trainable but cannot extrapolate
  ✓ RoPE — rotates Q/K vectors, strong relative position awareness (LLaMA)
  ✓ ALiBi — attention bias -m|i-j|, excellent extrapolation (BLOOM)

Key formula:
  PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
  PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

  Usage: embedding = token_embed(x) + PE[:seq_len]  ← element-wise ADD

Frequency intuition:
  Dimension 0 (highest freq): cycles every ~6 positions
  Dimension d-2 (lowest freq): cycles every ~62,832 positions
  → Like binary digits — different "bit widths" capture different resolutions

Next: 04_encoder_decoder_arch.py — combine attention + PE into full
  encoder and decoder blocks with LayerNorm, Feed-Forward Networks, and residuals.

Visualizations saved to: visuals/03_positional_encoding/
  01_sinusoidal_pe.png              — heatmap + individual waves + similarity matrix
  02_pe_comparison.png              — sinusoidal vs learned vs RoPE
  03_pe_in_action.png               — token embed + PE + combined
  04_positional_encoding_concept.png — concept diagram: why PE + matrix + addition
""")
