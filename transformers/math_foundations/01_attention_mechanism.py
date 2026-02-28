"""
Attention Mechanism — The Core of Transformers
===============================================

Learning Objectives:
  1. Understand why attention was invented: the bottleneck problem in seq2seq RNNs
  2. Implement scaled dot-product attention from scratch with numpy
  3. Understand Query, Key, Value intuition through a database analogy
  4. Build masked (causal) attention for decoder auto-regressive generation
  5. Visualize what the model "attends to" via attention weight heatmaps
  6. Compare soft attention vs hard attention and understand why soft is used

YouTube: Search "Attention Mechanism Transformer Explained" for companion videos
Time: ~35 minutes | Difficulty: Intermediate | Prerequisites: NLP Part 5, linear algebra
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "01_attention_mechanism")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("ATTENTION MECHANISM — The Core of Transformers")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Why Attention? The RNN Bottleneck
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Why Attention? The RNN Bottleneck Problem")
print("=" * 70)

print("""
Classic seq2seq (RNN encoder → RNN decoder) problem:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Encoder: reads entire input, compresses to ONE hidden vector   │
  │  ┌────┐   ┌────┐   ┌────┐   ┌────┐                            │
  │  │ h1 │→  │ h2 │→  │ h3 │→  │ h4 │ = CONTEXT VECTOR c         │
  │  └────┘   └────┘   └────┘   └────┘                            │
  │  "The"   "cat"    "sat"    "down"                              │
  │                                                                 │
  │  Decoder: generates output FROM ONLY the context vector c       │
  │  → For long sentences, c cannot capture all relevant info!      │
  │  → Called the "bottleneck problem"                              │
  └─────────────────────────────────────────────────────────────────┘

Attention solution (Bahdanau 2014):
  → Instead of one fixed vector, let the decoder CHOOSE which encoder
    hidden states to focus on at each decoding step.

  c_t = Σ α_{t,s} * h_s        (weighted sum of all encoder states)
  α_{t,s} = softmax(score(decoder_t, encoder_s))

  Now the decoder can "attend" to any part of the input! ✓
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Query, Key, Value — The Database Analogy
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 2: Query, Key, Value — The Database Analogy")
print("=" * 70)

print("""
Think of attention like a soft database lookup:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Hard database: SELECT value WHERE key == query (exact match)   │
  │                                                                 │
  │  Soft attention: weighted sum of ALL values,                    │
  │                  weighted by similarity of query to each key    │
  │                                                                 │
  │  Query (Q): "What am I looking for?" (from decoder state)       │
  │  Key   (K): "What does each position describe?" (from encoder)  │
  │  Value (V): "What is the actual content?" (from encoder)        │
  └─────────────────────────────────────────────────────────────────┘

Example: Translating "The cat sat"
  - Generating "Le" → Q asks about subject → K["The"/"cat"] score high
  - Generating "chat" → Q asks about noun → K["cat"] scores highest
  - Generating "s'est" → Q asks about verb → K["sat"] scores highest

Scaled Dot-Product Attention (Vaswani 2017 — "Attention Is All You Need"):

  Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V

  Why divide by sqrt(d_k)?
  → Dot products grow large when d_k is large (variance = d_k)
  → Large values push softmax into near-zero gradient regions
  → Dividing by sqrt(d_k) stabilizes gradients during training
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: From Scratch — Scaled Dot-Product Attention
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 3: Scaled Dot-Product Attention — From Scratch")
print("=" * 70)


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Args:
        Q: Queries  shape (seq_len_q, d_k) or (batch, heads, seq_q, d_k)
        K: Keys     shape (seq_len_k, d_k) or (batch, heads, seq_k, d_k)
        V: Values   shape (seq_len_k, d_v) or (batch, heads, seq_k, d_v)
        mask: Optional boolean mask — True means "attend to", False means "ignore"
              shape (seq_len_q, seq_len_k)

    Returns:
        output:  weighted sum of values  shape (..., seq_q, d_v)
        weights: attention weights       shape (..., seq_q, seq_k)
    """
    d_k = Q.shape[-1]

    # Step 1: Compute attention scores — Q × K^T
    # scores[i,j] = how much query i attends to key j
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)     # (..., seq_q, seq_k)

    # Step 2: Apply mask (for causal/padding masking)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)            # masked positions → -∞ before softmax

    # Step 3: Softmax — convert scores to probabilities
    weights = softmax(scores, axis=-1)                   # (..., seq_q, seq_k)

    # Step 4: Weighted sum of Values
    output = weights @ V                                 # (..., seq_q, d_v)

    return output, weights


print("Attention function defined. Let's trace through a mini example:")
print()

# Toy example: 4-token sequence, d_model=6, d_k=d_v=6
np.random.seed(42)
SEQ_LEN = 4
D_K = 6
TOKENS = ["The", "cat", "sat", "down"]

# Normally Q, K, V come from linear projections of word embeddings.
# Here we set them manually to show the mechanics.
Q = np.random.randn(SEQ_LEN, D_K)  # queries
K = np.random.randn(SEQ_LEN, D_K)  # keys
V = np.random.randn(SEQ_LEN, D_K)  # values

output, weights = scaled_dot_product_attention(Q, K, V)

print(f"  Input shapes  → Q:{Q.shape}  K:{K.shape}  V:{V.shape}")
print(f"  Output shape  → {output.shape}")
print(f"  Weights shape → {weights.shape}")
print()
print("  Attention weight matrix (each row sums to 1.0):")
print(f"  {'':12s}", end="")
for tok in TOKENS:
    print(f"{tok:>10s}", end="")
print()
for i, tok_q in enumerate(TOKENS):
    print(f"  {tok_q:12s}", end="")
    for j in range(SEQ_LEN):
        print(f"{weights[i, j]:>10.3f}", end="")
    print(f"   (attended most to: {TOKENS[np.argmax(weights[i])]})")

# ── Scaling effect demo ────────────────────────────────────────────────
print("\n" + "-" * 50)
print("Scaling effect: why divide by sqrt(d_k)?")
print()
d_options = [1, 8, 64, 512]
for d in d_options:
    q = np.random.randn(1, d)
    k = np.random.randn(4, d)
    raw_scores = q @ k.T
    scaled_scores = raw_scores / np.sqrt(d)
    raw_std = float(raw_scores.std())
    scaled_std = float(scaled_scores.std())
    raw_max_softmax = float(softmax(raw_scores).max())
    scaled_max_softmax = float(softmax(scaled_scores).max())
    print(f"  d_k={d:4d} | raw std={raw_std:.2f}  scaled std={scaled_std:.2f} | "
          f"softmax peak: raw={raw_max_softmax:.3f}  scaled={scaled_max_softmax:.3f}")
print()
print("  → Large d_k without scaling collapses softmax to near one-hot!")
print("    (one position gets ~1.0, rest ~0.0 → vanishing gradient)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Self-Attention vs Cross-Attention
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Self-Attention vs Cross-Attention")
print("=" * 70)

print("""
Self-Attention (Encoder):  Q, K, V all come from the SAME sequence
  → Every token can attend to every other token in the same sequence
  → "The cat sat" — 'sat' can attend to 'cat' (subject-verb agreement)

Cross-Attention (Decoder): Q comes from decoder, K/V come from encoder
  → Decoder queries the encoder's representations
  → Classic seq2seq attention: decoder state queries encoder outputs

Masked Self-Attention (Decoder): Self-attention with CAUSAL MASK
  → Token i can only attend to tokens 0..i (not future tokens)
  → Required for auto-regressive generation (can't see the future!)
""")


def make_causal_mask(seq_len):
    """
    Causal (lower-triangular) mask.
    mask[i,j] = True means position i CAN attend to position j.
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


# Demonstrate self-attention with causal mask
SEQ = 5
mask = make_causal_mask(SEQ)
Q2 = np.random.randn(SEQ, D_K)
K2 = np.random.randn(SEQ, D_K)
V2 = np.random.randn(SEQ, D_K)

output_causal, weights_causal = scaled_dot_product_attention(Q2, K2, V2, mask=mask)

print("Causal mask (True=attend, False=blocked):")
for i in range(SEQ):
    row = " ".join("✓" if mask[i, j] else "✗" for j in range(SEQ))
    print(f"  Token {i}: [{row}]")

print()
print("Causal attention weights (upper triangle is 0):")
for i in range(SEQ):
    row = " ".join(f"{weights_causal[i,j]:.3f}" for j in range(SEQ))
    print(f"  Token {i}: [{row}]")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Additive (Bahdanau) vs Multiplicative (Luong/Vaswani) Attention
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Attention Variants")
print("=" * 70)

print("""
Three main attention score functions:

  1. Additive (Bahdanau 2014):
       score(q, k) = v^T * tanh(W_q*q + W_k*k)
       → Trainable parameters, works well for smaller d_k
       → Original "attention" from neural machine translation

  2. Dot-Product (Luong 2015):
       score(q, k) = q · k
       → No extra parameters, fast (matrix multiply)
       → Works well when Q and K have similar distributions

  3. Scaled Dot-Product (Vaswani 2017 — Transformers):
       score(q, k) = q · k / sqrt(d_k)
       → Dot-product + scaling to stabilize gradients
       → Used in all modern Transformers (BERT, GPT, etc.)

All three compute the same thing: "how similar is this query to this key?"
The result is always normalized by softmax to get attention weights.
""")


class BahdanauAttention:
    """Additive attention (Bahdanau 2014) from scratch."""

    def __init__(self, d_model, d_attn=32):
        self.W_q = np.random.randn(d_model, d_attn) * 0.1
        self.W_k = np.random.randn(d_model, d_attn) * 0.1
        self.v = np.random.randn(d_attn) * 0.1

    def score(self, q, K):
        """q: (d,)  K: (seq, d)  → scores: (seq,)"""
        # W_q*q + W_k*k_i for each k_i
        combined = np.tanh(q @ self.W_q + K @ self.W_k)  # (seq, d_attn)
        return combined @ self.v                           # (seq,)

    def forward(self, q, K, V):
        scores = self.score(q, K)                         # (seq,)
        weights = softmax(scores)                         # (seq,)
        context = weights @ V                             # (d_v,)
        return context, weights


# Quick comparison
np.random.seed(0)
d = 16
seq = 6
q_vec = np.random.randn(d)
K_mat = np.random.randn(seq, d)
V_mat = np.random.randn(seq, d)

# Dot-product attention (single query)
dot_scores = K_mat @ q_vec / np.sqrt(d)
dot_weights = softmax(dot_scores)

# Bahdanau
bahdanau = BahdanauAttention(d, d_attn=8)
_, bah_weights = bahdanau.forward(q_vec, K_mat, V_mat)

print("Dot-product attention weights:", " ".join(f"{w:.3f}" for w in dot_weights))
print("Bahdanau  attention weights  :", " ".join(f"{w:.3f}" for w in bah_weights))
print()
print("Both produce probability distributions over the sequence.")
print("The key difference is HOW the score is computed, not the output format.")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Generating Visualizations")
print("=" * 70)

# ── Visualization 1: Attention Weights + Scaling Effect ───────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Scaled Dot-Product Attention", fontsize=14, fontweight="bold")

# 1a: Attention weight heatmap
ax = axes[0]
im = ax.imshow(weights, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(SEQ_LEN))
ax.set_yticks(range(SEQ_LEN))
ax.set_xticklabels(TOKENS, fontsize=11)
ax.set_yticklabels(TOKENS, fontsize=11)
ax.set_xlabel("Keys (attending TO)", fontsize=11)
ax.set_ylabel("Queries (attending FROM)", fontsize=11)
ax.set_title("Attention Weight Heatmap\n(each row sums to 1.0)", fontsize=11, fontweight="bold")
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center",
                fontsize=9, color="white" if weights[i, j] > 0.5 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 1b: Causal mask + weights
ax = axes[1]
im2 = ax.imshow(weights_causal, cmap="Greens", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(SEQ))
ax.set_yticks(range(SEQ))
ax.set_xticklabels([f"t{i}" for i in range(SEQ)], fontsize=10)
ax.set_yticklabels([f"t{i}" for i in range(SEQ)], fontsize=10)
ax.set_xlabel("Keys", fontsize=11)
ax.set_ylabel("Queries", fontsize=11)
ax.set_title("Causal (Masked) Attention\n(can't attend to future tokens)", fontsize=11, fontweight="bold")
for i in range(SEQ):
    for j in range(SEQ):
        val = weights_causal[i, j]
        ax.text(j, i, f"{val:.2f}" if val > 0 else "✗", ha="center", va="center",
                fontsize=8, color="white" if val > 0.5 else ("gray" if val == 0 else "black"))
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

# 1c: Scaling effect on softmax peakiness
ax = axes[2]
d_vals = [2, 8, 32, 128, 512]
colors = plt.cm.viridis(np.linspace(0, 1, len(d_vals)))
x_axis = np.linspace(-3, 3, 200)
for d_val, col in zip(d_vals, colors):
    # Simulate distribution of one score in d_k-dimensional space
    # and how softmax behaves with/without scaling
    np.random.seed(1)
    k_vecs = np.random.randn(8, d_val)
    q_vec2 = np.random.randn(d_val)
    raw = k_vecs @ q_vec2
    scaled = raw / np.sqrt(d_val)
    ax.plot(np.sort(softmax(scaled)), color=col, linewidth=2, label=f"d_k={d_val}")
ax.set_title("Softmax Distribution\nwith Scaling (more uniform = better)", fontsize=11, fontweight="bold")
ax.set_xlabel("Sorted key index")
ax.set_ylabel("Attention weight")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_attention_weights.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_attention_weights.png")

# ── Visualization 2: Q/K/V Intuition Diagram ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Attention: Query, Key, Value Intuition", fontsize=14, fontweight="bold")

# 2a: Database analogy diagram
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f8f9fa")

# Query box
q_rect = mpatches.FancyBboxPatch((0.3, 7.5), 2.5, 1.2, boxstyle="round,pad=0.1",
                                   facecolor="#3498db", alpha=0.85)
ax.add_patch(q_rect)
ax.text(1.55, 8.1, "Query (Q)\n\"What am I\nlooking for?\"",
        ha="center", va="center", color="white", fontsize=8, fontweight="bold")

# Database rows
db_entries = [
    ("Key: 'The'",   "Value: article info",  4.5, 8.5),
    ("Key: 'cat'",   "Value: noun info",     4.5, 6.8),
    ("Key: 'sat'",   "Value: verb info",     4.5, 5.1),
    ("Key: 'down'",  "Value: adverb info",   4.5, 3.4),
]
colors_db = ["#bdc3c7", "#2ecc71", "#bdc3c7", "#bdc3c7"]
for (key_lbl, val_lbl, x, y), col in zip(db_entries, colors_db):
    krect = mpatches.FancyBboxPatch((3.0, y - 0.6), 2.2, 1.1, boxstyle="round,pad=0.05",
                                     facecolor=col, alpha=0.85)
    ax.add_patch(krect)
    ax.text(4.1, y, key_lbl, ha="center", va="center", fontsize=8, fontweight="bold")
    vrect = mpatches.FancyBboxPatch((5.5, y - 0.6), 2.8, 1.1, boxstyle="round,pad=0.05",
                                     facecolor=col, alpha=0.6)
    ax.add_patch(vrect)
    ax.text(6.9, y, val_lbl, ha="center", va="center", fontsize=8)

ax.text(4.1, 9.7, "Keys (K)", ha="center", fontsize=9, fontweight="bold", color="#2c3e50")
ax.text(6.9, 9.7, "Values (V)", ha="center", fontsize=9, fontweight="bold", color="#2c3e50")
ax.text(1.55, 9.7, "Query (Q)", ha="center", fontsize=9, fontweight="bold", color="#3498db")

# Arrow from Q to keys
ax.annotate("", xy=(3.0, 6.8), xytext=(2.8, 7.9),
            arrowprops=dict(arrowstyle="->", lw=2, color="#e74c3c"))
ax.text(2.0, 7.2, "similarity\nscores", fontsize=7, color="#e74c3c", ha="center")

# Weighted output arrow
ax.annotate("", xy=(9.5, 5.5), xytext=(8.3, 5.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="#9b59b6"))
ax.text(9.6, 5.5, "Output\n(weighted\nsum of V)", fontsize=7.5, color="#9b59b6")

# Score annotations
scores_lbl = ["0.05", "0.75", "0.15", "0.05"]
for (_, _, x, y), s in zip(db_entries, scores_lbl):
    ax.text(2.7, y, f"score={s}", fontsize=7, color="#e74c3c", ha="right")

ax.set_title("Soft Database Lookup\n(Q matches 'cat' most → highest score → most V weight)",
             fontsize=10, fontweight="bold")

# 2b: Attention computation flow
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("Attention Computation Flow", fontsize=11, fontweight="bold")

STEPS = [
    (5, 9.2, "Q × K^T", "#3498db", "Matrix multiply: query vs all keys"),
    (5, 7.4, "÷ √d_k", "#e67e22", "Scale to stabilize softmax gradients"),
    (5, 5.6, "softmax(·)", "#2ecc71", "Convert scores → probabilities (sum=1)"),
    (5, 3.8, "× V", "#9b59b6", "Weighted sum of value vectors"),
    (5, 2.0, "Output", "#e74c3c", "Context-aware representation"),
]

for i, (x, y, label, color, desc) in enumerate(STEPS):
    rect = mpatches.FancyBboxPatch((2.5, y - 0.55), 5.0, 1.0,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=11,
            fontweight="bold", color="white")
    ax.text(x, y - 0.75, desc, ha="center", va="top", fontsize=8, color="#7f8c8d")
    if i < len(STEPS) - 1:
        ax.annotate("", xy=(x, STEPS[i + 1][1] + 0.55),
                    xytext=(x, y - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#7f8c8d"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_qkv_intuition.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_qkv_intuition.png")

# ── Visualization 3: Attention patterns on a sentence ────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Attention Patterns in Natural Language", fontsize=13, fontweight="bold")

# Simulate linguistically meaningful attention patterns for demonstration
WORDS = ["The", "hungry", "cat", "chased", "the", "mouse"]
N = len(WORDS)

# Synthetic plausible self-attention matrix (would come from a trained model)
# Diagonal + syntactic relationships
raw_attn = np.array([
    [0.6, 0.1, 0.2, 0.0, 0.05, 0.05],  # "The" attends to itself, then "cat"
    [0.1, 0.5, 0.3, 0.05, 0.0, 0.05],  # "hungry" attends to "cat" (modifies it)
    [0.2, 0.2, 0.4, 0.1, 0.05, 0.05],  # "cat" attends to itself + "hungry"
    [0.05, 0.05, 0.2, 0.5, 0.05, 0.15], # "chased" attends to "cat" (subject) + "mouse"
    [0.05, 0.0, 0.05, 0.0, 0.7, 0.2],   # "the" (2nd) attends to itself + "mouse"
    [0.05, 0.05, 0.1, 0.2, 0.15, 0.45], # "mouse" attends to itself + "chased"
])

ax = axes[0]
im = ax.imshow(raw_attn, cmap="YlOrRd", vmin=0, vmax=0.7, aspect="auto")
ax.set_xticks(range(N))
ax.set_yticks(range(N))
ax.set_xticklabels(WORDS, fontsize=11, rotation=20)
ax.set_yticklabels(WORDS, fontsize=11)
ax.set_xlabel("Attending TO →", fontsize=10)
ax.set_ylabel("Attending FROM →", fontsize=10)
ax.set_title("Self-Attention Weights\n(simulated linguistic patterns)", fontsize=11, fontweight="bold")
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{raw_attn[i,j]:.2f}", ha="center", va="center",
                fontsize=8, color="white" if raw_attn[i, j] > 0.4 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 3b: Attention as arrows diagram
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f9f9f9")
ax.set_title("Attention Arrows\n(thicker = higher weight)", fontsize=11, fontweight="bold")

word_positions = [(1.2, 6), (2.7, 6), (4.2, 6), (5.7, 6), (7.2, 6), (8.7, 6)]
for i, (x, y) in enumerate(word_positions):
    ax.text(x, y, WORDS[i], ha="center", va="center", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.7),
            color="white")

# Draw top-2 attention arrows for selected words
arrow_examples = [
    (2, 1, raw_attn[2, 1], "hungry→cat"),  # hungry modifies cat
    (3, 2, raw_attn[3, 2], "cat→chased"),  # cat is subject of chased
    (3, 5, raw_attn[3, 5], "mouse→chased"),
    (5, 3, raw_attn[5, 3], "chased→mouse"),
]
for (q_idx, k_idx, weight, lbl) in arrow_examples:
    qx, qy = word_positions[q_idx]
    kx, ky = word_positions[k_idx]
    lw = weight * 8
    ax.annotate("", xy=(kx, ky - 0.3), xytext=(qx, qy - 0.3),
                arrowprops=dict(arrowstyle="->", lw=lw, color="#e74c3c", alpha=0.7,
                                connectionstyle="arc3,rad=0.3"))
    mid_x = (qx + kx) / 2
    mid_y = qy - 0.8
    ax.text(mid_x, mid_y, f"{weight:.2f}", fontsize=8, ha="center", color="#e74c3c")

ax.text(5, 2.5,
        "Attention lets each token\ndynamically gather information\nfrom relevant other tokens.",
        ha="center", va="center", fontsize=10, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_attention_patterns.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_attention_patterns.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Attention Mechanism")
print("=" * 70)
print("""
What we covered:
  ✓ RNN bottleneck problem and why attention was invented
  ✓ Query, Key, Value intuition (soft database lookup)
  ✓ Scaled dot-product attention from scratch: Attention(Q,K,V) = softmax(QK^T/√d_k)V
  ✓ Why scaling by √d_k matters for training stability
  ✓ Self-attention vs cross-attention vs masked (causal) attention
  ✓ Bahdanau (additive) vs Luong/Vaswani (multiplicative) attention

Key equations:
  Score(q,k)    = q · k / √d_k             (scaled dot-product)
  Weights(Q,K)  = softmax(Q K^T / √d_k)    (attention distribution)
  Attention     = Weights × V               (context vector)

Coming up next:
  02_multi_head_attention.py — run multiple attention functions in parallel,
  each learning different types of relationships within the sequence.

Visualizations saved to: visuals/01_attention_mechanism/
  01_attention_weights.png  — heatmap + causal mask + scaling effect
  02_qkv_intuition.png      — Q/K/V database analogy + computation flow
  03_attention_patterns.png — linguistic attention patterns + arrows
""")
