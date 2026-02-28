"""
Multi-Head Attention — Parallel Attention in Multiple Subspaces
===============================================================

Learning Objectives:
  1. Understand why multiple attention heads outperform a single head
  2. Implement multi-head attention from scratch with numpy
  3. Trace the full forward pass: Linear projections → attention → concat → project
  4. Understand head specialization: syntax head, coreference head, position head
  5. Analyze the parameter efficiency of multi-head vs single-head attention
  6. Visualize how different heads attend to different linguistic relationships

YouTube: Search "Multi-Head Attention Transformer PyTorch" for companion videos
Time: ~35 minutes | Difficulty: Intermediate | Prerequisites: 01_attention_mechanism
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "02_multi_head_attention")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("MULTI-HEAD ATTENTION — Parallel Attention in Multiple Subspaces")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Motivation — Why Multiple Heads?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Motivation — Why Multiple Heads?")
print("=" * 70)

print("""
With a single attention head, every query-key comparison happens in the
same representation space. But language has MULTIPLE types of relationships:

  • Syntactic: "cat" and "chased" (subject-verb)
  • Semantic:  "hungry" and "cat" (adjective-noun modifier)
  • Coreference: "it" and "the cat" (pronoun reference)
  • Positional: each token attending to nearby tokens

A single attention head must capture ALL of these simultaneously,
which can lead to competition and under-representation.

Multi-Head Attention solution:
  → Run h independent attention functions in parallel
  → Each head projects Q, K, V into a smaller subspace (d_k = d_model / h)
  → Each head can specialize in a different type of relationship
  → Concatenate all heads, project back to d_model

  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

  where head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)

  Parameters per head:
    W_Q_i: (d_model, d_k)    d_k = d_model / h
    W_K_i: (d_model, d_k)
    W_V_i: (d_model, d_v)    d_v = d_model / h

  Output projection:
    W_O:   (h × d_v, d_model)
""")

print("Parameter count comparison (d_model=512, h=8):")
d_model = 512
h = 8
d_k = d_model // h   # 64
d_v = d_model // h   # 64

single_head_params = d_model * d_model * 3 + d_model * d_model  # Q,K,V + out
multi_head_params = h * (d_model * d_k + d_model * d_k + d_model * d_v) + (h * d_v) * d_model

print(f"  d_model={d_model}, h={h}, d_k=d_v={d_k}")
print(f"  Single head: {single_head_params:,} parameters")
print(f"  Multi-head:  {multi_head_params:,} parameters")
print(f"  They are the SAME — multi-head splits the representation, doesn't add params!")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: From Scratch — Multi-Head Attention
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Multi-Head Attention from Scratch")
print("=" * 70)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q,K,V: (..., seq, d_k). Returns (output, weights)."""
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


class MultiHeadAttention:
    """
    Multi-head attention from scratch.

    Splits d_model into h heads, each of dimension d_k = d_model // h.
    Runs attention independently in each head's subspace.
    Concatenates and projects back to d_model.
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.h = num_heads
        self.d_k = d_model // num_heads  # dimension per head

        # Weight matrices — initialized with Xavier/Glorot scaling
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale  # (d_model, h*d_k)
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale  # output projection

    def split_heads(self, X, batch_size, seq_len):
        """
        Reshape from (batch, seq, d_model) → (batch, h, seq, d_k).

        This is the key operation that creates h parallel attention computations.
        Each head gets a slice of the projected representation.
        """
        # X: (batch, seq, d_model)  →  (batch, seq, h, d_k)
        X = X.reshape(batch_size, seq_len, self.h, self.d_k)
        # (batch, seq, h, d_k)  →  (batch, h, seq, d_k)
        return X.transpose(0, 2, 1, 3)

    def forward(self, Q_in, K_in, V_in, mask=None):
        """
        Full multi-head attention forward pass.

        Args:
            Q_in: (batch, seq_q, d_model)
            K_in: (batch, seq_k, d_model)
            V_in: (batch, seq_v, d_model)
            mask: Optional (batch, 1, seq_q, seq_k) or (seq_q, seq_k)

        Returns:
            output:       (batch, seq_q, d_model)
            attn_weights: (batch, h, seq_q, seq_k)  — one per head
        """
        batch_size, seq_q = Q_in.shape[:2]
        seq_k = K_in.shape[1]

        # Step 1: Linear projections — map d_model → d_model (h separate d_k-dim spaces)
        Q = Q_in @ self.W_Q   # (batch, seq_q, d_model)
        K = K_in @ self.W_K   # (batch, seq_k, d_model)
        V = V_in @ self.W_V   # (batch, seq_v, d_model)

        # Step 2: Split into h heads — (batch, h, seq, d_k)
        Q = self.split_heads(Q, batch_size, seq_q)
        K = self.split_heads(K, batch_size, seq_k)
        V = self.split_heads(V, batch_size, seq_k)

        # Step 3: Attention in each head independently — (batch, h, seq_q, d_k)
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Step 4: Concatenate heads — (batch, seq_q, h*d_k) = (batch, seq_q, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3)          # (batch, seq_q, h, d_k)
        attn_out = attn_out.reshape(batch_size, seq_q, self.d_model)

        # Step 5: Final linear projection
        output = attn_out @ self.W_O                        # (batch, seq_q, d_model)

        return output, attn_weights


# ── Trace through a forward pass ────────────────────────────────────────
print("Tracing through Multi-Head Attention forward pass:")
print()

np.random.seed(42)
BATCH = 2
SEQ = 6
D_MODEL = 64
N_HEADS = 4
D_K_H = D_MODEL // N_HEADS  # 16

# Simulate token embeddings
X = np.random.randn(BATCH, SEQ, D_MODEL)
mha = MultiHeadAttention(d_model=D_MODEL, num_heads=N_HEADS)

print(f"  Input:        X.shape = {X.shape}  (batch={BATCH}, seq={SEQ}, d_model={D_MODEL})")
print(f"  Heads:        h={N_HEADS}, d_k per head={D_K_H}")
print()

# Manually trace
Q_proj = X @ mha.W_Q
K_proj = X @ mha.W_K
V_proj = X @ mha.W_V
print(f"  After W_Q projection: Q_proj.shape = {Q_proj.shape}")

Q_split = mha.split_heads(Q_proj, BATCH, SEQ)
print(f"  After split_heads:    Q_split.shape = {Q_split.shape}  (batch, h, seq, d_k)")

output, attn_weights = mha.forward(X, X, X)  # self-attention
print(f"  After attention:      attn_weights.shape = {attn_weights.shape}  (batch, h, seq_q, seq_k)")
print(f"  After concat+W_O:     output.shape = {output.shape}")
print()
print(f"  Input shape == Output shape: {X.shape == output.shape} ✓")
print(f"  (Multi-head attention is shape-preserving — key for residual connections)")

# ── Parameter counting ────────────────────────────────────────────────────
print("\n" + "-" * 50)
print("Parameter count breakdown:")
W_Q_params = D_MODEL * D_MODEL
W_K_params = D_MODEL * D_MODEL
W_V_params = D_MODEL * D_MODEL
W_O_params = D_MODEL * D_MODEL
total = W_Q_params + W_K_params + W_V_params + W_O_params
print(f"  W_Q: {W_Q_params:,}  ({D_MODEL}×{D_MODEL})")
print(f"  W_K: {W_K_params:,}  ({D_MODEL}×{D_MODEL})")
print(f"  W_V: {W_V_params:,}  ({D_MODEL}×{D_MODEL})")
print(f"  W_O: {W_O_params:,}  ({D_MODEL}×{D_MODEL})")
print(f"  Total: {total:,} parameters")
print(f"  Each head effectively has {total // N_HEADS:,} params allocated to it")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Head Specialization — What Do Different Heads Learn?
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Head Specialization in Trained Models")
print("=" * 70)

print("""
Research (Voita et al. 2019, Clark et al. 2019) shows that in trained models,
different BERT/Transformer heads learn to attend to different patterns:

  Head types found through probing and visualization:

  "Syntactic heads":   verb → subject, noun → modifier
  "Positional heads":  each token attends to [prev token] or [next token]
  "Coreference heads": pronouns attend to their antecedents
  "Rare word heads":   infrequent words attend to [CLS] (global context)
  "Separator heads":   attend to [SEP] or [PAD] (background attention)

  In GPT/GPT-2:
  "Induction heads":   learn to copy patterns: if "... A B ... A" then predict B
  (mechanistic interpretability research — Olsson et al. 2022)
""")

# Simulate 4 heads with different specializations for visualization
WORDS = ["The", "hungry", "cat", "chased", "the", "small", "mouse"]
N_W = len(WORDS)

# Simulated plausible head patterns (would come from a trained model)
HEAD_PATTERNS = {
    "Head 1\n(Syntactic)": np.array([
        [0.6, 0.1, 0.2, 0.0, 0.05, 0.0, 0.05],
        [0.1, 0.5, 0.3, 0.05, 0.0, 0.0, 0.05],
        [0.15, 0.25, 0.4, 0.1, 0.05, 0.0, 0.05],
        [0.05, 0.05, 0.35, 0.35, 0.05, 0.05, 0.1],
        [0.05, 0.0, 0.05, 0.0, 0.55, 0.2, 0.15],
        [0.0, 0.0, 0.0, 0.05, 0.2, 0.55, 0.2],
        [0.05, 0.05, 0.1, 0.15, 0.1, 0.25, 0.3],
    ]),
    "Head 2\n(Positional)": np.array([
        [0.7, 0.25, 0.05, 0.0, 0.0, 0.0, 0.0],
        [0.3, 0.5, 0.2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.3, 0.5, 0.2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.5, 0.15, 0.05, 0.0],
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.2, 0.05],
        [0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.25],
        [0.0, 0.0, 0.0, 0.0, 0.05, 0.3, 0.65],
    ]),
    "Head 3\n(Noun Phrase)": np.array([
        [0.4, 0.05, 0.4, 0.05, 0.05, 0.0, 0.05],
        [0.1, 0.3, 0.55, 0.0, 0.0, 0.0, 0.05],
        [0.25, 0.15, 0.5, 0.05, 0.0, 0.0, 0.05],
        [0.05, 0.05, 0.15, 0.5, 0.05, 0.05, 0.15],
        [0.05, 0.0, 0.05, 0.0, 0.35, 0.2, 0.35],
        [0.0, 0.0, 0.0, 0.05, 0.15, 0.4, 0.4],
        [0.05, 0.05, 0.1, 0.1, 0.2, 0.2, 0.3],
    ]),
    "Head 4\n(Verb-Object)": np.array([
        [0.5, 0.1, 0.15, 0.1, 0.05, 0.05, 0.05],
        [0.1, 0.4, 0.25, 0.1, 0.0, 0.1, 0.05],
        [0.1, 0.1, 0.35, 0.3, 0.05, 0.0, 0.1],
        [0.05, 0.05, 0.2, 0.3, 0.05, 0.05, 0.3],
        [0.05, 0.0, 0.05, 0.1, 0.4, 0.1, 0.3],
        [0.0, 0.05, 0.0, 0.05, 0.15, 0.45, 0.3],
        [0.05, 0.05, 0.15, 0.35, 0.1, 0.15, 0.15],
    ]),
}

print("Simulated head specializations:")
for head_name, pattern in HEAD_PATTERNS.items():
    # Check what each head attends to most on average
    avg_weight = pattern.mean(axis=0)
    dominant = WORDS[np.argmax(avg_weight)]
    print(f"  {head_name.replace(chr(10), ' ')}: average dominant attention → '{dominant}'")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Computational Complexity
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Computational Complexity")
print("=" * 70)

print("""
Attention complexity:

  Self-attention:   O(n² × d)   — quadratic in sequence length n
  RNN (LSTM):       O(n × d²)   — linear in n, quadratic in d

  For typical NLP (n=512, d=768):
    Attention: 512² × 768 = 201M operations per layer
    LSTM:      512  × 768² = 302M operations per layer

  ✓ Attention is faster for typical NLP sequence lengths
  ✗ Attention becomes bottleneck for very long sequences (n > 2000)
    → Motivation for efficient transformers: Longformer, BigBird, FlashAttention

  Parallelism advantage:
  • RNN: must process tokens SEQUENTIALLY (h_t depends on h_{t-1})
  • Attention: ALL queries computed SIMULTANEOUSLY → GPU-friendly!
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 5: Generating Visualizations")
print("=" * 70)

# ── Visualization 1: Multi-Head Architecture Diagram ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Multi-Head Attention", fontsize=14, fontweight="bold")

# 1a: Architecture flow
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("Architecture: h=4 heads, d_model=512, d_k=128 per head",
             fontsize=10, fontweight="bold")

# Input
ax.text(5, 10.5, "Input X  (batch, seq, 512)", ha="center", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#3498db", alpha=0.8), color="white")

# Linear projections
proj_cols = [1.5, 3.5, 5.5, 7.5, 9.5]
proj_labels = ["W_Q", "W_K", "W_V"]
for i, (lbl, x) in enumerate(zip(proj_labels * 4, [1.5, 3.5, 5.5, 1.5, 3.5, 5.5, 1.5, 3.5, 5.5, 1.5, 3.5, 5.5])):
    pass  # skip complex layout, use simpler boxes

# Simpler diagram: input → 4 parallel heads → concat → output
HEAD_COLORS = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
for i, (head_name, color) in enumerate(zip(["Head 1", "Head 2", "Head 3", "Head 4"], HEAD_COLORS)):
    x = 1.0 + i * 2.2
    # head box
    rect = mpatches.FancyBboxPatch((x - 0.8, 5.5), 1.6, 1.8,
                                    boxstyle="round,pad=0.1", facecolor=color, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, 6.4, head_name, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    # proj label
    ax.text(x, 5.1, f"W_Q, W_K, W_V\n({512}→{128})", ha="center", fontsize=7, color=color)
    # arrow from input to head
    ax.annotate("", xy=(x, 7.3), xytext=(5, 10.2),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color, alpha=0.6))
    # arrow from head to concat
    ax.annotate("", xy=(5, 3.8), xytext=(x, 5.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=color, alpha=0.6))

# Concat box
rect_concat = mpatches.FancyBboxPatch((2.5, 2.5), 5.0, 1.2,
                                       boxstyle="round,pad=0.1", facecolor="#e74c3c", alpha=0.8)
ax.add_patch(rect_concat)
ax.text(5, 3.1, "Concat  (batch, seq, 4×128 = 512)", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white")

# Output projection
ax.annotate("", xy=(5, 1.8), xytext=(5, 2.5),
            arrowprops=dict(arrowstyle="->", lw=2, color="#7f8c8d"))

rect_out = mpatches.FancyBboxPatch((2.5, 0.8), 5.0, 1.0,
                                    boxstyle="round,pad=0.1", facecolor="#1abc9c", alpha=0.8)
ax.add_patch(rect_out)
ax.text(5, 1.3, "W_O projection  (batch, seq, 512)", ha="center", va="center",
        fontsize=9, fontweight="bold", color="white")
ax.text(5, 0.4, "Output (same shape as input)", ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#1abc9c"))

# 1b: Single vs multi-head comparison
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("Single Head vs Multi-Head\nRepresentation Space", fontsize=11, fontweight="bold")

# Single head — one large space
rect_s = mpatches.FancyBboxPatch((0.5, 6.5), 8.5, 2.5,
                                   boxstyle="round,pad=0.1", facecolor="#e74c3c", alpha=0.5)
ax.add_patch(rect_s)
ax.text(4.75, 7.75, "Single Head — d_model=512\nMust capture ALL relationship types at once",
        ha="center", va="center", fontsize=10)
ax.text(0.3, 9.2, "SINGLE HEAD", fontsize=10, fontweight="bold", color="#e74c3c")

# Multi-head — four smaller, specialized spaces
colors_mh = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
labels_mh = ["Syntactic\n(d_k=128)", "Positional\n(d_k=128)",
              "Coreference\n(d_k=128)", "Semantic\n(d_k=128)"]
ax.text(0.3, 5.7, "MULTI-HEAD", fontsize=10, fontweight="bold", color="#2c3e50")
for i, (col, lbl) in enumerate(zip(colors_mh, labels_mh)):
    x = 0.5 + i * 2.3
    rect_m = mpatches.FancyBboxPatch((x, 3.2), 1.9, 2.2,
                                      boxstyle="round,pad=0.1", facecolor=col, alpha=0.7)
    ax.add_patch(rect_m)
    ax.text(x + 0.95, 4.3, lbl, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")

ax.text(4.75, 2.7, "Each head specializes → richer representations", ha="center",
        fontsize=9, style="italic")

# Benefits list
benefits = [
    "✓ Captures multiple relationship types in parallel",
    "✓ Same total parameters as single head",
    "✓ Each subspace can specialize independently",
    "✓ Ensemble-like effect → more robust representations",
]
for i, b in enumerate(benefits):
    ax.text(0.5, 2.0 - i * 0.45, b, fontsize=9, color="#2c3e50")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_multi_head_architecture.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_multi_head_architecture.png")

# ── Visualization 2: Head Specialization Heatmaps ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Attention Head Specialization\n(simulated — real patterns from BERT probing studies)",
             fontsize=13, fontweight="bold")

short_words = [w[:4] for w in WORDS]
for ax, (head_name, pattern) in zip(axes.flat, HEAD_PATTERNS.items()):
    im = ax.imshow(pattern, cmap="YlOrRd", vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(N_W))
    ax.set_yticks(range(N_W))
    ax.set_xticklabels(short_words, fontsize=9)
    ax.set_yticklabels(WORDS, fontsize=9)
    ax.set_title(head_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Attending TO", fontsize=9)
    ax.set_ylabel("Attending FROM", fontsize=9)
    for i in range(N_W):
        for j in range(N_W):
            val = pattern[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.4 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_head_specialization.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_head_specialization.png")

# ── Visualization 3: Computational comparison ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Multi-Head Attention — Computational Properties", fontsize=13, fontweight="bold")

# 3a: Complexity vs sequence length
ax = axes[0]
seq_lens = np.arange(64, 2049, 64)
d = 768
attn_cost = seq_lens ** 2 * d / 1e9         # O(n² d)
rnn_cost = seq_lens * d ** 2 / 1e9          # O(n d²)
ax.plot(seq_lens, attn_cost, color="#3498db", linewidth=2.5, label="Attention O(n²d)")
ax.plot(seq_lens, rnn_cost, color="#e74c3c", linewidth=2.5, label="RNN O(nd²)")
ax.axvline(512, color="gray", linestyle="--", alpha=0.7, label="n=512 (typical NLP)")
ax.set_title("Compute Cost vs Sequence Length\n(d=768)", fontsize=11, fontweight="bold")
ax.set_xlabel("Sequence Length n")
ax.set_ylabel("GFLOPs (approx)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 3b: Effect of number of heads on diversity (entropy of attention)
ax = axes[1]
np.random.seed(7)
n_heads_list = [1, 2, 4, 8, 16]
entropies = []
for nh in n_heads_list:
    dk_h = 64 // nh if 64 >= nh else 4
    # simulate attention weights for nh heads
    head_entropies = []
    for _ in range(nh):
        w = softmax(np.random.randn(8))
        ent = -np.sum(w * np.log(w + 1e-9))
        head_entropies.append(ent)
    entropies.append(np.mean(head_entropies))

ax.bar(range(len(n_heads_list)), entropies, color="#9b59b6", alpha=0.8, edgecolor="white", linewidth=2)
ax.set_xticks(range(len(n_heads_list)))
ax.set_xticklabels([f"h={n}" for n in n_heads_list])
ax.set_title("Average Attention Entropy by Head Count\n(higher = more distributed attention)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Shannon Entropy")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for i, e in enumerate(entropies):
    ax.text(i, e + 0.01, f"{e:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_computational_properties.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_computational_properties.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Multi-Head Attention")
print("=" * 70)
print(f"""
What we covered:
  ✓ Motivation: multiple relationship types need multiple subspaces
  ✓ Multi-head attention: h parallel attention functions
  ✓ Each head: projects to d_k={D_K_H}, attends, concatenates back to d_model={D_MODEL}
  ✓ Same parameter count as single head — efficiency from parallelism
  ✓ Head specialization: syntactic, positional, coreference, semantic
  ✓ O(n²d) complexity — better than RNN for typical NLP lengths

Key formula:
  MultiHead(Q,K,V) = Concat(head_1,...,head_h) × W_O
  head_i = Attention(Q×W_Q_i, K×W_K_i, V×W_V_i)

Implementation shape trace:
  Input:    (batch={BATCH}, seq={SEQ}, d_model={D_MODEL})
  After W_Q: (batch, seq, d_model) → split to (batch, h={N_HEADS}, seq, d_k={D_K_H})
  Attention: (batch, h, seq, d_k)
  Concat:   (batch, seq, h*d_k={D_MODEL})
  Output:   (batch, seq, d_model={D_MODEL}) ← same as input ✓

Next: 03_positional_encoding.py — how Transformers track word order
  without recurrence using sinusoidal functions.

Visualizations saved to: visuals/02_multi_head_attention/
  01_multi_head_architecture.png  — architecture diagram + single vs multi comparison
  02_head_specialization.png      — 4-head attention pattern heatmaps
  03_computational_properties.png — complexity + entropy analysis
""")
