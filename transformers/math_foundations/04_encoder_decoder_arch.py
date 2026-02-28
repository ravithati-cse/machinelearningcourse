"""
Encoder-Decoder Architecture — The Full Transformer
====================================================

Learning Objectives:
  1. Build a complete Transformer encoder block from scratch (numpy)
  2. Build a complete Transformer decoder block with cross-attention
  3. Understand Layer Normalization and why it's preferred over Batch Norm in NLP
  4. Implement the Position-wise Feed-Forward Network (FFN)
  5. Understand residual connections: why they are critical for training deep models
  6. Compare Pre-LN vs Post-LN (modern vs original Transformer)

YouTube: Search "Transformer Architecture Encoder Decoder" for companion videos
Time: ~40 minutes | Difficulty: Intermediate | Prerequisites: 01–03 math foundations
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "04_encoder_decoder_arch")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("ENCODER-DECODER ARCHITECTURE — The Full Transformer")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: The Transformer Blueprint
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: The Transformer Blueprint")
print("=" * 70)

print("""
The original Transformer (Vaswani 2017) has two stacks:

  ENCODER (N=6 layers):
  ┌─────────────────────────────────────────────┐
  │  Input Embeddings + Positional Encoding     │
  │  ↓                                          │
  │  [× N]  ┌──────────────────────────────┐   │
  │         │ Multi-Head Self-Attention     │   │
  │         │ + Add & LayerNorm            │   │
  │         │ ↓                            │   │
  │         │ Feed-Forward Network (FFN)   │   │
  │         │ + Add & LayerNorm            │   │
  │         └──────────────────────────────┘   │
  │  → Encoder Output (batch, seq, d_model)     │
  └─────────────────────────────────────────────┘

  DECODER (N=6 layers):
  ┌─────────────────────────────────────────────┐
  │  Output Embeddings + Positional Encoding    │
  │  ↓                                          │
  │  [× N]  ┌──────────────────────────────┐   │
  │         │ Masked Multi-Head Self-Attn  │   │
  │         │ + Add & LayerNorm            │   │
  │         │ ↓                            │   │
  │         │ Multi-Head Cross-Attention   │   │  ← K,V from encoder
  │         │ + Add & LayerNorm            │   │
  │         │ ↓                            │   │
  │         │ Feed-Forward Network (FFN)   │   │
  │         │ + Add & LayerNorm            │   │
  │         └──────────────────────────────┘   │
  │  → Linear + Softmax → Vocab distribution    │
  └─────────────────────────────────────────────┘

Modern encoder-only (BERT) and decoder-only (GPT) are simplifications.
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Layer Normalization — The Crucial Normalizer
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 2: Layer Normalization")
print("=" * 70)

print("""
Why NOT Batch Normalization for NLP?
  BatchNorm normalizes across the BATCH dimension.
  In NLP, batch sizes are small and sequences have variable lengths.
  → Statistics are noisy and inconsistent.

Layer Normalization (Ba et al. 2016):
  Normalizes across the FEATURE dimension (d_model) for EACH token independently.

  LayerNorm(x) = γ × (x - μ) / (σ + ε) + β

  where:
    μ = mean of x across d_model      (per token, per layer)
    σ = std of x across d_model
    γ, β = learnable scale and shift (shape: d_model)
    ε = small constant for numerical stability (1e-6)

  Key properties:
    ✓ Independent per sample — no batch statistics needed
    ✓ Works with any batch size, including batch_size=1
    ✓ Stable for variable-length sequences
    ✓ Critical for training stability in deep Transformers (24-96 layers)
""")


class LayerNorm:
    """Layer Normalization from scratch."""

    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)   # scale — initialized to 1
        self.beta = np.zeros(d_model)   # shift — initialized to 0
        self.eps = eps

    def forward(self, x):
        """
        x: (..., d_model)
        Normalizes across last dimension (d_model) independently per token.
        """
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + self.eps)
        return self.gamma * x_norm + self.beta


# Demo
np.random.seed(42)
x_demo = np.random.randn(2, 4, 16)  # (batch, seq, d_model)
ln = LayerNorm(16)
x_normed = ln.forward(x_demo)
print("LayerNorm demo:")
print(f"  Input:  mean={x_demo[0, 0].mean():.3f}  std={x_demo[0, 0].std():.3f}")
print(f"  Output: mean={x_normed[0, 0].mean():.6f}  std={x_normed[0, 0].std():.6f}")
print(f"  Output is normalized per token independently ✓")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Position-wise Feed-Forward Network (FFN)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Position-wise Feed-Forward Network")
print("=" * 70)

print("""
After attention collects information from other positions,
the FFN processes each position INDEPENDENTLY:

  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

  W_1: (d_model, d_ff)     d_ff = 4 × d_model  (typically)
  W_2: (d_ff, d_model)

  For BERT (d_model=768):   d_ff = 3072  (4×)
  For GPT-2 (d_model=768):  d_ff = 3072  (4×)
  For GPT-3 (d_model=12288): d_ff = 49152 (4×)

Why the 4× expansion?
  → Creates a "thinking" step: expand to higher dim, process, compress back
  → Analogous to the hidden layer in an MLP
  → This is where ~2/3 of Transformer parameters live!

Modern FFN variants:
  GELU (GPT-2, BERT): 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
  SwiGLU (LLaMA):     x × sigmoid(x) — gated variant, often better
""")


def gelu(x):
    """GELU activation — used in BERT and GPT-2."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class FeedForward:
    """Position-wise Feed-Forward Network from scratch."""

    def __init__(self, d_model, d_ff, activation="relu"):
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
        self.activation = activation

    def forward(self, x):
        """x: (batch, seq, d_model) → (batch, seq, d_model)"""
        h = x @ self.W1 + self.b1                            # (batch, seq, d_ff)
        if self.activation == "relu":
            h = np.maximum(0, h)
        elif self.activation == "gelu":
            h = gelu(h)
        return h @ self.W2 + self.b2                         # (batch, seq, d_model)


np.random.seed(0)
d_model_demo = 32
d_ff_demo = 128
ffn = FeedForward(d_model_demo, d_ff_demo, activation="relu")
x_ffn = np.random.randn(2, 6, d_model_demo)
y_ffn = ffn.forward(x_ffn)
print(f"FFN demo: input {x_ffn.shape} → output {y_ffn.shape} (shape preserved ✓)")
print(f"  Parameters: W1={ffn.W1.shape}  W2={ffn.W2.shape}")
print(f"  Total FFN params: {ffn.W1.size + ffn.W2.size + ffn.b1.size + ffn.b2.size:,}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Residual Connections — Why Deep Networks Work
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Residual Connections")
print("=" * 70)

print("""
The "Add & Norm" in each transformer block is crucial:

  x_new = LayerNorm(x + Sublayer(x))   ← POST-LN (original)
  x_new = x + Sublayer(LayerNorm(x))   ← PRE-LN  (modern, more stable)

  The residual connection (x + ...) allows:

  1. Gradient highway:
     ∂L/∂x_early = ∂L/∂x_late × ∏ (1 + ∂sublayer/∂x)
     The "+1" means gradients can flow unchanged through many layers!
     Without residuals: gradients vanish through 24+ layers.

  2. Identity shortcut:
     If sublayer is unhelpful, it can learn to output ~0
     → x_new ≈ x  (identity function)
     → Makes optimization easier: network only needs to learn residuals

  3. Ensemble interpretation:
     A deep residual network behaves like an ensemble of many shallow networks
     (each path through the network is a different sub-network)

Empirical evidence:
  ResNet (He 2016): added residuals → went from 19 to 152 layers
  Transformers:     residuals allow 96-layer GPT-3, 48-layer BERT-Large
""")


def residual_add(x, sublayer_output):
    """Add residual connection."""
    return x + sublayer_output

# Demonstrate gradient flow benefit numerically
print("Gradient flow comparison (10-layer network):")
n_layers = 10
grad_no_residual = 1.0
grad_with_residual = 1.0
weight_grad = 0.7  # simulated gradient magnitude per layer

for layer in range(n_layers):
    grad_no_residual *= weight_grad          # multiplied by weight gradient each layer
    grad_with_residual *= (1 + weight_grad)  # residual adds 1 to gradient

print(f"  No residuals:   gradient after {n_layers} layers = {grad_no_residual:.6f} (vanished!)")
print(f"  With residuals: gradient after {n_layers} layers = {grad_with_residual:.2f} (preserved!)")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Complete Transformer Encoder Block
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Complete Transformer Encoder Block")
print("=" * 70)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    w = softmax(scores, axis=-1)
    return w @ V, w


class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.h = num_heads
        self.d_k = d_model // num_heads
        self.d_model = d_model
        scale = np.sqrt(2.0 / (d_model + self.d_k))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def forward(self, Q, K, V, mask=None):
        B, Sq = Q.shape[:2]
        Sk = K.shape[1]
        Qp = (Q @ self.W_Q).reshape(B, Sq, self.h, self.d_k).transpose(0, 2, 1, 3)
        Kp = (K @ self.W_K).reshape(B, Sk, self.h, self.d_k).transpose(0, 2, 1, 3)
        Vp = (V @ self.W_V).reshape(B, Sk, self.h, self.d_k).transpose(0, 2, 1, 3)
        out, w = scaled_dot_product_attention(Qp, Kp, Vp, mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, Sq, self.d_model)
        return out @ self.W_O, w


class EncoderBlock:
    """
    Single Transformer Encoder Block (Pre-LN variant — more stable).

    x → LayerNorm → MultiHeadSelfAttention → Add residual
      → LayerNorm → FeedForward → Add residual
      → output
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff, activation="relu")
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Pre-LN forward pass (more numerically stable than original post-LN).

        x: (batch, seq, d_model)
        """
        # Sub-layer 1: Self-attention with residual
        x_norm = self.ln1.forward(x)
        attn_out, attn_weights = self.mha.forward(x_norm, x_norm, x_norm, mask)
        x = x + attn_out                    # residual connection

        # Sub-layer 2: FFN with residual
        x_norm = self.ln2.forward(x)
        ffn_out = self.ffn.forward(x_norm)
        x = x + ffn_out                     # residual connection

        return x, attn_weights


class TransformerEncoder:
    """Stack of N EncoderBlocks."""

    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.final_ln = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """x: (batch, seq, d_model)"""
        all_attn_weights = []
        for layer in self.layers:
            x, weights = layer.forward(x, mask)
            all_attn_weights.append(weights)
        x = self.final_ln.forward(x)
        return x, all_attn_weights


# Run a forward pass
np.random.seed(42)
BATCH, SEQ, D_MODEL, N_HEADS, D_FF, N_LAYERS = 2, 8, 32, 4, 128, 3

print(f"Transformer Encoder config:")
print(f"  d_model={D_MODEL}, heads={N_HEADS}, d_ff={D_FF}, layers={N_LAYERS}")
print(f"  d_k per head = {D_MODEL // N_HEADS}")
print()

encoder_input = np.random.randn(BATCH, SEQ, D_MODEL)
encoder = TransformerEncoder(D_MODEL, N_HEADS, D_FF, N_LAYERS)
encoder_output, all_weights = encoder.forward(encoder_input)

print(f"Input  shape: {encoder_input.shape}")
print(f"Output shape: {encoder_output.shape}  (shape preserved ✓)")
print(f"  {N_LAYERS} attention weight tensors, each: {all_weights[0].shape}  (batch, heads, seq, seq)")

# Parameter count
mha_params = 4 * D_MODEL * D_MODEL  # W_Q, W_K, W_V, W_O
ffn_params = D_MODEL * D_FF + D_FF * D_MODEL + D_FF + D_MODEL  # W1, W2, b1, b2
ln_params = 2 * 2 * D_MODEL  # 2 LayerNorms, each has gamma+beta
block_params = mha_params + ffn_params + ln_params
total_params = block_params * N_LAYERS + 2 * D_MODEL  # final LN
print(f"\nParameter count per encoder block:")
print(f"  MHA (W_Q+W_K+W_V+W_O): {mha_params:,}")
print(f"  FFN (W1+W2+b1+b2):     {ffn_params:,}")
print(f"  LayerNorms:            {ln_params:,}")
print(f"  Per-block total:       {block_params:,}")
print(f"  Full encoder ({N_LAYERS} layers): {total_params:,}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Decoder Block
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Transformer Decoder Block")
print("=" * 70)

print("""
The decoder block adds ONE extra sub-layer: Cross-Attention.

  Decoder Block:
  1. Masked Self-Attention (causal — can't see future tokens)
     + Add & LayerNorm
  2. Cross-Attention (Q=decoder, K,V=encoder output)
     + Add & LayerNorm
  3. Feed-Forward Network
     + Add & LayerNorm

  The cross-attention is what connects encoder to decoder:
    Q: from current decoder state (what the decoder is generating)
    K, V: from encoder output (the full input encoding)
    → Decoder "reads" the encoder representation at each generation step
""")


def make_causal_mask(seq_len):
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


class DecoderBlock:
    """
    Single Transformer Decoder Block.
    Three sub-layers: masked self-attn, cross-attn, FFN.
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        x:          (batch, tgt_seq, d_model)  — decoder input
        enc_output: (batch, src_seq, d_model)  — encoder output
        src_mask:   padding mask for source
        tgt_mask:   causal mask for target
        """
        # 1. Masked self-attention (causal)
        xn = self.ln1.forward(x)
        sa_out, _ = self.self_attn.forward(xn, xn, xn, tgt_mask)
        x = x + sa_out

        # 2. Cross-attention: Q from decoder, K/V from encoder
        xn = self.ln2.forward(x)
        ca_out, cross_weights = self.cross_attn.forward(xn, enc_output, enc_output, src_mask)
        x = x + ca_out

        # 3. FFN
        xn = self.ln3.forward(x)
        x = x + self.ffn.forward(xn)

        return x, cross_weights


# Run decoder forward pass
SRC_SEQ = 8   # encoder sequence length
TGT_SEQ = 5   # decoder sequence length (tokens generated so far)

dec_block = DecoderBlock(D_MODEL, N_HEADS, D_FF)
decoder_input = np.random.randn(BATCH, TGT_SEQ, D_MODEL)
causal_mask = make_causal_mask(TGT_SEQ)[np.newaxis, np.newaxis, :, :]  # (1, 1, tgt, tgt)

dec_output, cross_attn_weights = dec_block.forward(
    decoder_input, encoder_output, tgt_mask=causal_mask
)
print(f"Decoder Block:")
print(f"  Decoder input:  {decoder_input.shape}")
print(f"  Encoder output: {encoder_output.shape}")
print(f"  Decoder output: {dec_output.shape}")
print(f"  Cross-attn weights: {cross_attn_weights.shape}  (batch, heads, tgt_seq, src_seq)")
print()
print("  Each decoder position attends to ALL encoder positions via cross-attention ✓")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Pre-LN vs Post-LN
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: Pre-LN vs Post-LN")
print("=" * 70)

print("""
Original Transformer (Post-LN):
  x = LayerNorm(x + Sublayer(x))
  → Normalizes AFTER the residual addition
  → Original paper uses this
  → Requires careful learning rate warmup (often unstable early in training)

Modern Transformer (Pre-LN, used in GPT-2, T5, BERT variants):
  x = x + Sublayer(LayerNorm(x))
  → Normalizes BEFORE the sublayer
  → Gradient magnitude at each layer is bounded by LayerNorm
  → More stable training — often no warmup needed
  → Slightly weaker final performance in some settings, but more reliable

  GPT-2, PaLM, LLaMA all use Pre-LN.

Rule of thumb:
  Training from scratch → Pre-LN (stability)
  Fine-tuning BERT → Post-LN (matching pretraining convention)
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 8: Generating Visualizations")
print("=" * 70)

# ── Visualization 1: Full Architecture Diagram ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 12))
fig.suptitle("Transformer Encoder-Decoder Architecture", fontsize=14, fontweight="bold")

def draw_block(ax, x, y, w, h, label, color, sublabels=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    facecolor=color, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white")
    if sublabels:
        ax.text(x + w / 2, y - 0.25, sublabels, ha="center", va="top",
                fontsize=7, color=color, style="italic")

# Encoder column
ax = axes[0]
ax.set_xlim(0, 6)
ax.set_ylim(0, 16)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("ENCODER (×6 layers)", fontsize=12, fontweight="bold", color="#3498db")

# Input
draw_block(ax, 1.0, 14.5, 4.0, 0.9, "Input Embeddings + Positional Encoding", "#7f8c8d")
ax.annotate("", xy=(3.0, 14.5), xytext=(3.0, 14.0), arrowprops=dict(arrowstyle="->", lw=2))

# Encoder layer components (repeated N times)
components_enc = [
    (13.2, "Multi-Head\nSelf-Attention", "#3498db"),
    (11.4, "Add & LayerNorm", "#95a5a6"),
    (10.2, "Feed-Forward\nNetwork (FFN)", "#2ecc71"),
    (8.4, "Add & LayerNorm", "#95a5a6"),
]
for y_pos, label, color in components_enc:
    draw_block(ax, 1.0, y_pos, 4.0, 1.0, label, color)
    if y_pos > 8.4:
        ax.annotate("", xy=(3.0, y_pos), xytext=(3.0, y_pos + 1.0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#7f8c8d"))

# Repeat bracket
rect_repeat = mpatches.FancyBboxPatch((0.3, 8.2), 5.4, 6.2,
                                       boxstyle="round,pad=0.1",
                                       facecolor="none", edgecolor="#3498db",
                                       linewidth=2.5, linestyle="--")
ax.add_patch(rect_repeat)
ax.text(5.7, 11.3, "×N\n(N=6)", fontsize=12, fontweight="bold", color="#3498db")

ax.annotate("", xy=(3.0, 8.2), xytext=(3.0, 7.6), arrowprops=dict(arrowstyle="->", lw=2))
draw_block(ax, 1.0, 6.8, 4.0, 0.8, "Encoder Output  (batch, src_seq, d_model)", "#e74c3c")

# Decoder column
ax = axes[1]
ax.set_xlim(0, 6)
ax.set_ylim(0, 16)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("DECODER (×6 layers)", fontsize=12, fontweight="bold", color="#9b59b6")

draw_block(ax, 1.0, 14.5, 4.0, 0.9, "Output Embeddings + Positional Encoding", "#7f8c8d")
ax.annotate("", xy=(3.0, 14.5), xytext=(3.0, 14.0), arrowprops=dict(arrowstyle="->", lw=2))

components_dec = [
    (13.2, "Masked Multi-Head\nSelf-Attention", "#9b59b6"),
    (11.8, "Add & LayerNorm", "#95a5a6"),
    (10.5, "Multi-Head\nCross-Attention", "#e67e22"),
    (9.1, "Add & LayerNorm", "#95a5a6"),
    (7.8, "Feed-Forward\nNetwork (FFN)", "#2ecc71"),
    (6.4, "Add & LayerNorm", "#95a5a6"),
]
for y_pos, label, color in components_dec:
    draw_block(ax, 1.0, y_pos, 4.0, 1.0, label, color)

ax.annotate("", xy=(3.0, 13.2), xytext=(3.0, 14.0), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(3.0, 11.8), xytext=(3.0, 13.2), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(3.0, 10.5), xytext=(3.0, 11.8), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(3.0, 9.1), xytext=(3.0, 10.5), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(3.0, 7.8), xytext=(3.0, 9.1), arrowprops=dict(arrowstyle="->", lw=1.5))
ax.annotate("", xy=(3.0, 6.4), xytext=(3.0, 7.8), arrowprops=dict(arrowstyle="->", lw=1.5))

# Cross-attention arrow from encoder
ax.annotate("K,V from\nEncoder", xy=(1.0, 11.0), xytext=(-0.1, 11.0),
            fontsize=8, color="#e67e22", ha="right",
            arrowprops=dict(arrowstyle="->", color="#e67e22", lw=2))

# Repeat bracket
rect_dec = mpatches.FancyBboxPatch((0.3, 6.2), 5.4, 7.3,
                                    boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor="#9b59b6",
                                    linewidth=2.5, linestyle="--")
ax.add_patch(rect_dec)
ax.text(5.7, 9.85, "×N\n(N=6)", fontsize=12, fontweight="bold", color="#9b59b6")

ax.annotate("", xy=(3.0, 6.2), xytext=(3.0, 5.5), arrowprops=dict(arrowstyle="->", lw=2))
draw_block(ax, 1.0, 4.6, 4.0, 0.9, "Linear (d_model → vocab_size)", "#3498db")
ax.annotate("", xy=(3.0, 4.6), xytext=(3.0, 4.0), arrowprops=dict(arrowstyle="->", lw=2))
draw_block(ax, 1.0, 3.1, 4.0, 0.9, "Softmax → Token Probabilities", "#e74c3c")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_architecture_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_architecture_diagram.png")

# ── Visualization 2: LayerNorm + Residuals ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Layer Normalization & Residual Connections", fontsize=13, fontweight="bold")

# 2a: Batch vs Layer norm comparison
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f9f9f9")
ax.set_title("BatchNorm vs LayerNorm\nnormalization direction", fontsize=11, fontweight="bold")

# Draw a 3D matrix (batch × seq × feature)
batch_colors = ["#3498db", "#2ecc71", "#e74c3c"]
for b, bc in enumerate(batch_colors):
    for s in range(4):
        ax.add_patch(plt.Rectangle((1 + s * 1.5, 7 - b * 2.2), 1.2, 1.8,
                                    facecolor=bc, alpha=0.4, edgecolor=bc))
        ax.text(1.6 + s * 1.5, 7.9 - b * 2.2, f"B{b}S{s}", fontsize=7, ha="center")

ax.text(5, 9.5, "BatchNorm: normalize across BATCH (column)", ha="center", fontsize=9,
        color="#3498db", fontweight="bold")
ax.annotate("", xy=(8.5, 7.9), xytext=(8.5, 3.5),
            arrowprops=dict(arrowstyle="<->", lw=2, color="#3498db"))
ax.text(8.7, 5.7, "Batch\ndim", fontsize=8, color="#3498db")

ax.text(5, 1.8, "LayerNorm: normalize across FEATURES (row)", ha="center", fontsize=9,
        color="#e74c3c", fontweight="bold")
ax.annotate("", xy=(1.0, 7.9), xytext=(7.2, 7.9),
            arrowprops=dict(arrowstyle="<->", lw=2, color="#e74c3c"))
ax.text(3.5, 8.3, "Feature/d_model dim →", fontsize=8, color="#e74c3c")

# 2b: Residual gradient flow
ax = axes[1]
layers = np.arange(1, 13)
grad_no_res = 0.9 ** layers
grad_with_res = (1 + 0.9) ** layers / (1 + 0.9) ** 12 * 1.5
ax.semilogy(layers, grad_no_res, color="#e74c3c", linewidth=2.5, marker="o",
            markersize=5, label="Without residuals")
ax.semilogy(layers, np.ones_like(layers) * 0.15, color="#2ecc71", linewidth=2.5,
            linestyle="--", label="With residuals (approx)")
ax.set_title("Gradient Magnitude Through Layers\n(log scale)", fontsize=11, fontweight="bold")
ax.set_xlabel("Layer (from output)")
ax.set_ylabel("Gradient magnitude (log)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.text(6, 0.5, "Vanishing\ngradient", fontsize=9, color="#e74c3c", ha="center")
ax.text(6, 0.08, "Preserved\ngradient", fontsize=9, color="#2ecc71", ha="center")

# 2c: Pre-LN vs Post-LN
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#f9f9f9")
ax.set_title("Pre-LN vs Post-LN", fontsize=12, fontweight="bold")

# Post-LN (original)
post_steps = [(5, 8.5, "x", "#7f8c8d"), (2.5, 7.0, "Sublayer(x)", "#3498db"),
              (5, 5.5, "x + Sublayer(x)", "#e67e22"), (5, 4.0, "LayerNorm(·)", "#9b59b6")]
ax.text(1, 9.5, "POST-LN (original, 2017)", fontsize=9, fontweight="bold", color="#9b59b6")
for (x, y, lbl, col) in post_steps:
    ax.text(x, y, lbl, ha="center", va="center", fontsize=9, color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.8))

# Pre-LN (modern)
pre_steps = [(5, 2.5, "x", "#7f8c8d"), (2.5, 1.5, "LayerNorm(x)", "#9b59b6"),
             (2.5, 0.5, "Sublayer(LN(x))", "#3498db")]
ax.text(1, 3.5, "PRE-LN (modern GPT-2, LLaMA)", fontsize=9, fontweight="bold", color="#2ecc71")
for (x, y, lbl, col) in pre_steps:
    ax.text(x, y, lbl, ha="center", va="center", fontsize=9, color=col, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=col, alpha=0.8))
ax.text(6.5, 1.5, "x + Sublayer(LN(x))", fontsize=9, color="#2ecc71", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2ecc71", alpha=0.8))

ax.text(5, 9.0, "LayerNorm AFTER add", ha="center", fontsize=8, color="#7f8c8d")
ax.text(5, 3.0, "LayerNorm BEFORE sublayer", ha="center", fontsize=8, color="#7f8c8d")
ax.text(5, 5.5, "✓ More stable training\n✓ No warmup needed", ha="center",
        fontsize=8, color="#2ecc71",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2ecc71"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_layernorm_residuals.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_layernorm_residuals.png")

# ── Visualization 3: Cross-attention heatmap ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Cross-Attention: Decoder Reads Encoder", fontsize=13, fontweight="bold")

# Simulate cross-attention pattern for "The cat sat" → "Le chat"
src_words = ["The", "hungry", "cat", "chased", "the", "mouse", "<EOS>", "<PAD>"]
tgt_words = ["<BOS>", "Le", "chat", "a", "pourchassé"]

cross_attn_sim = np.array([
    [0.7, 0.05, 0.1, 0.05, 0.05, 0.02, 0.02, 0.01],  # <BOS> attends to "The"
    [0.6, 0.05, 0.1, 0.05, 0.1, 0.05, 0.03, 0.02],   # "Le" (the) → "The" + "the"
    [0.05, 0.2, 0.6, 0.05, 0.03, 0.04, 0.02, 0.01],  # "chat" (cat) → "hungry" + "cat"
    [0.05, 0.02, 0.03, 0.02, 0.03, 0.03, 0.01, 0.01],  # "a" (aux verb)
    [0.03, 0.05, 0.15, 0.55, 0.05, 0.1, 0.05, 0.02],  # "pourchassé" → "chased"
])
cross_attn_sim = cross_attn_sim / cross_attn_sim.sum(axis=1, keepdims=True)

ax = axes[0]
im = ax.imshow(cross_attn_sim, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.7)
ax.set_xticks(range(len(src_words)))
ax.set_yticks(range(len(tgt_words)))
ax.set_xticklabels(src_words, fontsize=9, rotation=20)
ax.set_yticklabels(tgt_words, fontsize=9)
ax.set_xlabel("Source (Encoder) tokens →", fontsize=10)
ax.set_ylabel("Target (Decoder) tokens →", fontsize=10)
ax.set_title("Cross-Attention Weights\n(EN→FR translation, simulated)", fontsize=11, fontweight="bold")
for i in range(len(tgt_words)):
    for j in range(len(src_words)):
        val = cross_attn_sim[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=7, color="white" if val > 0.4 else "black")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f9f9f9")
ax.set_title("Encoder-Decoder Information Flow", fontsize=11, fontweight="bold")

# Encoder tokens at top
for i, w in enumerate(src_words[:6]):
    ax.text(1.0 + i * 1.4, 7.0, w, ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#3498db", alpha=0.7), color="white")

# Decoder tokens at bottom
for i, w in enumerate(tgt_words):
    ax.text(1.5 + i * 1.6, 1.5, w, ha="center", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#9b59b6", alpha=0.7), color="white")

# Cross-attention arrows (strongest connections)
cross_arrows = [(2, 0, "Le→The"), (3, 2, "chat→cat"), (4, 3, "a→chased")]
for (src_i, tgt_i, lbl) in cross_arrows:
    sx = 1.0 + src_i * 1.4
    tx = 1.5 + tgt_i * 1.6
    ax.annotate("", xy=(tx, 2.0), xytext=(sx, 6.5),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#e67e22", alpha=0.8))

ax.text(5, 4.3, "Cross-attention lets each\ndecoder position read the\nfull encoder sequence",
        ha="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#e67e22"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_cross_attention.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_cross_attention.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY — Encoder-Decoder Architecture")
print("=" * 70)
print(f"""
What we covered:
  ✓ Transformer blueprint: N encoder layers + N decoder layers
  ✓ Layer Normalization: normalizes per token across d_model (not batch)
  ✓ Position-wise FFN: d_model → 4×d_model → d_model (2/3 of all params)
  ✓ Residual connections: gradient highway through deep networks
  ✓ Pre-LN (modern) vs Post-LN (original) — Pre-LN is more stable
  ✓ Encoder block: Self-Attention + FFN + 2× (Add & LN)
  ✓ Decoder block: Masked Self-Attn + Cross-Attn + FFN + 3× (Add & LN)
  ✓ Cross-attention: decoder queries encoder's output at each step

Key shapes (d_model={D_MODEL}, seq={SEQ}, batch={BATCH}):
  Encoder input/output: ({BATCH}, {SEQ}, {D_MODEL}) — shape preserved
  FFN intermediate:     ({BATCH}, {SEQ}, {D_FF})    — 4× expansion
  Cross-attn weights:   ({BATCH}, {N_HEADS}, {TGT_SEQ}, {SRC_SEQ})   — decoder reads encoder

Next: algorithms/transformer_from_scratch.py
  Build a complete working Transformer encoder for classification,
  combining all four math foundation modules into one system.

Visualizations saved to: visuals/04_encoder_decoder_arch/
  01_architecture_diagram.png  — full encoder-decoder blueprint
  02_layernorm_residuals.png   — LN comparison + gradient flow + Pre vs Post LN
  03_cross_attention.png       — cross-attention heatmap + information flow
""")
