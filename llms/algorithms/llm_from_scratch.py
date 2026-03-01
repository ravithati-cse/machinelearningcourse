"""
Build a Mini-LLM (Character-Level GPT) From Scratch
=====================================================
Learning Objectives:
  1. Implement a character-level tokenizer from scratch
  2. Build the full GPT architecture in pure NumPy (embedding + transformer blocks + LM head)
  3. Train with cross-entropy loss and stochastic gradient descent
  4. Implement temperature, top-K, and nucleus sampling for text generation
  5. Understand how scaling (depth, width, data) affects LLM capabilities
  6. Compare generation quality across different sampling strategies
  7. Visualize attention patterns and training dynamics
YouTube: Search "Andrej Karpathy build GPT from scratch nanoGPT"
Time: ~90 min | Difficulty: Expert | Prerequisites: Parts 3, 6 (Transformers)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "llm_from_scratch")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

# ===========================================================================
print("\n" + "="*70)
print("SECTION 1: CORPUS AND TOKENIZER")
print("="*70)
# ===========================================================================

CORPUS = """
machine learning trains models on data to make predictions without explicit programming.
neural networks learn representations through layers of weighted connections and activations.
transformers use self-attention to model relationships between all tokens simultaneously.
the loss function measures prediction error and gradient descent minimizes it iteratively.
language models learn to predict the next token given its preceding context window.
pre-training on large corpora enables transfer learning to downstream tasks efficiently.
fine-tuning adapts pre-trained weights to specific tasks with smaller labeled datasets.
attention weights reveal which tokens the model focuses on when making each prediction.
scaling laws show that loss decreases predictably with more parameters and training data.
large language models exhibit emergent capabilities not present in smaller model variants.
""".strip()

# Character-level tokenizer
chars = sorted(set(CORPUS))
vocab_size = len(chars)
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

def encode(text):
    return [char_to_id[c] for c in text if c in char_to_id]

def decode(ids):
    return "".join(id_to_char.get(i, "?") for i in ids)

print(f"Corpus length    : {len(CORPUS)} characters")
print(f"Unique characters: {vocab_size}")
print(f"Character set    : {repr(''.join(chars))}")

sample = "machine learning"
encoded = encode(sample)
decoded = decode(encoded)
print(f"\nTokenizer demo:")
print(f"  Input   : '{sample}'")
print(f"  Encoded : {encoded}")
print(f"  Decoded : '{decoded}'")
print(f"  Round-trip OK: {sample == decoded}")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 2: MINIGPT ARCHITECTURE (Pure NumPy)")
print("="*70)
# ===========================================================================

def embed(token_ids, W_emb):
    """Embedding lookup.
    W_emb : (vocab_size, d_model)
    token_ids : (T,) int array
    Returns : (T, d_model)
    """
    return W_emb[token_ids]

def pos_encoding(T, d_model):
    """Sinusoidal positional encoding. Returns (T, d_model)."""
    pe = np.zeros((T, d_model))
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div[:d_model // 2])
    return pe

def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalisation over the last axis."""
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def causal_attention(x, W_q, W_k, W_v, W_o):
    """Single-head causal (masked) self-attention.
    x : (T, d_model)
    Returns : (T, d_model), (T, T) attention weights
    """
    T, d = x.shape
    Q = x @ W_q          # (T, d_model)
    K = x @ W_k
    V = x @ W_v
    # Scaled dot-product with causal mask
    scores = Q @ K.T / np.sqrt(d)           # (T, T)
    mask = np.triu(np.ones((T, T)), k=1) * -1e9
    scores = scores + mask
    # Stable softmax
    scores_shifted = scores - scores.max(axis=-1, keepdims=True)
    attn = np.exp(scores_shifted)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    out = attn @ V @ W_o                    # (T, d_model)
    return out, attn

def ffn(x, W1, b1, W2, b2):
    """Position-wise feed-forward with GELU activation."""
    h = x @ W1 + b1
    # GELU approximation
    h = 0.5 * h * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h**3)))
    return h @ W2 + b2


class MiniGPT:
    """Minimal GPT: token embedding + positional encoding + N transformer blocks + LM head."""

    def __init__(self, vocab_size, d_model=32, n_layers=2, context_len=48, seed=42):
        rng = np.random.RandomState(seed)
        scale = 0.02
        self.d_model = d_model
        self.context_len = context_len
        self.n_layers = n_layers

        P = {}
        P['W_emb'] = rng.randn(vocab_size, d_model) * scale
        for l in range(n_layers):
            P[f'W_q_{l}'] = rng.randn(d_model, d_model) * scale
            P[f'W_k_{l}'] = rng.randn(d_model, d_model) * scale
            P[f'W_v_{l}'] = rng.randn(d_model, d_model) * scale
            P[f'W_o_{l}'] = rng.randn(d_model, d_model) * scale
            P[f'W1_{l}']  = rng.randn(d_model, 4 * d_model) * scale
            P[f'b1_{l}']  = np.zeros(4 * d_model)
            P[f'W2_{l}']  = rng.randn(4 * d_model, d_model) * scale
            P[f'b2_{l}']  = np.zeros(d_model)
            P[f'g1_{l}']       = np.ones(d_model)
            P[f'b_ln1_{l}']    = np.zeros(d_model)
            P[f'g2_{l}']       = np.ones(d_model)
            P[f'b_ln2_{l}']    = np.zeros(d_model)
        P['W_head']   = rng.randn(d_model, vocab_size) * scale
        P['b_head']   = np.zeros(vocab_size)
        P['g_final']  = np.ones(d_model)
        P['b_final']  = np.zeros(d_model)
        self.P = P
        self.last_attns  = []
        self.last_hidden = None

    def forward(self, token_ids):
        """Full forward pass. Returns logits (T, vocab_size)."""
        P = self.P
        T = len(token_ids)
        x = embed(token_ids, P['W_emb'])           # (T, d_model)
        x = x + pos_encoding(T, self.d_model)      # add PE

        self.last_attns = []
        for l in range(self.n_layers):
            # Pre-LN self-attention
            x_norm   = layer_norm(x, P[f'g1_{l}'], P[f'b_ln1_{l}'])
            attn_out, attn_weights = causal_attention(
                x_norm,
                P[f'W_q_{l}'], P[f'W_k_{l}'], P[f'W_v_{l}'], P[f'W_o_{l}']
            )
            x = x + attn_out
            self.last_attns.append(attn_weights)
            # Pre-LN FFN
            x_norm = layer_norm(x, P[f'g2_{l}'], P[f'b_ln2_{l}'])
            x = x + ffn(x_norm, P[f'W1_{l}'], P[f'b1_{l}'], P[f'W2_{l}'], P[f'b2_{l}'])

        x = layer_norm(x, P['g_final'], P['b_final'])
        self.last_hidden = x                        # (T, d_model) — store for backward
        logits = x @ P['W_head'] + P['b_head']     # (T, vocab_size)
        return logits

    def param_count(self):
        return sum(v.size for v in self.P.values())


# Instantiate and inspect
model = MiniGPT(vocab_size, d_model=32, n_layers=2, context_len=48)
total_params = model.param_count()

print("\nMiniGPT Architecture:")
print(f"  Vocabulary size : {vocab_size}")
print(f"  d_model         : 32")
print(f"  n_layers        : 2")
print(f"  context_len     : 48")
print(f"  Total params    : {total_params:,}")

print("\nComponent breakdown:")
print(f"  {'Component':<30} {'Params':>10}")
print(f"  {'-'*42}")
print(f"  {'Token embedding (W_emb)':<30} {model.P['W_emb'].size:>10,}")
for l in range(model.n_layers):
    attn_p = (model.P[f'W_q_{l}'].size + model.P[f'W_k_{l}'].size +
              model.P[f'W_v_{l}'].size + model.P[f'W_o_{l}'].size)
    ffn_p  = (model.P[f'W1_{l}'].size + model.P[f'b1_{l}'].size +
              model.P[f'W2_{l}'].size + model.P[f'b2_{l}'].size)
    ln_p   = (model.P[f'g1_{l}'].size + model.P[f'b_ln1_{l}'].size +
              model.P[f'g2_{l}'].size + model.P[f'b_ln2_{l}'].size)
    print(f"  {'Layer '+str(l)+' attention':<30} {attn_p:>10,}")
    print(f"  {'Layer '+str(l)+' FFN':<30} {ffn_p:>10,}")
    print(f"  {'Layer '+str(l)+' LayerNorm':<30} {ln_p:>10,}")
head_p = model.P['W_head'].size + model.P['b_head'].size
print(f"  {'LM head (W_head + b_head)':<30} {head_p:>10,}")
print(f"  {'Final LayerNorm':<30} {model.P['g_final'].size + model.P['b_final'].size:>10,}")
print(f"  {'TOTAL':<30} {total_params:>10,}")

# Quick sanity check
sample_ids = np.array(encode("machine learning"))
logits_test = model.forward(sample_ids)
print(f"\nForward pass test:")
print(f"  Input tokens  : {len(sample_ids)}")
print(f"  Output logits : {logits_test.shape}  (T x vocab_size)")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 3: TRAINING")
print("="*70)
# ===========================================================================

print("\nPreparing training sequences...")
data = encode(CORPUS)
T_ctx = 32  # context length
xs, ys = [], []
for i in range(0, len(data) - T_ctx - 1, 4):
    xs.append(data[i : i + T_ctx])
    ys.append(data[i + 1 : i + T_ctx + 1])

print(f"  Corpus tokens : {len(data)}")
print(f"  Training pairs: {len(xs)}")
print(f"  Context length: {T_ctx}")
print(f"\nTraining MiniGPT (W_head + b_head fine-tuned; base features frozen)...")
print(f"  Strategy: update only LM head — pedagogically tractable, shows real learning")

lr     = 0.05
losses = []
rng_train = np.random.RandomState(0)

for step in range(400):
    idx   = rng_train.randint(0, len(xs))
    x_tok = np.array(xs[idx])
    y_tok = np.array(ys[idx])

    # Forward
    logits = model.forward(x_tok)           # (T, vocab_size)

    # Numerically stable cross-entropy
    logits_s = logits - logits.max(axis=1, keepdims=True)
    probs    = np.exp(logits_s) / np.exp(logits_s).sum(axis=1, keepdims=True)
    T_s      = len(y_tok)
    loss     = -np.log(probs[np.arange(T_s), y_tok] + 1e-9).mean()
    losses.append(float(loss))

    # Gradient of cross-entropy w.r.t. logits
    d_logits = probs.copy()
    d_logits[np.arange(T_s), y_tok] -= 1
    d_logits /= T_s

    # Exact gradients for W_head and b_head
    # logits = last_hidden @ W_head + b_head
    # d_W_head = last_hidden.T @ d_logits   shape: (d_model, vocab_size)
    # d_b_head = d_logits.sum(0)            shape: (vocab_size,)
    x_hid  = model.last_hidden             # (T, d_model)
    d_W_head = x_hid.T @ d_logits         # (d_model, vocab_size)
    d_b_head = d_logits.sum(0)            # (vocab_size,)

    model.P['W_head'] -= lr * d_W_head
    model.P['b_head'] -= lr * d_b_head

    if step % 100 == 0:
        print(f"  Step {step:4d} | Loss: {loss:.4f}")

print(f"  Step  399 | Loss: {losses[-1]:.4f}")
print(f"\nTraining complete.")
print(f"  Initial loss : {losses[0]:.4f}  (random ≈ ln({vocab_size}) ≈ {np.log(vocab_size):.2f})")
print(f"  Final loss   : {losses[-1]:.4f}")
print(f"  Improvement  : {losses[0] - losses[-1]:.4f} nats")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 4: TEXT GENERATION")
print("="*70)
# ===========================================================================

def sample_next(logits_t, temperature=1.0, top_k=None, top_p=None):
    """Sample the next token from logits with optional temperature / top-K / top-P."""
    logits = logits_t / max(temperature, 1e-9)

    if top_k is not None:
        top_k = min(top_k, len(logits))
        threshold = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits >= threshold, logits, -1e9)

    # Softmax
    logits_shifted = logits - logits.max()
    probs = np.exp(logits_shifted)
    probs = probs / probs.sum()

    if top_p is not None:
        sorted_idx  = np.argsort(probs)[::-1]
        cumsum      = np.cumsum(probs[sorted_idx])
        # Keep tokens up to the first one that pushes cumsum above top_p
        cutoff_mask = cumsum > top_p
        if cutoff_mask.any():
            first_over = np.argmax(cutoff_mask)
            keep       = sorted_idx[:first_over + 1]
        else:
            keep = sorted_idx
        mask        = np.zeros_like(probs)
        mask[keep]  = 1.0
        probs       = probs * mask
        probs       = probs / probs.sum()

    return int(np.random.choice(len(probs), p=probs))


def generate(model, seed_text, n_tokens=80, temperature=0.8, top_k=None, top_p=None):
    """Auto-regressive character generation."""
    token_ids = encode(seed_text)
    for _ in range(n_tokens):
        ctx      = token_ids[-model.context_len:]
        logits   = model.forward(np.array(ctx))
        next_tok = sample_next(logits[-1], temperature=temperature, top_k=top_k, top_p=top_p)
        token_ids.append(next_tok)
    return decode(token_ids)


seed = "learning "
print(f"\nSeed text: '{seed}'")
print(f"Generating 80 characters with each sampling strategy...\n")

# Store outputs for visualisation
gen_outputs = {}

gen_greedy = generate(model, seed, n_tokens=80, temperature=0.01)
gen_outputs['Greedy (T=0.01)'] = gen_greedy
print(f"[Greedy (T=0.01)]")
print(f"  {gen_greedy}\n")

gen_temp = generate(model, seed, n_tokens=80, temperature=0.8)
gen_outputs['Temperature=0.8'] = gen_temp
print(f"[Temperature=0.8 — balanced]")
print(f"  {gen_temp}\n")

gen_topk = generate(model, seed, n_tokens=80, temperature=1.0, top_k=10)
gen_outputs['Top-K (k=10)'] = gen_topk
print(f"[Top-K (k=10, T=1.0)]")
print(f"  {gen_topk}\n")

gen_topp = generate(model, seed, n_tokens=80, top_p=0.9, temperature=1.0)
gen_outputs['Nucleus p=0.9'] = gen_topp
print(f"[Nucleus / Top-P (p=0.9, T=1.0)]")
print(f"  {gen_topp}\n")

# Diversity metric: distinct character ratio
def distinct_ratio(text):
    if not text:
        return 0.0
    return len(set(text)) / len(text)

print("Diversity (distinct character ratio):")
for label, text in gen_outputs.items():
    print(f"  {label:<25} : {distinct_ratio(text):.3f}")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 5: SCALING INTUITION")
print("="*70)
# ===========================================================================

print("""
Scaling table: from our toy Mini-GPT to frontier models
""")
print(f"{'Model Config':<18} | {'d_model':>7} | {'Layers':>6} | {'Heads':>5} | {'Params':>10} | Trained on")
print("-" * 80)
rows = [
    ("Mini (ours)",   32,    2,   1, "~50K",    "toy corpus (800 chars)"),
    ("GPT-2 Small",  768,   12,  12, "117M",    "40GB web text"),
    ("GPT-2 Large", 1280,   36,  20, "774M",    "40GB web text"),
    ("GPT-3",       12288,  96,  96, "175B",    "300B tokens"),
    ("LLaMA-7B",    4096,   32,  32, "7B",      "1T tokens"),
    ("LLaMA-70B",   8192,   80,  64, "70B",     "2T tokens"),
]
for (name, d, L, H, params, data_str) in rows:
    print(f"{name:<18} | {d:>7,} | {L:>6} | {H:>5} | {params:>10} | {data_str}")

print("\nChinchilla scaling rule:")
print("  optimal_tokens = 20 × parameters")
print("  e.g. 7B model → train on 140B tokens for compute-optimal training")
print("\nKey insight: scaling model size alone is insufficient — data must scale together.")
print("Loss follows a power law: L ∝ (N × D)^{-0.1} (approximately)")
print("Emergent capabilities (chain-of-thought, in-context learning) appear above ~10B params.")

# ===========================================================================
print("\n" + "="*70)
print("VISUALISATIONS")
print("="*70)
# ===========================================================================

# ------------------------------------------------------------------
# VIZ 1: Architecture diagram
# ------------------------------------------------------------------
print("\nSaving 01_architecture.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
fig.patch.set_facecolor('#0d1117')

# --- Left panel: token flow ---
ax = axes[0]
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')
ax.set_title("MiniGPT Token Flow", color='white', fontsize=13, fontweight='bold', pad=10)

boxes = [
    (5, 10.5, "Input Tokens\n['m','a','c','h',...]", '#264653', 'white', "T integers"),
    (5,  8.5, "Token Embedding\nW_emb lookup",        '#2a9d8f', 'white', f"(T, {model.d_model})"),
    (5,  6.5, "+ Positional Encoding\nsinusoidal PE", '#457b9d', 'white', f"(T, {model.d_model})"),
    (5,  4.5, "× 2 Transformer Blocks\n(Attn + FFN)",  '#e76f51', 'white', f"(T, {model.d_model})"),
    (5,  2.5, "Final LayerNorm",                       '#8ecae6', '#111', f"(T, {model.d_model})"),
    (5,  0.7, "LM Head (linear)\nW_head projection",  '#f4a261', '#111', f"(T, {vocab_size})"),
]
for (cx, cy, label, color, tc, dim) in boxes:
    fancy = FancyBboxPatch((cx - 2.5, cy - 0.7), 5.0, 1.4,
                           boxstyle="round,pad=0.1", facecolor=color, edgecolor='white', lw=1.2)
    ax.add_patch(fancy)
    ax.text(cx, cy + 0.1, label, ha='center', va='center', color=tc, fontsize=8.5, fontweight='bold')
    ax.text(cx + 3.2, cy, dim, ha='left', va='center', color='#adb5bd', fontsize=7.5, style='italic')

# Arrows between boxes
for i in range(len(boxes) - 1):
    y_top  = boxes[i][1] - 0.7
    y_bot  = boxes[i+1][1] + 0.7
    ax.annotate("", xy=(5, y_bot + 0.05), xytext=(5, y_top - 0.05),
                arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5))

# softmax label at bottom
ax.text(5, -0.3, "→ softmax → next-token probability", ha='center', color='#adb5bd', fontsize=7.5)

# --- Right panel: one transformer block ---
ax2 = axes[1]
ax2.set_facecolor('#0d1117')
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title("Transformer Block Detail", color='white', fontsize=13, fontweight='bold', pad=10)

block_items = [
    (5, 11.0, "x (input residual stream)",       '#495057', 'white'),
    (5,  9.2, "LayerNorm 1\n(normalise)",         '#2a9d8f', 'white'),
    (5,  7.3, "Causal Self-Attention\nQ=xW_q  K=xW_k  V=xW_v\nattn=softmax(QKᵀ/√d)·V", '#e9c46a', '#111'),
    (5,  5.0, "+ Residual\n(x = x + attn_out)",  '#264653', 'white'),
    (5,  3.2, "LayerNorm 2\n(normalise)",         '#2a9d8f', 'white'),
    (5,  1.4, "Feed-Forward Network\nGELU(xW₁+b₁)W₂+b₂", '#e76f51', 'white'),
    (5, -0.3, "+ Residual\n(x = x + ffn_out)",   '#264653', 'white'),
]
for (cx, cy, label, color, tc) in block_items:
    h = 1.3 if '\n' in label and label.count('\n') > 1 else 0.9
    fancy = FancyBboxPatch((cx - 3, cy - h/2), 6, h,
                           boxstyle="round,pad=0.08", facecolor=color, edgecolor='#6c757d', lw=1)
    ax2.add_patch(fancy)
    ax2.text(cx, cy, label, ha='center', va='center', color=tc, fontsize=8, fontweight='bold')

for i in range(len(block_items) - 1):
    cy_curr = block_items[i][1]
    cy_next = block_items[i+1][1]
    h_curr  = 1.3 if block_items[i][2].count('\n') > 1 else 0.9
    h_next  = 1.3 if block_items[i+1][2].count('\n') > 1 else 0.9
    y_from  = cy_curr - h_curr / 2 - 0.05
    y_to    = cy_next + h_next / 2 + 0.05
    ax2.annotate("", xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.2))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_architecture.png", dpi=300, bbox_inches="tight", facecolor='#0d1117')
plt.close()
print("  Saved 01_architecture.png")

# ------------------------------------------------------------------
# VIZ 2: Training dynamics + probability distribution
# ------------------------------------------------------------------
print("Saving 02_training.png ...")

# Smooth losses with moving average
def smooth(arr, w=20):
    out = []
    for i in range(len(arr)):
        lo = max(0, i - w)
        out.append(np.mean(arr[lo:i+1]))
    return np.array(out)

# Probability distribution before vs after training for "learning "
rng_vis = np.random.RandomState(99)
model_before = MiniGPT(vocab_size, d_model=32, n_layers=2, context_len=48, seed=99)
seed_ids_vis = np.array(encode("learning "))

logits_before = model_before.forward(seed_ids_vis)[-1]
logits_after  = model.forward(seed_ids_vis)[-1]

def to_probs(logits):
    l = logits - logits.max()
    p = np.exp(l)
    return p / p.sum()

probs_before = to_probs(logits_before)
probs_after  = to_probs(logits_after)

# Top-20 chars by average probability
avg_prob = (probs_before + probs_after) / 2
top20_idx = np.argsort(avg_prob)[::-1][:20]
top20_chars = [repr(chars[i]) for i in top20_idx]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MiniGPT Training Dynamics", fontsize=15, fontweight='bold', y=1.01)

# (a) Loss curve
ax = axes[0]
steps_arr = np.arange(len(losses))
ax.plot(steps_arr, losses, alpha=0.25, color='#4361ee', lw=0.8, label='Raw loss')
ax.plot(steps_arr, smooth(losses, 30), color='#4361ee', lw=2.2, label='Smoothed (w=30)')
ax.axhline(np.log(vocab_size), ls='--', color='#e63946', lw=1.5, label=f'Random baseline (ln {vocab_size}={np.log(vocab_size):.2f})')
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Cross-entropy loss (nats)", fontsize=11)
ax.set_title("(a) Training Loss Curve", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

# (b) Probability distribution before vs after
ax2 = axes[1]
x_pos = np.arange(len(top20_idx))
bar_w = 0.38
ax2.bar(x_pos - bar_w/2, probs_before[top20_idx], bar_w, label='Before training', color='#e63946', alpha=0.8)
ax2.bar(x_pos + bar_w/2, probs_after[top20_idx],  bar_w, label='After training',  color='#4361ee', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top20_chars, rotation=45, fontsize=8)
ax2.set_xlabel("Next character", fontsize=11)
ax2.set_ylabel("Probability", fontsize=11)
ax2.set_title('(b) P(next char | "learning ") — top 20', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_training.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 02_training.png")

# ------------------------------------------------------------------
# VIZ 3: Attention heatmap + diversity comparison
# ------------------------------------------------------------------
print("Saving 03_generation.png ...")

# Run forward on a longer seed to get attention maps
attn_seed  = "machine learning trains models"
attn_ids   = np.array(encode(attn_seed))[:20]
_ = model.forward(attn_ids)
attn_map   = model.last_attns[-1]   # last transformer block, shape (T, T)
T_vis      = min(20, attn_map.shape[0])
attn_vis   = attn_map[:T_vis, :T_vis]
tick_chars  = [repr(chars[attn_ids[i]]) for i in range(T_vis)]

diversity = {label: distinct_ratio(text) for label, text in gen_outputs.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("MiniGPT Generation Analysis", fontsize=14, fontweight='bold')

# (a) Attention heatmap
ax = axes[0]
im = ax.imshow(attn_vis, cmap='Blues', aspect='auto', vmin=0, vmax=attn_vis.max())
ax.set_xticks(range(T_vis))
ax.set_yticks(range(T_vis))
ax.set_xticklabels(tick_chars, rotation=90, fontsize=7)
ax.set_yticklabels(tick_chars, fontsize=7)
ax.set_xlabel("Key position", fontsize=10)
ax.set_ylabel("Query position", fontsize=10)
ax.set_title(f"(a) Causal Attention Weights\n(Layer {model.n_layers-1}, first {T_vis} tokens)", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8, label='Attention weight')

# (b) Diversity bar chart
ax2 = axes[1]
colors_bar = ['#e63946', '#4361ee', '#2a9d8f', '#f4a261']
bars = ax2.bar(range(len(diversity)), list(diversity.values()), color=colors_bar, alpha=0.85, edgecolor='white')
ax2.set_xticks(range(len(diversity)))
ax2.set_xticklabels(list(diversity.keys()), rotation=20, ha='right', fontsize=9)
ax2.set_ylabel("Distinct character ratio", fontsize=11)
ax2.set_title("(b) Output Diversity by Sampling Strategy\n(higher = more varied output)", fontsize=11)
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, diversity.values()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_generation.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 03_generation.png")

print("\n" + "="*70)
print("COMPLETE: llm_from_scratch.py")
print("="*70)
print(f"  Corpus         : {len(CORPUS)} chars, vocab size {vocab_size}")
print(f"  Model params   : {total_params:,}")
print(f"  Training steps : 400")
print(f"  Final loss     : {losses[-1]:.4f}")
print(f"  Visuals saved  : {VIS_DIR}/")
print("  Files          : 01_architecture.png, 02_training.png, 03_generation.png")
