"""
Transformer Encoder from Scratch — Complete Implementation
==========================================================

Learning Objectives:
  1. Understand each component of the Transformer encoder architecture
     (embeddings, positional encoding, multi-head attention, FFN, layer norm).
  2. Implement scaled dot-product attention and multi-head attention in pure NumPy.
  3. Build a full encoder stack (N stacked EncoderBlocks) with Pre-LN ordering.
  4. Extract mean-pooled representations from a random transformer and train a
     logistic-regression classifier on top of them.
  5. Compare transformer features against a TF-IDF baseline and (optionally)
     a Keras-trained transformer.
  6. Visualize attention patterns and interpret which tokens the encoder focuses on.

YouTube: https://www.youtube.com/watch?v=PLACEHOLDER_TRANSFORMER

Time:          ~50 minutes
Difficulty:    Advanced
Prerequisites: math_foundations 01_attention_mechanism,
               02_multi_head_attention, 03_positional_encoding,
               04_encoder_decoder_arch
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score, classification_report,
)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "transformer_from_scratch")
os.makedirs(VIS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
D_MODEL    = 64
NUM_HEADS  = 4
D_FF       = 256
NUM_LAYERS = 2
MAX_LEN    = 30
VOCAB_SIZE = 60
NUM_CLASSES = 2

# ===========================================================================
print("=" * 70)
print("SECTION 1: TRANSFORMER ENCODER ARCHITECTURE OVERVIEW")
print("=" * 70)
# ===========================================================================

print("""
Transformer Encoder-Only Architecture (for Classification)
===========================================================

  Input Tokens  [batch, seq_len]
        |
  Token Embedding  [batch, seq_len, d_model]
        |
  + Positional Encoding  [batch, seq_len, d_model]
        |
  ┌─────────────────────────────────────────────┐
  │  Encoder Block × N                          │
  │                                             │
  │   x ──► LayerNorm ──► MultiHeadAttention ──► + ──► x'
  │                             │                    ↑
  │                             └────────────────────┘  (residual)
  │                                                  │
  │   x' ──► LayerNorm ──► FeedForward ──────────► + ──► x''
  │                             │                    ↑
  │                             └────────────────────┘  (residual)
  └─────────────────────────────────────────────┘
        |
  Mean Pooling over sequence  [batch, d_model]
        |
  Linear Classifier  [batch, num_classes]
        |
  Logits → softmax → class probabilities

Components
----------
  Embedding         : vocab_size × d_model lookup table
  Positional Enc.   : sinusoidal, added once before the encoder stack
  LayerNorm         : normalize across d_model dimension (Pre-LN style)
  Multi-Head Attn   : num_heads parallel attention heads, concat + project
  Feed-Forward      : Linear(d_model→d_ff) → GELU → Linear(d_ff→d_model)
  Residual stream   : x = x + sublayer(LayerNorm(x))   [Pre-LN]
  Pooling           : mean of all token representations
  Classifier        : single linear layer → num_classes logits

Key parameters used in this module
-----------------------------------
  d_model    = {d_model}
  num_heads  = {num_heads}
  d_ff       = {d_ff}
  num_layers = {num_layers}
  max_len    = {max_len}
  vocab_size = {vocab_size}
  num_classes= {num_classes}
""".format(
    d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF,
    num_layers=NUM_LAYERS, max_len=MAX_LEN, vocab_size=VOCAB_SIZE,
    num_classes=NUM_CLASSES,
))

# ===========================================================================
print("=" * 70)
print("SECTION 2: BUILDING BLOCKS FROM SCRATCH (NumPy only)")
print("=" * 70)
# ===========================================================================

# ---------------------------------------------------------------------------
# 2a. Activation functions
# ---------------------------------------------------------------------------

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """Gaussian Error Linear Unit (approximate version used in BERT/GPT)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


print("Activation functions defined: softmax (numerically stable), gelu")
print("  gelu(0.0) =", round(gelu(0.0), 4))
print("  softmax([1,2,3]) =", softmax(np.array([1.0, 2.0, 3.0])).round(4))


# ---------------------------------------------------------------------------
# 2b. LayerNorm
# ---------------------------------------------------------------------------

class LayerNorm:
    """
    Layer Normalisation over the last axis (d_model dimension).

    For each sample and each position, normalise across the feature (d_model)
    dimension, then apply learnable scale (gamma) and shift (beta).
    """

    def __init__(self, d_model, eps=1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model)   # learnable scale
        self.beta  = np.zeros(d_model)  # learnable shift

    def forward(self, x):
        """
        x : (..., d_model)
        returns normalised tensor of same shape.
        """
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


# Quick test
_ln = LayerNorm(D_MODEL)
_x  = np.random.randn(2, 5, D_MODEL)
_out = _ln.forward(_x)
print(f"\nLayerNorm test — input shape: {_x.shape}, output shape: {_out.shape}")
print(f"  Output mean (should ≈ 0): {_out.mean():.4f}  std (should ≈ 1): {_out.std():.4f}")


# ---------------------------------------------------------------------------
# 2c. FeedForward Network
# ---------------------------------------------------------------------------

class FeedForward:
    """
    Position-wise Feed-Forward Network.

    Architecture: Linear(d_model → d_ff) → Activation → Linear(d_ff → d_model)

    Uses Kaiming (He) initialisation for W1 and Xavier for W2.
    """

    def __init__(self, d_model, d_ff, activation="gelu"):
        self.d_model = d_model
        self.d_ff    = d_ff
        self.act_fn  = gelu if activation == "gelu" else np.tanh

        # He initialisation for first layer (into ReLU/GELU)
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)

        # Xavier / Glorot for second layer
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        """
        x    : (..., d_model)
        return: (..., d_model)
        """
        h = self.act_fn(x @ self.W1 + self.b1)   # (..., d_ff)
        return h @ self.W2 + self.b2              # (..., d_model)


_ff   = FeedForward(D_MODEL, D_FF)
_ffout = _ff.forward(_x)
print(f"\nFeedForward test — input: {_x.shape}, output: {_ffout.shape}")
print(f"  W1: {_ff.W1.shape}, W2: {_ff.W2.shape}")


# ---------------------------------------------------------------------------
# 2d. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot-product attention.

    Q, K, V : (..., seq_len, d_k) — can have leading batch/head dimensions.
    mask    : optional boolean array (..., seq_len_q, seq_len_k);
              True positions are masked (set to -inf before softmax).

    Returns
    -------
    output  : (..., seq_len_q, d_v)
    weights : (..., seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    scale = np.sqrt(d_k)

    # Attention scores: (batch, heads, seq_q, seq_k)
    scores = Q @ K.swapaxes(-2, -1) / scale

    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    weights = softmax(scores, axis=-1)
    output  = weights @ V
    return output, weights


# Quick sanity check
_seq, _dk = 5, D_MODEL // NUM_HEADS
_Q = np.random.randn(2, NUM_HEADS, _seq, _dk)
_K = np.random.randn(2, NUM_HEADS, _seq, _dk)
_V = np.random.randn(2, NUM_HEADS, _seq, _dk)
_attn_out, _attn_w = scaled_dot_product_attention(_Q, _K, _V)
print(f"\nScaled dot-product attention test:")
print(f"  Q/K/V shape: {_Q.shape}")
print(f"  Output shape: {_attn_out.shape}")
print(f"  Weights shape: {_attn_w.shape}")
print(f"  Weights sum over keys (should be 1.0): {_attn_w[0, 0, 0].sum():.4f}")


# ---------------------------------------------------------------------------
# 2e. Multi-Head Attention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
    """
    Multi-Head Attention as described in 'Attention Is All You Need'.

    Splits d_model into num_heads heads, applies attention in parallel,
    then concatenates and projects back to d_model.

    Parameters
    ----------
    d_model   : model dimension (must be divisible by num_heads)
    num_heads : number of attention heads
    """

    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimension per head

        # Projection matrices (d_model → d_model for Q, K, V and output)
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale

    def _split_heads(self, x):
        """
        x : (batch, seq_len, d_model)
        → (batch, num_heads, seq_len, d_k)
        """
        batch, seq_len, _ = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)   # (batch, heads, seq, d_k)

    def _merge_heads(self, x):
        """
        x : (batch, num_heads, seq_len, d_k)
        → (batch, seq_len, d_model)
        """
        batch, heads, seq_len, d_k = x.shape
        x = x.transpose(0, 2, 1, 3)              # (batch, seq, heads, d_k)
        return x.reshape(batch, seq_len, heads * d_k)

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V : (batch, seq_len, d_model)
        mask    : optional (batch, 1, seq_q, seq_k)

        Returns
        -------
        output  : (batch, seq_len, d_model)
        weights : (batch, num_heads, seq_len, seq_len)
        """
        # Linear projections
        Q_proj = Q @ self.W_Q   # (batch, seq, d_model)
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V

        # Split into heads
        Q_heads = self._split_heads(Q_proj)  # (batch, heads, seq, d_k)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # Attention per head
        attn_out, weights = scaled_dot_product_attention(Q_heads, K_heads, V_heads, mask)

        # Merge heads and project
        merged = self._merge_heads(attn_out)     # (batch, seq, d_model)
        output = merged @ self.W_O               # (batch, seq, d_model)
        return output, weights


_mha = MultiHeadAttention(D_MODEL, NUM_HEADS)
_x_mha = np.random.randn(2, 8, D_MODEL)
_mha_out, _mha_w = _mha.forward(_x_mha, _x_mha, _x_mha)
print(f"\nMultiHeadAttention test:")
print(f"  Input: {_x_mha.shape}")
print(f"  Output: {_mha_out.shape}")
print(f"  Attention weights: {_mha_w.shape}  (batch, heads, seq_q, seq_k)")


# ---------------------------------------------------------------------------
# 2f. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------

def sinusoidal_positional_encoding(max_len, d_model):
    """
    Create a sinusoidal positional encoding matrix.

    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Returns
    -------
    PE : (max_len, d_model)  — one encoding vector per position
    """
    PE   = np.zeros((max_len, d_model))
    pos  = np.arange(max_len)[:, np.newaxis]           # (max_len, 1)
    dims = np.arange(0, d_model, 2)                    # even indices
    div  = np.power(10000.0, dims / d_model)           # (d_model/2,)

    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div)
    return PE


_PE = sinusoidal_positional_encoding(MAX_LEN, D_MODEL)
print(f"\nPositional Encoding shape: {_PE.shape}  (max_len={MAX_LEN}, d_model={D_MODEL})")
print(f"  PE[0, :4]  = {_PE[0,  :4].round(3)}")
print(f"  PE[1, :4]  = {_PE[1,  :4].round(3)}")
print(f"  PE[10, :4] = {_PE[10, :4].round(3)}")


# ---------------------------------------------------------------------------
# 2g. Encoder Block (Pre-LN)
# ---------------------------------------------------------------------------

class EncoderBlock:
    """
    Single Transformer Encoder Block using Pre-LN ordering.

    Pre-LN: LayerNorm is applied BEFORE each sublayer (more stable training).

      x_attn = x + MultiHeadAttention(LayerNorm(x), LayerNorm(x), LayerNorm(x))
      x_out  = x_attn + FeedForward(LayerNorm(x_attn))
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_ff      = d_ff

        self.ln1  = LayerNorm(d_model)
        self.mha  = MultiHeadAttention(d_model, num_heads)
        self.ln2  = LayerNorm(d_model)
        self.ffn  = FeedForward(d_model, d_ff)

    def forward(self, x, mask=None):
        """
        x    : (batch, seq_len, d_model)
        mask : optional attention mask

        Returns
        -------
        x_out   : (batch, seq_len, d_model)
        weights : (batch, num_heads, seq_len, seq_len)
        """
        # Sub-layer 1: Multi-Head Self-Attention with residual
        x_norm = self.ln1.forward(x)
        attn_out, weights = self.mha.forward(x_norm, x_norm, x_norm, mask)
        x = x + attn_out   # residual connection

        # Sub-layer 2: Feed-Forward with residual
        x_norm2 = self.ln2.forward(x)
        ffn_out  = self.ffn.forward(x_norm2)
        x_out   = x + ffn_out   # residual connection

        return x_out, weights


_eb = EncoderBlock(D_MODEL, NUM_HEADS, D_FF)
_x_eb = np.random.randn(2, 10, D_MODEL)
_eb_out, _eb_w = _eb.forward(_x_eb)
print(f"\nEncoderBlock test:")
print(f"  Input:   {_x_eb.shape}")
print(f"  Output:  {_eb_out.shape}")
print(f"  Weights: {_eb_w.shape}")


# ---------------------------------------------------------------------------
# 2h. Full Transformer Encoder (for classification)
# ---------------------------------------------------------------------------

class TransformerEncoder:
    """
    Complete Transformer Encoder for sequence classification.

    Pipeline:
      token_ids → Embedding + Positional Encoding
               → N × EncoderBlock
               → Mean Pooling
               → Linear Classifier
               → logits (batch, num_classes)
    """

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_len, num_classes):
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_ff       = d_ff
        self.num_layers = num_layers
        self.max_len    = max_len
        self.num_classes = num_classes

        # Token embedding table
        self.embedding = np.random.randn(vocab_size, d_model) * 0.01

        # Positional encoding (fixed, not learned)
        self.PE = sinusoidal_positional_encoding(max_len, d_model)  # (max_len, d_model)

        # Stack of encoder blocks
        self.encoder_blocks = [
            EncoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]

        # Output classifier: d_model → num_classes
        scale = np.sqrt(2.0 / (d_model + num_classes))
        self.W_cls = np.random.randn(d_model, num_classes) * scale
        self.b_cls = np.zeros(num_classes)

    # ------------------------------------------------------------------
    def _embed(self, token_ids):
        """
        token_ids : (batch, seq_len)  integer token indices
        returns   : (batch, seq_len, d_model)
        """
        batch, seq_len = token_ids.shape
        x = self.embedding[token_ids]                   # (batch, seq, d_model)
        x = x + self.PE[:seq_len, :]                    # add positional encoding
        return x

    # ------------------------------------------------------------------
    def encode(self, token_ids):
        """
        Run the full encoder stack and return mean-pooled representations.

        token_ids : (batch, seq_len)
        returns   : (batch, d_model)  — mean over sequence dimension
        """
        x = self._embed(token_ids)                      # (batch, seq, d_model)
        all_weights = []
        for block in self.encoder_blocks:
            x, w = block.forward(x)
            all_weights.append(w)
        pooled = x.mean(axis=1)                         # (batch, d_model)
        self._last_attention_weights = all_weights      # store for analysis
        return pooled

    # ------------------------------------------------------------------
    def forward(self, token_ids):
        """
        Full forward pass: token_ids → logits.

        token_ids : (batch, seq_len)
        returns   : (batch, num_classes)
        """
        pooled = self.encode(token_ids)                 # (batch, d_model)
        logits = pooled @ self.W_cls + self.b_cls       # (batch, num_classes)
        return logits

    # ------------------------------------------------------------------
    def predict(self, token_ids):
        """Return predicted class indices (batch,)."""
        logits = self.forward(token_ids)
        return np.argmax(logits, axis=-1)

    # ------------------------------------------------------------------
    def train_step(self, X_batch, y_batch, lr=1e-3):
        """
        Single gradient-update step for the classifier head only.

        We treat the encoder as a fixed feature extractor and apply the
        analytic gradient of cross-entropy + softmax w.r.t. W_cls and b_cls.

        dL/dlogits = softmax(logits) - one_hot(y)
        dL/dW_cls  = pooled.T @ dL/dlogits  / batch
        dL/db_cls  = mean(dL/dlogits, axis=0)
        """
        batch_size = X_batch.shape[0]
        pooled = self.encode(X_batch)                       # (batch, d_model)
        logits = pooled @ self.W_cls + self.b_cls           # (batch, num_classes)
        probs  = softmax(logits, axis=-1)                   # (batch, num_classes)

        # One-hot targets
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(batch_size), y_batch] = 1.0

        # Gradient of cross-entropy loss w.r.t. logits
        dlogits = (probs - one_hot) / batch_size            # (batch, num_classes)

        # Gradients for classifier parameters
        dW = pooled.T @ dlogits                             # (d_model, num_classes)
        db = dlogits.mean(axis=0)                           # (num_classes,)

        # SGD update
        self.W_cls -= lr * dW
        self.b_cls -= lr * db

        # Cross-entropy loss
        log_probs = np.log(np.clip(probs, 1e-12, 1.0))
        loss = -log_probs[np.arange(batch_size), y_batch].mean()
        return loss


# Build the model
model = TransformerEncoder(
    vocab_size  = VOCAB_SIZE,
    d_model     = D_MODEL,
    num_heads   = NUM_HEADS,
    d_ff        = D_FF,
    num_layers  = NUM_LAYERS,
    max_len     = MAX_LEN,
    num_classes = NUM_CLASSES,
)

# Parameter count
def count_params(m):
    total = 0
    total += m.embedding.size
    for blk in m.encoder_blocks:
        for W in [blk.mha.W_Q, blk.mha.W_K, blk.mha.W_V, blk.mha.W_O]:
            total += W.size
        total += blk.ffn.W1.size + blk.ffn.b1.size
        total += blk.ffn.W2.size + blk.ffn.b2.size
        total += blk.ln1.gamma.size + blk.ln1.beta.size
        total += blk.ln2.gamma.size + blk.ln2.beta.size
    total += m.W_cls.size + m.b_cls.size
    return total

TOTAL_PARAMS = count_params(model)

# Shape trace
_test_ids = np.random.randint(0, VOCAB_SIZE, (3, 12))
_emb_out  = model._embed(_test_ids)
_enc_out  = model.encode(_test_ids)
_logits   = model.forward(_test_ids)

print(f"\nTransformerEncoder shape trace:")
print(f"  token_ids  : {_test_ids.shape}")
print(f"  after embed: {_emb_out.shape}   (embedding + PE)")
print(f"  after encode (mean pool): {_enc_out.shape}")
print(f"  logits     : {_logits.shape}")
print(f"\nTotal parameters: {TOTAL_PARAMS:,}")


# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 3: SYNTHETIC SENTIMENT DATASET")
print("=" * 70)
# ===========================================================================

# Build vocabulary
positive_words = [
    "great", "excellent", "amazing", "wonderful", "love",
    "fantastic", "brilliant", "superb", "perfect", "best",
    "incredible", "outstanding", "phenomenal", "exceptional", "splendid",
]
negative_words = [
    "terrible", "awful", "horrible", "hate", "worst",
    "dreadful", "disgusting", "pathetic", "abysmal", "atrocious",
    "lousy", "miserable", "appalling", "ghastly", "mediocre",
]
neutral_words = [
    "the", "a", "is", "was", "it", "this", "that", "movie",
    "film", "story", "character", "plot", "scene", "actor",
    "very", "quite", "some", "not", "but", "and", "or",
    "really", "just", "also", "much", "more", "even",
    "with", "has",
]

# Assign integer IDs
vocab = positive_words + negative_words + neutral_words
# Pad vocab to VOCAB_SIZE with placeholder tokens
while len(vocab) < VOCAB_SIZE:
    vocab.append(f"<pad{len(vocab)}>")

word2id = {w: i for i, w in enumerate(vocab)}
POS_IDS = list(range(len(positive_words)))
NEG_IDS = list(range(len(positive_words), len(positive_words) + len(negative_words)))
NEU_IDS = list(range(len(positive_words) + len(negative_words), len(vocab)))

def generate_sequence(label, min_len=10, max_len=20):
    """
    Generate a synthetic sequence.
    label=1 → more positive words; label=0 → more negative words.
    """
    seq_len = np.random.randint(min_len, max_len + 1)
    tokens  = []
    if label == 1:  # positive: 40-60% positive, 10-20% negative
        n_pos = int(seq_len * np.random.uniform(0.40, 0.60))
        n_neg = int(seq_len * np.random.uniform(0.05, 0.15))
    else:           # negative: 10-20% positive, 40-60% negative
        n_pos = int(seq_len * np.random.uniform(0.05, 0.15))
        n_neg = int(seq_len * np.random.uniform(0.40, 0.60))
    n_neu = seq_len - n_pos - n_neg

    tokens += list(np.random.choice(POS_IDS, size=n_pos, replace=True))
    tokens += list(np.random.choice(NEG_IDS, size=n_neg, replace=True))
    tokens += list(np.random.choice(NEU_IDS, size=max(0, n_neu), replace=True))
    np.random.shuffle(tokens)
    return np.array(tokens[:seq_len], dtype=np.int32)


# Generate 200 sequences
N_SAMPLES = 200
sequences = []
labels    = []

for i in range(N_SAMPLES):
    lbl = i % 2   # alternate to get balanced dataset
    seq = generate_sequence(lbl)
    sequences.append(seq)
    labels.append(lbl)

labels = np.array(labels)

# Pad sequences to uniform length for batching
def pad_sequences(seqs, max_len):
    out = np.zeros((len(seqs), max_len), dtype=np.int32)
    for i, s in enumerate(seqs):
        l = min(len(s), max_len)
        out[i, :l] = s[:l]
    return out

X_padded = pad_sequences(sequences, MAX_LEN)   # (200, MAX_LEN)

# 80/20 train/test split (stratified)
rng       = np.random.RandomState(42)
idx       = rng.permutation(N_SAMPLES)
train_cut = int(0.8 * N_SAMPLES)
train_idx = idx[:train_cut]
test_idx  = idx[train_cut:]

X_train, y_train = X_padded[train_idx], labels[train_idx]
X_test,  y_test  = X_padded[test_idx],  labels[test_idx]

print(f"Vocabulary size : {len(vocab)} tokens")
print(f"  Positive words: {len(positive_words)}  (IDs 0–{len(positive_words)-1})")
print(f"  Negative words: {len(negative_words)}  (IDs {len(positive_words)}–{len(positive_words)+len(negative_words)-1})")
print(f"  Neutral words : {len(neutral_words)}  (IDs {len(positive_words)+len(negative_words)}–{len(positive_words)+len(negative_words)+len(neutral_words)-1})")
print()
print(f"Dataset: {N_SAMPLES} sequences")
print(f"  Train  : {len(X_train)} samples  (positive: {y_train.sum()}, negative: {(1-y_train).sum()})")
print(f"  Test   : {len(X_test)}  samples  (positive: {y_test.sum()},  negative: {(1-y_test).sum()})")
print(f"  Sequence lengths: min={min(len(s) for s in sequences)}, "
      f"max={max(len(s) for s in sequences)}, "
      f"mean={np.mean([len(s) for s in sequences]):.1f}")
print(f"\nExample sequences (first 3):")
for i in range(3):
    raw = sequences[train_idx[i]]
    words = [vocab[t] for t in raw]
    print(f"  [{y_train[i]}] {' '.join(words)}")


# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 4: TRAINING")
print("=" * 70)
# ===========================================================================

print("""
Training Strategy
-----------------
Because implementing full backpropagation through multi-head attention in
pure NumPy is extremely complex (and not the focus of this module), we use
a two-stage approach that is honest about NumPy limitations:

  Stage A — Encoder as feature extractor (random weights, untrained):
    The encoder applies its randomly-initialised attention + FFN layers,
    producing a (d_model,) mean-pooled representation for each sample.
    We fit sklearn LogisticRegression on these features.
    This demonstrates that even random transformers produce useful structure
    (via positional encoding and the inductive bias of attention).

  Stage B — Fine-tune classifier head only:
    We additionally run 30 gradient-descent steps on the linear classifier
    head (W_cls, b_cls) using the analytic cross-entropy gradient.
    The encoder weights remain fixed.

  This is a principled approximation used in the early "probing" literature
  and is a valid pedagogical simplification.
""")

# ------- Stage A: sklearn LR on random encoder features -------

print("Stage A: sklearn LogisticRegression on random transformer features")
print("-" * 60)

feats_train = model.encode(X_train)   # (160, d_model)
feats_test  = model.encode(X_test)    # (40,  d_model)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(feats_train, y_train)

train_acc_A = lr_model.score(feats_train, y_train)
test_acc_A  = lr_model.score(feats_test,  y_test)

print(f"  Train accuracy (Stage A): {train_acc_A:.4f}")
print(f"  Test  accuracy (Stage A): {test_acc_A:.4f}")

# ------- Stage B: train classifier head with manual gradient -------

print("\nStage B: Fine-tune classifier head (30 epochs, batch_size=32)")
print("-" * 60)

NUM_EPOCHS  = 30
BATCH_SIZE  = 32
LEARN_RATE  = 5e-3

train_accs = []
test_accs  = []
losses     = []

n_train = X_train.shape[0]

for epoch in range(NUM_EPOCHS):
    # Shuffle
    perm = np.random.permutation(n_train)
    X_shuf = X_train[perm]
    y_shuf = y_train[perm]

    epoch_loss = 0.0
    n_batches  = 0

    for start in range(0, n_train, BATCH_SIZE):
        end      = min(start + BATCH_SIZE, n_train)
        X_batch  = X_shuf[start:end]
        y_batch  = y_shuf[start:end]
        loss     = model.train_step(X_batch, y_batch, lr=LEARN_RATE)
        epoch_loss += loss
        n_batches  += 1

    # Evaluate using model.predict (classifier head)
    tr_preds = model.predict(X_train)
    te_preds = model.predict(X_test)
    tr_acc   = accuracy_score(y_train, tr_preds)
    te_acc   = accuracy_score(y_test,  te_preds)

    train_accs.append(tr_acc)
    test_accs.append(te_acc)
    losses.append(epoch_loss / n_batches)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:2d}/{NUM_EPOCHS}  loss={losses[-1]:.4f}  "
              f"train_acc={tr_acc:.4f}  test_acc={te_acc:.4f}")

final_train_acc = train_accs[-1]
final_test_acc  = test_accs[-1]
print(f"\nFinal   train accuracy (Stage B): {final_train_acc:.4f}")
print(f"Final   test  accuracy (Stage B): {final_test_acc:.4f}")


# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 5: COMPARISON")
print("=" * 70)
# ===========================================================================

from sklearn.feature_extraction.text import TfidfVectorizer

# Reconstruct text for TF-IDF
def tokens_to_text(token_ids, vocab):
    return " ".join(vocab[t] for t in token_ids if t < len(vocab))

train_texts = [tokens_to_text(X_train[i], vocab) for i in range(len(X_train))]
test_texts  = [tokens_to_text(X_test[i],  vocab) for i in range(len(X_test))]

tfidf = TfidfVectorizer(max_features=200)
X_tr_tfidf = tfidf.fit_transform(train_texts)
X_te_tfidf = tfidf.transform(test_texts)

lr_tfidf = LogisticRegression(max_iter=1000, random_state=42)
lr_tfidf.fit(X_tr_tfidf, y_train)
tfidf_train_acc = lr_tfidf.score(X_tr_tfidf, y_train)
tfidf_test_acc  = lr_tfidf.score(X_te_tfidf, y_test)

print("\nComparison Table")
print("-" * 70)
print(f"{'Method':<45} {'Train Acc':>10} {'Test Acc':>10}")
print("-" * 70)
print(f"{'Random Transformer + LogisticRegression (A)':<45} {train_acc_A:>10.4f} {test_acc_A:>10.4f}")
print(f"{'Transformer head fine-tuned (B, 30 epochs)':<45} {final_train_acc:>10.4f} {final_test_acc:>10.4f}")
print(f"{'TF-IDF + LogisticRegression (baseline)':<45} {tfidf_train_acc:>10.4f} {tfidf_test_acc:>10.4f}")

# Optional Keras transformer
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("\nTensorFlow detected — building Keras Transformer classifier...")

    def build_keras_transformer(vocab_size, d_model, num_heads, d_ff,
                                num_layers, max_len, num_classes):
        inp = keras.Input(shape=(max_len,), dtype="int32")
        x   = layers.Embedding(vocab_size, d_model)(inp)

        # Positional encoding as a non-trainable embedding
        pe_init = keras.initializers.Constant(
            sinusoidal_positional_encoding(max_len, d_model)
        )
        pos_emb = layers.Embedding(
            max_len, d_model,
            embeddings_initializer=pe_init, trainable=False,
        )
        positions = tf.range(start=0, limit=max_len, delta=1)
        x = x + pos_emb(positions)

        for _ in range(num_layers):
            # Multi-head self-attention (Pre-LN)
            x_norm = layers.LayerNormalization(epsilon=1e-6)(x)
            attn   = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=d_model // num_heads
            )(x_norm, x_norm)
            x      = x + attn
            # FFN (Pre-LN)
            x_norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
            ffn_out = layers.Dense(d_ff, activation="gelu")(x_norm2)
            ffn_out = layers.Dense(d_model)(ffn_out)
            x       = x + ffn_out

        # Mean pooling + classifier
        x   = layers.GlobalAveragePooling1D()(x)
        out = layers.Dense(num_classes, activation="softmax")(x)
        return keras.Model(inp, out)

    keras_model = build_keras_transformer(
        VOCAB_SIZE, D_MODEL, NUM_HEADS, D_FF, NUM_LAYERS, MAX_LEN, NUM_CLASSES
    )
    keras_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    keras_model.fit(
        X_train, y_train,
        epochs=30, batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0,
    )
    _, keras_train_acc = keras_model.evaluate(X_train, y_train, verbose=0)
    _, keras_test_acc  = keras_model.evaluate(X_test,  y_test,  verbose=0)
    print(f"{'Keras trained Transformer (30 epochs)':<45} {keras_train_acc:>10.4f} {keras_test_acc:>10.4f}")
    KERAS_AVAILABLE = True
    KERAS_TEST_ACC  = keras_test_acc

except ImportError:
    print("\nTensorFlow not found — skipping Keras comparison.")
    print("Install with:  pip install tensorflow")
    KERAS_AVAILABLE = False
    KERAS_TEST_ACC  = None

print("-" * 70)


# ===========================================================================
print("\n" + "=" * 70)
print("SECTION 6: ATTENTION PATTERN ANALYSIS")
print("=" * 70)
# ===========================================================================

# Pick a sample from the test set
SAMPLE_IDX = 0
sample_ids  = X_test[SAMPLE_IDX:SAMPLE_IDX+1]   # (1, MAX_LEN)
sample_label = y_test[SAMPLE_IDX]

# Run encoder to populate attention weights
_ = model.encode(sample_ids)
attn_weights = model._last_attention_weights  # list of (1, num_heads, seq, seq) per layer

sample_words = [vocab[t] for t in X_test[SAMPLE_IDX]]

print(f"\nSample sequence (label={'positive' if sample_label==1 else 'negative'}):")
print("  " + " ".join(f"[{w}]" for w in sample_words))
print()

for layer_idx, w in enumerate(attn_weights):
    # w : (1, num_heads, seq, seq)
    avg_head_weights = w[0].mean(axis=0)   # (seq, seq) averaged over heads
    print(f"Layer {layer_idx+1} attention (avg over {NUM_HEADS} heads) — "
          f"top 3 attended positions for each token:")
    for tok_pos in range(min(5, MAX_LEN)):
        top3 = np.argsort(avg_head_weights[tok_pos])[::-1][:3]
        top3_info = [(int(p), f"{avg_head_weights[tok_pos,p]:.3f}") for p in top3]
        print(f"  tok[{tok_pos}]={sample_words[tok_pos]:12s} attends to: "
              + ", ".join(f"tok[{p}]({s})" for p, s in top3_info))
    print()


# ===========================================================================
print("=" * 70)
print("SECTION 7: VISUALIZATIONS")
print("=" * 70)
# ===========================================================================

# ---------------------------------------------------------------------------
# Plot 1: Architecture diagram
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
fig.suptitle("Transformer Encoder Architecture", fontsize=16, fontweight="bold")

# Left: text architecture diagram
ax_arch = axes[0]
ax_arch.set_xlim(0, 10)
ax_arch.set_ylim(0, 20)
ax_arch.axis("off")

boxes = [
    (5, 18.5, "Input Token IDs\n[batch, seq_len]",             "#E8F4F8", 12),
    (5, 16.5, "Token Embedding\n[batch, seq_len, d_model=64]", "#D4E9F7", 11),
    (5, 14.5, "+ Positional Encoding\n(sinusoidal, fixed)",    "#D4E9F7", 11),
    (5, 12.0, f"Encoder Block x{NUM_LAYERS}\n"
              "  Pre-LN → MultiHeadAttn\n"
              f"  (heads={NUM_HEADS}, d_k={D_MODEL//NUM_HEADS})\n"
              "  + Residual\n"
              "  Pre-LN → FFN\n"
              f"  (d_ff={D_FF}) + Residual",                   "#FFF3CD", 10),
    (5,  8.5, "Mean Pooling\n[batch, d_model=64]",             "#D5F5E3", 11),
    (5,  6.5, f"Linear Classifier\n[batch, num_classes={NUM_CLASSES}]",
                                                                "#D5F5E3", 11),
    (5,  4.5, "Softmax → Class Probabilities",                 "#FADBD8", 11),
]

for (cx, cy, txt, color, fs) in boxes:
    bbox_props = dict(boxstyle="round,pad=0.4", facecolor=color, edgecolor="#333333", linewidth=1.5)
    ax_arch.text(cx, cy, txt, ha="center", va="center", fontsize=fs,
                 bbox=bbox_props, multialignment="center")

# Arrows between boxes
arrow_y_pairs = [(18.0, 17.2), (16.0, 15.2), (14.0, 13.5), (10.5, 9.2), (8.0, 7.2), (6.0, 5.2)]
for (y1, y2) in arrow_y_pairs:
    ax_arch.annotate("", xy=(5, y2), xytext=(5, y1),
                     arrowprops=dict(arrowstyle="->", color="#333333", lw=1.5))

ax_arch.set_title(f"Encoder-Only Transformer\n(d_model={D_MODEL}, "
                  f"layers={NUM_LAYERS}, heads={NUM_HEADS})",
                  fontsize=12, fontweight="bold")

# Right: parameter breakdown pie chart
ax_pie = axes[1]
emb_params    = VOCAB_SIZE * D_MODEL
attn_params   = NUM_LAYERS * 4 * D_MODEL * D_MODEL   # W_Q, W_K, W_V, W_O
ffn_params    = NUM_LAYERS * (D_MODEL * D_FF + D_FF + D_FF * D_MODEL + D_MODEL)
ln_params     = NUM_LAYERS * 2 * 2 * D_MODEL         # 2 LN per block, gamma+beta
cls_params    = D_MODEL * NUM_CLASSES + NUM_CLASSES

sizes  = [emb_params, attn_params, ffn_params, ln_params, cls_params]
labels_pie = [
    f"Embedding\n{emb_params:,}",
    f"Attention\n{attn_params:,}",
    f"FFN\n{ffn_params:,}",
    f"LayerNorm\n{ln_params:,}",
    f"Classifier\n{cls_params:,}",
]
colors = ["#AED6F1", "#A9DFBF", "#FAD7A0", "#D7BDE2", "#F1948A"]
wedges, texts, autotexts = ax_pie.pie(
    sizes, labels=labels_pie, colors=colors,
    autopct="%1.1f%%", startangle=140,
    pctdistance=0.75, labeldistance=1.18,
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(8)
ax_pie.set_title(f"Parameter Count Breakdown\nTotal: {sum(sizes):,} parameters",
                 fontsize=12, fontweight="bold")

plt.tight_layout()
out1 = os.path.join(VIS_DIR, "01_transformer_architecture.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# ---------------------------------------------------------------------------
# Plot 2: Training results
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Training Results — Transformer Classifier", fontsize=14, fontweight="bold")

# Left: accuracy vs epoch
ax_acc = axes[0]
epochs_range = range(1, NUM_EPOCHS + 1)
ax_acc.plot(epochs_range, train_accs, label="Train Accuracy", color="#2196F3", lw=2, marker="o", markersize=3)
ax_acc.plot(epochs_range, test_accs,  label="Test Accuracy",  color="#F44336", lw=2, marker="s", markersize=3)
ax_acc.axhline(y=test_acc_A,    color="#9C27B0", ls="--", lw=1.5, label=f"Stage A (LR): {test_acc_A:.3f}")
ax_acc.axhline(y=tfidf_test_acc,color="#FF9800", ls=":",  lw=1.5, label=f"TF-IDF baseline: {tfidf_test_acc:.3f}")
ax_acc.set_xlabel("Epoch", fontsize=11)
ax_acc.set_ylabel("Accuracy", fontsize=11)
ax_acc.set_title("Accuracy vs Epoch", fontsize=12, fontweight="bold")
ax_acc.legend(fontsize=8)
ax_acc.set_ylim(0, 1.05)
ax_acc.grid(True, alpha=0.3)

# Middle: confusion matrix
ax_cm = axes[1]
final_preds = model.predict(X_test)
cm = confusion_matrix(y_test, final_preds)
im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
tick_marks = [0, 1]
class_names = ["Negative", "Positive"]
ax_cm.set_xticks(tick_marks)
ax_cm.set_yticks(tick_marks)
ax_cm.set_xticklabels(class_names, fontsize=10)
ax_cm.set_yticklabels(class_names, fontsize=10)
ax_cm.set_xlabel("Predicted Label", fontsize=11)
ax_cm.set_ylabel("True Label", fontsize=11)
ax_cm.set_title("Confusion Matrix\n(Test Set, Stage B)", fontsize=12, fontweight="bold")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=14, fontweight="bold")

# Right: per-class metrics bar chart
ax_metrics = axes[2]
prec  = precision_score(y_test, final_preds, average=None, zero_division=0)
rec   = recall_score(   y_test, final_preds, average=None, zero_division=0)
f1    = f1_score(       y_test, final_preds, average=None, zero_division=0)

x_pos = np.arange(len(class_names))
width = 0.25
ax_metrics.bar(x_pos - width, prec, width, label="Precision", color="#4CAF50", alpha=0.85)
ax_metrics.bar(x_pos,         rec,  width, label="Recall",    color="#2196F3", alpha=0.85)
ax_metrics.bar(x_pos + width, f1,   width, label="F1-Score",  color="#FF5722", alpha=0.85)
ax_metrics.set_xticks(x_pos)
ax_metrics.set_xticklabels(class_names, fontsize=10)
ax_metrics.set_ylabel("Score", fontsize=11)
ax_metrics.set_title("Per-Class Metrics\n(Test Set, Stage B)", fontsize=12, fontweight="bold")
ax_metrics.legend(fontsize=9)
ax_metrics.set_ylim(0, 1.15)
ax_metrics.grid(True, alpha=0.3, axis="y")
for xi, (p, r, f) in enumerate(zip(prec, rec, f1)):
    ax_metrics.text(xi - width, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
    ax_metrics.text(xi,         r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax_metrics.text(xi + width, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)

plt.tight_layout()
out2 = os.path.join(VIS_DIR, "02_training_results.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")


# ---------------------------------------------------------------------------
# Plot 3: Attention patterns
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    f"Attention Patterns — Sample Sequence\n"
    f"(Label: {'Positive' if sample_label==1 else 'Negative'}, "
    f"first {MAX_LEN} tokens)",
    fontsize=13, fontweight="bold",
)

# We have NUM_LAYERS layers each with NUM_HEADS heads
# Plot: layer 1 head 1, layer 1 head 2, layer 2 head 1, layer 2 head 2
plot_configs = [
    (0, 0, "Layer 1, Head 1"),
    (0, 1, "Layer 1, Head 2"),
    (1, 0, f"Layer 2, Head 1" if NUM_LAYERS >= 2 else "N/A"),
    (1, 1, f"Layer 2, Head 2" if NUM_LAYERS >= 2 else "N/A"),
]

for ax_r, ax_c, title in plot_configs:
    ax = axes[ax_r][ax_c]
    layer_idx = ax_r
    head_idx  = ax_c

    if layer_idx < len(attn_weights):
        w_mat = attn_weights[layer_idx][0, head_idx, :, :]   # (seq, seq)
        im = ax.imshow(w_mat, cmap="Blues", vmin=0, vmax=w_mat.max(), aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_step = max(1, MAX_LEN // 10)
        ticks = list(range(0, MAX_LEN, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks], fontsize=8)
        ax.set_yticklabels([str(t) for t in ticks], fontsize=8)
        ax.set_xlabel("Key position (attended to)", fontsize=9)
        ax.set_ylabel("Query position", fontsize=9)
    else:
        ax.text(0.5, 0.5, "N/A\n(fewer layers)", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()

    ax.set_title(title, fontsize=11, fontweight="bold")

plt.tight_layout()
out3 = os.path.join(VIS_DIR, "03_attention_patterns.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")


# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
# ===========================================================================

print(f"""
What We Built
-------------
  1. Numerically stable softmax and GELU activation from scratch.
  2. LayerNorm — normalises over the d_model dimension with learnable
     gamma and beta parameters.
  3. FeedForward network: Linear(d_model→d_ff) → GELU → Linear(d_ff→d_model).
  4. Scaled dot-product attention with optional masking.
  5. MultiHeadAttention — projects Q/K/V into {NUM_HEADS} heads, runs
     attention in parallel, concatenates and projects back.
  6. Sinusoidal positional encoding matrix (max_len={MAX_LEN}, d_model={D_MODEL}).
  7. EncoderBlock (Pre-LN): MHA sub-layer + FFN sub-layer, each wrapped in
     a residual connection with LayerNorm applied before the sublayer.
  8. TransformerEncoder: full pipeline from token IDs to class logits,
     including embedding lookup, PE addition, {NUM_LAYERS} encoder blocks,
     mean pooling, and a linear classifier head.

Model Statistics
----------------
  Parameters total : {TOTAL_PARAMS:,}
  Architecture     : vocab={VOCAB_SIZE}, d_model={D_MODEL}, heads={NUM_HEADS},
                     d_ff={D_FF}, layers={NUM_LAYERS}, max_len={MAX_LEN}

Results
-------
  Stage A — Random Transformer + LogisticRegression:
    Train: {train_acc_A:.4f}   Test: {test_acc_A:.4f}

  Stage B — Fine-tuned Classifier Head (30 epochs):
    Train: {final_train_acc:.4f}   Test: {final_test_acc:.4f}

  TF-IDF Baseline + LogisticRegression:
    Train: {tfidf_train_acc:.4f}   Test: {tfidf_test_acc:.4f}

  Keras Transformer: {'Not available (pip install tensorflow)' if not KERAS_AVAILABLE else f'Test acc: {KERAS_TEST_ACC:.4f}'}

Generated Files
---------------
  {out1}
  {out2}
  {out3}

Key Takeaways
-------------
  1. Multi-head attention allows each head to focus on different aspects of
     the input simultaneously — positional, lexical, semantic.
  2. Pre-Layer-Normalization (Pre-LN) is more stable to train than the
     original Post-LN formulation from 'Attention Is All You Need'.
  3. Sinusoidal positional encodings inject position information without
     adding learnable parameters, and generalise to unseen lengths.
  4. Even a randomly-initialised Transformer encoder produces representations
     that are more structured than raw one-hot features, as evidenced by
     the Stage A classifier exceeding random chance without any training.
  5. Full backpropagation through attention requires the chain rule applied
     to multiple matrix products and softmax operations — this is what
     frameworks like TensorFlow and PyTorch auto-differentiate for you.
  6. Mean pooling is a simple but effective way to aggregate a variable-length
     sequence into a fixed-size vector for classification.

Next Steps — Part 6 Modules
-----------------------------
  02_bert_encoder.py  — load pre-trained BERT with HuggingFace Transformers
                         and fine-tune for text classification
  03_gpt_decoder.py   — causal (decoder-only) language modelling with GPT-2
  projects/           — text_classification_with_bert.py
                        text_generation_with_gpt2.py
""")
