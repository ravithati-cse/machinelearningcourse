# MLForBeginners — Instructor Guide
## Part 6 · Module 05: Transformer Encoder from Scratch
### Two-Session Teaching Script

> **Prerequisites:** All 4 math foundations complete. They can explain
> attention, multi-head attention, positional encoding, and the full
> encoder-decoder architecture. They are comfortable with NumPy matrix
> operations.
> **Payoff today:** They are going to build a working Transformer encoder
> in pure NumPy. Every weight matrix, every dot product, every residual —
> written by hand. This is the biggest coding session in Part 6.

---

# SESSION 1 (~90 min)
## "Building the pieces — attention, FFN, and one encoder block in NumPy"

## Before They Arrive
- Terminal open in `transformers/algorithms/`
- Whiteboard with the Pre-LN encoder block diagram from Module 04
- Write at the top: `D_MODEL=64  NUM_HEADS=4  D_FF=256  NUM_LAYERS=2`
- Have pride in your voice — this session is a celebration

---

## OPENING (10 min)

> *"I want you to take a second and think about where you started.*
>
> *Part 1: you fit a line through data points.*
> *Part 3: you built a neural network from scratch.*
> *Part 5: you tokenized text and classified sentiment.*
>
> *Today you are building a Transformer. From scratch. In NumPy.*
>
> *Not using a library. Not calling model.fit(). Writing every matrix
> multiplication, every softmax, every residual connection yourself.*
>
> *This is one of the most celebrated architectures in the history of AI.
> And you are going to understand it at the level of every single number.*
>
> *Let's go."*

Point at the hyperparameters on the board:
```
D_MODEL    = 64    (embedding dimension)
NUM_HEADS  = 4     (attention heads)
D_FF       = 256   (FFN hidden size = 4 × D_MODEL)
NUM_LAYERS = 2     (encoder blocks stacked)
MAX_LEN    = 30    (max sequence length)
VOCAB_SIZE = 60    (number of tokens)
```

> *"These are tiny numbers — a real BERT uses D_MODEL=768, 12 heads, 12 layers.
> Ours is a teaching transformer. The math is identical."*

---

## SECTION 1: Scaled Dot-Product Attention — Review and Code (25 min)

Write the formula:
```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

Q: [seq, d_k]    Queries  — "what am I looking for?"
K: [seq, d_k]    Keys     — "what do I offer to be found?"
V: [seq, d_v]    Values   — "what information do I provide?"

Step 1:  scores = Q · Kᵀ           [seq, seq]
Step 2:  scores = scores / √d_k    (scale to prevent softmax saturation)
Step 3:  weights = softmax(scores)  [seq, seq]   ← attention distribution
Step 4:  output = weights · V       [seq, d_v]   ← weighted information
```

> *"The spotlight analogy one more time:*
>
> *Q is the spotlight. It asks: where should I look?*
> *K is each actor on stage. It says: here's what I'm about.*
> *V is what the actor actually says when the spotlight hits.*
>
> *Score = spotlight direction · actor position = how aligned are they.*
> *Softmax = turn scores into a distribution: which actors get how much light.*
> *Output = weighted sum of what actors say."*

**Code together:**
```python
import numpy as np

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))  # numerically stable
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)  # [batch, heads, seq, seq]
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)   # mask out forbidden positions
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

# Quick test
seq, d_k = 4, 8
Q = np.random.randn(seq, d_k)
K = np.random.randn(seq, d_k)
V = np.random.randn(seq, d_k)
out, attn = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {out.shape}")        # (4, 8)
print(f"Attention weights sum:  {attn.sum(axis=-1)}")  # all 1.0
```

**Ask the room:** *"Why do we subtract the max before computing exp?
What goes wrong without it?"*

> *"Numerical stability. exp(800) causes overflow — the number is too large
> for floating point. Subtracting the max doesn't change the softmax output
> mathematically, but keeps every exponent near zero where it's safe."*

---

## SECTION 2: Multi-Head Attention — Split, Compute, Recombine (25 min)

Draw the multi-head split:
```
INPUT x:  [seq, d_model]   d_model = 64

SPLIT into NUM_HEADS = 4 heads:
  d_k = d_model / num_heads = 64 / 4 = 16

W_Q: [d_model, d_model]    W_K: [d_model, d_model]    W_V: [d_model, d_model]
  ↓                           ↓                           ↓
Q = x·W_Q  [seq, 64]       K = x·W_K  [seq, 64]       V = x·W_V  [seq, 64]

Reshape to 4 heads:
  Q: [4, seq, 16]            K: [4, seq, 16]            V: [4, seq, 16]

4 × scaled_dot_product_attention:
  head_1: [seq, 16]
  head_2: [seq, 16]
  head_3: [seq, 16]
  head_4: [seq, 16]

Concatenate along last axis: [seq, 64]
Final projection W_O [64, 64]: [seq, 64]
```

> *"Each head runs independent attention with its own W_Q, W_K, W_V.
> But we do it all in one big matrix multiplication and reshape —
> which is why it's efficient.*
>
> *Head 1 might focus on syntactic structure.
> Head 2 might focus on coreference — which 'it' refers to.
> Head 3 might focus on local context — adjacent words.
> They run in parallel. They see the same input but ask different questions."*

**Code together:**
```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Random initialization (in training, these get optimized)
        self.W_Q = np.random.randn(d_model, d_model) * 0.02
        self.W_K = np.random.randn(d_model, d_model) * 0.02
        self.W_V = np.random.randn(d_model, d_model) * 0.02
        self.W_O = np.random.randn(d_model, d_model) * 0.02

    def forward(self, x, mask=None):
        B, S, D = x.shape
        Q = (x @ self.W_Q).reshape(B, S, self.num_heads, self.d_k).transpose(0,2,1,3)
        K = (x @ self.W_K).reshape(B, S, self.num_heads, self.d_k).transpose(0,2,1,3)
        V = (x @ self.W_V).reshape(B, S, self.num_heads, self.d_k).transpose(0,2,1,3)
        # Q,K,V: [B, num_heads, S, d_k]
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(0,2,1,3).reshape(B, S, D)  # [B, S, d_model]
        return out @ self.W_O, weights

# Test
mha = MultiHeadAttention(d_model=64, num_heads=4)
x_test = np.random.randn(1, 5, 64)   # batch=1, seq=5, d_model=64
out, w = mha.forward(x_test)
print(f"MHA output: {out.shape}")    # (1, 5, 64)
print(f"Attention weights: {w.shape}")  # (1, 4, 5, 5)
```

---

## CLOSING SESSION 1 (5 min)

```
TODAY:
  scaled_dot_product_attention  →  [seq, d_k] → [seq, d_v]
  MultiHeadAttention            →  [B, S, D] → [B, S, D]
  (pre-LN) encoder sub-layer   →  LayerNorm → transform → + residual

NEXT SESSION:
  Layer normalization in NumPy
  Feed-Forward Network
  Full EncoderBlock = MHA + FFN + both residuals
  Full EncoderStack = N blocks
```

**Homework:** Look at the `reshape` + `transpose` calls in MHA.
Draw on paper exactly what shape the tensor has at each step.

---

# SESSION 2 (~90 min)
## "Layer norm, FFN, full encoder stack, and attention heatmaps"

## OPENING (10 min)

> *"Last session we built scaled dot-product attention and multi-head
> attention in NumPy. Today we add the last pieces: layer normalization,
> the feed-forward network, and assemble the complete encoder stack.*
>
> *By the end of today's session, you will have a function that takes a
> batch of token IDs and produces contextual embeddings.*
>
> *Then we'll visualize what the attention heads are actually looking at —
> the heatmap that shows you the model's 'thoughts'.*
>
> *And to prove the transformer actually works, we'll train a logistic
> regression classifier on its output. Even with random weights, the
> structure itself encodes useful patterns."*

---

## SECTION 1: Layer Norm and FFN in NumPy (20 min)

**LayerNorm:**
```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)   # scale
        self.beta  = np.zeros(d_model)  # shift
        self.eps = eps

    def forward(self, x):
        # x: [batch, seq, d_model]
        mu    = x.mean(axis=-1, keepdims=True)    # mean over d_model
        sigma = x.std(axis=-1, keepdims=True)     # std over d_model
        x_hat = (x - mu) / (sigma + self.eps)
        return self.gamma * x_hat + self.beta

ln = LayerNorm(64)
x_test = np.random.randn(1, 5, 64) * 10 + 3  # wildly scaled input
out = ln.forward(x_test)
print(f"Mean after LN: {out.mean(axis=-1).round(3)}")   # ≈ 0
print(f"Std  after LN: {out.std(axis=-1).round(3)}")    # ≈ 1
```

> *"Notice: we normalize over the last axis — d_model — for each token
> independently. A 64-dimensional token vector gets mapped to mean≈0, std≈1.
> Then gamma and beta (learned) can rescale it to whatever range the model needs."*

**FeedForward:**
```python
def relu(x):
    return np.maximum(0, x)

class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # x: [batch, seq, d_model]
        return relu(x @ self.W1 + self.b1) @ self.W2 + self.b2
```

> *"64 → 256 → 64. Each token gets the same two linear layers applied.
> Applied position-wise — independently per token, same weights everywhere."*

---

## SECTION 2: The Full Encoder Block (20 min)

```python
class EncoderBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn  = FeedForward(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Sub-layer 1: Pre-LN + MHA + residual
        attn_out, weights = self.attn.forward(self.ln1.forward(x), mask)
        x = x + attn_out

        # Sub-layer 2: Pre-LN + FFN + residual
        x = x + self.ffn.forward(self.ln2.forward(x))

        return x, weights
```

Draw on board exactly what this maps to:
```
EncoderBlock.forward(x):
  x ─── ln1 ─── MHA ─── + ─── x'
  │                      ▲
  └──────────────────────┘ (residual skip)

  x' ─── ln2 ─── FFN ─── + ─── output
  │                       ▲
  └───────────────────────┘ (residual skip)
```

> *"Six lines of Python. That's one full encoder block.
> The original Transformer stacks 6 of these. BERT-Large stacks 24."*

---

## SECTION 3: Full Encoder Stack + Mean Pooling (15 min)

```python
class TransformerEncoder:
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, max_len):
        # Token embeddings: lookup table
        self.embedding = np.random.randn(vocab_size, d_model) * 0.02

        # Positional encoding (sinusoidal — from Module 03)
        pos = np.arange(max_len)[:, np.newaxis]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        self.pos_enc = pe

        # Stack of encoder blocks
        self.blocks = [EncoderBlock(d_model, num_heads, d_ff)
                       for _ in range(num_layers)]
        self.final_ln = LayerNorm(d_model)

    def forward(self, token_ids):
        # token_ids: [batch, seq_len]
        x = self.embedding[token_ids]          # [batch, seq, d_model]
        x = x + self.pos_enc[:token_ids.shape[1]]
        all_weights = []
        for block in self.blocks:
            x, w = block.forward(x)
            all_weights.append(w)
        x = self.final_ln.forward(x)
        pooled = x.mean(axis=1)                # [batch, d_model]  — mean pool
        return pooled, x, all_weights

encoder = TransformerEncoder(60, 64, 4, 256, 2, 30)
token_ids = np.random.randint(0, 60, (8, 10))  # batch=8, seq=10
pooled, full_out, weights = encoder.forward(token_ids)
print(f"Pooled representation: {pooled.shape}")   # (8, 64)
print(f"Full sequence output:  {full_out.shape}") # (8, 10, 64)
```

> *"Mean pooling: we average across all sequence positions.
> Each position has a 64-dimensional vector. We average them into one.
> That's the document-level representation.*
>
> *You could also just take position 0 — a learnable [CLS] token, like BERT does.
> Mean pooling works well for random-weight demonstration."*

---

## SECTION 4: Attention Heatmap — Seeing Inside the Model (15 min)

```python
import matplotlib.pyplot as plt

# Get attention weights from the first layer, first head
w = weights[0][0, 0, :, :]    # [seq, seq] — first batch, first layer, first head
tokens = [f"t{i}" for i in range(10)]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(w, cmap='Blues', vmin=0, vmax=w.max())
ax.set_xticks(range(10)); ax.set_xticklabels(tokens)
ax.set_yticks(range(10)); ax.set_yticklabels(tokens)
ax.set_xlabel("Key (source)"); ax.set_ylabel("Query (what attends)")
ax.set_title("Attention Head 0 — Layer 0")
plt.colorbar(im, ax=ax)
plt.savefig("visuals/transformer_from_scratch/attention_heatmap.png", dpi=150, bbox_inches='tight')
print("Saved attention heatmap.")
```

> *"Each row is a query token. Each column is a key token.
> The color shows how much the query token attends to each key.*
>
> *Bright blue = high attention. Dark = ignored.*
>
> *With random weights these patterns don't mean much.
> But in a trained BERT, you'd see clear linguistic patterns:
> subjects attending to verbs, pronouns attending to their antecedents.*
>
> *This is what interpretability researchers spend their careers on."*

**Ask the room:** *"If row i has a very bright column j, what does that mean
in plain English?"*

> *"Token i is 'looking at' token j. When building its contextual
> representation, token i is heavily influenced by token j's information."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
WHAT WE BUILT TODAY (pure NumPy):

  scaled_dot_product_attention(Q,K,V)
          ↑
  MultiHeadAttention (split → attend → concat → project)
          ↑
  EncoderBlock (Pre-LN + MHA + residual, Pre-LN + FFN + residual)
          ↑
  TransformerEncoder (embed → pos_enc → N blocks → mean pool)
          ↑
  LogisticRegression on top of pooled reps → works!

FROM SCRATCH means:
  No TensorFlow. No PyTorch. No sklearn attention.
  Every dot product, every softmax — you wrote it.
```

**Homework:** Run `python3 transformer_from_scratch.py` and look at the
generated attention heatmap. Can you find any token that attends mostly to
itself? What positions tend to attract the most attention?

---

## INSTRUCTOR TIPS

**"The shapes are confusing me — I keep losing track"**
> *"Write the shape in a comment after every single operation.
> Make it a habit. [batch, seq, d_model]. When shapes don't match,
> the operation tells you exactly what's wrong.*
>
> *A good rule: matmul requires the LAST axis of left to equal
> the SECOND-TO-LAST axis of right. Check that always."*

**"Why mean pooling instead of using the first token?"**
> *"Both work. Mean pooling averages information from all positions —
> good for representing the whole document.*
> *First token is what BERT does with [CLS] — it learns to gather global
> information during pretraining. With random weights (no training),
> [CLS] hasn't learned anything yet, so mean pooling is more informative.*
>
> *Next session we'll see BERT actually use [CLS] the right way."*

**"Our transformer has random weights. Why does logistic regression work?"**
> *"Even random weights create structured representations.
> The architecture itself — the residual connections, the normalization,
> the attention pattern — imposes structure on the data.*
> It's like how a random forest with random thresholds still does
> better than random guessing. The structure helps."*

**"This is a lot of code. How do I remember all of it?"**
> *"You don't memorize code. You understand the architecture.
> If you can draw the encoder block from memory, you can re-derive
> every line of code. That's what we've been doing all session."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — celebrate the milestone   10 min
├── Scaled dot-product attention        25 min
├── Multi-head attention + code         25 min
├── Break (if needed)                    5 min
├── Shapes deep dive + questions        20 min
└── Close + homework                     5 min

SESSION 2  (90 min)
├── Opening bridge                      10 min
├── LayerNorm + FFN in NumPy            20 min
├── Full EncoderBlock                   20 min
├── TransformerEncoder + mean pool      15 min
├── Attention heatmap visualization     15 min
└── Close + board summary               10 min
```

---
*MLForBeginners · Part 6: Transformers · Module 05*
