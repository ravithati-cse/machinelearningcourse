# MLForBeginners — Instructor Guide
## Part 6 · Module 04: Encoder-Decoder Architecture
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–03 complete. They know attention mechanism,
> multi-head attention, and positional encoding. They can compute Q·Kᵀ by hand
> and explain why we scale by 1/√d_k.
> **Payoff today:** They will see the full Transformer stack — encoder AND
> decoder — and understand every single layer inside each block.

---

# SESSION 1 (~90 min)
## "Inside the encoder block — MHA, FFN, residuals, and LayerNorm"

## Before They Arrive
- Terminal open in `transformers/math_foundations/`
- Whiteboard with two large empty rectangles labeled ENCODER and DECODER
- Have the original Vaswani 2017 paper figure ready (google "Attention Is All You Need figure 1")

---

## OPENING (10 min)

> *"Over the last three sessions you've built three pieces of a puzzle.
> You know how attention works — the spotlight on a stage.
> You know how multi-head attention runs eight spotlights at once.
> You know how positional encoding tells the model where words live.*
>
> *Today we assemble the whole machine.*
>
> *The original Transformer paper — 'Attention Is All You Need' — describes
> a stack of N=6 encoder blocks and N=6 decoder blocks. You're going to
> understand every single layer in every single block. After today,
> that famous architecture diagram will make complete sense."*

Draw on board:
```
TRANSFORMER
┌─────────────────────┐     ┌─────────────────────┐
│     ENCODER         │     │      DECODER        │
│  [reads the input]  │────▶│ [generates output]  │
│                     │     │                     │
│  "The cat sat on"   │     │  "Le chat s'est"    │
└─────────────────────┘     └─────────────────────┘
```

> *"English → French translation. The encoder reads and understands
> the English sentence. The decoder generates the French sentence,
> using the encoder's understanding at every step."*

---

## SECTION 1: The Full Encoder Block (25 min)

Write the encoder block on board piece by piece:

```
ONE ENCODER BLOCK  (repeated N=6 times)
════════════════════════════════════════

INPUT: x  [seq_len, d_model]

   x ──────────────────────────────────────────────────┐
   │                                                    │
   ▼                                                    │ (residual)
LayerNorm(x)                                           │
   │                                                    │
   ▼                                                    │
Multi-Head Self-Attention                              │
   │                                                    │
   └──────────────────────────── + ◀──────────────────┘
                                  │
                                  x'  (after first Add & Norm)
                                  │
   ┌──────────────────────────────┘
   │                                                    │
   ▼                                                    │ (residual)
LayerNorm(x')                                          │
   │                                                    │
   ▼                                                    │
Feed-Forward Network                                   │
   │                                                    │
   └──────────────────────────── + ◀──────────────────┘
                                  │
                                OUTPUT: x''  [seq_len, d_model]
```

> *"Two sub-layers per encoder block. Each sub-layer has the same
> three-part recipe: normalize, transform, add the original back.*
>
> *The 'add the original back' part — that's the residual connection.
> That's what makes training deep networks possible. We'll come back to that."*

**The Feed-Forward Network (FFN):**
```
FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

Where:
  W₁:  [d_model → d_ff]    d_ff is usually 4 × d_model
  W₂:  [d_ff → d_model]

For d_model=512:  d_ff = 2048
  512 → 2048 → 512

Applied INDEPENDENTLY to each position.
"Each word thinks for itself after seeing the whole sentence."
```

> *"Attention is about mixing information between positions.
> FFN is about transforming each position's representation after mixing.
> They do different jobs."*

**Ask the room:** *"The FFN expands to 4x the width then compresses back.
Why not just use one layer? What does the middle expanded space give us?"*

Pause. Let them think.

> *"The expanded middle layer gives the network room to work. It can form
> complex intermediate representations — combinations of features — that
> it then compresses back into the d_model space. Think of it like
> a scratch pad."*

---

## SECTION 2: Layer Normalization — Why Not Batch Norm? (20 min)

Draw on board:

```
BATCH NORMALIZATION          LAYER NORMALIZATION
(normalizes across batch)    (normalizes across features)

Batch:   sentence1 ──┐       sentence1:  word1 word2 word3
         sentence2 ──┼──▶ μ             │────────────────│
         sentence3 ──┘                  normalize ACROSS features
                                         for each token independently

Problem: What if batch=1?    Works for ANY batch size.
         Padding messes up μ  Sequences can be different lengths.
         RNNs/Transformers    No issue with variable-length text.
         have variable lengths.
```

Write the formula:
```
LayerNorm(x) = γ × (x - μ) / (σ + ε) + β

μ = mean over the d_model features of ONE token
σ = std dev over the d_model features of ONE token
γ, β = learned scale and shift parameters
ε = small number for numerical stability (e.g. 1e-6)
```

> *"Every token in every sentence gets normalized independently.
> It doesn't matter how long the sentence is, how big the batch is,
> or whether you're training or testing. LayerNorm always works.*
>
> *Batch Norm needs big batches to estimate μ and σ reliably.
> In NLP, sequences have wildly different lengths. LayerNorm is the solution."*

**Live Demo — feel LayerNorm in 6 lines:**
```python
import numpy as np

x = np.array([2.0, 4.0, 6.0, 8.0])  # one token's embedding
mu = x.mean()
sigma = x.std()
eps = 1e-6
gamma, beta = 1.0, 0.0  # learnable, start at identity

x_norm = gamma * (x - mu) / (sigma + eps) + beta
print(f"Input:      {x}")
print(f"Mean: {mu:.1f}  Std: {sigma:.1f}")
print(f"Normalized: {x_norm.round(3)}")
```

---

## SECTION 3: Residual Connections — The Secret to Depth (15 min)

> *"Here's a question: why do we add x back after every sub-layer?
> Why not just pass the transformed output forward?*
>
> *In 2015, people discovered that very deep networks stopped training.
> Gradients vanished before they reached the early layers.
> The fix: give the gradient a highway — a direct path backwards.*
>
> *The residual connection is that highway."*

Draw on board:
```
WITHOUT RESIDUALS:           WITH RESIDUALS:
 x                            x ──────────────────────────┐
 │                            │                           │
 ▼                            ▼                           │
transform(x)                transform(x)                  │
 │                            │                           │
 ▼                            └──────────────── + ◀───────┘
output                        output = transform(x) + x

Gradient flows through transform only.    Gradient can skip directly:
Easy to vanish in deep networks.           ∂L/∂x = ∂L/∂output (direct)
                                           + gradient through transform
                                           = MUCH harder to vanish
```

> *"The residual path says: 'whatever you can't figure out how to change,
> just pass through unchanged.' It's an identity shortcut.*
>
> *In the worst case, the transform layer can learn to output near-zero
> and the residual just flows through. In the best case, the transform
> learns something genuinely useful and adds it to x.*
>
> *This is why ResNets work. This is why Transformers work.
> Residuals are one of the most important ideas in deep learning."*

---

## CLOSING SESSION 1 (5 min)

```
TODAY'S ENCODER BLOCK:
  LayerNorm → Multi-Head Self-Attention → Add (residual)
  LayerNorm → Feed-Forward Network       → Add (residual)

  LayerNorm:  normalize across features, not batch
  FFN:        d_model → 4×d_model → d_model, per position
  Residuals:  gradient highway, makes deep networks trainable
```

**Homework:** Draw an encoder block from memory. Label every component.
Check against today's board diagram.

---

# SESSION 2 (~90 min)
## "The decoder, cross-attention, and Pre-LN vs Post-LN"

## OPENING (10 min)

> *"The encoder reads. The decoder writes.*
>
> *Last time we built the encoder block from the inside out.
> The decoder block is almost the same — but with one extra piece
> that makes translation possible: cross-attention.*
>
> *Cross-attention is how the decoder looks at the encoder's output.
> It's how the model connects 'what the English means' to
> 'what French word comes next'.*
>
> *Today we build the full stack, then I'll show you the difference
> between the original Transformer (Post-LN) and the modern version (Pre-LN).
> One change. Huge impact."*

---

## SECTION 1: The Full Decoder Block (25 min)

Draw on board in full:

```
ONE DECODER BLOCK  (repeated N=6 times)
════════════════════════════════════════

INPUT: y (partial output so far)  ← "Le chat s'est..."
ENCODER OUTPUT: memory            ← encoder's full understanding of English

STEP 1: MASKED Self-Attention
   y ─────────────────────────────────────────────────┐
   │                                                   │
   ▼                                                   │ (residual)
LayerNorm(y)                                          │
   │                                                   │
   ▼                                                   │
Masked Multi-Head Self-Attention                      │
(can only see positions ≤ current — future is hidden) │
   │                                                   │
   └─────────────────────────── + ◀───────────────────┘
                                 │
                                 y'

STEP 2: CROSS-Attention (the bridge to the encoder)
   y' ────────────────────────────────────────────────┐
   │                                                   │
   ▼                                                   │ (residual)
LayerNorm(y')                                         │
   │     Q = from y' (decoder)                        │
   │     K = from memory (encoder) ◀── encoder output │
   │     V = from memory (encoder) ◀── encoder output │
   ▼                                                   │
Multi-Head Cross-Attention                            │
(spotlight from decoder query onto encoder keys)      │
   │                                                   │
   └─────────────────────────── + ◀───────────────────┘
                                 │
                                 y''

STEP 3: Feed-Forward
   y'' ───────────────────────────────────────────────┐
   │                                                   │ (residual)
   ▼                                                   │
LayerNorm(y'')                                        │
   │                                                   │
   ▼                                                   │
Feed-Forward Network                                  │
   │                                                   │
   └─────────────────────────── + ◀───────────────────┘
                                 │
                               OUTPUT: y'''
```

> *"The decoder has THREE sub-layers versus the encoder's TWO.*
>
> *Sub-layer 1: Masked self-attention. The decoder reads what it has
> generated so far. 'Masked' means it can't peek at future tokens —
> it must predict them one at a time.*
>
> *Sub-layer 2: Cross-attention. This is where the decoder looks at
> the encoder's output. Q comes from the decoder, K and V come from
> the encoder. The spotlight analogy: the decoder is the spotlight operator,
> the encoder's tokens are the actors on stage.*
>
> *Sub-layer 3: FFN. Same as the encoder — per-position transformation."*

**Ask the room:** *"Why do we mask in sub-layer 1? What would happen
if the decoder could attend to future tokens?"*

> *"If the decoder could see future tokens, it could cheat — just copy
> the answer. The model would never learn to generate. The causal mask
> forces it to predict based only on what came before."*

---

## SECTION 2: Cross-Attention — The Bridge (15 min)

Write on board — the spotlight diagram with source labeled:

```
CROSS-ATTENTION

Q  ←  decoder's current state  (what am I trying to say?)
K  ←  encoder output           (what did the input mean?)
V  ←  encoder output           (what information can I extract?)

Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V

           spotlight              film reel of input meaning
              ↑                         ↑
   "What does the French need?"   "English token info"

Scores (Q·Kᵀ): how relevant is each English token to my French word?
Weights (softmax): distribution over English tokens
Output (·V): weighted sum of English token information
```

> *"When the decoder generates the word 'chat' (cat), the cross-attention
> weights should be high on 'cat' in the English encoder output.*
>
> *That's the mechanism by which translation works.
> The decoder spotlights the relevant part of the source sentence
> at each decoding step."*

**Live Demo:**
```python
import numpy as np

# Tiny example: 3-token encoder output, decoder at step 2
d_k = 4
encoder_output = np.random.randn(3, d_k)  # 3 English tokens
decoder_query  = np.random.randn(1, d_k)  # 1 French token being generated

scores = decoder_query @ encoder_output.T / np.sqrt(d_k)  # [1, 3]
weights = np.exp(scores) / np.exp(scores).sum()
print("Cross-attention weights over 3 English tokens:", weights.round(3))
print("Sum:", weights.sum().round(4))
```

---

## SECTION 3: Pre-LN vs Post-LN — A Critical Detail (20 min)

Draw both side by side:

```
POST-LN (original 2017 paper)    PRE-LN (modern, e.g. GPT-2, BERT)
══════════════════════════════   ══════════════════════════════════
x ──────────────────────┐        x ──────────────────────────────┐
│                        │        │                               │
▼                        │        ▼                               │
MHA(x)                   │        LayerNorm(x)                    │
│                        │        │                               │
└── + ◀──────────────────┘        ▼                               │
    │                             MHA(LayerNorm(x))               │
    ▼                             │                               │
LayerNorm(x + MHA(x))             └── + ◀──────────────────────── ┘
    │                                 │
    x'                                x'

Norm AFTER adding residual.       Norm BEFORE computing MHA/FFN.
```

> *"In Post-LN, the residual x gets added first, THEN normalized.
> This means the raw x goes through the attention unscaled.*
>
> *In Pre-LN, we normalize x first, THEN compute attention,
> THEN add the original raw x back. The shortcut path x
> is never transformed — it flows through clean.*
>
> *Why does this matter? Training stability.*
>
> *Post-LN requires very careful learning rate warmup schedules.
> Get the warmup wrong and the model diverges early in training.*
>
> *Pre-LN is much more stable. The gradients are better conditioned
> from the start. GPT-2, GPT-3, BERT, and most modern transformers
> use Pre-LN.*"

**Ask the room:** *"Which of the two is easier to train from scratch?
Which one matches what we saw in the code this week?"*

> *"Pre-LN. The code uses Pre-LN. It's what you should default to."*

---

## SECTION 4: The Full Stack (10 min)

Write the complete picture:
```
ENCODER STACK (N=6)              DECODER STACK (N=6)
═══════════════════               ═══════════════════
Input + PosEnc                   Partial output + PosEnc
     │                                │
[Encoder Block 1]                [Decoder Block 1] ◀── encoder_output
     │                                │
[Encoder Block 2]                [Decoder Block 2] ◀── encoder_output
     │                                │
    ...                              ...
     │                                │
[Encoder Block 6]                [Decoder Block 6] ◀── encoder_output
     │                                │
encoder_output                   Linear layer (d_model → vocab_size)
                                       │
                                  Softmax → probability over vocab
```

> *"Every decoder block gets the SAME encoder output — the full
> encoded representation of the input. Cross-attention into it
> at every layer.*
>
> *BERT drops the decoder entirely — pure encoder, bidirectional.
> GPT drops the encoder entirely — pure decoder, autoregressive.*
>
> *Both are simplifications of this full architecture.
> Now you understand why."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
ENCODER BLOCK:
  [Pre-LN] LayerNorm → Self-Attention → + residual
  [Pre-LN] LayerNorm → FFN            → + residual

DECODER BLOCK:
  [Pre-LN] LayerNorm → Masked Self-Attention  → + residual
  [Pre-LN] LayerNorm → Cross-Attention (Q=dec, K,V=enc) → + residual
  [Pre-LN] LayerNorm → FFN                    → + residual

PRE-LN vs POST-LN:
  Post-LN: original paper, needs careful warmup
  Pre-LN:  modern standard, more stable training
```

**Homework:** Run `python3 04_encoder_decoder_arch.py`. Look at the
visualizations. Can you identify which colors correspond to which sub-layers
in the architecture diagram?

---

## INSTRUCTOR TIPS

**"Why does the decoder need self-attention if it already has cross-attention?"**
> *"Cross-attention looks at the encoder. Self-attention looks at what the
> decoder has generated so far. The decoder needs both: understand the
> source (cross) AND build coherent output (self).*
>
> *Think of a translator who reads the English paragraph (cross-attention)
> while also keeping track of the French they've written so far (self-attention).
> Both are necessary."*

**"I'm confused about what Q, K, V are in cross-attention"**
> *"The rule: Q always comes from the layer asking the question.
> K and V always come from the source being queried.*
>
> *Cross-attention: the decoder is asking a question about the input.
> Q = decoder. K, V = encoder.*
>
> *Self-attention: the sequence is asking questions about itself.
> Q = K = V = the same sequence."*

**"Why do we need LayerNorm at all? Can't we just use residuals?"**
> *"Without normalization, the scale of activations drifts as you stack
> more layers. Some neurons blow up to huge values, others shrink to zero.
> LayerNorm keeps every token's representation in a reasonable range
> at every layer. Residuals prevent vanishing gradients.
> Together, they make 6-layer (or 96-layer) networks trainable."*

**"What's the causal mask exactly?"**
> *"A lower-triangular matrix of 1s. Position i can only attend to
> positions j ≤ i. We set masked positions to -infinity before softmax,
> so they get weight 0 after softmax. Future tokens become invisible."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                  10 min
├── Full encoder block            25 min
├── Layer Normalization           20 min
├── Residual connections          15 min
├── Live code: LayerNorm          10 min
└── Close + homework               5 min  (approx. — flex with demos)

SESSION 2  (90 min)
├── Opening bridge                10 min
├── Full decoder block            25 min
├── Cross-attention deep dive     15 min
├── Pre-LN vs Post-LN             20 min
├── Full stack overview           10 min
└── Close + homework              10 min
```

---
*MLForBeginners · Part 6: Transformers · Module 04*
