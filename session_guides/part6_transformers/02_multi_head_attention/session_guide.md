# MLForBeginners — Instructor Guide
## Module 2: Multi-Head Attention  ·  Two-Session Teaching Script
### Part 6: Transformers

> **Who this is for:** You, teaching close friends who completed Part 6 Module 1.
> **Their background:** They understand Q/K/V, scaled dot-product attention, and have seen the attention heatmap.
> **Tone:** Casual, curious, conversational.
> **Goal by end of both sessions:** Everyone understands why multiple heads beat one, can trace the full forward pass, and has seen per-head attention patterns in the visualizations.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "What if eight people watched the same movie?"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/transformers/math_foundations/`
- `02_multi_head_attention.py` open in your editor
- Whiteboard — you will draw the head diagram several times
- Generated visuals from `transformers/visuals/02_multi_head_attention/` (run the script beforehand)
- Write this on the board before they arrive:

```
Single head:   1 person watching a movie, taking notes
Multi-head:    8 people watching, each focused on something different
```

---

## OPENING  (10 min)

### Hook — The movie analogy

> *"Quick question — if you wanted to fully understand a complex movie,
> would you rather watch it alone once, or watch it with 7 friends
> who each pay attention to different things?*
>
> *Friend 1 tracks the plot structure.*
> *Friend 2 notices all the character relationships.*
> *Friend 3 catches the visual symbolism.*
> *Friend 4 listens to the score and how it changes.*
> *...*
>
> *Then you all compare notes. You end up with a much richer understanding
> than any one of you could have gotten alone.*
>
> *That's multi-head attention. And it's why Transformers are so powerful."*

Write on the board:

```
Multi-Head Attention (h=8):
  head_1: attends to syntactic relationships
  head_2: attends to coreference (it → the cat)
  head_3: attends to nearby positions (local context)
  head_4: attends to rare, long-distance dependencies
  ... 4 more heads, each specializing differently

  Concat all heads → project back → rich representation
```

> *"Nobody programs these specializations. The model learns them during training.*
> *That's the magic — different heads spontaneously discover different linguistic patterns."*

---

## SECTION 1: Why One Head Isn't Enough  (20 min)

> *"Last session we built attention with one Q, one K, one V.*
> *One head computes one attention pattern.*
> *But language has many types of relationships happening simultaneously.*"

**Draw on whiteboard:**

```
The sentence: "The hungry cat chased the startled bird."

Relationships in this sentence:
  Subject-verb:   "cat" → "chased"
  Adjective-noun: "hungry" → "cat"
  Object:         "chased" → "bird"
  Adjective-noun: "startled" → "bird"
  Determiner:     "The" → "cat"
  Determiner:     "the" → "bird"

With ONE attention head:
  It must represent ALL of these in one pattern.
  The attention matrix is one fixed grid.
  Competing relationships interfere with each other.

With EIGHT heads:
  Head 1 can focus on subject-verb
  Head 2 can focus on adjective-noun
  Head 3 can focus on determiner-noun
  ... etc.
  No interference. Specialization.
```

> *"Think of it like this: if you had to describe a city using only ONE photograph,
> you'd have to pick one angle, one moment, one perspective.*
> *With eight photographers, you get eight angles.*
> *The combination is much more informative."*

**Ask the room:**

> *"What kinds of relationships do you think a Transformer for CODE would learn?*
> *Different from English sentences — what would the heads specialize in?"*

Good answers:
- Function calls and their arguments
- Variable definitions and their usages
- Opening brackets and closing brackets
- Loop variables and loop bodies

> *"Exactly. The architecture is language-agnostic. Python code, legal contracts,*
> *medical notes — the heads always find the relevant structure."*

---

## SECTION 2: The Multi-Head Forward Pass  (25 min)

### Walk through the architecture step by step

> *"Let me trace exactly what happens in a multi-head attention layer.*
> *I'm going to draw this in detail — follow along."*

**Draw on board — step by step:**

```
INPUT: x  [seq_len, d_model]      e.g., [10, 512]
           (sequence of token embeddings)

STEP 1: For each head i (1..8), project to smaller subspace:
  Q_i = x × W_Q_i    [10, 64]    (d_k = 512/8 = 64)
  K_i = x × W_K_i    [10, 64]
  V_i = x × W_V_i    [10, 64]

STEP 2: Run scaled dot-product attention per head:
  head_i = Attention(Q_i, K_i, V_i)    [10, 64]

STEP 3: Concatenate all heads:
  Concat(head_1, ..., head_8)    [10, 512]    (8 × 64 = 512)

STEP 4: Final linear projection:
  Output = Concat(...) × W_O    [10, 512]

Total parameters:
  8 × (W_Q + W_K + W_V) = 8 × 3 × (512×64) = 786,432
  W_O = 512×512 = 262,144
  SAME as single head with d_k=512!
```

> *"Here's the beautiful part — the total parameter count is IDENTICAL*
> *to a single attention head with full dimension.*
> *You get 8 perspectives for free. No extra parameters.*
> *You're splitting the representation space, not adding to it."*

**Write the formula clearly:**

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O

where head_i = Attention(Q × W_Q_i,  K × W_K_i,  V × W_V_i)
```

**Interactive moment:**

> *"Walk me through the shapes. If d_model = 512 and h = 8:*
> *What is d_k? What shape is each W_Q_i?*
> *What shape is the concatenated output before W_O?"*

Let them work it out:
- d_k = 512 / 8 = 64
- W_Q_i shape: (512, 64)
- Concat output: (seq_len, 512) — 8 heads × 64 each

> *"Exactly right. Now you can read any paper that mentions multi-head attention.*
> *It's always this same pattern."*

---

## SECTION 3: Live Python Demo  (25 min)

> *"Let's run the module and see this computed in real numpy."*

**Open terminal:**

```bash
cd /Users/ravithati/MLForBeginners/transformers/math_foundations
python3 02_multi_head_attention.py
```

**As it runs, point at key moments:**

When Section 1 prints (parameter count):
> *"See — it's computing 786,432 parameters for single-head and multi-head.*
> *They're the same. That's the key insight we just worked through."*

When Section 2 prints (from-scratch code):
> *"Here's the actual multi-head implementation in numpy.*
> *Notice the reshape — it splits the d_model dimension into h heads.*
> *That's how you do h parallel attention computations efficiently:*
> *no for-loop, just a reshape and batch matrix multiply."*

When Section 3 prints (head specialization):
> *"Now look at this — it's showing which tokens each head attends to most.*
> *In a trained model, you'd see real specialization here.*
> *In our random-weight model, it's just demonstrating the STRUCTURE.*
> *But the mechanism is identical to what GPT and BERT do."*

**Open the generated visuals:**

> *"This is what we care about — the per-head attention heatmaps.*
> *Each subplot is one head's attention pattern.*
> *See how they're different from each other?*
> *That's the specialization. Even without training, the random projections*
> *create different patterns. With training, these become meaningful."*

**Ask the room:**

> *"If you had to guess — which head is tracking syntax*
> *and which is tracking local context?*
> *How would you tell from the heatmap?"*

Guide: local context heads have a strong diagonal. Long-range heads have scattered high-weight cells.

---

## CLOSING SESSION 1  (10 min)

### Recap board

```
MULTI-HEAD ATTENTION — KEY IDEAS
────────────────────────────────────
Single head: one attention pattern in full d_model space
Multi-head: h patterns in d_model/h subspaces each

Forward pass:
  1. Project Q, K, V into h subspaces (learned projections)
  2. Run attention independently per head
  3. Concatenate head outputs
  4. Final linear projection W_O

Parameters: SAME as single head (no free lunch in params,
            but huge gain in expressive power)

Specialization emerges: syntax, coreference, position, etc.
```

---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Complexity, head analysis, and connecting to the full Transformer"
# ─────────────────────────────────────────────

## Opening  (10 min)

> *"Last session we built multi-head attention from scratch.*
> *Today I want to go deeper on three things:*
> *What do trained heads actually learn,*
> *how does the complexity compare to LSTMs,*
> *and how does this fit into the full Transformer.*"

**Quick quiz — no pressure:**

> *"Without looking at your notes: what are the four steps of multi-head attention?*
> *Shout them out."*

Walk through as they answer. Fill in gaps on the board.

---

## SECTION 1: Head Specialization — What Research Shows  (20 min)

> *"Researchers have actually analyzed what attention heads learn in trained models.*
> *The results are fascinating — let me share a few."*

**Draw this on the board:**

```
FINDINGS FROM BERT (Clark et al. 2019 — "What Does BERT Look At?"):

  Head type           What it attends to
  ────────────────    ────────────────────────────────────────
  Direct object       "kicked" → "ball" (verb → direct object)
  Coreference         "it" → "the cat" (pronoun resolution)
  Next/prev token     Strong off-diagonal pattern
  [SEP] token         Many heads dump attention on separator tokens
  Syntactic tree      Some heads approximate dependency parse
```

> *"These heads were never told what to learn.*
> *They emerged from training on billions of words.*
> *The model discovered grammar.*
>
> *This is one of the most exciting results in NLP:*
> *linguistic structure appears spontaneously in attention heads.*
> *You can use attention to ANALYZE language structure as a side effect."*

**Ask the room:**

> *"What would you expect from heads in a model trained on Python code?*
> *Or a model trained on legal documents?*
> *Would the heads specialize in different things?"*

Discussion: yes — coding models show bracket matching, variable reference tracking, function signature heads. Legal models show citation reference, defined-term tracking.

---

## SECTION 2: Computational Complexity  (15 min)

> *"One of the key advantages of multi-head attention over LSTM —*
> *and one key disadvantage. Both matter in practice."*

**Draw on board:**

```
COMPARISON:

               LSTM              Transformer (Multi-Head Attn)
               ─────────────     ──────────────────────────────
Sequential?    YES (must         NO — all positions computed
               process in        simultaneously
               order)

Time (train):  O(n × d²)        O(n² × d)
               n=sequence len,  n²: all pairs compared
               d=hidden dim     d: model dimension

Memory:        O(n × d)         O(n² + n × d)

Parallelism:   Limited          FULLY parallel on GPU

Path length    O(n)             O(1)
between        signal must      any two positions connect
distant words: travel n steps   in one attention step
```

> *"For n < 512, Transformer wins decisively on training speed.*
> *For very long sequences (n > 1000), the n² cost becomes a problem.*
> *That's why you have maximum sequence lengths in BERT (512 tokens)*
> *and why handling book-length documents requires special tricks."*

---

## SECTION 3: Connecting to the Full Architecture  (20 min)

> *"Multi-head attention is a layer — it fits into a larger structure.*
> *Let me preview how, because next session we build the full Encoder-Decoder.*"

**Draw on board — preview of the full Transformer:**

```
TOKEN INPUT:
  "The" "cat" "sat"
       ↓
  Token Embeddings [seq, d_model]
       ↓
  + Positional Encoding (next session!)
       ↓
┌─────────────────────────────────────────┐
│         ENCODER BLOCK (×N)              │
│                                         │
│  x ──→ LayerNorm ──→ MultiHeadAttn ──→ + ──→ x'
│                           ↑ Q=K=V=x         │
│                      (SELF-ATTENTION)   residual
│                                         │
│  x' ──→ LayerNorm ──→ FeedForward  ──→ + ──→ x''
│                                    residual  │
└─────────────────────────────────────────┘
       ↓
  [ENCODER OUTPUT] — rich contextual representations
```

> *"Multi-head self-attention is the core of EVERY Transformer encoder block.*
> *It runs N times — 6 layers in the original paper, 12 in BERT-base, 96 in GPT-3.*
> *Each layer refines the representation using what it learned from all positions.*
> *Layer 1 might learn basic syntax. Layer 12 might learn abstract semantics.*"

---

## SECTION 4: Common Implementation Details  (15 min)

> *"When you read Transformer code in PyTorch or TensorFlow,*
> *here are the details that trip people up."*

**Write on board:**

```
IMPLEMENTATION DETAILS:

1. DROPOUT in attention:
   Apply dropout to attention WEIGHTS (not values)
   Randomly zero out some attention connections during training
   Prevents over-reliance on specific positions

   weights = dropout(softmax(scores / sqrt(d_k)))

2. BATCH DIMENSION:
   In practice, input is [batch, seq, d_model]
   All matrix multiplies are batched
   Shapes: Q [batch, heads, seq, d_k]

3. EFFICIENT RESHAPING (no for-loop over heads):
   # Project all heads at once:
   Q = x @ W_Q  # [batch, seq, d_model]
   # Reshape to separate heads:
   Q = Q.view(batch, seq, h, d_k).transpose(1, 2)
   # Now: [batch, heads, seq, d_k]
   # Batch matmul handles all heads simultaneously

4. PADDING MASK:
   For variable-length sequences in a batch,
   shorter sequences are padded with zeros
   Padding tokens must get zero attention weight
   → Additional mask on top of causal mask (for decoder)
```

> *"When you look at PyTorch's nn.MultiheadAttention,*
> *this is exactly what's inside. Now you can read the source code*
> *and understand every line."*

---

## CLOSING SESSION 2  (10 min)

### Full recap board

```
WHAT WE NOW KNOW
────────────────────────────────────────────────────────
Module 1: Attention(Q,K,V) = softmax(Q×K^T / √d_k) × V

Module 2: MultiHead(Q,K,V) = Concat(head_1...head_h) × W_O
          where head_i = Attention(x×W_Qi, x×W_Ki, x×W_Vi)

Key facts:
  → Same params as single head, but h independent perspectives
  → Heads specialize: syntax / coreference / position / semantics
  → Fully parallel → trains much faster than LSTM
  → O(n²) cost: problem for very long sequences
  → Fits inside Encoder block: MultiHead + FFN, each with residual + LayerNorm
```

### The Road Ahead

```
WHERE WE ARE:
  ✅ Attention mechanism
  ✅ Multi-head attention

NEXT:
  → Positional encoding: how does the Transformer know word order?
    (It doesn't — we have to tell it. That's the problem we solve next.)
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## Common Confusions and What to Say

**"So we run attention 8 times? Isn't that 8x slower?"**
> *"No — and this is the key insight. All 8 heads run in PARALLEL.*
> *We do one big matrix multiply that handles all heads simultaneously*
> *using the reshape + batched-matmul trick.*
> *It's not 8 separate passes — it's one vectorized operation.*
> *That's what makes GPUs so good at this."*

**"If the parameter count is the same, why are multiple heads better?"**
> *"Think of it like this: a single head with dimension 512 must use all 512 dimensions*
> *for ONE type of relationship. The 512 dimensions are all competing.*
> *Multi-head splits into 8 independent subspaces of 64 each.*
> *Each subspace learns one type of relationship without interference.*
> *Same budget, better allocation."*

**"How does the model know which head should learn what?"**
> *"It doesn't prescribe this — it emerges from backpropagation.*
> *If two heads learn the same thing, the gradient signal pushes them apart*
> *because there's no benefit from redundancy.*
> *The optimization pressure causes specialization naturally."*

**"What is W_O doing? Why do we need that final projection?"**
> *"After concatenating 8 heads, you have [seq, d_model] again.*
> *W_O mixes the contributions from all heads.*
> *Without it, each head's output would be in a separate 'lane' —*
> *they'd never interact. W_O allows head 3's output to inform head 7's region.*
> *It's the integration step."*

**"Do all Transformers use 8 heads? What about 12 or 16?"**
> *"h is a hyperparameter. BERT-base uses 12, BERT-large uses 16.*
> *GPT-3 uses 96. The constraint is d_model must be divisible by h.*
> *More heads = finer-grained specialization but smaller per-head dimension.*
> *In practice, 8-16 is the sweet spot for most tasks."*

## Energy Management

- **Session 1 at 45 min:** Draw the full forward pass diagram. Physical activity (drawing) resets energy.
- **Session 2 at 30 min:** The "what do trained heads learn" section is inherently interesting — lean into it.
- **If someone is lost on shapes:** Stop. Do one concrete example on the board with small numbers (d_model=4, h=2, d_k=2). Size doesn't matter for understanding the pattern.
- **If ahead of schedule:** Ask them to implement a 2-head attention in a Python REPL live. Very instructive.

## The Golden Rule

> The movie analogy works better than any equation for the "why."
> Use it at the start of Session 1 and again at the start of Session 2.
> After two exposures, it sticks.

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening — movie analogy              10 min
├── Section 1: Why one head isn't enough 20 min
├── Section 2: Full forward pass walkthrough 25 min
├── Section 3: Live Python demo          25 min
└── Closing recap                        10 min

SESSION 2  (90 min)
├── Quick quiz recap                     10 min
├── Section 1: Head specialization research 20 min
├── Section 2: Complexity comparison     15 min
├── Section 3: Full architecture preview 20 min
├── Section 4: Implementation details   15 min
└── Closing + road ahead                 10 min
```

---

*Generated for MLForBeginners — Module 02 · Part 6: Transformers*
*Source: transformers/math_foundations/02_multi_head_attention.py*
