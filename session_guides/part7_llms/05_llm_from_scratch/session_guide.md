# MLForBeginners — Instructor Guide
## Part 7 · Module 05: Build a Mini-LLM (Character-Level GPT) From Scratch
### Two-Session Teaching Script  *** THE CROWN JEWEL ***

> **Prerequisites:** Part 6 Transformers complete (attention, multi-head attention,
> positional encoding, encoder-decoder). Part 3 DNNs. They understand gradient
> descent, cross-entropy loss, and the GPT architecture conceptually.
> **Payoff today:** They will build ChatGPT's ancestor — every component — in
> pure NumPy. No libraries. No abstraction. Just math and code.
> This is the most important session in the entire course.

---

# SESSION 1 (~90 min)
## "Building the brain — tokenizer, embeddings, and the MiniGPT architecture"

## Before They Arrive
- Terminal open in `llms/algorithms/`
- Whiteboard CLEAN — you will fill the entire board with the MiniGPT architecture
- Write at the top of the board in large letters: "YOU ARE ABOUT TO UNDERSTAND WHAT IS INSIDE CHATGPT"
- Have a physical book and some sticky notes on the desk (for analogies)
- Run the module once ahead of time so you know what the output looks like

---

## OPENING (15 min)

Pause. Let people settle. Then:

> *"I want to start today's session differently.*
>
> *You have been on a journey. You started with y = mx + b — a line through
> some points. You added more features, then you added neurons, then layers,
> then convolutions for images, then attention for sequences.*
>
> *Today is the summit. Today you are going to build — from absolute scratch,
> no libraries, pure NumPy — a small but real language model. The same
> architecture that powers GPT-2, GPT-3, and at its core, GPT-4.*
>
> *When this module runs, it will generate text. Novel text. Text that never
> existed before. The model will have learned it from a corpus of ten sentences
> about machine learning — and it will produce new sentences in that style.*
>
> *That's what 'language model' means: a distribution over sequences.
> You are about to build that distribution from scratch."*

Let it land.

> *"Let me draw the full architecture before we write a single line of code.
> This is the most important diagram of the entire course."*

Take 10 minutes. Draw on board — large, clear, with labels:
```
                    MiniGPT ARCHITECTURE
                    ====================

  Input tokens: [4, 17, 2, 9, ...]   (character IDs)
        |
        v
  [EMBEDDING LAYER]
    W_emb: (vocab_size, d_model)
    token_ids → dense vectors of size d_model
        |
        v
  [POSITIONAL ENCODING] (added, not concatenated)
    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
        |
        v
  [TRANSFORMER BLOCK]  ← repeated N_layers times
    ┌─────────────────────────────────────────┐
    │  [CAUSAL SELF-ATTENTION]                │
    │    Q = x @ W_Q,  K = x @ W_K,  V = x @ W_V │
    │    scores = QKᵀ / sqrt(d_k)            │
    │    MASK future positions → -inf         │
    │    attn = softmax(masked_scores) @ V    │
    │                                         │
    │  [LAYER NORM + RESIDUAL]                │
    │    x = LayerNorm(x + attn_output)       │
    │                                         │
    │  [FEED-FORWARD NETWORK]                 │
    │    x = x @ W1 (with ReLU) @ W2         │
    │                                         │
    │  [LAYER NORM + RESIDUAL]                │
    │    x = LayerNorm(x + ffn_output)        │
    └─────────────────────────────────────────┘
        |
        v
  [LM HEAD]  W_out: (d_model, vocab_size)
    x → logits (one score per character in vocabulary)
        |
        v
  [SOFTMAX] → probability over next character
```

> *"Every single one of these components — you understand. You built attention
> in Part 6. You built feedforward layers in Part 3. Layer norm is just
> mean-centering plus scaling. Positional encoding — Part 6 module 3.*
>
> *Today we wire them all together and give them a gradient descent training
> loop. That's it. That's a language model."*

---

## SECTION 1: Corpus and Character-Level Tokenizer (15 min)

> *"Real LLMs use subword tokenizers like BPE — byte-pair encoding.
> We're going to use character-level tokenization. Every character
> in our vocabulary is a token.*
>
> *This gives us a tiny vocabulary (32 unique characters vs 50,000 for GPT-4)
> so training is fast enough to see in this session."*

Show the tokenizer logic on the board:
```python
CORPUS = """
machine learning trains models on data...
...
""".strip()

chars = sorted(set(CORPUS))        # all unique characters
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for ch, i in char_to_id.items()}

def encode(text):  → list of integer IDs
def decode(ids):   → string
```

**Live demo — run just the tokenizer section:**
```bash
python3 llm_from_scratch.py 2>&1 | head -30
```

> *"Notice: 'machine learning' encodes to a list of integers — one per character.
> Decode those integers and you get back the original string. Round-trip verified.*
>
> *That's the tokenizer. Simple, exact, reversible."*

---

## SECTION 2: Embeddings — Turning Integers into Geometry (20 min)

> *"Here's the deep question: integers aren't meaningful for neural networks.
> Character ID 17 doesn't mean anything. We need to convert each ID into a
> vector of real numbers that can participate in matrix multiplications.*
>
> *That's the embedding layer: a lookup table.*
> *W_emb has one row per character. Token 17 → we just grab row 17.*
> *Those rows are parameters — learned during training."*

Draw on board:
```
W_emb: (vocab_size=32, d_model=16)

  char_id=0  [ 0.12, -0.34, 0.88, ... ]   ← 'd' embedding
  char_id=1  [ 0.45,  0.21, 0.03, ... ]   ← 'a' embedding
  ...
  char_id=17 [-0.09,  0.67, 0.44, ... ]   ← 'n' embedding

Lookup for token sequence [4, 17, 2]:
  → stack rows 4, 17, 2
  → shape: (seq_len=3, d_model=16)
  → this is the input X to the transformer block
```

> *"Initially, these are random numbers. After training, similar characters
> appear in similar regions of the embedding space. Characters that appear
> in similar contexts — 'a' and 'e' for example — end up with similar
> embedding vectors. That's what learning means here."*

**Ask the room:** *"Why do we need positional encoding if the input already
has an order — character 1, then 2, then 3?"*

Expected answer: The attention mechanism has no inherent notion of position.
It's a set operation — Q, K, V don't know which token came first unless we
add position information.

---

## SECTION 3: Causal Self-Attention — The Core (25 min)

> *"You've seen attention before in Part 6. But there's a crucial difference
> in a language model: we can only attend to PAST tokens.*
>
> *When predicting the next character after 'machne', the model may not
> peek at 'learning'. That would be cheating — at inference time, we don't
> have those future characters yet.*
>
> *This is causal masking — we mask out future positions by setting their
> attention scores to -infinity before the softmax."*

Draw on board:
```
CAUSAL ATTENTION MASK (seq_len=5):

         attend to positions:
         0    1    2    3    4
pos 0  [ OK  -inf -inf -inf -inf ]
pos 1  [ OK   OK  -inf -inf -inf ]
pos 2  [ OK   OK   OK  -inf -inf ]
pos 3  [ OK   OK   OK   OK  -inf ]
pos 4  [ OK   OK   OK   OK   OK  ]

After softmax: -inf → 0.0
→ position i can only attend to positions 0..i
→ this is the "causal" in "causal language model"
```

Write the full attention computation:
```
Q = x @ W_Q    (queries)
K = x @ W_K    (keys)
V = x @ W_V    (values)

scores = Q @ Kᵀ / sqrt(d_k)       (scale to prevent gradient vanishing)
scores[mask] = -1e9                (mask future positions)
attn = softmax(scores, axis=-1)    (probabilities over past positions)
output = attn @ V                  (weighted sum of values)
```

> *"Everything from Part 6, but with that masking step added. That one line —
> 'scores[mask] = -1e9' — is what makes this an autoregressive model
> instead of a bidirectional encoder like BERT."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
TODAY:
  Character tokenizer    → integers to integers
  Embedding layer        → integers to dense vectors (lookup table)
  Positional encoding    → adds position information
  Causal self-attention  → attends only to past positions (masking)
  FFN                    → per-position nonlinear transformation
  LM head                → final dense layer → logits over vocabulary

NEXT SESSION:
  Cross-entropy loss + gradient descent training loop
  Temperature / top-K / nucleus sampling
  Watch the loss drop. Watch the model generate text.
```

**Homework:** Draw the MiniGPT architecture from memory on paper.
Label every component and what shape flows through it.

---

# SESSION 2 (~90 min)
## "Training the model — cross-entropy, backprop, and text generation"

## OPENING (10 min)

> *"Last session we built the architecture. Every component exists.
> Today we plug in the training loop and watch the model learn.*
>
> *Fair warning: this might be the most exciting thing you run in this
> entire course. When the loss starts dropping, it means the model
> is genuinely learning the statistical structure of language from scratch.*
>
> *Keep your eyes on the console. We're going to celebrate when it happens."*

---

## SECTION 1: Cross-Entropy Loss for Language Modeling (20 min)

> *"The training objective: at each position, predict the NEXT character.*
>
> *Given the sequence 'machine', the model sees 'machin' and should
> predict 'e'. Given 'machin', it sees 'machi' and should predict 'n'.*
> *We train on all positions simultaneously — that's the power of
> the transformer: it processes every position in parallel."*

Draw on board:
```
Input:   m  a  c  h  i  n  e
         ↓  ↓  ↓  ↓  ↓  ↓  ↓
Model:   ?  ?  ?  ?  ?  ?  ?
         ↓  ↓  ↓  ↓  ↓  ↓  ↓
Target:  a  c  h  i  n  e  _

Loss = CrossEntropy(logits[0], 'a')
     + CrossEntropy(logits[1], 'c')
     + CrossEntropy(logits[2], 'h')
     + ...
     / seq_len    (average across all positions)

CrossEntropy(logits, true_id) = -log(softmax(logits)[true_id])
```

> *"Cross-entropy loss measures how surprised the model is by the correct answer.
> If the model assigns 0.01 probability to the correct next character,
> loss = -log(0.01) = 4.6. That's high — the model is very surprised.*
>
> *If the model assigns 0.8 probability, loss = -log(0.8) = 0.22.
> Low loss — the model expected this character. Training pushes us toward low loss."*

**Ask the room:** *"What is the loss for a random model with uniform probabilities
over 32 characters?"*

Expected answer: -log(1/32) = log(32) ≈ 3.46. This is the baseline loss.
When you run training, watch the initial loss — it should start near 3.5.

---

## SECTION 2: The Training Loop (20 min)

Walk through the training loop logic together (don't type — trace through
the module output):
```python
for epoch in range(N_EPOCHS):
    # 1. Sample a random batch of windows from the corpus
    # 2. Forward pass: x → logits
    # 3. Compute cross-entropy loss
    # 4. Backward pass: compute gradients manually
    # 5. Update all parameters: W -= lr * dW
    # 6. Print loss every 100 epochs
```

> *"This is the same gradient descent loop you wrote in Part 1 for linear
> regression — just applied to millions of parameters instead of two.*
>
> *The gradients flow backward through the LM head, through the transformer
> block, through the embeddings. Every weight nudges slightly to predict
> the next character better."*

Now run it:
```bash
python3 llm_from_scratch.py
```

> *"Watch the loss. It starts near 3.5 — random model level. Watch it drop."*

When the loss starts falling, point at it:

> *"That's learning. The model is discovering that 'machine' is often
> followed by 'learning'. That 'gradient' often precedes 'descent'.
> The statistical structure of language — compressed into weight matrices."*

When you see the loss drop below 2.5, CELEBRATE. Ask everyone to
pause and appreciate what just happened.

> *"This is what it felt like at Google in 2017 when Vaswani et al.
> first saw the transformer loss curve drop. Same thing. You just
> reproduced it in 200 lines of NumPy."*

---

## SECTION 3: Text Generation — Temperature, Top-K, Nucleus (25 min)

> *"The model is trained. Now we generate. The LM head gives us logits —
> unnormalized scores for each character. We need to convert those to a
> probability distribution and sample from it.*
>
> *How we sample is a huge creative choice."*

Draw on board:
```
LOGITS (raw scores):  [2.1, 0.3, -1.2, 0.8, ...]
SOFTMAX:              [0.52, 0.09, 0.02, 0.15, ...]

STRATEGY 1: GREEDY (temperature → 0)
  Always pick the highest probability character
  Deterministic, but boring and repetitive
  "machine learning trains models models models models..."

STRATEGY 2: TEMPERATURE SAMPLING
  Divide logits by T before softmax
    T < 1: sharpen distribution (more confident, less creative)
    T = 1: use logits as-is (original distribution)
    T > 1: flatten distribution (more random, more creative)

  logits_scaled = logits / T
  probs = softmax(logits_scaled)
  next_char = sample(probs)

STRATEGY 3: TOP-K SAMPLING
  Keep only the top-K highest probability characters
  Zero out the rest, renormalize, then sample
  K=5: only consider 5 candidates at each step
  Prevents very unlikely characters from ever being sampled

STRATEGY 4: NUCLEUS (TOP-P) SAMPLING
  Keep the smallest set of chars whose cumulative probability ≥ p
  p=0.9: keep enough chars to cover 90% of probability mass
  Adapts the cutoff: if model is confident → few chars, if uncertain → many
```

Show generation output from the module. Read some generated text aloud.

> *"Notice how the generated text sounds vaguely like ML writing — it has
> the right words, roughly the right word order, even some real phrases.
> It's not coherent over long spans yet — we're training on 10 sentences
> with a tiny model. But the statistical structure is there.*
>
> *Scale this to 175 billion parameters, 450 billion tokens of text —
> and you have GPT-3. Same architecture. 10,000× bigger."*

---

## SECTION 4: Scaling Laws (10 min)

Draw the table:
```
MODEL      PARAMS     LAYERS  D_MODEL  HEADS  CONTEXT
------------------------------------------------------
MiniGPT    ~10K       2       64       2      32 chars
GPT-2 sm   124M       12      768      12     1024 tokens
GPT-2 xl   1.5B       48      1600     25     1024 tokens
GPT-3      175B       96      12288    96     2048 tokens
GPT-4      ~1T(est)   ?       ?        ?      128K tokens
------------------------------------------------------

Scaling law (Kaplan et al. 2020):
  Loss ∝ N^(-0.076)  (N = parameters)
  Loss ∝ D^(-0.095)  (D = dataset tokens)
  → Loss decreases predictably with scale
  → More data helps as much as more parameters
```

> *"The incredible thing about scaling laws: they're smooth and predictable.
> You can extrapolate — train a 10M parameter model, measure its loss curve,
> and accurately predict how a 10B parameter model will perform — before
> building it. This is how OpenAI, Google, and Anthropic decide to commit
> hundreds of millions of dollars to training runs."*

**Ask the room:** *"If loss decreases predictably with more parameters —
why don't we just keep scaling forever? What are the limits?"*

Expected answers: compute cost, energy, data availability, diminishing returns,
alignment/safety concerns as models get more capable.

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
YOU JUST BUILT:
  Character tokenizer        → text to integer sequences
  Embedding table            → integers to dense vectors
  Causal self-attention      → attends to past positions only
  Feed-forward network       → per-position transformation
  LM head                    → logits over next character
  Cross-entropy training     → gradient descent on next-char prediction
  Sampling strategies        → greedy / temperature / top-K / nucleus

THIS IS CHATGPT'S ANCESTOR.
Same architecture.
Same training objective.
You understand every single line.
```

**Homework:** Modify the corpus — add 5 new sentences on a topic of your
choice. Retrain the model. How does the generated text change?

---

## INSTRUCTOR TIPS

**"I don't understand how loss can drop so fast"**
> *"Each gradient update nudges every weight slightly. With 100+ parameters
> and hundreds of training examples, the cumulative signal is strong.
> Think of it as: every training step, 200 parameters each move a tiny bit
> toward the right answer. After 1000 steps — that's 200,000 small corrections."*

**"The generated text doesn't make sense — is something wrong?"**
> *"No — that's expected. We're training on 10 sentences with a 10K parameter
> model for a few hundred epochs. The model is learning short-range patterns.
> Coherent long-range generation requires much more scale. This is the point —
> you see the mechanism without needing to wait for a $1M training run."*

**"How is this different from a Markov chain?"**
> *"A Markov chain uses the last N characters to predict the next one — fixed
> window, no learned features. Our transformer attends to ALL previous characters
> with learned attention weights — it can focus on relevant context anywhere in
> the sequence. That non-local attention is what makes transformers so powerful."*

**"What's the point of multiple transformer blocks?"**
> *"Each block processes the representation at a different level of abstraction.
> Earlier blocks learn low-level patterns (character-level n-grams).
> Later blocks learn higher-level semantic structure.
> It's analogous to how convolutional layers in CNNs go from edges → textures → objects."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening + full architecture on board   15 min
├── Corpus and character tokenizer         15 min
├── Embeddings and lookup tables           20 min
├── Causal self-attention + masking        25 min
└── Close + architecture homework          15 min

SESSION 2  (90 min)
├── Opening bridge                         10 min
├── Cross-entropy loss for LM              20 min
├── Training loop — run it, celebrate      20 min
├── Text generation sampling strategies   25 min
├── Scaling laws table                     10 min
└── Close + corpus modification homework   5 min
```

---
*MLForBeginners · Part 7: LLMs · Module 05*
