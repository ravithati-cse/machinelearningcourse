"""
GPT Decoder — Autoregressive Language Generation
=================================================

Learning Objectives:
  1. Understand GPT's decoder-only, causal (left-to-right) transformer architecture
  2. Implement autoregressive generation with a causal mask from scratch in numpy
  3. Compare and implement text sampling strategies: greedy, temperature,
     top-K, top-P (nucleus), and beam search
  4. Use HuggingFace GPT-2 for real text generation with different strategies
  5. Contrast BERT (encoder, bidirectional) vs GPT (decoder, causal) design choices
  6. Visualize sampling strategies, perplexity vs temperature, and beam search trees

Time:         ~45 minutes
Difficulty:   Intermediate-Advanced
Prerequisites: math_foundations 01-04 (attention, multi-head attention,
               positional encoding, encoder-decoder architecture),
               bert_encoder.py
YouTube:      https://www.youtube.com/watch?v=kCc8FmEb1nY  (GPT from scratch — Karpathy)
              https://www.youtube.com/watch?v=ISNdQcPhsts  (How GPT works)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "gpt_decoder")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

# =============================================================================
# SECTION 1: GPT Architecture
# =============================================================================
print("=" * 70)
print("SECTION 1: GPT Architecture — Decoder-Only Transformer")
print("=" * 70)

print("""
GPT — Generative Pre-trained Transformer (OpenAI, 2018)
Paper: "Improving Language Understanding by Generative Pre-Training"
       Radford et al., 2018   https://openai.com/blog/language-unsupervised

Core Architecture: Decoder-Only Transformer
  Unlike BERT (encoder) which reads text bidirectionally,
  GPT uses a CAUSAL / AUTOREGRESSIVE attention mechanism:

  Each token can ONLY attend to tokens BEFORE it (including itself).
  This makes GPT naturally suited for GENERATION tasks.

  Why Causal?
    Training objective: predict the NEXT token given ALL previous tokens.
      P(token_5 | token_1, token_2, token_3, token_4)
    During generation, we autoregressively feed our own output back in:
      Step 1: [START] → predict "The"
      Step 2: [START, The] → predict "quick"
      Step 3: [START, The, quick] → predict "brown"
      ...

  Causal Mask (lower triangular):
    Token 1 can see: [1]
    Token 2 can see: [1, 2]
    Token 3 can see: [1, 2, 3]
    Token 4 can see: [1, 2, 3, 4]
    (Masked positions → -inf before softmax → 0 attention weight)

No Cross-Attention:
  BERT uses the full encoder stack to encode context, then a decoder
  attends to the encoder output (cross-attention). GPT has NO encoder
  and NO cross-attention — it's purely a stack of masked self-attention
  + feed-forward layers.

Pre-training Objective:
  Simple next-token prediction over massive text corpora:
    L = -Σ log P(token_t | token_1, …, token_{t-1})
  GPT-2 was trained on WebText (40GB of Reddit links ≥ 3 karma).
  GPT-3 was trained on ~570GB of filtered Common Crawl + books + Wikipedia.
""")

print("GPT-2 Model Variants:")
print("-" * 65)
print(f"  {'Variant':<12} {'Layers':<10} {'d_model':<10} {'Heads':<10} {'Params'}")
print("-" * 65)
gpt2_variants = [
    ("small",    "12",  "768",  "12",  "117M"),
    ("medium",   "24",  "1024", "16",  "345M"),
    ("large",    "36",  "1280", "20",  "762M"),
    ("xl",       "48",  "1600", "25",  "1558M"),
]
for variant, layers, d_model, heads, params in gpt2_variants:
    print(f"  gpt2-{variant:<8} {layers:<10} {d_model:<10} {heads:<10} {params}")
print("-" * 65)

print("""
GPT Family Evolution:
  GPT-1  (2018): 117M params, demonstrated pretraining + fine-tuning works
  GPT-2  (2019): 1.5B params, "too dangerous to release" → now fully open
  GPT-3  (2020): 175B params, few-shot learning, in-context learning
  InstructGPT (2022): RLHF alignment — GPT-3 that follows instructions
  ChatGPT (2022): Dialogue-tuned InstructGPT — changed AI forever
  GPT-4  (2023): Multimodal, ~1T params (estimated), state-of-the-art
""")

# =============================================================================
# SECTION 2: Autoregressive Generation from Scratch
# =============================================================================
print("=" * 70)
print("SECTION 2: Autoregressive Generation — Numpy Implementation")
print("=" * 70)


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + eps)


def scaled_dot_product_attention(Q, W_q, W_k, W_v, causal=True):
    """
    Q, W_q, W_k, W_v: (seq_len, d_model)
    Returns context vectors (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    Qs = Q @ W_q   # (seq, d_model)
    Ks = Q @ W_k
    Vs = Q @ W_v

    scale = np.sqrt(d_model)
    scores = (Qs @ Ks.T) / scale   # (seq, seq)

    if causal:
        # Mask upper triangle: future tokens not visible
        mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
        scores[mask] = -1e9

    attn_weights = softmax(scores, axis=-1)   # (seq, seq)
    return attn_weights @ Vs                  # (seq, d_model)


class CausalTransformerLM:
    """
    Tiny causal language model (decoder-only transformer) in pure numpy.
    Architecture: Embedding → L × (CausalSelfAttention + FFN) → LM Head

    This is a FORWARD PASS ONLY implementation to demonstrate generation.
    Weights are random (not trained), so generated text is random — but
    the architecture and sampling logic are identical to GPT-2.
    """

    def __init__(self, vocab_size, d_model=32, num_heads=2, d_ff=64,
                 num_layers=2, max_len=50):
        self.vocab_size = vocab_size
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_ff       = d_ff
        self.num_layers = num_layers
        self.max_len    = max_len

        # Token embeddings + positional embeddings
        self.tok_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_emb = np.random.randn(max_len, d_model) * 0.02

        # Per-layer weights
        self.W_q = [np.random.randn(d_model, d_model) * 0.02 for _ in range(num_layers)]
        self.W_k = [np.random.randn(d_model, d_model) * 0.02 for _ in range(num_layers)]
        self.W_v = [np.random.randn(d_model, d_model) * 0.02 for _ in range(num_layers)]
        self.W_o = [np.random.randn(d_model, d_model) * 0.02 for _ in range(num_layers)]
        self.W1  = [np.random.randn(d_model, d_ff)   * 0.02 for _ in range(num_layers)]
        self.W2  = [np.random.randn(d_ff, d_model)   * 0.02 for _ in range(num_layers)]

        # LM head: maps from d_model → vocab_size
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02

    def forward(self, token_ids):
        """
        token_ids: list or array of integer token ids, length T
        Returns: logits (T, vocab_size)
        """
        T = len(token_ids)
        # Embeddings
        x = self.tok_emb[token_ids] + self.pos_emb[:T]   # (T, d_model)

        for layer in range(self.num_layers):
            # Causal self-attention (simplified: single-head for clarity)
            attn_out = scaled_dot_product_attention(
                x, self.W_q[layer], self.W_k[layer], self.W_v[layer], causal=True
            )
            attn_out = attn_out @ self.W_o[layer]
            x = layer_norm(x + attn_out)   # residual + norm

            # Feed-forward
            ff = np.maximum(0, x @ self.W1[layer])   # ReLU activation
            ff = ff @ self.W2[layer]
            x = layer_norm(x + ff)                   # residual + norm

        logits = x @ self.lm_head   # (T, vocab_size)
        return logits

    def generate(self, prompt_ids, max_new=20, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressively generate tokens.
        prompt_ids: list of seed token ids
        Returns: list of generated token ids (including prompt)
        """
        ids = list(prompt_ids)

        for _ in range(max_new):
            # Forward pass on the current context
            logits = self.forward(ids)       # (T, vocab_size)
            next_logits = logits[-1]         # (vocab_size,) — last token's prediction

            # Apply sampling strategy
            if top_p is not None:
                next_id = nucleus_sample(next_logits, top_p, temperature)
            elif top_k is not None:
                next_id = topk_sample(next_logits, top_k, temperature)
            else:
                next_id = temperature_sample(next_logits, temperature)

            ids.append(next_id)

            # Stop at max_len
            if len(ids) >= self.max_len:
                break

        return ids


# Build a tiny vocabulary from a few simple sentences
CORPUS_SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the sun rose over the hill",
    "a bird sang in the tree",
    "the wind blew through the leaves",
    "the moon shines over the lake",
    "a fox ran across the field",
    "the child played near the river",
]

# Build word-level vocabulary
all_words = set()
for sent in CORPUS_SENTENCES:
    all_words.update(sent.split())
all_words = sorted(all_words)
vocab = ["<PAD>", "<BOS>", "<EOS>"] + all_words
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
VOCAB_SIZE = len(vocab)

print(f"Tiny vocabulary size: {VOCAB_SIZE}")
print(f"Vocabulary: {vocab}\n")

# Instantiate the tiny LM
tiny_lm = CausalTransformerLM(
    vocab_size=VOCAB_SIZE, d_model=32, num_heads=2, d_ff=64,
    num_layers=2, max_len=30
)

# Demo forward pass
prompt_words = ["<BOS>", "the", "cat"]
prompt_ids   = [word2id[w] for w in prompt_words]
logits_demo  = tiny_lm.forward(prompt_ids)

print(f"Forward pass demo:")
print(f"  Prompt      : {prompt_words}")
print(f"  Token IDs   : {prompt_ids}")
print(f"  Logits shape: {logits_demo.shape}  (seq_len × vocab_size)")
print(f"  Last-token logits (first 8): {logits_demo[-1, :8].round(4)}")

probs_demo = softmax(logits_demo[-1])
top5_ids   = np.argsort(-probs_demo)[:5]
print(f"  Top-5 next token predictions:")
for rank, idx in enumerate(top5_ids, 1):
    print(f"    {rank}. '{id2word[idx]}'  prob={probs_demo[idx]:.4f}")

print("""
NOTE: Weights are randomly initialized — output is meaningless until trained.
      The architecture (causal mask, residual connections, layer norm, LM head)
      is identical to GPT-2. Training would require backprop through the sequence,
      which is complex in numpy. In practice: use PyTorch autograd.
""")

# =============================================================================
# SECTION 3: Text Generation Sampling Strategies
# =============================================================================
print("=" * 70)
print("SECTION 3: Text Generation Sampling Strategies")
print("=" * 70)

# Base example logits over 8 tokens
EXAMPLE_LOGITS = np.array([4.0, 2.5, 1.8, 1.2, 0.8, 0.3, 0.1, -0.5])
EXAMPLE_TOKENS = ["the", "a", "some", "my", "this", "her", "an", "those"]


def greedy_decode(logits):
    return np.argmax(logits), np.zeros(len(logits))


def temperature_sample(logits, temperature=1.0):
    if temperature == 0.0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs  = softmax(scaled)
    return int(np.random.choice(len(probs), p=probs))


def topk_sample(logits, k=5, temperature=1.0):
    top_k_idx = np.argsort(-logits)[:k]
    filtered  = np.full(len(logits), -1e9)
    filtered[top_k_idx] = logits[top_k_idx]
    if temperature != 1.0:
        filtered[top_k_idx] = filtered[top_k_idx] / temperature
    probs = softmax(filtered)
    return int(np.random.choice(len(probs), p=probs))


def nucleus_sample(logits, p=0.9, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    probs    = softmax(logits)
    sorted_idx   = np.argsort(-probs)
    sorted_probs = probs[sorted_idx]
    cumulative   = np.cumsum(sorted_probs)
    # Find smallest set where cumulative prob >= p
    cutoff = np.searchsorted(cumulative, p) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs = nucleus_probs / nucleus_probs.sum()
    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


def beam_search(logits_fn, prompt_ids, beam_size=3, max_new=5, vocab_size=None):
    """
    Simplified beam search. logits_fn(ids) → logits for next token.
    Returns list of (score, token_ids) for top beams.
    """
    beams = [(0.0, list(prompt_ids))]  # (cumulative_log_prob, ids)

    for step in range(max_new):
        all_candidates = []
        for score, ids in beams:
            logits = logits_fn(ids)[-1]
            log_probs = logits - np.log(np.sum(np.exp(logits - logits.max())) ) - (logits.max() - logits.max())
            # Proper log softmax
            log_probs = logits - (np.log(np.sum(np.exp(logits - logits.max()))) + logits.max())
            top_k = np.argsort(-log_probs)[:beam_size]
            for token_id in top_k:
                new_score = score + log_probs[token_id]
                new_ids   = ids + [int(token_id)]
                all_candidates.append((new_score, new_ids))
        # Keep top beam_size
        all_candidates.sort(key=lambda x: -x[0])
        beams = all_candidates[:beam_size]

    return beams


print("Base distribution (8 tokens):")
base_probs = softmax(EXAMPLE_LOGITS)
print(f"  Tokens: {EXAMPLE_TOKENS}")
print(f"  Logits: {EXAMPLE_LOGITS.tolist()}")
print(f"  Probs:  {base_probs.round(4).tolist()}")

print("\n--- Strategy 1: Greedy Decoding ---")
greedy_idx = np.argmax(EXAMPLE_LOGITS)
print(f"  Always pick argmax → token '{EXAMPLE_TOKENS[greedy_idx]}' (prob={base_probs[greedy_idx]:.4f})")
print("  Deterministic. Can produce repetitive / degenerate text.")

print("\n--- Strategy 2: Temperature Sampling ---")
for T in [0.5, 1.0, 2.0]:
    scaled_probs = softmax(EXAMPLE_LOGITS / T)
    entropy = -np.sum(scaled_probs * np.log(scaled_probs + 1e-10))
    print(f"  T={T}: probs={scaled_probs.round(3).tolist()}  entropy={entropy:.3f}")
print("  T→0: like greedy (sharper).  T=1: standard.  T>1: more uniform (creative).")

print("\n--- Strategy 3: Top-K Sampling (K=3) ---")
K = 3
top_k_idx = np.argsort(-EXAMPLE_LOGITS)[:K]
topk_probs = np.zeros(len(EXAMPLE_LOGITS))
topk_probs[top_k_idx] = base_probs[top_k_idx]
topk_probs /= topk_probs.sum()
print(f"  Keep top {K} tokens: {[EXAMPLE_TOKENS[i] for i in top_k_idx]}")
print(f"  Re-normalized probs: {topk_probs.round(4).tolist()}")
print("  Fixed K can include too many (flat distribution) or too few (peaked).")

print("\n--- Strategy 4: Top-P / Nucleus Sampling (p=0.9) ---")
P = 0.9
sorted_idx   = np.argsort(-base_probs)
sorted_probs = base_probs[sorted_idx]
cumulative   = np.cumsum(sorted_probs)
cutoff       = np.searchsorted(cumulative, P) + 1
nucleus_idx  = sorted_idx[:cutoff]
print(f"  Sorted tokens: {[EXAMPLE_TOKENS[i] for i in sorted_idx]}")
print(f"  Cumulative probs: {cumulative.round(4).tolist()}")
print(f"  Nucleus (first tokens summing to >= {P}): {[EXAMPLE_TOKENS[i] for i in nucleus_idx]}")
print("  Nucleus size adapts to the distribution — better than fixed K.")

print("\n--- Strategy 5: Beam Search (beam_size=3, 3 steps) ---")
beams = beam_search(tiny_lm.forward, prompt_ids, beam_size=3, max_new=3,
                    vocab_size=VOCAB_SIZE)
print(f"  Prompt: {prompt_words}")
print(f"  Top-3 beams after 3 additional tokens:")
for rank, (score, ids) in enumerate(beams[:3], 1):
    words = [id2word[i] for i in ids]
    print(f"    Beam {rank} (score={score:.4f}): {' '.join(words)}")
print("  Beam search explores multiple paths, avoids greedy's myopic choices.")

# Generation demo with tiny LM (random weights but illustrates the process)
print("\n--- Tiny LM Generation Demo (random weights — process demonstration) ---")
strategies = [
    ("Greedy",         dict(temperature=0.01)),
    ("Temperature 0.8",dict(temperature=0.8)),
    ("Top-K (k=5)",    dict(top_k=5, temperature=0.8)),
    ("Nucleus (p=0.9)",dict(top_p=0.9, temperature=0.8)),
]
seed = [word2id["<BOS>"], word2id["the"]]
for name, kwargs in strategies:
    np.random.seed(0)
    generated_ids = tiny_lm.generate(seed, max_new=8, **kwargs)
    generated_words = [id2word[i] for i in generated_ids]
    print(f"  {name:<22}: {' '.join(generated_words)}")

# =============================================================================
# SECTION 4: HuggingFace GPT-2 Generation
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: HuggingFace GPT-2 Text Generation")
print("=" * 70)

GPT2_AVAILABLE = False
try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch
    GPT2_AVAILABLE = True
    print("HuggingFace transformers detected — using real GPT-2.\n")
except ImportError:
    print("transformers not installed. Install with:  pip install transformers torch")
    print("Showing simulated output below.\n")

PROMPTS = [
    "The scientists discovered that",
    "Once upon a time in a land far away",
    "The best approach to machine learning is",
]

if GPT2_AVAILABLE:
    print("Loading gpt2 (smallest variant, ~117M params)...")
    gpt2_tok   = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_tok.pad_token = gpt2_tok.eos_token

    def generate_text(prompt, max_new_tokens=50, temperature=0.8, top_k=50, top_p=0.9,
                      do_sample=True):
        inputs = gpt2_tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = gpt2_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 0,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=gpt2_tok.eos_token_id,
            )
        return gpt2_tok.decode(out[0], skip_special_tokens=True)

    for prompt in PROMPTS:
        print(f"\nPrompt: {prompt!r}")
        greedy_out  = generate_text(prompt, do_sample=False, max_new_tokens=40)
        sampled_out = generate_text(prompt, do_sample=True, temperature=0.8,
                                    top_k=50, top_p=0.9, max_new_tokens=40)
        print(f"  Greedy : {greedy_out}")
        print(f"  Sampled: {sampled_out}")

else:
    # Hardcoded example outputs (representative of real GPT-2 output)
    simulated = [
        (
            "The scientists discovered that",
            "The scientists discovered that the new compound could be used to treat cancer "
            "in a way that had not been seen before, according to a new study.",
            "The scientists discovered that a single gene mutation is responsible for the "
            "development of a rare form of blindness in mice, suggesting a potential treatment.",
        ),
        (
            "Once upon a time in a land far away",
            "Once upon a time in a land far away there was a kingdom ruled by a wise king "
            "who had three daughters, each more beautiful than the last.",
            "Once upon a time in a land far away, there lived a young wizard who could "
            "speak to animals and bend the wind to his will.",
        ),
        (
            "The best approach to machine learning is",
            "The best approach to machine learning is to start with a simple baseline and "
            "iteratively improve it using more complex models and larger datasets.",
            "The best approach to machine learning is not just about algorithms — it's "
            "about understanding the data and the problem you are trying to solve.",
        ),
    ]

    for prompt, greedy_out, sampled_out in simulated:
        print(f"\nPrompt: {prompt!r}")
        print(f"  Greedy (simulated) : {greedy_out}")
        print(f"  Sampled (simulated): {sampled_out}")

print("""
Key Observation:
  - Greedy output is often fluent but repetitive and predictable.
  - Sampling (temperature + top-p) produces more varied and creative text.
  - Temperature 0.7–0.9 + nucleus p=0.9 is a popular production default.
  - Very high temperature produces incoherent text (too random).
  - Very low temperature collapses to greedy (too deterministic).
""")

# =============================================================================
# SECTION 5: BERT vs GPT Comparison
# =============================================================================
print("=" * 70)
print("SECTION 5: BERT vs GPT — Comprehensive Comparison")
print("=" * 70)

print()
print(f"  {'Aspect':<30} {'BERT':<30} {'GPT'}")
print("  " + "-" * 90)
comparison = [
    ("Architecture",           "Encoder-only",              "Decoder-only"),
    ("Attention Direction",    "Bidirectional",             "Causal (left-to-right)"),
    ("Pre-training Task",      "MLM + NSP",                 "Next-token prediction"),
    ("Context",                "Full sentence simultaneously","Tokens seen so far only"),
    ("Best For",               "Classification, NER, QA",   "Generation, completion"),
    ("Generation?",            "No (not autoregressive)",   "Yes (native)"),
    ("Typical Fine-tuning",    "Add linear head on [CLS]",  "Prompt + generate / classify"),
    ("Token special",          "[CLS], [SEP], [MASK]",      "<|endoftext|>"),
    ("Max context (base)",     "512 tokens",                "1024 tokens (GPT-2)"),
    ("Tokenization",           "WordPiece",                 "Byte-Pair Encoding (BPE)"),
    ("Embedding type",         "Token + Pos + Segment",     "Token + Positional"),
    ("Typical LR fine-tune",   "2e-5",                      "1e-5 to 5e-5"),
    ("GLUE benchmark",         "88.5 (BERT-large)",         "72.8 (GPT, less fine-tuned)"),
    ("Popular successor",      "RoBERTa, DistilBERT, ALBERT","GPT-2, GPT-3, GPT-4, LLaMA"),
]
for aspect, bert_val, gpt_val in comparison:
    print(f"  {aspect:<30} {bert_val:<30} {gpt_val}")
print("  " + "-" * 90)

print("""
The Encoder-Decoder Family:
  BERT (encoder) and GPT (decoder) represent two ends of the spectrum.
  T5 and BART combine both: encoder encodes input, decoder generates output.
  This is powerful for: translation, summarization, question answering.

GPT-3 and Beyond:
  GPT-3 (175B params) demonstrated "few-shot learning" — the ability to
  perform tasks from just a few examples in the prompt, without fine-tuning.
  
  InstructGPT added RLHF (Reinforcement Learning from Human Feedback):
    1. Supervised fine-tuning on human-written demonstrations
    2. Train a reward model on human preference pairs
    3. Fine-tune GPT with PPO to maximize the reward model
  
  This is what made ChatGPT feel so much more helpful and aligned.
  GPT-4 extended this to multimodal inputs (text + images).

  Modern open-source equivalents:
    - LLaMA (Meta): 7B–65B, fully open weights
    - Mistral 7B: outperforms LLaMA-2 13B at 7B params
    - Phi-2 (Microsoft): 2.7B, surprisingly capable
""")

# =============================================================================
# SECTION 6: Visualizations
# =============================================================================
print("=" * 70)
print("SECTION 6: Generating Visualizations")
print("=" * 70)

# ── Plot 1: GPT Architecture + Causal Mask ───────────────────────────────
fig1, (ax_gpt, ax_mask) = plt.subplots(1, 2, figsize=(16, 8))
fig1.suptitle("GPT Architecture and Causal Attention Mask", fontsize=14, fontweight="bold")

# Left: BERT vs GPT architecture boxes
ax_gpt.set_xlim(0, 14)
ax_gpt.set_ylim(0, 12)
ax_gpt.axis("off")
ax_gpt.set_title("BERT (Encoder) vs GPT (Decoder) Architecture", fontsize=11,
                  fontweight="bold")

def draw_rect(ax, cx, cy, w, h, text, facecolor, text_color="white", fontsize=8):
    rect = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor="white", linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            multialignment="center")

def draw_arr(ax, x, y0, y1, color="gray"):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Column headers
ax_gpt.text(3.5, 11.5, "BERT (Encoder)", ha="center", fontsize=11,
             fontweight="bold", color="#1565C0")
ax_gpt.text(10.5, 11.5, "GPT (Decoder)", ha="center", fontsize=11,
             fontweight="bold", color="#B71C1C")

# BERT stack (left)
bert_layers = [
    (0.9,  "Input Text",              "#607D8B"),
    (2.1,  "WordPiece Tokenizer\n[CLS] tokens [SEP]", "#455A64"),
    (3.4,  "Token + Pos + Seg Emb",   "#1565C0"),
    (4.7,  "Bidirectional\nSelf-Attention (×12)", "#1976D2"),
    (5.9,  "Feed-Forward (×12)",      "#1E88E5"),
    (7.2,  "All Hidden States\n(T × 768)", "#42A5F5"),
    (8.5,  "[CLS] Embedding\n(768-dim)", "#0288D1"),
    (9.8,  "Task Head\n(Classification / NER / QA)", "#00897B"),
]
for y, text, color in bert_layers:
    draw_rect(ax_gpt, 3.5, y, 4.5, 0.85, text, color, fontsize=8)
for i in range(len(bert_layers) - 1):
    y_from = bert_layers[i][0] + 0.43
    y_to   = bert_layers[i + 1][0] - 0.43
    draw_arr(ax_gpt, 3.5, y_from, y_to, color="#1565C0")

# GPT stack (right)
gpt_layers = [
    (0.9,  "Input Text",              "#607D8B"),
    (2.1,  "BPE Tokenizer",           "#455A64"),
    (3.4,  "Token + Positional Emb",  "#B71C1C"),
    (4.7,  "CAUSAL Self-Attention\n(×12, lower-tri mask)", "#C62828"),
    (5.9,  "Feed-Forward (×12)",      "#D32F2F"),
    (7.2,  "All Hidden States\n(T × 768)", "#EF5350"),
    (8.5,  "LM Head (linear)\n→ vocab logits", "#E53935"),
    (9.8,  "Softmax → Next Token\n(Autoregressive)", "#C0392B"),
]
for y, text, color in gpt_layers:
    draw_rect(ax_gpt, 10.5, y, 4.5, 0.85, text, color, fontsize=8)
for i in range(len(gpt_layers) - 1):
    y_from = gpt_layers[i][0] + 0.43
    y_to   = gpt_layers[i + 1][0] - 0.43
    draw_arr(ax_gpt, 10.5, y_from, y_to, color="#B71C1C")

# Key difference annotation
ax_gpt.annotate("Bidirectional\n(sees all tokens)", xy=(3.5, 4.7),
                xytext=(3.5, 4.7), ha="center", va="center", fontsize=7,
                color="white")
ax_gpt.annotate("Causal mask\n(left-to-right only)", xy=(10.5, 4.7),
                xytext=(10.5, 4.7), ha="center", va="center", fontsize=7,
                color="white")

# Right: Causal mask visualization
ax_mask.set_title("Causal (Lower-Triangular) Attention Mask\n"
                   "0=visible, -inf=masked (future tokens)",
                   fontsize=11, fontweight="bold")

N = 8
causal_mask = np.tril(np.ones((N, N)))
token_labels = [f"t{i+1}" for i in range(N)]

im = ax_mask.imshow(causal_mask, cmap="Blues", vmin=0, vmax=1.2, aspect="auto")
ax_mask.set_xticks(range(N))
ax_mask.set_yticks(range(N))
ax_mask.set_xticklabels(token_labels, fontsize=9)
ax_mask.set_yticklabels(token_labels, fontsize=9)
ax_mask.set_xlabel("Key (attending to)", fontsize=10)
ax_mask.set_ylabel("Query (current token)", fontsize=10)

for i in range(N):
    for j in range(N):
        val = "1" if causal_mask[i, j] == 1 else "-∞"
        color = "white" if causal_mask[i, j] == 1 else "#CCCCCC"
        ax_mask.text(j, i, val, ha="center", va="center",
                     fontsize=9, color=color, fontweight="bold")

plt.colorbar(im, ax=ax_mask, fraction=0.046, pad=0.04, label="Attention allowed (1) / masked (0)")

plt.tight_layout()
path1 = os.path.join(VIS_DIR, "01_gpt_architecture.png")
plt.savefig(path1, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")

# ── Plot 2: Sampling Strategies ───────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 11))
fig2.suptitle("Text Generation Sampling Strategies", fontsize=14, fontweight="bold")

tokens_short = ["the", "a", "some", "my", "this", "her", "an", "those"]
base_logits  = EXAMPLE_LOGITS.copy()
base_probs   = softmax(base_logits)
x_pos        = np.arange(len(tokens_short))

# Top-left: Greedy
ax_gr = axes2[0, 0]
ax_gr.set_title("Greedy Decoding\n(argmax — always pick highest prob)", fontsize=10,
                 fontweight="bold")
bar_colors = ["#E53935" if i == np.argmax(base_probs) else "#90CAF9"
              for i in range(len(base_probs))]
ax_gr.bar(x_pos, base_probs, color=bar_colors, edgecolor="white")
ax_gr.set_xticks(x_pos)
ax_gr.set_xticklabels(tokens_short, fontsize=9)
ax_gr.set_ylabel("Probability", fontsize=10)
ax_gr.set_ylim(0, 0.85)
ax_gr.annotate("argmax\nchosen", xy=(np.argmax(base_probs), base_probs[np.argmax(base_probs)]),
               xytext=(np.argmax(base_probs) + 1.2, base_probs[np.argmax(base_probs)] - 0.1),
               arrowprops=dict(arrowstyle="->", color="#E53935"),
               fontsize=9, color="#E53935", fontweight="bold")
ax_gr.spines[["top", "right"]].set_visible(False)
ax_gr.set_facecolor("#FAFAFA")

# Top-right: Temperature
ax_tmp = axes2[0, 1]
ax_tmp.set_title("Temperature Sampling\nT=0.5 (focused), T=1.0 (standard), T=2.0 (random)",
                  fontsize=10, fontweight="bold")
temps     = [0.5, 1.0, 2.0]
tmp_colors = ["#1565C0", "#43A047", "#E53935"]
width      = 0.25
for ti, (T, color) in enumerate(zip(temps, tmp_colors)):
    probs_t = softmax(base_logits / T)
    offset  = (ti - 1) * width
    ax_tmp.bar(x_pos + offset, probs_t, width=width, color=color,
               edgecolor="white", alpha=0.85, label=f"T={T}")
ax_tmp.set_xticks(x_pos)
ax_tmp.set_xticklabels(tokens_short, fontsize=9)
ax_tmp.set_ylabel("Probability", fontsize=10)
ax_tmp.legend(fontsize=9)
ax_tmp.spines[["top", "right"]].set_visible(False)
ax_tmp.set_facecolor("#FAFAFA")

# Bottom-left: Top-K
ax_topk = axes2[1, 0]
ax_topk.set_title("Top-K Sampling (K=3)\nOnly top-3 tokens remain, rest zeroed out",
                   fontsize=10, fontweight="bold")
K = 3
top_k_idx    = set(np.argsort(-base_probs)[:K])
topk_probs_2 = np.array([base_probs[i] if i in top_k_idx else 0 for i in range(len(base_probs))])
topk_probs_2 /= topk_probs_2.sum() + 1e-10

bar_colors_k = ["#E53935" if i in top_k_idx else "#CCCCCC" for i in range(len(base_probs))]
ax_topk.bar(x_pos, topk_probs_2, color=bar_colors_k, edgecolor="white")
ax_topk.set_xticks(x_pos)
ax_topk.set_xticklabels(tokens_short, fontsize=9)
ax_topk.set_ylabel("Re-normalized Probability", fontsize=10)
ax_topk.axvline(sorted(top_k_idx)[-1] + 0.5, color="#FF6F00", linestyle="--",
                linewidth=2, label=f"K={K} cutoff")

# Label kept vs zeroed
for i in range(len(base_probs)):
    label = "kept" if i in top_k_idx else "zeroed"
    color = "#B71C1C" if i in top_k_idx else "#888888"
    ax_topk.text(i, topk_probs_2[i] + 0.01, label,
                 ha="center", va="bottom", fontsize=7, color=color, rotation=45)

ax_topk.legend(fontsize=9)
ax_topk.spines[["top", "right"]].set_visible(False)
ax_topk.set_facecolor("#FAFAFA")

# Bottom-right: Top-P (nucleus)
ax_topp = axes2[1, 1]
ax_topp.set_title("Top-P / Nucleus Sampling (p=0.9)\nSmallest set of tokens whose cumulative prob ≥ 0.9",
                   fontsize=10, fontweight="bold")
P = 0.9
sorted_idx_2    = np.argsort(-base_probs)
sorted_probs_2  = base_probs[sorted_idx_2]
cumulative_2    = np.cumsum(sorted_probs_2)
cutoff_2        = int(np.searchsorted(cumulative_2, P)) + 1
nucleus_set     = set(sorted_idx_2[:cutoff_2])

bar_colors_p = ["#1976D2" if i in nucleus_set else "#CCCCCC" for i in range(len(base_probs))]
ax_topp.bar(x_pos, base_probs, color=bar_colors_p, edgecolor="white", label="Base probs")
ax_topp2 = ax_topp.twinx()
cumulative_full = np.zeros(len(base_probs))
for rank, idx in enumerate(sorted_idx_2):
    cumulative_full[idx] = cumulative_2[rank]
ax_topp2.plot(x_pos, cumulative_full, "o--", color="#E53935", linewidth=1.5,
              markersize=5, label="Cumulative prob")
ax_topp2.axhline(P, color="#FF6F00", linestyle=":", linewidth=2, label=f"p={P} threshold")
ax_topp2.set_ylabel("Cumulative Probability", fontsize=9, color="#E53935")
ax_topp2.set_ylim(0, 1.2)
ax_topp2.legend(loc="upper right", fontsize=8)

ax_topp.set_xticks(x_pos)
ax_topp.set_xticklabels(tokens_short, fontsize=9)
ax_topp.set_ylabel("Probability", fontsize=10)
ax_topp.legend(loc="upper left", fontsize=8)

in_nucleus = sum(1 for i in range(len(base_probs)) if i in nucleus_set)
ax_topp.set_title(
    f"Top-P / Nucleus Sampling (p={P})\n"
    f"{in_nucleus} tokens in nucleus (adaptive cutoff)",
    fontsize=10, fontweight="bold"
)
ax_topp.spines[["top", "right"]].set_visible(False)
ax_topp.set_facecolor("#FAFAFA")

plt.tight_layout()
path2 = os.path.join(VIS_DIR, "02_sampling_strategies.png")
plt.savefig(path2, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

# ── Plot 3: Generation Analysis ───────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle("GPT Generation Analysis", fontsize=14, fontweight="bold")

# Left: Perplexity vs Temperature
ax_ppl = axes3[0]
ax_ppl.set_title("Perplexity vs Temperature\n(lower = more predictable text)",
                  fontsize=11, fontweight="bold")

temps_range = np.linspace(0.1, 3.0, 50)

# Simulate perplexity: low T → low ppl (predictable), high T → high ppl (random)
# Perplexity of base distribution under temperature scaling
ppl_vals = []
for T in temps_range:
    scaled = softmax(base_logits / T)
    entropy = -np.sum(scaled * np.log(scaled + 1e-10))
    ppl_vals.append(np.exp(entropy))

ppl_vals = np.array(ppl_vals)
# Scale to realistic GPT-2 range (~20–200 on WikiText-103)
ppl_vals = 20 + 180 * (ppl_vals - ppl_vals.min()) / (ppl_vals.max() - ppl_vals.min() + 1e-10)

ax_ppl.plot(temps_range, ppl_vals, color="#1976D2", linewidth=2.5)
ax_ppl.axvspan(0.7, 1.1, alpha=0.15, color="#43A047", label="Sweet spot (0.7–1.1)")
ax_ppl.axvline(0.7, color="#43A047", linestyle="--", linewidth=1.5, alpha=0.7)
ax_ppl.axvline(1.1, color="#43A047", linestyle="--", linewidth=1.5, alpha=0.7)
ax_ppl.set_xlabel("Temperature", fontsize=10)
ax_ppl.set_ylabel("Perplexity", fontsize=10)
ax_ppl.legend(fontsize=9)
ax_ppl.spines[["top", "right"]].set_visible(False)
ax_ppl.set_facecolor("#FAFAFA")
ax_ppl.annotate("Too deterministic\n(repetitive)", xy=(0.2, ppl_vals[2]),
                xytext=(0.15, ppl_vals[2] + 30),
                fontsize=8, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935"))
ax_ppl.annotate("Too random\n(incoherent)", xy=(2.8, ppl_vals[-5]),
                xytext=(2.0, ppl_vals[-5] - 30),
                fontsize=8, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935"))

# Middle: Token probability heatmap over generation steps
ax_heat = axes3[1]
ax_heat.set_title("Top-Token Probabilities During Generation\n(rows=step, cols=top tokens)",
                   fontsize=11, fontweight="bold")

GENERATION_STEPS = 6
TOP_N_TOKENS = 8

np.random.seed(7)
# Simulate a peaked-then-diffuse probability pattern over steps
heat_data = np.zeros((GENERATION_STEPS, TOP_N_TOKENS))
for step in range(GENERATION_STEPS):
    # Each step: different sharpness
    sharpness = 3.0 - step * 0.3
    raw = np.random.randn(TOP_N_TOKENS) * sharpness
    heat_data[step] = softmax(raw)

heat_data /= heat_data.max(axis=1, keepdims=True)   # normalize for viz

im3 = ax_heat.imshow(heat_data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax_heat.set_xticks(range(TOP_N_TOKENS))
ax_heat.set_xticklabels([f"tok{i+1}" for i in range(TOP_N_TOKENS)], fontsize=9)
ax_heat.set_yticks(range(GENERATION_STEPS))
ax_heat.set_yticklabels([f"Step {i+1}" for i in range(GENERATION_STEPS)], fontsize=9)
ax_heat.set_xlabel("Top Token (by rank at each step)", fontsize=10)
ax_heat.set_ylabel("Generation Step", fontsize=10)

for i in range(GENERATION_STEPS):
    for j in range(TOP_N_TOKENS):
        ax_heat.text(j, i, f"{heat_data[i,j]:.2f}", ha="center", va="center",
                     fontsize=7.5, color="white" if heat_data[i, j] > 0.6 else "black")

plt.colorbar(im3, ax=ax_heat, fraction=0.046, pad=0.04, label="Relative prob (normalized)")

# Right: Beam search tree diagram
ax_beam = axes3[2]
ax_beam.set_xlim(-0.5, 4.5)
ax_beam.set_ylim(-0.5, 5.5)
ax_beam.axis("off")
ax_beam.set_title("Beam Search Tree (beam=3, 2 steps)\nExpand each beam → keep top-3",
                   fontsize=11, fontweight="bold")

# Step 0: root
root_x, root_y = 2.0, 5.0
root_rect = mpatches.FancyBboxPatch((root_x - 0.6, root_y - 0.25), 1.2, 0.5,
                                     boxstyle="round,pad=0.05",
                                     facecolor="#455A64", edgecolor="white")
ax_beam.add_patch(root_rect)
ax_beam.text(root_x, root_y, "the", ha="center", va="center",
             fontsize=10, color="white", fontweight="bold")

# Step 1: 3 beams
step1_tokens = [("cat", -0.3, "#1565C0"), ("dog", -0.7, "#1565C0"), ("sun", -1.1, "#1565C0")]
step1_x = [0.7, 2.0, 3.3]
step1_y = 3.3
for (token, score, color), sx in zip(step1_tokens, step1_x):
    rect = mpatches.FancyBboxPatch((sx - 0.55, step1_y - 0.25), 1.1, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="white")
    ax_beam.add_patch(rect)
    ax_beam.text(sx, step1_y, f"{token}\n({score:.1f})", ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")
    ax_beam.annotate("", xy=(sx, step1_y + 0.25), xytext=(root_x, root_y - 0.25),
                     arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.2))

# Step 2: 3 beams (top expansions)
step2_tokens = [
    ("sat", -0.5, "#1976D2"),
    ("ran", -0.9, "#1976D2"),
    ("rose", -1.4, "#1976D2"),
]
step2_x    = [0.7, 2.0, 3.3]
step2_y    = 1.5
step2_from = [0, 1, 2]  # which step1 beam each came from

for (token, score, color), sx, from_idx in zip(step2_tokens, step2_x, step2_from):
    rect = mpatches.FancyBboxPatch((sx - 0.6, step2_y - 0.25), 1.2, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, edgecolor="white")
    ax_beam.add_patch(rect)
    ax_beam.text(sx, step2_y, f"{token}\n({score:.1f})", ha="center", va="center",
                 fontsize=8.5, color="white", fontweight="bold")
    ax_beam.annotate("", xy=(sx, step2_y + 0.25), xytext=(step1_x[from_idx], step1_y - 0.25),
                     arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.2))

# Annotations
ax_beam.text(2.0, 4.3, "Step 0: Start", ha="center", fontsize=8.5,
             color="#607D8B", style="italic")
ax_beam.text(2.0, 2.55, "Step 1: 3 beams kept", ha="center", fontsize=8.5,
             color="#607D8B", style="italic")
ax_beam.text(2.0, 0.8, "Step 2: Best continuations", ha="center", fontsize=8.5,
             color="#607D8B", style="italic")
ax_beam.text(2.0, 0.3,
             "Best beam: 'the cat sat'  (score = -0.5)",
             ha="center", fontsize=9, color="#E53935", fontweight="bold")

ax_beam.set_facecolor("#FAFAFA")

plt.tight_layout()
path3 = os.path.join(VIS_DIR, "03_generation_analysis.png")
plt.savefig(path3, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY — GPT Decoder")
print("=" * 70)
print("""
What You Learned:
  1. GPT is a decoder-only transformer with CAUSAL (left-to-right) attention.
     Each token can only attend to previous tokens — enabling autoregressive
     next-token prediction.

  2. The causal mask is a lower-triangular matrix: upper triangle → -inf before
     softmax → zero attention weight on future tokens.

  3. Sampling Strategies:
       Greedy      — argmax. Fast, deterministic, often repetitive.
       Temperature — scale logits by 1/T. T<1=focused, T>1=random.
       Top-K       — keep only K highest-prob tokens. Fixed cutoff.
       Nucleus     — keep smallest set summing to prob p. Adaptive cutoff.
       Beam Search — maintain B candidate sequences. Better quality, slower.

  4. In practice: temperature 0.7–0.9 + nucleus p=0.9 is a solid default for
     open-ended text generation. Use greedy/beam for more factual tasks.

  5. BERT vs GPT:
       BERT: encoder, bidirectional, MLM+NSP, great for classification/NER/QA.
       GPT:  decoder, causal, next-token prediction, great for generation.
       T5/BART: encoder-decoder, combines both strengths.

  6. Modern LLMs (GPT-3/4, LLaMA, Mistral) are GPT-style decoders scaled to
     billions of parameters and aligned with RLHF for helpful behavior.

Next Steps:
  - See transformers/math_foundations/ for the attention mechanism deep-dive
  - Explore: text_generation_with_gpt2.py (fine-tuning GPT-2 on custom data)
  - Read: "Language Models are Few-Shot Learners" (Brown et al., 2020) — GPT-3

Visualizations saved to:
""")
for p in [path1, path2, path3]:
    print(f"  {p}")
