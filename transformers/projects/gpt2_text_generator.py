"""
Text Generation with GPT-2 — End-to-End Project
================================================

Learning Objectives:
  1. Build a complete autoregressive text generation pipeline
  2. Implement all major sampling strategies from scratch (greedy, temperature, top-k, top-p)
  3. Use HuggingFace GPT-2 for high-quality text generation
  4. Evaluate generation quality with perplexity and diversity metrics
  5. Explore prompt engineering and its effect on generated text style
  6. Build a deployable generate_text() function with configurable sampling

YouTube: Search "GPT-2 Text Generation HuggingFace Python" for companion videos
Time: ~60 minutes | Difficulty: Advanced | Prerequisites: transformers Parts 1-9

Dataset: In-memory prompts — no external download required
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "gpt2_text_generator")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("TEXT GENERATION WITH GPT-2 — End-to-End Project")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Autoregressive Generation — The Core Idea
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Autoregressive Generation")
print("=" * 70)

print("""
Autoregressive language models generate text token by token:

  P(x_1, x_2, ..., x_n) = ∏ P(x_t | x_1, ..., x_{t-1})

  At each step:
    1. Feed all previously generated tokens as input
    2. Model outputs logits over entire vocabulary
    3. Sample (or argmax) next token from the distribution
    4. Append to sequence, repeat

  GPT-2 architecture:
    • Decoder-only transformer (no cross-attention)
    • Causal masking: token t only attends to tokens 0..t-1
    • Trained on WebText: 40GB of internet text (8 million web pages)
    • 4 sizes: 117M / 345M / 762M / 1.5B parameters

  GPT-2 sizes:
  ┌──────────────┬────────┬─────────┬───────┬────────┐
  │ Variant      │ Layers │ d_model │ Heads │ Params │
  ├──────────────┼────────┼─────────┼───────┼────────┤
  │ gpt2 (small) │   12   │   768   │  12   │  117M  │
  │ gpt2-medium  │   24   │  1024   │  16   │  345M  │
  │ gpt2-large   │   36   │  1280   │  20   │  762M  │
  │ gpt2-xl      │   48   │  1600   │  25   │  1.5B  │
  └──────────────┴────────┴─────────┴───────┴────────┘
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Sampling Strategies from Scratch
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 2: Sampling Strategies from Scratch")
print("=" * 70)

print("""
Given logits over vocabulary at step t, how do we pick the next token?

  Greedy:         always pick argmax — deterministic but repetitive
  Temperature:    scale logits before softmax — controls randomness
  Top-K:          restrict to top K tokens — prevents low-probability choices
  Top-P (Nucleus): smallest set of tokens with cumulative prob ≥ p
  Beam Search:    keep beam_size partial sequences — balance quality/diversity
""")


def softmax(logits, temperature=1.0):
    """Apply temperature then softmax."""
    logits = np.array(logits, dtype=np.float64) / max(temperature, 1e-9)
    logits -= logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


def greedy_sample(logits):
    """Always pick the most probable token."""
    return int(np.argmax(logits))


def temperature_sample(logits, temperature=1.0):
    """Sample from softmax distribution with temperature scaling."""
    probs = softmax(logits, temperature)
    return int(np.random.choice(len(probs), p=probs))


def top_k_sample(logits, k=50, temperature=1.0):
    """
    Top-K sampling: restrict to k most probable tokens, then sample.
    k=1 → greedy. k=vocab_size → unrestricted sampling.
    """
    logits = np.array(logits, dtype=np.float64)
    top_k_idx = np.argsort(logits)[-k:]
    # Mask everything except top-k
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_idx] = logits[top_k_idx]
    probs = softmax(filtered, temperature)
    return int(np.random.choice(len(probs), p=probs))


def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    Top-P (Nucleus) sampling: use the smallest set of tokens
    whose cumulative probability >= p.

    Advantages over top-k:
      - Adapts to the sharpness of the distribution
      - For high-entropy steps: uses many tokens (creative)
      - For low-entropy steps: uses few tokens (focused)
    """
    probs = softmax(logits, temperature)
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative = np.cumsum(sorted_probs)
    # Remove tokens that push cumulative prob beyond p
    cutoff = np.searchsorted(cumulative, p) + 1
    nucleus_idx = sorted_idx[:cutoff]
    nucleus_probs = probs[nucleus_idx]
    nucleus_probs /= nucleus_probs.sum()
    return int(np.random.choice(nucleus_idx, p=nucleus_probs))


# Demo on a toy vocabulary of 10 tokens
np.random.seed(42)
VOCAB = ["the", "cat", "sat", "on", "mat", "dog", "run", "fast", "slow", "a"]
toy_logits = np.array([3.5, 2.1, 1.8, 0.9, 0.7, 0.4, 0.3, 0.2, 0.1, -0.5])

print("Toy vocabulary (10 tokens), base logits:")
for i, (w, l) in enumerate(zip(VOCAB, toy_logits)):
    bar = "█" * int(softmax(toy_logits)[i] * 40)
    print(f"  {w:6s}  logit={l:5.1f}  prob={softmax(toy_logits)[i]:.3f}  {bar}")

print()
print("Sampling strategy comparison (same logits, 10 samples each):")
print()
strategies = {
    "Greedy":           lambda: VOCAB[greedy_sample(toy_logits)],
    "Temp T=0.5":       lambda: VOCAB[temperature_sample(toy_logits, 0.5)],
    "Temp T=1.0":       lambda: VOCAB[temperature_sample(toy_logits, 1.0)],
    "Temp T=2.0":       lambda: VOCAB[temperature_sample(toy_logits, 2.0)],
    "Top-K (k=3)":      lambda: VOCAB[top_k_sample(toy_logits, k=3)],
    "Top-P (p=0.9)":    lambda: VOCAB[top_p_sample(toy_logits, p=0.9)],
}
for name, fn in strategies.items():
    samples = [fn() for _ in range(10)]
    print(f"  {name:16s}: {samples}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Tiny Word-Level LM from Scratch (numpy)
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Tiny Language Model from Scratch")
print("=" * 70)

print("""
We implement a tiny bigram + smoothed language model to demonstrate
autoregressive generation WITHOUT requiring TensorFlow.

This is NOT a Transformer — it's a simple n-gram model that lets us:
  • Practice the generate() loop with all sampling strategies
  • See how generation quality degrades with different temperatures
  • Understand the autoregressive framework before moving to GPT-2
""")

# Training corpus
CORPUS = """
The quick brown fox jumps over the lazy dog.
The cat sat on the mat and watched the birds fly.
Machine learning models learn patterns from data to make predictions.
Natural language processing enables computers to understand human text.
The transformer architecture revolutionized natural language understanding.
Attention mechanisms allow models to focus on relevant parts of the input.
Deep neural networks can learn complex representations from raw data.
The quick fox ran through the forest looking for food and shelter.
Language models predict the next word given the previous context.
Transfer learning allows pretrained models to be adapted to new tasks.
"""

def build_ngram_model(text, n=2):
    """Build an n-gram language model."""
    words = text.lower().split()
    words = [re.sub(r"[^a-z]", "", w) for w in words]
    words = [w for w in words if w]
    vocab = list(set(words))
    vocab.sort()
    w2i = {w: i for i, w in enumerate(vocab)}

    # Count n-grams
    counts = {}
    for i in range(len(words) - n + 1):
        ctx = tuple(words[i:i + n - 1])
        nxt = words[i + n - 1]
        if ctx not in counts:
            counts[ctx] = Counter()
        counts[ctx][nxt] += 1

    return vocab, w2i, counts


def ngram_logits(context, counts, vocab, alpha=0.1):
    """Compute log-probabilities for next word given context (with Laplace smoothing)."""
    V = len(vocab)
    ctx = tuple(context)
    ctx_counts = counts.get(ctx, Counter())
    total = sum(ctx_counts.values()) + alpha * V
    logits = np.array([math.log((ctx_counts.get(w, 0) + alpha) / total) for w in vocab])
    return logits


vocab, w2i, bigram_counts = build_ngram_model(CORPUS, n=2)
print(f"Vocabulary: {len(vocab)} words")
print(f"Unique contexts: {len(bigram_counts)}")


def generate_ngram(prompt_words, max_new=15, strategy="top_p", **kwargs):
    """Generate text using n-gram model with specified sampling strategy."""
    words = [w for w in prompt_words if w in w2i]
    if not words:
        words = ["the"]

    generated = list(words)
    for _ in range(max_new):
        ctx = (generated[-1],) if generated else ("the",)
        logits = ngram_logits(ctx, bigram_counts, vocab)
        if strategy == "greedy":
            idx = greedy_sample(logits)
        elif strategy == "temperature":
            idx = temperature_sample(logits, kwargs.get("temperature", 1.0))
        elif strategy == "top_k":
            idx = top_k_sample(logits, k=kwargs.get("k", 5))
        elif strategy == "top_p":
            idx = top_p_sample(logits, p=kwargs.get("p", 0.9))
        else:
            idx = greedy_sample(logits)
        generated.append(vocab[idx])

    return " ".join(generated)


prompt = ["the", "machine"]
print(f"\nGeneration demo (prompt: '{' '.join(prompt)}'):")
print(f"  Greedy    : {generate_ngram(prompt, 12, 'greedy')}")
print(f"  Temp=0.5  : {generate_ngram(prompt, 12, 'temperature', temperature=0.5)}")
print(f"  Temp=1.5  : {generate_ngram(prompt, 12, 'temperature', temperature=1.5)}")
print(f"  Top-K=3   : {generate_ngram(prompt, 12, 'top_k', k=3)}")
print(f"  Top-P=0.9 : {generate_ngram(prompt, 12, 'top_p', p=0.9)}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: GPT-2 with HuggingFace
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: GPT-2 Text Generation with HuggingFace")
print("=" * 70)

GPT2_AVAILABLE = False
gpt2_outputs = {}

try:
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    import torch

    print("Loading GPT-2 (small, 117M params)...")
    gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    GPT2_AVAILABLE = True
    print(f"  GPT-2 loaded: {sum(p.numel() for p in gpt2_model.parameters()):,} parameters")

    def generate_gpt2(prompt, max_new_tokens=60, temperature=0.8,
                      top_k=50, top_p=0.9, do_sample=True):
        """Generate text with GPT-2."""
        input_ids = gpt2_tok.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = gpt2_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=gpt2_tok.eos_token_id,
            )
        return gpt2_tok.decode(outputs[0], skip_special_tokens=True)

    PROMPTS = [
        "The future of artificial intelligence is",
        "Once upon a time in a land far away,",
        "Scientists discovered a new method to",
    ]
    print("\nGPT-2 generation demo:")
    for prompt in PROMPTS:
        result = generate_gpt2(prompt, max_new_tokens=40, temperature=0.8)
        gpt2_outputs[prompt] = result
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {result}")

    # Compare sampling strategies
    test_prompt = "The machine learning model"
    print(f"\nSampling strategy comparison (prompt: '{test_prompt}'):")
    strats = [
        ("Greedy",          dict(do_sample=False, temperature=1.0)),
        ("Temp=0.5",        dict(do_sample=True, temperature=0.5, top_k=0, top_p=1.0)),
        ("Temp=1.5",        dict(do_sample=True, temperature=1.5, top_k=0, top_p=1.0)),
        ("Top-K (k=10)",    dict(do_sample=True, temperature=1.0, top_k=10, top_p=1.0)),
        ("Top-P (p=0.9)",   dict(do_sample=True, temperature=1.0, top_k=0, top_p=0.9)),
    ]
    for name, kw in strats:
        out = generate_gpt2(test_prompt, max_new_tokens=20, **kw)
        continuation = out[len(test_prompt):].strip()
        print(f"  {name:16s}: ...{continuation[:60]}")

except ImportError:
    print("HuggingFace transformers / PyTorch not available.")
    print("Install: pip install transformers torch")
    print()
    print("Showing example GPT-2 outputs (hardcoded for illustration):")
    gpt2_outputs = {
        "The future of artificial intelligence is": (
            "The future of artificial intelligence is one of the most exciting fields "
            "of our time. AI systems are becoming increasingly capable of performing "
            "complex tasks that once required human expertise and judgment."
        ),
        "Once upon a time in a land far away,": (
            "Once upon a time in a land far away, there lived a wise old wizard who "
            "had spent his entire life studying the ancient secrets of mathematics "
            "and the mysteries of the natural world."
        ),
        "Scientists discovered a new method to": (
            "Scientists discovered a new method to detect early signs of disease using "
            "machine learning algorithms trained on millions of medical imaging scans "
            "from hospitals across twenty different countries."
        ),
    }
    for prompt, output in gpt2_outputs.items():
        print(f"\n  Prompt: {prompt}")
        print(f"  Output: {output[:120]}...")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Generation Quality Metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Generation Quality Metrics")
print("=" * 70)

print("""
How do we measure generation quality?

  1. Perplexity (PPL):
     PPL = exp(-1/N × Σ log P(x_t | x_<t))
     → Lower = model is more certain (better fit to reference text)
     → GPT-2 achieves ~29 PPL on WikiText-103

  2. BLEU Score (vs reference):
     Measures n-gram overlap between generated and reference text
     → Commonly used for machine translation evaluation
     → Less meaningful for open-ended generation

  3. Distinct-n:
     Proportion of unique n-grams in generated text
     → Measures diversity/repetition
     → distinct-1 = unique unigrams / total tokens
     → distinct-2 = unique bigrams / total bigrams

  4. Self-BLEU:
     BLEU of one generated sample against all others
     → Measures diversity across multiple generations
     → Lower self-BLEU = more diverse outputs
""")


def compute_distinct_n(text, n):
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(ngrams)) / max(len(ngrams), 1)


def estimate_perplexity_ngram(text, counts, vocab, alpha=0.1):
    """Estimate perplexity using our bigram model (rough proxy)."""
    words = text.lower().split()
    words = [re.sub(r"[^a-z]", "", w) for w in words if re.sub(r"[^a-z]", "", w) in w2i]
    if len(words) < 2:
        return float("inf")
    log_prob = 0.0
    for i in range(1, len(words)):
        ctx = (words[i - 1],)
        logits = ngram_logits(ctx, counts, vocab, alpha)
        idx = vocab.index(words[i]) if words[i] in vocab else 0
        log_prob += logits[idx]
    return math.exp(-log_prob / max(len(words) - 1, 1))


# Evaluate diversity across strategies
temps = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]
diversity_data = {"temperature": temps, "distinct1": [], "distinct2": [], "perplexity": []}

np.random.seed(42)
for T in temps:
    text = generate_ngram(["the", "model"], 30, "temperature", temperature=T)
    diversity_data["distinct1"].append(compute_distinct_n(text, 1))
    diversity_data["distinct2"].append(compute_distinct_n(text, 2))
    diversity_data["perplexity"].append(min(estimate_perplexity_ngram(text, bigram_counts, vocab), 50))

print("Diversity vs Temperature (bigram model):")
print(f"  {'Temp':>6} {'Distinct-1':>12} {'Distinct-2':>12} {'Perplexity':>12}")
print("  " + "-" * 46)
for i, T in enumerate(temps):
    print(f"  {T:>6.1f} {diversity_data['distinct1'][i]:>12.3f} "
          f"{diversity_data['distinct2'][i]:>12.3f} "
          f"{diversity_data['perplexity'][i]:>12.2f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Prompt Engineering
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Prompt Engineering")
print("=" * 70)

print("""
Prompt engineering: crafting the input text to guide model output.

  Zero-shot:    Just describe the task and let the model complete it
    Prompt: "Translate to French: The cat is on the mat."

  Few-shot:     Provide examples before the actual query (in-context learning)
    Prompt: "English: Hello  French: Bonjour
             English: Thank you  French: Merci
             English: Good morning  French: "

  Chain-of-Thought: Ask the model to reason step by step
    Prompt: "Q: A train travels 60 mph for 2 hours. How far?
             A: Let me think step by step. Speed = 60 mph, Time = 2 hours.
             Distance = Speed × Time = 60 × 2 = 120 miles."

  Role prompting: Assign a persona
    Prompt: "You are an expert machine learning researcher. Explain attention."

Key insights:
  • Specificity matters: vague prompts → vague outputs
  • Temperature affects creativity: lower T = more focused
  • Top-P 0.9–0.95 often gives the best balance of quality and diversity
  • Few-shot examples dramatically improve task performance (GPT-3 paper)
""")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Production generate_text() API
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SECTION 7: Production generate_text() API")
print("=" * 70)


def generate_text(prompt, max_tokens=50, temperature=0.8, top_k=50,
                  top_p=0.9, strategy="top_p", seed=None):
    """
    Production text generation function.

    Args:
        prompt:      Input text to continue
        max_tokens:  Maximum new tokens to generate
        temperature: Sampling temperature (0.1=focused, 2.0=creative)
        top_k:       Top-K filtering (0 = disabled)
        top_p:       Nucleus probability threshold
        strategy:    'greedy' | 'temperature' | 'top_k' | 'top_p'
        seed:        Random seed for reproducibility

    Returns:
        dict with 'generated_text', 'new_tokens', 'strategy_used'
    """
    if seed is not None:
        np.random.seed(seed)

    if GPT2_AVAILABLE:
        full_text = generate_gpt2(prompt, max_new_tokens=max_tokens,
                                   temperature=temperature,
                                   top_k=top_k if top_k > 0 else 0,
                                   top_p=top_p,
                                   do_sample=(strategy != "greedy"))
        new_text = full_text[len(prompt):].strip()
    else:
        # Fallback to n-gram model
        prompt_words = prompt.lower().split()[:3]
        full_text = generate_ngram(prompt_words, max_tokens, strategy,
                                    temperature=temperature, k=top_k, p=top_p)
        new_text = " ".join(full_text.split()[len(prompt_words):])

    return {
        "generated_text": full_text,
        "new_tokens": new_text,
        "strategy_used": strategy,
        "temperature": temperature,
    }


# Demo
print("generate_text() API demo:")
demo_prompts_gen = [
    ("The attention mechanism in transformers", dict(temperature=0.7, top_p=0.9, seed=1)),
    ("In the year 2050,", dict(temperature=1.0, top_k=40, strategy="top_k", seed=2)),
    ("The best approach to machine learning is", dict(strategy="greedy", seed=3)),
]
for prompt, kwargs in demo_prompts_gen:
    result = generate_text(prompt, max_tokens=30, **kwargs)
    print(f"\n  Prompt  : {prompt}")
    print(f"  Strategy: {result['strategy_used']} (T={result['temperature']})")
    print(f"  Output  : {result['generated_text'][:120]}...")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Generating Visualizations")
print("=" * 70)

COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

# ── Visualization 1: Sampling Strategy Comparison ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Text Generation Sampling Strategies", fontsize=14, fontweight="bold")

probs_base = softmax(toy_logits, 1.0)

# 1a: Greedy — single token highlighted
ax = axes[0, 0]
colors_bar = ["#e74c3c" if i == np.argmax(toy_logits) else "#3498db" for i in range(len(VOCAB))]
ax.bar(VOCAB, probs_base, color=colors_bar, alpha=0.8, edgecolor="white")
ax.set_title("Greedy Decoding\n(always pick argmax — red bar)", fontsize=11, fontweight="bold")
ax.set_ylabel("Probability")
ax.tick_params(axis="x", rotation=30)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1b: Temperature effect
ax = axes[0, 1]
temps_show = [0.3, 0.7, 1.0, 1.5]
t_colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
x_pos = np.arange(len(VOCAB))
w = 0.2
for i, (T, col) in enumerate(zip(temps_show, t_colors)):
    probs_t = softmax(toy_logits, T)
    ax.bar(x_pos + i * w - 0.3, probs_t, w, alpha=0.8, color=col, label=f"T={T}")
ax.set_xticks(x_pos)
ax.set_xticklabels(VOCAB, fontsize=8, rotation=30)
ax.set_title("Temperature Sampling\n(lower T = sharper, higher T = flatter)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Probability")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1c: Top-K
ax = axes[1, 0]
k_show = 4
sorted_idx = np.argsort(toy_logits)[::-1]
top_k_mask = np.zeros(len(VOCAB), dtype=bool)
top_k_mask[sorted_idx[:k_show]] = True
bar_colors_k = ["#2ecc71" if top_k_mask[i] else "#bdc3c7" for i in range(len(VOCAB))]
probs_topk = np.zeros(len(VOCAB))
probs_topk[top_k_mask] = probs_base[top_k_mask]
probs_topk[top_k_mask] /= probs_topk[top_k_mask].sum()
ax.bar(VOCAB, probs_topk, color=bar_colors_k, alpha=0.85, edgecolor="white")
ax.set_title(f"Top-K Sampling (K={k_show})\n(gray bars = excluded)", fontsize=11, fontweight="bold")
ax.set_ylabel("Probability (renormalized)")
ax.tick_params(axis="x", rotation=30)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1d: Top-P (nucleus)
ax = axes[1, 1]
p_thresh = 0.85
sorted_probs = probs_base[sorted_idx]
cumulative = np.cumsum(sorted_probs)
nucleus_size = np.searchsorted(cumulative, p_thresh) + 1
nucleus_mask = np.zeros(len(VOCAB), dtype=bool)
nucleus_mask[sorted_idx[:nucleus_size]] = True
bar_colors_p = ["#9b59b6" if nucleus_mask[i] else "#bdc3c7" for i in range(len(VOCAB))]
probs_nucleus = np.zeros(len(VOCAB))
probs_nucleus[nucleus_mask] = probs_base[nucleus_mask]
if probs_nucleus.sum() > 0:
    probs_nucleus[nucleus_mask] /= probs_nucleus[nucleus_mask].sum()
ax.bar(VOCAB, probs_nucleus, color=bar_colors_p, alpha=0.85, edgecolor="white")
ax.set_title(f"Top-P Sampling (p={p_thresh})\n({nucleus_size} tokens in nucleus — gray excluded)",
             fontsize=11, fontweight="bold")
ax.set_ylabel("Probability (renormalized)")
ax.tick_params(axis="x", rotation=30)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_sampling_strategies.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_sampling_strategies.png")

# ── Visualization 2: Diversity vs Temperature ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Generation Quality vs Temperature", fontsize=14, fontweight="bold")

ax = axes[0]
ax.plot(diversity_data["temperature"], diversity_data["distinct1"],
        color="#3498db", linewidth=2.5, marker="o", markersize=6, label="Distinct-1")
ax.plot(diversity_data["temperature"], diversity_data["distinct2"],
        color="#2ecc71", linewidth=2.5, marker="s", markersize=6, label="Distinct-2")
ax.set_title("Lexical Diversity vs Temperature\n(higher = more diverse vocabulary)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Temperature")
ax.set_ylabel("Distinct-n score")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(diversity_data["temperature"], diversity_data["perplexity"],
        color="#e74c3c", linewidth=2.5, marker="^", markersize=6)
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7, label="T=1.0")
ax.set_title("Perplexity vs Temperature\n(lower = more predictable)", fontsize=11, fontweight="bold")
ax.set_xlabel("Temperature")
ax.set_ylabel("Perplexity (bigram model)")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[2]
# Scatter: diversity vs perplexity (colored by temperature)
scatter = ax.scatter(diversity_data["distinct2"], diversity_data["perplexity"],
                     c=diversity_data["temperature"], cmap="RdYlGn_r",
                     s=120, alpha=0.85, edgecolors="white", linewidths=1.5)
for i, T in enumerate(diversity_data["temperature"]):
    ax.annotate(f"T={T}", (diversity_data["distinct2"][i], diversity_data["perplexity"][i]),
                textcoords="offset points", xytext=(5, 5), fontsize=8)
plt.colorbar(scatter, ax=ax, label="Temperature")
ax.set_title("Diversity–Quality Tradeoff\n(ideal: high distinct, low perplexity)",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Distinct-2 (diversity ↑)")
ax.set_ylabel("Perplexity (lower=better ↓)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_diversity_quality.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_diversity_quality.png")

# ── Visualization 3: GPT Architecture + Generation Flow ───────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 8))
fig.suptitle("GPT-2: Autoregressive Generation", fontsize=14, fontweight="bold")

# 3a: GPT decoder architecture
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("GPT-2 Architecture (Decoder-Only)", fontsize=12, fontweight="bold")

steps_gpt = [
    (5, 11.2, "Input Tokens + Positional Encoding", "#7f8c8d"),
    (5, 9.5,  "Causal Self-Attention\n(mask: can't see future)", "#9b59b6"),
    (5, 7.8,  "Add & LayerNorm", "#95a5a6"),
    (5, 6.5,  "Feed-Forward Network (FFN)", "#2ecc71"),
    (5, 5.2,  "Add & LayerNorm", "#95a5a6"),
    (5, 3.8,  "[Repeat × 12 layers]", "#e67e22"),
    (5, 2.5,  "Linear (d_model → vocab_size=50257)", "#3498db"),
    (5, 1.2,  "Softmax → Next Token Probabilities", "#e74c3c"),
]
for i, (x, y, lbl, col) in enumerate(steps_gpt):
    rect = mpatches.FancyBboxPatch((x - 3.8, y - 0.55), 7.6, 1.0,
                                    boxstyle="round,pad=0.05", facecolor=col, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, y, lbl, ha="center", va="center", fontsize=8.5,
            fontweight="bold", color="white")
    if i < len(steps_gpt) - 1:
        ax.annotate("", xy=(x, steps_gpt[i + 1][1] + 0.55),
                    xytext=(x, y - 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#7f8c8d"))

# Add "No Cross-Attention" note
ax.text(9.8, 7.5, "No cross-\nattention!\n(decoder only)", ha="center", fontsize=8,
        color="#e74c3c", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e74c3c"))

# 3b: Autoregressive generation step-by-step
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 9)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
ax.set_title("Autoregressive Generation Loop\n(each step generates one token)",
             fontsize=12, fontweight="bold")

GEN_STEPS = [
    "\"The future\"",
    "\"The future of\"",
    "\"The future of AI\"",
    "\"The future of AI is\"",
    "\"The future of AI is bright\"",
]
for step_i, text in enumerate(GEN_STEPS):
    y_pos = 7.5 - step_i * 1.4
    # Token sequence box
    rect = mpatches.FancyBboxPatch((0.3, y_pos - 0.4), 5.8, 0.9,
                                    boxstyle="round,pad=0.05",
                                    facecolor="#3498db" if step_i < len(GEN_STEPS) - 1 else "#2ecc71",
                                    alpha=0.7)
    ax.add_patch(rect)
    ax.text(3.2, y_pos, text, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white")
    if step_i < len(GEN_STEPS) - 1:
        new_tok = GEN_STEPS[step_i + 1].split()[-1].replace('"', '')
        ax.text(6.5, y_pos, f"→ sample\n'{new_tok}'", ha="left", va="center",
                fontsize=8.5, color="#9b59b6", fontweight="bold")
        ax.annotate("", xy=(3.2, y_pos - 0.4), xytext=(3.2, y_pos - 1.0 + 0.55),
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#7f8c8d"))

ax.text(5.0, 0.3,
        f"Strategy: Top-P (p=0.9), T=0.8\n{'GPT-2 (real)' if GPT2_AVAILABLE else 'N-gram (demo)'}",
        ha="center", va="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#bdc3c7"))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_gpt_generation.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_gpt_generation.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROJECT SUMMARY — GPT-2 Text Generator")
print("=" * 70)
print(f"""
What we built:
  ✓ Sampling strategy implementations from scratch:
      Greedy, Temperature, Top-K, Top-P (nucleus)
  ✓ Bigram n-gram language model for generation demo (no GPU needed)
  ✓ GPT-2 HuggingFace integration {'(real GPT-2 loaded ✓)' if GPT2_AVAILABLE else '(install transformers + torch)'}
  ✓ Diversity metrics: Distinct-1, Distinct-2, Perplexity
  ✓ Prompt engineering examples: zero-shot, few-shot, chain-of-thought
  ✓ Production generate_text() API with configurable sampling

Key takeaways:
  • Temperature < 1.0 → focused, repetitive; Temperature > 1.0 → creative, random
  • Top-P (nucleus) adapts to distribution entropy — usually best for quality
  • Greedy is deterministic but often leads to repetitive/degenerate output
  • Perplexity and diversity are in tension — optimal T is typically 0.7–1.0
  • Prompt engineering is crucial: specificity and examples guide output quality

Sampling strategy guide:
  Creative writing   → T=0.9–1.2, Top-P=0.95
  Factual Q&A        → T=0.3–0.5, Top-K=10–20
  Code generation    → T=0.2–0.4, Top-K=5–15  (deterministic preferred)
  Balanced general   → T=0.8, Top-P=0.9  (GPT-2 defaults)

BERT vs GPT-2 at a glance:
  BERT   → Encoder-only, bidirectional, great for CLASSIFICATION
  GPT-2  → Decoder-only, causal, great for GENERATION

Visualizations saved to: {VIS_DIR}/
  01_sampling_strategies.png  — greedy, temperature, top-k, top-p comparison
  02_diversity_quality.png    — diversity vs temperature curves
  03_gpt_generation.png       — architecture + autoregressive loop diagram
""")
