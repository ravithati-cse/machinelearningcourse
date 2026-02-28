"""
01_how_llms_work.py
===================
Part 7 - Large Language Models: How LLMs Work

Learning Objectives:
  1. Understand the decoder-only transformer architecture that underlies modern LLMs
  2. Implement a simple Byte-Pair Encoding (BPE)-style tokenizer from scratch
  3. Explain scaling laws (Chinchilla) and their implications for training
  4. Describe emergent abilities that arise at large scale
  5. Distinguish pre-training, fine-tuning, RLHF, and prompting paradigms
  6. Compute next-token prediction loss and interpret perplexity

YouTube: Search "How Large Language Models Work 3Blue1Brown" and
         "Andrej Karpathy Let's build GPT from scratch"

Time: ~25 min | Difficulty: Intermediate | Prerequisites: Parts 3-6
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import math
from collections import Counter, defaultdict

# ─── paths ────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(_HERE, "..", "visuals", "01_how_llms_work")
os.makedirs(VIS_DIR, exist_ok=True)

# ==============================================================================
print("\n" + "="*70)
print("SECTION 1: WHAT IS A LARGE LANGUAGE MODEL?")
print("="*70)
# ==============================================================================

print("""
A Large Language Model (LLM) is a neural network trained on massive text
corpora to predict the next token in a sequence. The key architecture is
the DECODER-ONLY TRANSFORMER (as in GPT), which processes tokens left-to-
right using causal (masked) self-attention.

Core training objective — NEXT-TOKEN PREDICTION (Causal Language Modeling):
  Given tokens t_1, t_2, ..., t_{i-1}, predict t_i.

Loss function:
  L = -1/T * sum_{i=1}^{T} log P(t_i | t_1, ..., t_{i-1})

This is cross-entropy loss, averaged over all token positions T.
Perplexity = exp(L) — the "effective vocabulary size" the model is confused by.
""")

# Model scale table
print("MODEL SCALE vs. CAPABILITIES:")
print("-" * 68)
print(f"{'Model':<20} {'Parameters':<18} {'Training Tokens':<20} {'Key Capability'}")
print("-" * 68)
models = [
    ("GPT-1",     "117M",   "~1B",    "Basic coherence"),
    ("GPT-2",     "1.5B",   "~10B",   "Coherent paragraphs"),
    ("GPT-3",     "175B",   "300B",   "Few-shot in-context learning"),
    ("Chinchilla","70B",    "1.4T",   "Compute-optimal 70B model"),
    ("LLaMA-2",   "70B",    "2T",     "Open, instruction-tuned"),
    ("GPT-4",     "~1T est","~13T est","Complex reasoning, multimodal"),
    ("Gemini 1.5","~1T est","~5T est","Long context (1M tokens)"),
]
for name, params, tokens, cap in models:
    print(f"  {name:<18} {params:<18} {tokens:<20} {cap}")
print("-" * 68)

print("""
KEY INSIGHT: Scale alone isn't sufficient. Chinchilla showed that many
large models were undertrained. Compute-optimal training requires
proportionally more data as models grow larger.

Architecture recap — Decoder-Only Transformer:
  Input tokens → Embedding + Positional Encoding
  → N x (Masked Self-Attention → Feed-Forward) blocks
  → Linear projection → Softmax over vocabulary
  → Next-token probability distribution

"Masked" means each token can only attend to tokens BEFORE it (causal mask).
This allows training on sequences of any length by predicting all positions
in one forward pass.
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 2: TOKENIZATION — FROM TEXT TO NUMBERS")
print("="*70)
# ==============================================================================

print("""
LLMs don't operate on characters or words — they use SUBWORD TOKENS.
The dominant algorithm is Byte-Pair Encoding (BPE), introduced by
Sennrich et al. (2016) and used in GPT-2/3/4, LLaMA, and most modern LLMs.

THREE TOKENIZATION STRATEGIES:
  1. Word-level:      "transformer" → ["transformer"]  — OOV problem for new words
  2. Character-level: "transformer" → ['t','r','a','n','s','f','o','r','m','e','r']
                      Very long sequences; loses word structure.
  3. Subword (BPE):   "transformer" → ["transform", "er"]  — best of both worlds
""")

def get_vocab(text):
    """Build character-level vocab from text, treating each word as chars + </w>."""
    vocab = defaultdict(int)
    for word in text.lower().split():
        word_chars = " ".join(list(word)) + " </w>"
        vocab[word_chars] += 1
    return vocab

def get_pairs(vocab):
    """Count all adjacent symbol pairs across all words in vocab."""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, vocab):
    """Merge the most frequent pair into a single symbol."""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

# Sample corpus for BPE demo
sample_corpus = (
    "the transformer model is a large language model "
    "transformers transform text transformer training transforms language "
    "the model learns from large amounts of text data "
    "language models learn language patterns from text"
)

print("BPE TOKENIZATION — FROM SCRATCH DEMONSTRATION")
print(f"Sample corpus: '{sample_corpus[:60]}...'")
print()

vocab = get_vocab(sample_corpus)
print("Initial character-level vocabulary (top 8 entries):")
for i, (word, count) in enumerate(list(vocab.items())[:8]):
    print(f"  '{word}'  (freq={count})")

print("\nRunning 10 BPE merge rounds...")
merges = []
for round_num in range(10):
    pairs = get_pairs(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    best_count = pairs[best_pair]
    merges.append((best_pair, best_count))
    vocab = merge_pair(best_pair, vocab)
    merged_symbol = "".join(best_pair)
    print(f"  Round {round_num+1}: Merge {best_pair} → '{merged_symbol}'  (freq={best_count})")

print("\nFinal vocabulary sample (top 10 subword units):")
all_symbols = set()
for word in vocab.keys():
    all_symbols.update(word.split())
sorted_symbols = sorted(all_symbols, key=lambda s: -len(s))[:10]
for sym in sorted_symbols:
    print(f"  '{sym}'")

# Tokenization comparison
word = "transformer"
char_tokens = list(word)
bpe_tokens = ["transform", "er"]          # approximate after merges
word_tokens = [word]

print("\nTOKENIZATION COMPARISON for the word 'transformer':")
print(f"  Word-level:  {word_tokens}  ({len(word_tokens)} token)")
print(f"  Char-level:  {char_tokens}  ({len(char_tokens)} tokens)")
print(f"  BPE approx:  {bpe_tokens}  ({len(bpe_tokens)} tokens)")

sample_sentence = "transformers are large language models trained on massive datasets"
print(f"\nFor the sentence: '{sample_sentence}'")
word_len = len(sample_sentence.split())
char_len = len(sample_sentence.replace(" ", ""))
bpe_len  = 13   # typical BPE estimate
print(f"  Word-level length:      {word_len} tokens")
print(f"  Character-level length: {char_len} tokens")
print(f"  BPE length (estimated): {bpe_len} tokens  ← efficient & handles unknown words")

print("""
GPT-4 uses ~100K BPE tokens (cl100k_base).
LLaMA 3 uses a ~128K vocabulary.
Larger vocabularies → shorter sequences → faster training & inference.
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 3: SCALING LAWS (CHINCHILLA)")
print("="*70)
# ==============================================================================

print("""
Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models"
(nicknamed the Chinchilla paper) showed:

  CHINCHILLA SCALING LAW:
    Optimal training tokens = 20 × number of parameters

  Before Chinchilla, the field trained large models with too little data.
  Chinchilla-70B (70B params, 1.4T tokens) outperformed Gopher (280B params)
  despite being 4× smaller — because it was trained on 4× more data.

  Compute budget C ≈ 6 × N × D
    where N = parameters, D = training tokens, C = FLOPs
  
  Optimal: N_opt ≈ sqrt(C / 120)    D_opt ≈ 20 * N_opt
""")

print("CHINCHILLA-OPTIMAL TRAINING TABLE:")
print("-" * 60)
print(f"  {'Parameters':<18} {'Optimal Tokens':<22} {'Compute (PF-days)'}")
print("-" * 60)
chinchilla_table = [
    (1e9,   "1B",   20e9,  "20B",   0.5),
    (7e9,   "7B",   140e9, "140B",  27),
    (13e9,  "13B",  260e9, "260B",  99),
    (34e9,  "34B",  680e9, "680B",  677),
    (70e9,  "70B",  1.4e12,"1.4T",  2800),
    (175e9, "175B", 3.5e12,"3.5T",  17500),
]
for n, n_str, d, d_str, pfd in chinchilla_table:
    print(f"  {n_str:<18} {d_str:<22} {pfd:.0f}")
print("-" * 60)

print("""
IMPLICATIONS:
  - GPT-3 (175B params, 300B tokens) was ~10× undertrained!
  - LLaMA-2-70B (70B, 2T tokens) ≈ compute-optimal.
  - Smaller, well-trained models can match larger undertrained ones.
  - More data collection is as important as model architecture.
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 4: EMERGENT ABILITIES")
print("="*70)
# ==============================================================================

print("""
EMERGENCE: Capabilities that appear suddenly at scale, not predicted by
extrapolating from smaller models (Wei et al., 2022).

Key emergent abilities and the scale at which they appear:
  - Few-shot in-context learning     (~10B+ params, GPT-3 era)
  - Chain-of-thought reasoning       (~100B+ params)
  - Instruction following            (~50B+ params + fine-tuning)
  - Code generation                  (~12B+ params)
  - Multi-step arithmetic            (~100B+ params with CoT)
  - Theory of mind (basic)           (~540B params, PaLM)

IN-CONTEXT LEARNING (ICL): The model adapts its behavior purely from
examples in the prompt — no gradient updates, no weight changes.
""")

print("ZERO-SHOT vs FEW-SHOT EXAMPLE:")
print()
print("Task: Translate English → French")
print()
print("  ZERO-SHOT PROMPT:")
print("  ┌─────────────────────────────────────────┐")
print("  │ Translate to French: 'The sky is blue.' │")
print("  │ → Le ciel est bleu.                     │")
print("  └─────────────────────────────────────────┘")
print()
print("  FEW-SHOT PROMPT (3-shot):")
print("  ┌──────────────────────────────────────────────────────┐")
print("  │ English: Hello → French: Bonjour                     │")
print("  │ English: Thank you → French: Merci                   │")
print("  │ English: Goodbye → French: Au revoir                 │")
print("  │ English: The sky is blue → French: ?                 │")
print("  │ → Le ciel est bleu.  (more reliable with examples)   │")
print("  └──────────────────────────────────────────────────────┘")
print()
print("CHAIN-OF-THOUGHT example (math):")
print()
print("  STANDARD PROMPT:  'If a train travels 60 mph for 2.5 hours,")
print("                     how far does it travel? Answer: ?'")
print("  → Standard model: '150 miles' (often correct, sometimes wrong)")
print()
print("  COT PROMPT: 'Let's think step by step.'")
print("  → Model: 'Speed = 60 mph. Time = 2.5 hours.")
print("             Distance = speed × time = 60 × 2.5 = 150 miles.'")
print("  → Dramatically more reliable on complex multi-step problems.")
print()

# ==============================================================================
print("\n" + "="*70)
print("SECTION 5: PRE-TRAINING vs FINE-TUNING vs PROMPTING")
print("="*70)
# ==============================================================================

print("""
FOUR PARADIGMS for deploying language model capabilities:

1. PRE-TRAINING
   - Train from scratch on massive unlabeled text (CommonCrawl, Books, Wikipedia)
   - Objective: next-token prediction (causal LM) or masked LM (BERT)
   - Cost: millions of dollars, thousands of GPU-days
   - Result: a general-purpose base model with broad world knowledge

2. SUPERVISED FINE-TUNING (SFT)
   - Take a pre-trained base model, train on (instruction, response) pairs
   - Teaches the model to follow instructions and be helpful
   - Cost: hundreds of GPU-hours on 1K–100K examples
   - Result: an "instruct" model (e.g., GPT-3 → InstructGPT)

3. RLHF (Reinforcement Learning from Human Feedback)
   - Human raters compare model outputs, train a reward model
   - Fine-tune the LLM with PPO to maximize reward while staying close to SFT
   - Aligns model with human preferences (helpful, harmless, honest)
   - Result: ChatGPT, Claude, Gemini — aligned assistants

4. PROMPTING
   - Use the frozen model as-is, engineer the input text
   - Zero-shot, few-shot, chain-of-thought, retrieval-augmented
   - Cost: only inference compute
   - Result: task-specific behavior without any weight updates
""")

print("COMPARISON TABLE:")
print("-" * 72)
print(f"  {'Paradigm':<22} {'Weight Update?':<16} {'Data Needed':<20} {'Cost'}")
print("-" * 72)
rows = [
    ("Pre-training",    "Yes (all weights)", "Trillions of tokens", "Millions $$$"),
    ("SFT",            "Yes (all or PEFT)", "1K–100K pairs",       "Hundreds GPU-hrs"),
    ("RLHF",           "Yes (policy net)", "Human comparisons",   "Thousands GPU-hrs"),
    ("Prompting",      "No",               "0–100 examples",      "Inference only"),
]
for paradigm, update, data, cost in rows:
    print(f"  {paradigm:<22} {update:<16} {data:<20} {cost}")
print("-" * 72)

print("""
WHEN TO USE EACH:
  - Prompting first: always try this before anything else
  - SFT: when prompting fails, you have labeled data, and need consistency
  - RLHF: when you need nuanced alignment and have human annotators
  - Pre-training: only if you have a unique domain (medical, legal, code)
    or need a fully open model
""")

# ==============================================================================
print("\n" + "="*70)
print("SECTION 6: GENERATING VISUALIZATIONS")
print("="*70)
# ==============================================================================

print(f"Saving visualizations to: {VIS_DIR}")

# ── Visualization 1: LLM Scale vs Benchmark ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

model_data = [
    ("GPT-2\n(1.5B)",   1.5e9,  55,  "#4C72B0"),
    ("GPT-3\n(175B)",   1.75e11, 72, "#DD8452"),
    ("GPT-3.5\n(~20B)", 2e10,   82,  "#55A868"),
    ("PaLM\n(540B)",    5.4e11, 85,  "#C44E52"),
    ("GPT-4\n(~1T)",    1e12,   92,  "#8172B2"),
    ("Gemini\nUltra",   1e12,   90,  "#937860"),
    ("LLaMA-2\n(70B)",  7e10,   80,  "#DA8BC3"),
    ("Chinchilla\n(70B)", 7e10, 79, "#8C8C8C"),
]

for name, params, score, color in model_data:
    ax.scatter(params, score, s=180, color=color, zorder=5)
    ax.annotate(name, (params, score), textcoords="offset points",
                xytext=(8, 4), fontsize=8, color=color)

# Trend line
param_vals = np.array([d[1] for d in model_data])
score_vals = np.array([d[2] for d in model_data])
log_params = np.log10(param_vals)
coeffs = np.polyfit(log_params, score_vals, 1)
x_fit = np.logspace(8.5, 12.5, 200)
y_fit = coeffs[0] * np.log10(x_fit) + coeffs[1]
ax.plot(x_fit, y_fit, "k--", alpha=0.35, linewidth=1.5, label="Log-linear trend")

ax.set_xscale("log")
ax.set_xlabel("Model Parameters (log scale)", fontsize=12)
ax.set_ylabel("Aggregate Benchmark Score (simulated %)", fontsize=12)
ax.set_title("LLM Scale vs. Capability\n(Simulated Data — Illustrative)", fontsize=13, fontweight="bold")
ax.set_xlim(5e8, 3e12)
ax.set_ylim(45, 100)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "01_llm_scale.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 01_llm_scale.png")

# ── Visualization 2: Scaling Laws ─────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot a: compute-optimal tokens vs params
params_B = np.array([1, 7, 13, 34, 70, 175])
opt_tokens_B = params_B * 20
ax1.plot(params_B, opt_tokens_B, "o-", color="#4C72B0", linewidth=2, markersize=8, label="Chinchilla optimal (20x)")
actual_tokens = np.array([20, 100, 200, 300, 500, 300])
ax1.plot(params_B, actual_tokens, "s--", color="#DD8452", linewidth=2, markersize=8, label="Actual training (pre-Chinchilla)")
ax1.fill_between(params_B, opt_tokens_B, actual_tokens,
                  where=opt_tokens_B > actual_tokens,
                  alpha=0.15, color="red", label="Under-training gap")
ax1.set_xlabel("Model Parameters (B)", fontsize=11)
ax1.set_ylabel("Training Tokens (B)", fontsize=11)
ax1.set_title("Compute-Optimal Token Budget\n(Chinchilla Law: 20x params)", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot b: loss vs compute
compute = np.logspace(19, 24, 300)   # FLOPs
# Simplified power-law loss curves for different model sizes
for n_b, label, color in [(7, "7B model", "#55A868"), (70, "70B model", "#4C72B0"), (175, "175B model", "#C44E52")]:
    n = n_b * 1e9
    # Irreducible loss + power law in compute
    loss = 1.8 + 3e23 / compute**0.5 + n * 1e-12
    ax2.plot(compute, loss, color=color, linewidth=2, label=label)

ax2.set_xscale("log")
ax2.set_xlabel("Compute Budget (FLOPs, log scale)", fontsize=11)
ax2.set_ylabel("Language Model Loss", fontsize=11)
ax2.set_title("Loss vs Compute Budget\n(Power-law scaling, simulated)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(1.5, 5.0)

fig.suptitle("Chinchilla Scaling Laws", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "02_scaling_laws.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 02_scaling_laws.png")

# ── Visualization 3: Tokenization Comparison ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart: sequence lengths per strategy
strategies  = ["Word-level", "Character-level", "BPE (GPT-4)"]
seq_lengths = [12, 61, 15]
colors      = ["#DD8452", "#C44E52", "#55A868"]
bars = ax1.bar(strategies, seq_lengths, color=colors, edgecolor="white", linewidth=1.2, width=0.5)
for bar, val in zip(bars, seq_lengths):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
ax1.set_ylabel("Sequence Length (tokens)", fontsize=11)
ax1.set_title("Tokenization Strategy Comparison\n('transformers are large language models...')",
              fontsize=10, fontweight="bold")
ax1.set_ylim(0, 75)
ax1.grid(True, axis="y", alpha=0.3)

# BPE merge steps diagram (text-based in matplotlib)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("BPE Merge Steps (Example)", fontsize=11, fontweight="bold")

steps = [
    (9.0, "Initial:  t r a n s f o r m e r"),
    (7.8, "Round 1:  t r a n s f o r m er   [merge: e+r]"),
    (6.6, "Round 2:  t r a n s f orm er     [merge: o+r→or; f+orm]"),
    (5.4, "Round 3:  t r a n s form er      [merge: f+orm]"),
    (4.2, "Round 4:  t r ans form er        [merge: a+n→an; n+s→ns]"),
    (3.0, "Round 5:  tr ans form er         [merge: t+r]"),
    (1.5, "Final:   ['transform', 'er']     ← 2 subword tokens!"),
]
for y_pos, text in steps:
    color = "#2d6a4f" if "Final" in text else "#1a1a2e"
    weight = "bold" if "Final" in text else "normal"
    ax2.text(0.3, y_pos, text, fontsize=9, color=color, fontweight=weight,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", alpha=0.7) if "Final" in text else None)

fig.suptitle("Tokenization: From Text to Token IDs", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(VIS_DIR, "03_tokenization.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: 03_tokenization.png")

print(f"\nAll 3 visualizations saved to: {VIS_DIR}")
print("\n" + "="*70)
print("MODULE 01 COMPLETE: How LLMs Work")
print("="*70)
print("""
KEY TAKEAWAYS:
  1. LLMs are decoder-only transformers trained with next-token prediction loss
  2. BPE tokenization balances vocabulary size with sequence length
  3. Chinchilla law: train on 20x as many tokens as parameters
  4. Emergent abilities appear at scale — not predictable from small models
  5. Prompting is the cheapest adaptation; pre-training the most expensive
  6. RLHF aligns model behavior with human preferences via reward modeling

NEXT: 02_prompt_engineering.py — master the art of prompting
""")
