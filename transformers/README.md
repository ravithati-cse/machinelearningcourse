# Part 6: Transformers

A hands-on, visual-first introduction to Transformer architecture — from the
scaled dot-product attention formula through production BERT and GPT-2 pipelines.
No prior experience with attention mechanisms required.

---

## Quick Start

```bash
# Math foundations (run in order)
cd transformers/math_foundations
python3 01_attention_mechanism.py
python3 02_multi_head_attention.py
python3 03_positional_encoding.py
python3 04_encoder_decoder_arch.py

# Algorithms
cd ../algorithms
python3 transformer_from_scratch.py
python3 bert_encoder.py
python3 gpt_decoder.py

# Projects
cd ../projects
python3 bert_text_classifier.py
python3 gpt2_text_generator.py

# View all generated visuals
open ../visuals/
```

---

## Module Map

### Math Foundations

| # | File | Concepts | Time | Difficulty |
|---|------|----------|------|------------|
| 01 | `math_foundations/01_attention_mechanism.py` | Scaled dot-product attention · Q/K/V as soft lookup · causal masking · Bahdanau vs Vaswani | 20 min | Intermediate |
| 02 | `math_foundations/02_multi_head_attention.py` | Multi-head attention · head splitting/merging · W_Q/K/V/O projections · shape trace (B,S,D)→(B,h,S,d_k) | 25 min | Intermediate |
| 03 | `math_foundations/03_positional_encoding.py` | Sinusoidal PE formula · learned embeddings · RoPE · position similarity analysis | 20 min | Intermediate |
| 04 | `math_foundations/04_encoder_decoder_arch.py` | LayerNorm · FeedForward (GELU) · residual connections · Pre-LN vs Post-LN · EncoderBlock · DecoderBlock + cross-attention | 30 min | Advanced |

### Algorithms

| File | What it demonstrates | Time | Difficulty |
|------|---------------------|------|------------|
| `algorithms/transformer_from_scratch.py` | Full encoder-only transformer in NumPy · synthetic sentiment dataset · mean-pool → sklearn LR · end-to-end forward pass | 40 min | Advanced |
| `algorithms/bert_encoder.py` | BERT architecture deep-dive · WordPiece tokenization · [CLS] token · HuggingFace `BertModel` · feature extraction vs fine-tuning | 35 min | Advanced |
| `algorithms/gpt_decoder.py` | GPT decoder-only architecture · causal masking · autoregressive generation · greedy / temperature / top-K / top-P / beam search | 40 min | Advanced |

### Projects

| File | Task | Time | Difficulty |
|------|------|------|------------|
| `projects/bert_text_classifier.py` | 5-class news classification (Technology / Sports / Politics / Business / Entertainment) with frozen BERT embeddings + logistic regression head | 60 min | Advanced |
| `projects/gpt2_text_generator.py` | Creative text generation with all sampling strategies implemented from scratch + HuggingFace GPT-2 | 60 min | Advanced |

---

## Concept Cheat Sheet

| Concept | Formula / Key Idea |
|---------|--------------------|
| Scaled dot-product attention | `Attention(Q,K,V) = softmax(QK^T / √d_k) V` |
| Why scale by `√d_k`? | Prevents dot products from growing large → softmax saturation |
| Multi-head attention | h parallel attention heads, each in d_k = d_model/h subspace |
| Causal mask | Lower-triangular matrix; future positions set to −∞ before softmax |
| Sinusoidal PE | `PE(pos, 2i) = sin(pos / 10000^(2i/d))`, `PE(pos, 2i+1) = cos(...)` |
| LayerNorm | Normalizes across feature dim (not batch dim); stable for variable-length sequences |
| FFN sublayer | `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`; hidden dim = 4 × d_model |
| Residual connections | `output = LayerNorm(x + Sublayer(x))`; enables deep networks |
| [CLS] token (BERT) | Prepended sentinel; its final hidden state = sequence-level representation |
| Feature extraction | Freeze BERT weights; train only the task head (fast, small-data friendly) |
| Fine-tuning | Update all BERT weights end-to-end on the task (slower, more accurate) |
| Autoregressive generation | Predict one token at a time; each output fed back as next input |
| Temperature sampling | Divide logits by T before softmax: high T → uniform, low T → sharp |
| Top-K sampling | Keep only K most likely tokens; renormalize; sample |
| Top-P (nucleus) sampling | Keep smallest set whose cumulative prob ≥ P; sample from it |
| Beam search | Maintain B hypothesis beams; expand + prune each step; no randomness |

---

## Prerequisites

- **Part 3 (DNNs)** — backprop, activation functions, gradient descent
- **Part 4 (CNNs)** — convolutional operations, feature maps (helpful but not required)
- **Part 5 (NLP)** — tokenization, TF-IDF, word embeddings, RNN intuition
- **Python** — numpy, matplotlib, scikit-learn
- **Optional (for real BERT/GPT-2)** — `pip install transformers torch`

---

## Installation

```bash
# Base dependencies (already installed if you did Parts 1-5)
pip install numpy pandas matplotlib seaborn scikit-learn

# For real BERT / GPT-2 (optional — all modules degrade gracefully without)
pip install transformers torch
```

All modules work without `transformers` / `torch` installed — they simulate
model outputs and clearly label results as simulated.

---

## Visualization Index

Every module auto-generates 3 PNG visualizations (300 dpi) in `visuals/`:

| Directory | Visualizations |
|-----------|---------------|
| `visuals/01_attention_mechanism/` | attention weight heatmap · scaling effect on softmax · Bahdanau vs Vaswani comparison |
| `visuals/02_multi_head_attention/` | multi-head attention diagram · head specialization patterns · complexity comparison |
| `visuals/03_positional_encoding/` | sinusoidal encoding heatmap · position similarity matrix · learned vs sinusoidal vs RoPE |
| `visuals/04_encoder_decoder_arch/` | encoder block diagram · decoder block with cross-attention · Pre-LN vs Post-LN gradient flow |
| `visuals/transformer_from_scratch/` | full architecture diagram · attention weights · training loss curve |
| `visuals/bert_encoder/` | BERT architecture diagram · embedding space (PCA) · tokenization examples |
| `visuals/gpt_decoder/` | GPT architecture diagram · sampling strategy comparison · token probability distributions |
| `visuals/bert_text_classifier/` | BERT pipeline diagram · confusion matrix + per-class P/R/F1 · PCA embedding space |
| `visuals/gpt2_text_generator/` | sampling strategy comparison · temperature effect · diversity metrics (distinct-n) |

---

## How Transformers Fit into the Course

```
Part 1: Regression       →  linear models, gradient descent basics
Part 2: Classification   →  logistic regression, decision boundaries
Part 3: DNNs             →  multi-layer networks, backpropagation, activations
Part 4: CNNs             →  spatial feature extraction, weight sharing
Part 5: NLP              →  text processing, word embeddings, RNN/LSTM
Part 6: Transformers ← YOU ARE HERE
  Math foundations:  attention, multi-head attention, positional encoding, encoder-decoder
  Algorithms:        transformer from scratch, BERT (encoder), GPT (decoder)
  Projects:          text classification with BERT, text generation with GPT-2
Part 7: LLMs (planned)  →  prompt engineering, fine-tuning, RAG
```

---

## Key Papers

| Paper | Year | What it introduced |
|-------|------|--------------------|
| *Attention Is All You Need* (Vaswani et al.) | 2017 | Transformer architecture |
| *BERT* (Devlin et al.) | 2018 | Bidirectional encoder, [CLS] token, masked LM pre-training |
| *Language Models are Unsupervised Multitask Learners* (Radford et al.) | 2019 | GPT-2, decoder-only generation |
| *RoFormer* (Su et al.) | 2021 | Rotary Position Embedding (RoPE) |
| *Training language models to follow instructions* (Ouyang et al.) | 2022 | InstructGPT / RLHF |

---

*Part of the [MLForBeginners](../README.md) course — a visual-first ML curriculum from algebra to production.*
