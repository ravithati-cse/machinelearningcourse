# Part 7: Large Language Models (LLMs)

A hands-on, visual-first deep dive into Large Language Models — from the
mathematics of next-token prediction through prompt engineering, fine-tuning
with LoRA, retrieval-augmented generation (RAG), and production deployment.
No prior LLM experience required; Parts 3–6 are the only prerequisites.

---

## Quick Start

```bash
# Math foundations (run in order)
cd llms/math_foundations
python3 01_how_llms_work.py
python3 02_prompt_engineering.py
python3 03_fine_tuning_basics.py
python3 04_rag_pipeline.py

# Algorithms
cd ../algorithms
python3 llm_api_usage.py
python3 build_a_chatbot.py
python3 llm_evaluation.py

# Projects
cd ../projects
python3 qa_system_with_rag.py
python3 llm_powered_classifier.py

# View all generated visuals
open ../visuals/
```

---

## Module Map

### Math Foundations

| # | File | Concepts | Time | Difficulty |
|---|------|----------|------|------------|
| 01 | `math_foundations/01_how_llms_work.py` | Next-token prediction · cross-entropy loss · token probability distributions · scaling laws (Chinchilla) · emergent capabilities | 25 min | Intermediate |
| 02 | `math_foundations/02_prompt_engineering.py` | Zero-shot prompting · few-shot in-context learning · chain-of-thought · role prompting · prompt templates · systematic prompt evaluation | 25 min | Intermediate |
| 03 | `math_foundations/03_fine_tuning_basics.py` | Full fine-tuning vs parameter-efficient · LoRA math (low-rank decomposition) · adapter layers · catastrophic forgetting · PEFT workflows | 30 min | Advanced |
| 04 | `math_foundations/04_rag_pipeline.py` | Document chunking · embedding & indexing · retrieval strategies · context injection · RAG vs fine-tuning tradeoffs · evaluation metrics | 30 min | Advanced |

### Algorithms

| File | What it demonstrates | Time | Difficulty |
|------|---------------------|------|------------|
| `algorithms/llm_api_usage.py` | OpenAI / HuggingFace API patterns · structured outputs · streaming · rate limiting · cost estimation · error handling | 35 min | Intermediate |
| `algorithms/build_a_chatbot.py` | Conversation history management · system prompts · persona design · context window budgeting · memory strategies | 40 min | Intermediate |
| `algorithms/llm_evaluation.py` | BLEU · ROUGE · BERTScore · LLM-as-judge · human evaluation rubrics · hallucination detection · benchmark datasets | 40 min | Advanced |

### Projects

| File | Task | Time | Difficulty |
|------|------|------|------------|
| `projects/qa_system_with_rag.py` | End-to-end Q&A system: 20-doc ML knowledge base · TF-IDF retrieval with topic filtering · query preprocessing · intent classification · P@1/P@3/MRR evaluation | 60 min | Advanced |
| `projects/llm_powered_classifier.py` | Zero-shot and few-shot text classification across 5 classes · template-based similarity · Jaccard k-NN few-shot · comparison vs supervised TF-IDF+LR baseline | 60 min | Advanced |

---

## Concept Cheat Sheet

| Concept | Formula / Key Idea |
|---------|--------------------|
| Next-token prediction loss | `L = -∑ log P(x_t | x_<t)` — cross-entropy over the vocabulary at each position |
| Perplexity | `PPL = exp(L)` — lower is better; measures how "surprised" the model is |
| Scaling laws (Chinchilla) | Optimal tokens ≈ 20 × parameters; both model size and data matter equally |
| LoRA formula | `W' = W + ΔW = W + BA` where B ∈ R^(d×r), A ∈ R^(r×k), rank r ≪ min(d,k) |
| LoRA parameter savings | Trainable params: r(d+k) vs d×k full; at r=8, d=k=4096 → 99.6% reduction |
| RAG pipeline steps | Chunk docs → embed chunks → index (FAISS/BM25) → retrieve top-K → inject context → generate |
| Zero-shot | Classify / answer using only a natural-language task description — no examples |
| Few-shot | Provide 2–10 labeled examples in the prompt context; no gradient updates |
| Fine-tuning | Update model weights end-to-end on task data; needs labeled data, GPU time |
| Temperature sampling | Divide logits by T before softmax: T→0 = greedy, T→∞ = uniform random |
| Top-K sampling | Keep K most likely tokens, renormalize, sample; reduces incoherent tails |
| Top-P (nucleus) sampling | Keep smallest token set with cumulative probability ≥ P; adaptive cutoff |
| RLHF Stage 1 | Supervised fine-tuning (SFT) on human-written demonstrations |
| RLHF Stage 2 | Train reward model on human preference rankings of model outputs |
| RLHF Stage 3 | PPO (or DPO) updates the policy to maximize reward model score |
| Retrieval P@K | Fraction of top-K retrieved docs that are relevant |
| MRR | `MRR = (1/|Q|) ∑ 1/rank_i` — mean reciprocal rank of the first relevant doc |

---

## Prerequisites

- **Part 3 (DNNs)** — backpropagation, activation functions, gradient descent
- **Part 4 (CNNs)** — optional; useful for understanding feature extraction
- **Part 5 (NLP)** — tokenization, TF-IDF, word embeddings, sequence modeling
- **Part 6 (Transformers)** — attention, BERT, GPT architecture, autoregressive generation
- **Python** — numpy, matplotlib, scikit-learn

---

## Installation

```bash
# Base dependencies (already installed from Parts 1–5)
pip install numpy pandas matplotlib seaborn scikit-learn

# For HuggingFace models (optional — all modules degrade gracefully)
pip install transformers torch

# For parameter-efficient fine-tuning
pip install peft

# For vector search (optional, used in RAG pipeline)
pip install faiss-cpu
```

All modules work without `transformers`, `torch`, or `peft` installed.
They simulate model outputs, clearly labelling results as simulated, and
print install instructions when optional packages are missing.

---

## Visualization Index

Every module auto-generates 3 PNG visualizations (300 dpi) in `visuals/`:

| Directory | Visualizations |
|-----------|---------------|
| `visuals/01_how_llms_work/` | token probability distributions · scaling law curves (loss vs compute) · emergent capability threshold diagram |
| `visuals/02_prompt_engineering/` | prompt template anatomy · few-shot context window layout · chain-of-thought reasoning trace |
| `visuals/03_fine_tuning_basics/` | LoRA low-rank decomposition diagram · trainable parameter comparison · loss curves (full vs LoRA fine-tuning) |
| `visuals/04_rag_pipeline/` | RAG architecture flowchart · chunk-to-embedding space scatter · retrieval precision vs chunk-size |
| `visuals/llm_api_usage/` | API call lifecycle diagram · token cost breakdown · streaming latency analysis |
| `visuals/build_a_chatbot/` | conversation history management diagram · context window budget · memory strategy comparison |
| `visuals/llm_evaluation/` | BLEU/ROUGE score distributions · hallucination rate by prompt type · LLM-as-judge calibration |
| `visuals/qa_system_with_rag/` | full system architecture diagram · per-topic P@1/P@3/MRR bar chart + confidence histogram · KB topic-centroid similarity heatmap |
| `visuals/llm_powered_classifier/` | three-approach comparison diagram · overall + per-class accuracy heatmap · confidence vs correctness scatter |

---

## How LLMs Fit into the Course

```
Part 1: Regression       →  linear models, gradient descent, loss functions
Part 2: Classification   →  logistic regression, decision boundaries, metrics
Part 3: DNNs             →  multi-layer networks, backpropagation, regularization
Part 4: CNNs             →  spatial feature extraction, convolution, pooling
Part 5: NLP              →  tokenization, TF-IDF, word embeddings, RNN/LSTM
Part 6: Transformers     →  attention, BERT, GPT, positional encoding
Part 7: LLMs ← YOU ARE HERE
  Math foundations:  how LLMs work, prompt engineering, fine-tuning, RAG
  Algorithms:        API usage, chatbot, evaluation
  Projects:          Q&A system with RAG, LLM-powered classifier
```

LLMs are the culmination of every concept taught in Parts 1–6:
- **Regression** foundations → cross-entropy loss, gradient descent
- **Classification** → token prediction as a 50,000-class classification task
- **DNNs** → deep residual transformer blocks
- **CNNs** → patch embeddings in vision transformers (ViT)
- **NLP** → tokenization, embeddings, sequential modeling
- **Transformers** → the core architecture powering every modern LLM

---

## Key Papers

| Paper | Year | What it introduced |
|-------|------|--------------------|
| *Language Models are Unsupervised Multitask Learners* (Radford et al.) | 2019 | GPT-2: decoder-only generation, zero-shot task transfer |
| *Language Models are Few-Shot Learners* (Brown et al.) | 2020 | GPT-3 (175B), in-context learning, few-shot prompting |
| *Training language models to follow instructions* (Ouyang et al.) | 2022 | InstructGPT: RLHF alignment, SFT + reward model + PPO |
| *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al.) | 2022 | LoRA: parameter-efficient fine-tuning with low-rank matrices |
| *Retrieval-Augmented Generation* (Lewis et al.) | 2020 | RAG: grounding generation in retrieved documents |
| *Training Compute-Optimal Large Language Models* (Hoffmann et al.) | 2022 | Chinchilla scaling laws: optimal data-to-parameter ratio |

---

*Part of the [MLForBeginners](../README.md) course — a visual-first ML curriculum from algebra to production.*
