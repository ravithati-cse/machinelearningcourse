"""
MLForBeginners — Mind Map Generator  (v2 — clean horizontal-tree layout)
=========================================================================
Produces 20 readable, non-overlapping PNG mind maps.

Layout strategy
---------------
  course_overview  → simple radial wheel (centre + 7 part spokes, no leaves)
  per-part maps    → horizontal tree: centre | branches | leaves
  module maps      → horizontal tree: centre | branches | leaves

The horizontal tree guarantees zero overlap:
  • each leaf gets LEAF_H vertical units
  • each branch is centred on its own leaf cluster
  • branches are separated by BRANCH_GAP
  • the centre node sits at the mean of all branch y-positions

Run from anywhere:
    python3 mind_maps/generate_mindmaps.py
"""

import os
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
os.makedirs(MODULES_DIR, exist_ok=True)
GENERATED   = []

# ── palette ───────────────────────────────────────────────────────────────────
PART_COLORS = [
    "#C0392B",   # 1 Regression        — brick red
    "#2980B9",   # 2 Classification     — steel blue
    "#27AE60",   # 3 DNNs               — forest green
    "#D35400",   # 4 CNNs               — burnt orange
    "#8E44AD",   # 5 NLP                — purple
    "#16A085",   # 6 Transformers       — teal
    "#E67E22",   # 7 LLMs               — amber
]

BG   = "#F4F6F8"   # light background
DARK = "#1C2833"   # near-black for centre node


# ── helpers ───────────────────────────────────────────────────────────────────
def _wrap(text, w=20):
    """Soft-wrap text at w characters."""
    return "\n".join(textwrap.wrap(str(text), width=w))


def _darken(hex_c, f=0.72):
    r, g, b = [int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(int(r*f), int(g*f), int(b*f))


def _lighten(hex_c, f=0.55):
    r, g, b = [int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)]
    return "#{:02x}{:02x}{:02x}".format(
        min(255, r + int((255-r)*f)),
        min(255, g + int((255-g)*f)),
        min(255, b + int((255-b)*f)),
    )


def _node(ax, x, y, label, bg, fs=10, bold=False, tc="white",
          alpha=1.0, zorder=5, wrap_w=20):
    """Draw a rounded-rect node centred at (x, y)."""
    ax.text(
        x, y, _wrap(label, wrap_w),
        fontsize=fs,
        color=tc,
        fontweight="bold" if bold else "normal",
        ha="center", va="center",
        zorder=zorder + 1,
        multialignment="center",
        linespacing=1.3,
        bbox=dict(
            boxstyle="round,pad=0.45",
            facecolor=bg,
            alpha=alpha,
            edgecolor="white",
            linewidth=1.5,
        ),
    )


def _curve(ax, x1, y1, x2, y2, col, lw=1.8, alpha=0.55):
    """Cubic S-bezier — horizontal tangents at both ends (good for tree layout)."""
    t   = np.linspace(0, 1, 60)
    mx  = (x1 + x2) / 2
    # control points: pull horizontally to create smooth S
    c1x, c1y = mx, y1
    c2x, c2y = mx, y2
    px = (1-t)**3*x1 + 3*(1-t)**2*t*c1x + 3*(1-t)*t**2*c2x + t**3*x2
    py = (1-t)**3*y1 + 3*(1-t)**2*t*c1y + 3*(1-t)*t**2*c2y + t**3*y2
    ax.plot(px, py, color=col, lw=lw, alpha=alpha, zorder=2,
            solid_capstyle="round")


# ── horizontal-tree layout engine ────────────────────────────────────────────
LEAF_H    = 1.15   # vertical space per leaf (units)
BRANCH_GAP = 0.65  # extra gap between branch groups


def _tree_layout(branches_dict):
    """
    Compute non-overlapping y-positions for branches and leaves.

    Returns
    -------
    center_y  : float
    rows      : list of (branch_name, branch_y, [(leaf_name, leaf_y), ...])
    total_h   : float   (total height consumed)
    """
    items   = list(branches_dict.items())
    n_b     = len(items)
    n_total = sum(len(v) for _, v in items)
    total_h = n_total * LEAF_H + max(0, n_b - 1) * BRANCH_GAP

    rows    = []
    y_top   = total_h / 2   # start from the very top

    for bname, leaves in items:
        n = len(leaves)
        lys = [y_top - (i + 0.5) * LEAF_H for i in range(n)]
        rows.append((bname, float(np.mean(lys)), list(zip(leaves, lys))))
        y_top -= n * LEAF_H + BRANCH_GAP

    cy = (rows[0][1] + rows[-1][1]) / 2
    return cy, rows, total_h


# ── x-column positions (data units) ──────────────────────────────────────────
CX  =  1.5    # centre node
BX  =  6.5    # branch nodes
LX  = 12.0    # leaf nodes


# ===========================================================================
# 1.  Course overview  (radial wheel — clean, just 7 spokes)
# ===========================================================================

PARTS_OVERVIEW = [
    ("Part 1\nRegression",       PART_COLORS[0], "Algebra → Linear Model → Projects"),
    ("Part 2\nClassification",   PART_COLORS[1], "Sigmoid → Trees → Forests → Projects"),
    ("Part 3\nDeep Neural\nNets",PART_COLORS[2], "Neurons → Backprop → Keras → MNIST"),
    ("Part 4\nCNNs",             PART_COLORS[3], "Conv → Pooling → ResNet → Transfer"),
    ("Part 5\nNLP",              PART_COLORS[4], "BoW → Embeddings → LSTM → NER"),
    ("Part 6\nTransformers",     PART_COLORS[5], "Attention → BERT → GPT → Projects"),
    ("Part 7\nLLMs",             PART_COLORS[6], "Scaling → LoRA → RAG → Classifier"),
]


def make_course_overview():
    fig, ax = plt.subplots(figsize=(22, 22))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.set_aspect("equal"); ax.axis("off")

    fig.suptitle("MLForBeginners — Full Course Overview",
                 fontsize=24, fontweight="bold", color=DARK, y=0.97)

    R_part = 6.0   # radius to part node
    R_sub  = 9.0   # radius to subtitle node
    cx, cy = 0, 0

    # Centre
    _node(ax, cx, cy, "Machine\nLearning\nfor Beginners",
          DARK, fs=17, bold=True, tc="white", zorder=8, wrap_w=18)

    n = len(PARTS_OVERVIEW)
    for i, (label, col, subtitle) in enumerate(PARTS_OVERVIEW):
        theta = np.pi/2 - 2*np.pi * i / n   # clockwise from top

        # spoke: centre → part
        px, py = R_part * np.cos(theta), R_part * np.sin(theta)
        t  = np.linspace(0, 1, 60)
        sx_c = cx + (px - cx)*0.45; sy_c = cy + (py - cy)*0.45
        bx_c = cx + (px - cx)*0.55; by_c = cy + (py - cy)*0.55
        spx  = (1-t)**3*cx + 3*(1-t)**2*t*sx_c + 3*(1-t)*t**2*bx_c + t**3*px
        spy  = (1-t)**3*cy + 3*(1-t)**2*t*sy_c + 3*(1-t)*t**2*by_c + t**3*py
        ax.plot(spx, spy, color=col, lw=2.8, alpha=0.65, zorder=2)

        # part node
        _node(ax, px, py, label, col, fs=13, bold=True, tc="white",
              zorder=7, wrap_w=15)

        # subtitle  (further out, same angle)
        qx = R_sub * np.cos(theta)
        qy = R_sub * np.sin(theta)
        t2 = np.linspace(0, 1, 40)
        spx2 = (1-t2)*px + t2*qx
        spy2 = (1-t2)*py + t2*qy
        ax.plot(spx2, spy2, color=col, lw=1.2, alpha=0.40, zorder=2)
        _node(ax, qx, qy, subtitle, _lighten(col, 0.35), fs=9.5,
              tc="white", zorder=6, wrap_w=22, alpha=0.90)

    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12)
    out = os.path.join(BASE_DIR, "course_overview.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig); GENERATED.append(out)
    print(f"  [OK] course_overview.png")


# ===========================================================================
# 2.  Per-part mind maps  (horizontal tree layout)
# ===========================================================================

PART_DATA = [
    # ── Part 1 ──
    {
        "title":     "Part 1: Regression",
        "filename":  "part1_regression.png",
        "color_idx": 0,
        "center":    "Regression",
        "branches": {
            "Math Foundations": [
                "Algebra Basics",
                "Statistics",
                "Calculus Basics",
                "Probability",
                "Linear Algebra",
            ],
            "Algorithms": [
                "Linear Regression",
                "Multiple Regression",
            ],
            "Examples & Projects": [
                "Data Exploration",
                "Simple Examples",
                "Model Evaluation",
                "Housing Analysis",
                "House Price\nPrediction",
            ],
        },
    },
    # ── Part 2 ──
    {
        "title":     "Part 2: Classification",
        "filename":  "part2_classification.png",
        "color_idx": 1,
        "center":    "Classification",
        "branches": {
            "Math Foundations": [
                "Sigmoid Function",
                "Probability Basics",
                "Decision Boundaries",
                "KNN Distance Metrics",
                "Ensemble Methods",
            ],
            "Algorithms": [
                "Logistic Regression",
                "K-Nearest Neighbours",
                "Decision Trees",
                "Random Forests",
                "Evaluation Metrics",
            ],
            "Projects": [
                "Spam Classifier",
                "Churn Prediction",
                "Model Comparison",
            ],
        },
    },
    # ── Part 3 ──
    {
        "title":     "Part 3: Deep Neural Networks",
        "filename":  "part3_dnns.png",
        "color_idx": 2,
        "center":    "Deep Neural\nNetworks",
        "branches": {
            "Math Foundations": [
                "Neurons & Activations",
                "Forward Propagation",
                "Backpropagation",
                "Loss & Optimizers",
                "Regularization",
            ],
            "Algorithms": [
                "Perceptron from Scratch",
                "MLP from Scratch",
                "MLP with Keras",
                "HP Tuning",
            ],
            "Projects": [
                "MNIST Classifier",
                "Tabular Deep Learning",
            ],
        },
    },
    # ── Part 4 ──
    {
        "title":     "Part 4: Convolutional Neural Networks",
        "filename":  "part4_cnns.png",
        "color_idx": 3,
        "center":    "CNNs",
        "branches": {
            "Math Foundations": [
                "Image Basics",
                "Convolution Operation",
                "Pooling & Depth",
            ],
            "Algorithms": [
                "Conv Layer from Scratch",
                "CNN with Keras",
                "Classic Architectures",
                "Transfer Learning",
            ],
            "Projects": [
                "CIFAR-10 Classifier",
                "Custom Image Classifier",
            ],
        },
    },
    # ── Part 5 ──
    {
        "title":     "Part 5: Natural Language Processing",
        "filename":  "part5_nlp.png",
        "color_idx": 4,
        "center":    "NLP",
        "branches": {
            "Math Foundations": [
                "Text Processing",
                "BoW & TF-IDF",
                "Word Embeddings",
                "RNN Intuition",
            ],
            "Algorithms": [
                "Text Classification\nPipeline",
                "Sentiment Analysis",
                "LSTM Classifier",
                "Named Entity\nRecognition",
            ],
            "Projects": [
                "Movie Review Sentiment",
                "News Article Classifier",
            ],
        },
    },
    # ── Part 6 ──
    {
        "title":     "Part 6: Transformers",
        "filename":  "part6_transformers.png",
        "color_idx": 5,
        "center":    "Transformers",
        "branches": {
            "Math Foundations": [
                "Attention Mechanism",
                "Multi-Head Attention",
                "Positional Encoding",
                "Encoder-Decoder Arch",
            ],
            "Algorithms": [
                "Transformer from Scratch",
                "BERT Encoder",
                "GPT Decoder",
            ],
            "Projects": [
                "BERT Text Classifier",
                "GPT-2 Text Generator",
            ],
        },
    },
    # ── Part 7 ──
    {
        "title":     "Part 7: Large Language Models",
        "filename":  "part7_llms.png",
        "color_idx": 6,
        "center":    "LLMs",
        "branches": {
            "Math Foundations": [
                "How LLMs Work",
                "Prompt Engineering",
                "Fine-Tuning & LoRA",
                "Retrieval-Augmented\nGeneration",
            ],
            "Algorithms": [
                "LLM from Scratch\n(MiniGPT in NumPy)",
                "LoRA Fine-Tuning",
                "RAG Pipeline",
            ],
            "Projects": [
                "Q&A System with RAG",
                "LLM-Powered Classifier",
            ],
        },
    },
]


def make_part_mindmap(part: dict):
    color  = PART_COLORS[part["color_idx"]]
    dark_b = _darken(color, 0.72)     # branch node background
    light_l= _lighten(color, 0.30)    # leaf node background (slightly lighter)

    cy, rows, total_h = _tree_layout(part["branches"])

    fig_h = max(9, total_h + 2.5)
    fig, ax = plt.subplots(figsize=(24, fig_h))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.axis("off")

    fig.suptitle(f"MLForBeginners — {part['title']}",
                 fontsize=17, fontweight="bold", color=DARK, y=0.99)

    # centre node
    _node(ax, CX, cy, part["center"], color,
          fs=16, bold=True, tc="white", zorder=8, wrap_w=16)

    for bname, by, leaves in rows:
        # centre → branch
        _curve(ax, CX, cy, BX, by, color, lw=2.2, alpha=0.60)
        # branch node
        _node(ax, BX, by, bname, dark_b,
              fs=12, bold=True, tc="white", zorder=7, wrap_w=18)

        for lname, ly in leaves:
            # branch → leaf
            _curve(ax, BX, by, LX, ly, color, lw=1.3, alpha=0.40)
            # leaf node
            _node(ax, LX, ly, lname, light_l,
                  fs=10, tc="white", zorder=6, wrap_w=22, alpha=0.92)

    margin = 1.2
    ax.set_xlim(0, LX + 3.5)
    ax.set_ylim(cy - total_h/2 - margin, cy + total_h/2 + margin)

    out = os.path.join(BASE_DIR, part["filename"])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig); GENERATED.append(out)
    print(f"  [OK] {part['filename']}")


# ===========================================================================
# 3.  Per-module concept maps  (horizontal tree layout)
# ===========================================================================

MODULE_MAPS = [
    # ── Attention Mechanism ──
    {
        "title":    "Attention Mechanism",
        "filename": "modules/attention_mechanism.png",
        "center":   "Attention\nMechanism",
        "color":    PART_COLORS[5],
        "branches": {
            "Core Formula": [
                "Attn(Q,K,V) = softmax(QKᵀ/√dₖ)·V",
                "Scale factor: 1/√dₖ",
                "Softmax → attention weights",
                "Weighted sum of Values",
            ],
            "Q / K / V Roles": [
                "Query — what am I looking for?",
                "Key   — what do I contain?",
                "Value — what do I return?",
                "All learned via linear projection",
            ],
            "Causal Masking": [
                "Upper-triangle set to −∞",
                "Prevents attending to future",
                "Used in decoder (GPT-style)",
                "Enables auto-regressive generation",
            ],
            "Bahdanau vs Dot-Product": [
                "Bahdanau: additive, RNN-based",
                "Dot-product: multiplicative, parallel",
                "Vaswani (2017): scaled dot-product",
                "Dot-product scales better",
            ],
        },
    },
    # ── Transformer Architecture ──
    {
        "title":    "Transformer Architecture",
        "filename": "modules/transformer_architecture.png",
        "center":   "Transformer\nArchitecture",
        "color":    PART_COLORS[5],
        "branches": {
            "Encoder Stack": [
                "Multi-Head Self-Attention",
                "Add & LayerNorm",
                "Position-wise FFN",
                "Add & LayerNorm",
            ],
            "Decoder Stack": [
                "Masked Multi-Head Self-Attn",
                "Cross-Attention (enc→dec)",
                "Position-wise FFN",
                "Add & LayerNorm at each step",
            ],
            "Positional Encoding": [
                "Sine & cosine of position",
                "PE(pos,2i)   = sin(pos/10000^(2i/d))",
                "PE(pos,2i+1) = cos(…)",
                "Added to token embeddings",
            ],
            "Feed-Forward Network": [
                "Linear → GELU/ReLU → Linear",
                "d_model → 4×d_model → d_model",
                "Applied position-wise",
                "Independent per token",
            ],
        },
    },
    # ── BERT vs GPT ──
    {
        "title":    "BERT vs GPT",
        "filename": "modules/bert_vs_gpt.png",
        "center":   "BERT vs GPT",
        "color":    "#8E44AD",
        "branches": {
            "BERT": [
                "Encoder-only architecture",
                "Bidirectional context",
                "Masked Language Modelling",
                "CLS token → classification",
                "Best for: classify, NER, QA",
            ],
            "GPT": [
                "Decoder-only architecture",
                "Causal (left-to-right) context",
                "Next-token prediction",
                "Autoregressive generation",
                "Best for: text generation",
            ],
            "Pre-training": [
                "BERT: Masked LM + Next-Sentence",
                "GPT: Causal LM on web text",
                "Both: large unlabelled corpora",
                "Both: transformer backbone",
            ],
            "Fine-tuning": [
                "BERT: add task head on CLS",
                "GPT: prompt-based task framing",
                "Both: few labelled examples needed",
                "LoRA / PEFT for efficient tuning",
            ],
        },
    },
    # ── Neural Network Concepts ──
    {
        "title":    "Neural Network Concepts",
        "filename": "modules/neural_network_concepts.png",
        "center":   "Neural Network\nConcepts",
        "color":    PART_COLORS[2],
        "branches": {
            "Neurons & Activations": [
                "Weighted sum: z = Wx + b",
                "ReLU: max(0, z)",
                "Sigmoid: 1 / (1 + e^−z)",
                "Tanh: (e^z − e^−z) / (e^z + e^−z)",
            ],
            "Forward Pass": [
                "Input → Hidden layers → Output",
                "Matrix multiply at each layer",
                "Apply activation function",
                "Produce prediction ŷ",
            ],
            "Backpropagation": [
                "Compute loss L",
                "∂L/∂W via chain rule",
                "Gradient flows backwards",
                "W ← W − α · ∇W",
            ],
            "Loss & Optimizers": [
                "MSE (regression)",
                "Cross-entropy (classification)",
                "SGD / Adam / RMSProp",
                "Learning rate & mini-batches",
            ],
            "Regularization": [
                "L1 / L2 weight decay",
                "Dropout (randomly zero neurons)",
                "Batch Normalisation",
                "Early Stopping",
            ],
        },
    },
    # ── CNN Concepts ──
    {
        "title":    "CNN Concepts",
        "filename": "modules/cnn_concepts.png",
        "center":   "CNN Concepts",
        "color":    PART_COLORS[3],
        "branches": {
            "Convolution": [
                "Filter slides over input",
                "Element-wise multiply & sum",
                "Output = feature map",
                "Learns edges, textures, shapes",
            ],
            "Filters & Feature Maps": [
                "Each filter → one feature map",
                "Depth = number of filters",
                "Shared weights (parameter efficient)",
                "Translation equivariant",
            ],
            "Pooling": [
                "Max pooling: take the maximum",
                "Average pooling: take the mean",
                "Reduces spatial dimensions",
                "Adds translation invariance",
            ],
            "Stride & Padding": [
                "Stride: step size of filter",
                "Padding SAME: output = input size",
                "Padding VALID: output shrinks",
                "Output size: ⌊(n + 2p − f)/s⌋ + 1",
            ],
        },
    },
    # ── NLP Concepts ──
    {
        "title":    "NLP Concepts",
        "filename": "modules/nlp_concepts.png",
        "center":   "NLP Concepts",
        "color":    PART_COLORS[4],
        "branches": {
            "Tokenisation": [
                "Word-level (simple, large vocab)",
                "Subword — BPE (GPT-style)",
                "Subword — WordPiece (BERT)",
                "Character-level (tiny vocab)",
            ],
            "TF-IDF": [
                "TF = term frequency in doc",
                "IDF = log(N / df)",
                "Score = TF × IDF",
                "Sparse, high-dim representation",
            ],
            "Word Embeddings": [
                "Dense vector per word",
                "Word2Vec / GloVe (static)",
                "BERT embeddings (contextual)",
                "Similarity via cosine distance",
            ],
            "RNN & LSTM": [
                "Hidden state h_t passed forward",
                "Vanishing gradient problem",
                "LSTM gates: input / forget / output",
                "Cell state = long-term memory",
            ],
        },
    },
    # ── LLM Concepts ── (NEW)
    {
        "title":    "LLM Concepts",
        "filename": "modules/llm_concepts.png",
        "center":   "LLM\nConcepts",
        "color":    PART_COLORS[6],
        "branches": {
            "How LLMs Work": [
                "Next-token prediction objective",
                "Transformer (decoder-only) backbone",
                "BPE tokenisation",
                "Chinchilla: tokens ≈ 20 × params",
                "Emergent abilities above ~1B params",
            ],
            "Prompt Engineering": [
                "Zero-shot: task in prompt only",
                "Few-shot: examples in prompt",
                "Chain-of-Thought: show reasoning",
                "System vs user vs assistant roles",
                "Temperature controls randomness",
            ],
            "Fine-Tuning": [
                "SFT: cross-entropy on (prompt, reply)",
                "LoRA: ΔW = (α/r) · B · A",
                "Only A & B trained, W frozen",
                "RLHF: reward model + PPO",
                "Instruction tuning for alignment",
            ],
            "RAG Pipeline": [
                "Chunk documents → embed → index",
                "Query → retrieve top-k docs",
                "Augment prompt with context",
                "LLM generates grounded answer",
                "Eval: P@k, MRR, faithfulness",
            ],
        },
    },
    # ── LoRA & Fine-Tuning ── (NEW)
    {
        "title":    "LoRA & Fine-Tuning",
        "filename": "modules/lora_and_finetuning.png",
        "center":   "LoRA &\nFine-Tuning",
        "color":    PART_COLORS[6],
        "branches": {
            "Full Fine-Tuning": [
                "Update all model weights",
                "High GPU memory cost",
                "Risk of catastrophic forgetting",
                "Best accuracy on target task",
            ],
            "LoRA (Low-Rank Adaptation)": [
                "Freeze pre-trained W",
                "Add ΔW = B · A  (rank r ≪ d)",
                "B initialised to 0 → ΔW starts at 0",
                "Scale: α/r applied to ΔW",
                "Merge at inference: W' = W + ΔW",
            ],
            "RLHF Pipeline": [
                "Stage 1: SFT on demonstrations",
                "Stage 2: Train reward model",
                "Stage 3: PPO optimisation",
                "KL penalty: β · KL(π_RL ‖ π_SFT)",
                "Prevents reward hacking",
            ],
            "Choosing a Strategy": [
                "Tiny data → prompt engineering",
                "< 10 K examples → LoRA / QLoRA",
                "Large domain shift → full fine-tune",
                "Need safety / alignment → RLHF",
            ],
        },
    },
    # ── RAG Pipeline ── (NEW)
    {
        "title":    "Retrieval-Augmented Generation",
        "filename": "modules/rag_pipeline.png",
        "center":   "RAG\nPipeline",
        "color":    PART_COLORS[6],
        "branches": {
            "Indexing (Offline)": [
                "Load & clean documents",
                "Chunk: fixed / sentence / paragraph",
                "Embed chunks → dense vectors",
                "Store in vector DB (FAISS, Chroma)",
            ],
            "Retrieval (Online)": [
                "Embed the user query",
                "Cosine-similarity search",
                "Return top-k chunks",
                "Optional: re-rank with cross-encoder",
            ],
            "Augmented Generation": [
                "Build prompt: context + question",
                "LLM generates grounded answer",
                "Citation: reference source chunks",
                "Hallucination ↓ vs pure LLM",
            ],
            "Evaluation Metrics": [
                "P@1 / P@3: precision at k",
                "MRR: mean reciprocal rank",
                "Faithfulness: answer in context?",
                "Answer relevance (RAGAs framework)",
            ],
        },
    },
]


def make_module_mindmap(mod: dict):
    color  = mod["color"]
    dark_b = _darken(color, 0.72)
    light_l= _lighten(color, 0.30)

    cy, rows, total_h = _tree_layout(mod["branches"])

    fig_h = max(9, total_h + 2.5)
    fig, ax = plt.subplots(figsize=(26, fig_h))
    ax.set_facecolor(BG); fig.patch.set_facecolor(BG)
    ax.axis("off")

    fig.suptitle(f"MLForBeginners — {mod['title']}",
                 fontsize=17, fontweight="bold", color=DARK, y=0.99)

    # centre node
    _node(ax, CX, cy, mod["center"], color,
          fs=15, bold=True, tc="white", zorder=8, wrap_w=16)

    for bname, by, leaves in rows:
        _curve(ax, CX, cy, BX, by, color, lw=2.2, alpha=0.60)
        _node(ax, BX, by, bname, dark_b,
              fs=12, bold=True, tc="white", zorder=7, wrap_w=20)

        for lname, ly in leaves:
            _curve(ax, BX, by, LX, ly, color, lw=1.3, alpha=0.40)
            _node(ax, LX, ly, lname, light_l,
                  fs=9.5, tc="white", zorder=6, wrap_w=28, alpha=0.92)

    margin = 1.2
    ax.set_xlim(0, LX + 4.5)
    ax.set_ylim(cy - total_h/2 - margin, cy + total_h/2 + margin)

    out = os.path.join(BASE_DIR, mod["filename"])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig); GENERATED.append(out)
    print(f"  [OK] {mod['filename']}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n=== MLForBeginners Mind Map Generator (v2 — clean layout) ===\n")

    print("Generating course overview …")
    make_course_overview()

    print("\nGenerating per-part mind maps …")
    for part in PART_DATA:
        make_part_mindmap(part)

    print("\nGenerating per-module concept maps …")
    for mod in MODULE_MAPS:
        make_module_mindmap(mod)

    print(f"\n{'='*60}")
    print(f"Done!  Generated {len(GENERATED)} mind map(s):")
    for f in GENERATED:
        rel     = os.path.relpath(f, BASE_DIR)
        size_kb = os.path.getsize(f) // 1024
        print(f"  {rel:55s}  {size_kb:>5} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
