"""
MLForBeginners — Mind Map Generator
====================================
Generates beautiful radial and hierarchical mind map PNG visualizations
for the entire MLForBeginners course using only matplotlib and numpy.

Outputs:
  mind_maps/course_overview.png
  mind_maps/part1_regression.png  …  part7_llms.png
  mind_maps/modules/attention_mechanism.png
  mind_maps/modules/transformer_architecture.png
  mind_maps/modules/bert_vs_gpt.png
  mind_maps/modules/neural_network_concepts.png
  mind_maps/modules/cnn_concepts.png
  mind_maps/modules/nlp_concepts.png
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
os.makedirs(MODULES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palette — one colour per part (7 parts)
# ---------------------------------------------------------------------------
PART_COLORS = [
    "#E74C3C",  # Part 1  Regression        – red
    "#E67E22",  # Part 2  Classification     – orange
    "#F39C12",  # Part 3  DNNs               – amber
    "#27AE60",  # Part 4  CNNs               – green
    "#3498DB",  # Part 5  NLP                – blue
    "#9B59B6",  # Part 6  Transformers       – purple
    "#1ABC9C",  # Part 7  LLMs               – teal
]

GENERATED = []   # track successfully written files

# ===========================================================================
# Low-level drawing helpers
# ===========================================================================

def _hex_to_rgba(hex_color: str, alpha: float = 1.0):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, alpha)


def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Return a lighter version of *hex_color* for node backgrounds."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


def draw_fancy_node(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    fontsize: float = 10,
    width: float = None,
    height: float = 0.45,
    alpha: float = 0.92,
    zorder: int = 4,
    text_color: str = "white",
    bold: bool = False,
    style: str = "round,pad=0.15",
):
    """Draw a FancyBboxPatch centred at (x, y) with *text* inside."""
    lines = text.split("\n")
    max_chars = max(len(ln) for ln in lines)
    n_lines = len(lines)

    if width is None:
        width = max(max_chars * fontsize * 0.013 + 0.25, 0.8)

    height = max(height, n_lines * fontsize * 0.018 + 0.20)

    bg = _lighten(color, 0.30) if text_color == "white" else _lighten(color, 0.78)

    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=style,
        facecolor=bg,
        edgecolor=color,
        linewidth=1.6,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color if text_color != "white" else "white",
        fontweight=weight,
        zorder=zorder + 1,
        wrap=False,
        clip_on=True,
    )
    return (x - width / 2, y - height / 2, width, height)


def draw_bezier(ax, x0, y0, x1, y1, color, lw=1.5, alpha=0.7, zorder=2):
    """Draw a smooth quadratic Bézier curve from (x0,y0) to (x1,y1)."""
    mx = (x0 + x1) / 2
    my = (y0 + y1) / 2

    # Control point: perpendicular nudge proportional to distance
    dx, dy = x1 - x0, y1 - y0
    dist = max(np.hypot(dx, dy), 0.01)
    perp_x = -dy / dist * dist * 0.18
    perp_y = dx / dist * dist * 0.18

    cx = mx + perp_x
    cy = my + perp_y

    t = np.linspace(0, 1, 60)
    bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x1
    by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y1

    ax.plot(bx, by, color=color, lw=lw, alpha=alpha, zorder=zorder, solid_capstyle="round")


def draw_straight(ax, x0, y0, x1, y1, color, lw=1.2, alpha=0.6, zorder=2):
    ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha, zorder=zorder)


# ===========================================================================
# Radial (polar) layout helpers
# ===========================================================================

def radial_positions(n: int, radius: float, angle_offset: float = 0.0):
    """Return list of (x, y) around a circle of *radius*, evenly spaced."""
    angles = [angle_offset + 2 * np.pi * i / n for i in range(n)]
    return [(radius * np.cos(a), radius * np.sin(a)) for a in angles], angles


def sub_positions(parent_x, parent_y, parent_angle, n_children, child_radius,
                  spread: float = np.pi / 3):
    """Fan *n_children* nodes around *parent_angle* at distance *child_radius*."""
    if n_children == 1:
        angles = [parent_angle]
    else:
        half = spread / 2
        angles = [parent_angle - half + spread * i / (n_children - 1)
                  for i in range(n_children)]
    positions = []
    for a in angles:
        px = parent_x + child_radius * np.cos(a)
        py = parent_y + child_radius * np.sin(a)
        positions.append((px, py, a))
    return positions


# ===========================================================================
# 1.  Full-course overview mind map
# ===========================================================================

PARTS = [
    {
        "label": "Part 1\nRegression",
        "sub": ["Linear Algebra", "Linear Regression", "Multiple Regression", "Model Evaluation"],
    },
    {
        "label": "Part 2\nClassification",
        "sub": ["Sigmoid & Prob.", "Logistic Regression", "KNN & Decision Trees", "Random Forests"],
    },
    {
        "label": "Part 3\nDeep Neural Nets",
        "sub": ["Neurons & Activations", "Backpropagation", "MLP with Keras", "HP Tuning"],
    },
    {
        "label": "Part 4\nCNNs",
        "sub": ["Convolution Op.", "Pooling & Depth", "Classic Archs", "Transfer Learning"],
    },
    {
        "label": "Part 5\nNLP",
        "sub": ["Text Processing", "Word Embeddings", "LSTM Classifier", "NER"],
    },
    {
        "label": "Part 6\nTransformers",
        "sub": ["Attention Mech.", "Multi-Head Attn", "BERT Encoder", "GPT Decoder"],
    },
    {
        "label": "Part 7\nLLMs",
        "sub": ["How LLMs Work", "Prompt Engineering", "Fine-Tuning / LoRA", "RAG Pipeline"],
    },
]


def make_course_overview():
    fig, ax = plt.subplots(figsize=(20, 16))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    fig.suptitle(
        "MLForBeginners — Full Course Mind Map",
        fontsize=22,
        fontweight="bold",
        color="#2C3E50",
        y=0.97,
    )

    # --- center node ---
    cx, cy = 0.0, 0.0
    draw_fancy_node(
        ax, cx, cy,
        "Machine Learning\nfor Beginners",
        "#2C3E50",
        fontsize=16,
        width=2.2,
        height=0.80,
        bold=True,
        text_color="white",
        style="round,pad=0.25",
        zorder=6,
    )

    n_parts = len(PARTS)
    main_r = 3.8          # radius for part nodes
    sub_r  = 2.6          # additional radius for sub-nodes

    positions, angles = radial_positions(n_parts, main_r, angle_offset=np.pi / 2)

    for i, (part, (px, py), ang) in enumerate(zip(PARTS, positions, angles)):
        color = PART_COLORS[i]

        # spoke from center to branch node
        draw_bezier(ax, cx, cy, px, py, color, lw=2.2, alpha=0.55)

        # branch node
        draw_fancy_node(
            ax, px, py, part["label"], color,
            fontsize=12, width=1.9, height=0.70,
            bold=True, text_color="white", zorder=5,
        )

        # sub-nodes
        sub_pos = sub_positions(px, py, ang, len(part["sub"]),
                                child_radius=sub_r, spread=np.pi / 2.2)
        for label, (sx, sy, sang) in zip(part["sub"], sub_pos):
            draw_bezier(ax, px, py, sx, sy, color, lw=1.3, alpha=0.50)
            draw_fancy_node(
                ax, sx, sy, label, color,
                fontsize=9.5, width=1.65, height=0.42,
                text_color="white", alpha=0.88, zorder=4,
            )

    # axis limits
    ax.set_xlim(-9, 9)
    ax.set_ylim(-8, 8)

    out = os.path.join(BASE_DIR, "course_overview.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    GENERATED.append(out)
    print(f"  [OK] {out}")


# ===========================================================================
# 2.  Per-part mind maps
# ===========================================================================

PART_DATA = [
    # Part 1 – Regression
    {
        "title": "Part 1: Regression",
        "filename": "part1_regression.png",
        "color_idx": 0,
        "center": "Regression",
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
            "Projects": [
                "Housing Analysis",
                "House Price Prediction",
            ],
        },
        "concepts": {
            "Linear Regression": ["slope & intercept", "MSE loss", "gradient descent"],
            "Multiple Regression": ["feature matrix", "normal equation", "multicollinearity"],
            "Housing Analysis": ["EDA", "feature engineering", "data viz"],
            "House Price Prediction": ["model pipeline", "cross-validation", "R²"],
        },
    },
    # Part 2 – Classification
    {
        "title": "Part 2: Classification",
        "filename": "part2_classification.png",
        "color_idx": 1,
        "center": "Classification",
        "branches": {
            "Math Foundations": [
                "Sigmoid Function",
                "Probability",
                "Decision Boundaries",
                "KNN Distance",
                "Ensemble Methods",
            ],
            "Algorithms": [
                "Logistic Regression",
                "KNN",
                "Decision Trees",
                "Random Forests",
            ],
            "Projects": [
                "Spam Classifier",
                "Churn Prediction",
                "Model Comparison",
            ],
        },
        "concepts": {
            "Logistic Regression": ["binary cross-entropy", "log-odds", "threshold"],
            "KNN": ["Euclidean dist.", "k-selection", "curse of dim."],
            "Decision Trees": ["Gini impurity", "information gain", "pruning"],
            "Random Forests": ["bagging", "feature importance", "OOB error"],
        },
    },
    # Part 3 – DNNs
    {
        "title": "Part 3: Deep Neural Networks",
        "filename": "part3_dnns.png",
        "color_idx": 2,
        "center": "Deep Neural\nNetworks",
        "branches": {
            "Math Foundations": [
                "Neurons & Activations",
                "Forward Propagation",
                "Backpropagation",
                "Loss & Optimizers",
                "Regularization",
            ],
            "Algorithms": [
                "Perceptron",
                "Multilayer Perceptron",
                "MLP with Keras",
                "HP Tuning",
            ],
            "Projects": [
                "MNIST Classifier",
                "Tabular Deep Learning",
            ],
        },
        "concepts": {
            "Neurons & Activations": ["ReLU", "sigmoid", "tanh"],
            "Backpropagation": ["chain rule", "gradients", "vanishing grad."],
            "Perceptron": ["step function", "weight update", "linearity limit"],
            "MLP with Keras": ["Dense layers", "compile/fit", "callbacks"],
        },
    },
    # Part 4 – CNNs
    {
        "title": "Part 4: Convolutional Neural Networks",
        "filename": "part4_cnns.png",
        "color_idx": 3,
        "center": "CNNs",
        "branches": {
            "Math Foundations": [
                "Image Basics",
                "Convolution Operation",
                "Pooling & Depth",
            ],
            "Algorithms": [
                "Conv from Scratch",
                "CNN with Keras",
                "Classic Architectures",
                "Transfer Learning",
            ],
            "Projects": [
                "CIFAR-10 Classifier",
                "Custom Image Classifier",
            ],
        },
        "concepts": {
            "Convolution Operation": ["filters/kernels", "stride", "padding"],
            "Pooling & Depth": ["max pooling", "feature maps", "receptive field"],
            "Classic Architectures": ["LeNet", "VGG", "ResNet"],
            "Transfer Learning": ["pre-trained weights", "fine-tuning", "feature extraction"],
        },
    },
    # Part 5 – NLP
    {
        "title": "Part 5: Natural Language Processing",
        "filename": "part5_nlp.png",
        "color_idx": 4,
        "center": "NLP",
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
                "NER",
            ],
            "Projects": [
                "Movie Review\nSentiment",
                "News Classifier",
            ],
        },
        "concepts": {
            "Text Processing": ["tokenization", "stemming", "stop words"],
            "Word Embeddings": ["Word2Vec", "GloVe", "cosine sim."],
            "LSTM Classifier": ["gates", "cell state", "seq-to-seq"],
            "Sentiment Analysis": ["polarity", "lexicon", "fine-grained"],
        },
    },
    # Part 6 – Transformers
    {
        "title": "Part 6: Transformers",
        "filename": "part6_transformers.png",
        "color_idx": 5,
        "center": "Transformers",
        "branches": {
            "Math Foundations": [
                "Attention Mechanism",
                "Multi-Head Attention",
                "Positional Encoding",
                "Encoder-Decoder\nArch",
            ],
            "Algorithms": [
                "Transformer\nfrom Scratch",
                "BERT Encoder",
                "GPT Decoder",
            ],
            "Projects": [
                "BERT Text Classifier",
                "GPT-2 Generator",
            ],
        },
        "concepts": {
            "Attention Mechanism": ["Q/K/V", "softmax(QKᵀ/√d)V", "context vector"],
            "Multi-Head Attention": ["parallel heads", "concat & project", "different subspaces"],
            "BERT Encoder": ["masked LM", "bidirectional", "CLS token"],
            "GPT Decoder": ["causal mask", "autoregressive", "next-token pred."],
        },
    },
    # Part 7 – LLMs
    {
        "title": "Part 7: Large Language Models",
        "filename": "part7_llms.png",
        "color_idx": 6,
        "center": "LLMs",
        "branches": {
            "Concepts": [
                "How LLMs Work",
                "Prompt Engineering",
                "Fine-Tuning Basics",
                "RAG",
            ],
            "Practical": [
                "Using LLM APIs",
                "Build a Chatbot",
                "LLM Evaluation",
            ],
            "Projects": [
                "Q&A System\nwith RAG",
                "LLM-Powered\nClassifier",
            ],
        },
        "concepts": {
            "How LLMs Work": ["pre-training", "scale laws", "emergent abilities"],
            "Prompt Engineering": ["zero-shot", "few-shot", "chain-of-thought"],
            "Fine-Tuning Basics": ["LoRA", "PEFT", "instruction tuning"],
            "RAG": ["retrieval", "vector DB", "generation"],
        },
    },
]


def _branch_angle_base(branch_idx, n_branches):
    """Spread branches horizontally: left third = Math, centre = Algos, right = Projects."""
    # We want: Math → left, Algorithms → right-top, Projects → right-bottom (roughly)
    # Generic: divide full circle evenly, starting at the left
    step = 2 * np.pi / n_branches
    base = np.pi - step * branch_idx
    return base


def make_part_mindmap(part: dict):
    color = PART_COLORS[part["color_idx"]]
    light = _lighten(color, 0.85)

    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    fig.suptitle(
        f"MLForBeginners — {part['title']}",
        fontsize=20,
        fontweight="bold",
        color="#2C3E50",
        y=0.97,
    )

    cx, cy = 0.0, 0.0

    # Center node
    draw_fancy_node(
        ax, cx, cy, part["center"], color,
        fontsize=16, width=2.0, height=0.80,
        bold=True, text_color="white",
        style="round,pad=0.25", zorder=6,
    )

    branches = part["branches"]
    n_branches = len(branches)
    branch_r = 3.5
    sub_r    = 2.4
    concept_r = 1.6

    # Branch angles: spread evenly
    b_angles, _ = radial_positions(n_branches, branch_r, angle_offset=np.pi / 2)
    b_angles_list = [np.pi / 2 + 2 * np.pi * i / n_branches for i in range(n_branches)]

    for b_idx, (branch_name, modules) in enumerate(branches.items()):
        bx, by = b_angles[b_idx]
        bang = b_angles_list[b_idx]

        # Spoke
        draw_bezier(ax, cx, cy, bx, by, color, lw=2.5, alpha=0.55)

        # Branch node (category)
        branch_colors = ["#2C3E50", "#34495E", "#5D6D7E"]
        b_color = branch_colors[b_idx % len(branch_colors)]
        draw_fancy_node(
            ax, bx, by, branch_name, b_color,
            fontsize=13, width=2.0, height=0.60,
            bold=True, text_color="white", zorder=5,
        )

        # Sub-nodes (module names)
        n_mods = len(modules)
        spread = min(np.pi * 0.7, np.pi * 0.15 * n_mods)
        sub_pos = sub_positions(bx, by, bang, n_mods,
                                child_radius=sub_r, spread=spread)

        for mod_label, (sx, sy, sang) in zip(modules, sub_pos):
            draw_bezier(ax, bx, by, sx, sy, color, lw=1.6, alpha=0.50)
            draw_fancy_node(
                ax, sx, sy, mod_label, color,
                fontsize=10, width=1.80, height=0.46,
                text_color="white", zorder=4,
            )

            # Concept leaf nodes (optional, only if defined)
            concepts = part.get("concepts", {}).get(mod_label, [])
            if concepts:
                n_c = len(concepts)
                spread_c = min(np.pi * 0.45, np.pi * 0.12 * n_c)
                c_pos = sub_positions(sx, sy, sang, n_c,
                                      child_radius=concept_r, spread=spread_c)
                for clabel, (cx2, cy2, _) in zip(concepts, c_pos):
                    draw_straight(ax, sx, sy, cx2, cy2, color, lw=1.0, alpha=0.40)
                    draw_fancy_node(
                        ax, cx2, cy2, clabel, color,
                        fontsize=8.5, width=1.35, height=0.36,
                        text_color="white", alpha=0.80, zorder=3,
                    )

    ax.set_xlim(-10, 10)
    ax.set_ylim(-9, 9)

    out = os.path.join(BASE_DIR, part["filename"])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    GENERATED.append(out)
    print(f"  [OK] {out}")


# ===========================================================================
# 3.  Per-module mind maps
# ===========================================================================

# Each module map is defined as a dict:
#   title, filename, center, color, branches: { branch_label: [leaf, ...] }

MODULE_MAPS = [
    # ---- Attention Mechanism ----
    {
        "title": "Attention Mechanism",
        "filename": "modules/attention_mechanism.png",
        "center": "Attention\nMechanism",
        "color": PART_COLORS[5],
        "branches": {
            "Core Formula": [
                "Attention(Q,K,V)",
                "= softmax(QKᵀ/√dₖ)V",
                "Scale: 1/√dₖ",
                "Softmax → weights",
            ],
            "Q / K / V": [
                "Query (what to find)",
                "Key   (what is there)",
                "Value (what to return)",
                "Learned projections",
            ],
            "Causal Masking": [
                "Upper-triangle = -∞",
                "Prevents future peek",
                "Used in decoder",
                "Auto-regressive gen.",
            ],
            "Bahdanau vs Vaswani": [
                "Bahdanau: additive",
                "Vaswani: dot-product",
                "Bahdanau: RNN-based",
                "Vaswani: parallel",
            ],
        },
    },
    # ---- Transformer Architecture ----
    {
        "title": "Transformer Architecture",
        "filename": "modules/transformer_architecture.png",
        "center": "Transformer\nArchitecture",
        "color": PART_COLORS[5],
        "branches": {
            "Encoder Stack": [
                "Multi-Head Attention",
                "Add & LayerNorm",
                "Feed-Forward Net",
                "Add & LayerNorm",
            ],
            "Decoder Stack": [
                "Masked Self-Attn",
                "Cross-Attention",
                "Feed-Forward Net",
                "Causal masking",
            ],
            "Residual Connections": [
                "x + Sublayer(x)",
                "Stable gradients",
                "Skip connections",
                "Enables depth",
            ],
            "FFN (Feed-Forward)": [
                "Linear → ReLU → Linear",
                "d_model → 4×d_model",
                "Position-wise",
                "Independent per token",
            ],
        },
    },
    # ---- BERT vs GPT ----
    {
        "title": "BERT vs GPT",
        "filename": "modules/bert_vs_gpt.png",
        "center": "BERT vs GPT",
        "color": "#8E44AD",
        "branches": {
            "BERT": [
                "Encoder-only",
                "Bidirectional context",
                "Masked Language Model",
                "CLS token for class.",
                "Tasks: classify, NER",
            ],
            "GPT": [
                "Decoder-only",
                "Causal (left-to-right)",
                "Next-token prediction",
                "Auto-regressive gen.",
                "Tasks: generation",
            ],
            "Pre-training": [
                "BERT: Masked LM + NSP",
                "GPT: Causal LM only",
                "Both: large corpora",
                "Both: transformer base",
            ],
            "Fine-tuning": [
                "BERT: add head on CLS",
                "GPT: prompt-based",
                "Both: few labeled ex.",
                "LoRA / PEFT for LLMs",
            ],
        },
    },
    # ---- Neural Network Concepts ----
    {
        "title": "Neural Network Concepts",
        "filename": "modules/neural_network_concepts.png",
        "center": "Neural Network\nConcepts",
        "color": PART_COLORS[2],
        "branches": {
            "Neurons & Activations": [
                "Weighted sum + bias",
                "ReLU: max(0,x)",
                "Sigmoid: 1/(1+e⁻ˣ)",
                "Tanh: (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ)",
            ],
            "Forward Pass": [
                "Input → Hidden → Output",
                "Matrix multiply: Wx+b",
                "Apply activation",
                "Predict ŷ",
            ],
            "Backpropagation": [
                "Compute loss L",
                "∂L/∂W via chain rule",
                "Gradient flows back",
                "Update W ← W - α·∇W",
            ],
            "Loss & Optimizers": [
                "MSE / Cross-Entropy",
                "SGD / Adam / RMSProp",
                "Learning rate α",
                "Mini-batch training",
            ],
            "Regularization": [
                "L1 / L2 weight decay",
                "Dropout (mask neurons)",
                "Batch Normalization",
                "Early Stopping",
            ],
        },
    },
    # ---- CNN Concepts ----
    {
        "title": "CNN Concepts",
        "filename": "modules/cnn_concepts.png",
        "center": "CNN\nConcepts",
        "color": PART_COLORS[3],
        "branches": {
            "Convolution": [
                "Filter slides over input",
                "Element-wise multiply",
                "Sum → feature value",
                "Learns edge/pattern",
            ],
            "Filters & Feature Maps": [
                "Each filter → one map",
                "Depth = # filters",
                "Shared weights",
                "Translation equivariant",
            ],
            "Pooling": [
                "Max / Average pooling",
                "Reduces spatial size",
                "Adds invariance",
                "Reduces parameters",
            ],
            "Stride & Padding": [
                "Stride: step size",
                "Padding: SAME/VALID",
                "Output size formula",
                "Controls shrinkage",
            ],
            "Receptive Field": [
                "Input region a neuron sees",
                "Grows with depth",
                "Deeper → global context",
                "Atrous/dilated conv.",
            ],
        },
    },
    # ---- NLP Concepts ----
    {
        "title": "NLP Concepts",
        "filename": "modules/nlp_concepts.png",
        "center": "NLP\nConcepts",
        "color": PART_COLORS[4],
        "branches": {
            "Tokenization": [
                "Word / sub-word / char",
                "BPE (Byte-Pair Enc.)",
                "WordPiece (BERT)",
                "Vocab size trade-off",
            ],
            "TF-IDF": [
                "TF: term frequency",
                "IDF: inverse doc freq.",
                "TF × IDF = importance",
                "Sparse representation",
            ],
            "Embeddings": [
                "Dense vector for word",
                "Word2Vec / GloVe",
                "Contextual (BERT)",
                "Cosine similarity",
            ],
            "RNN & LSTM": [
                "Hidden state h_t",
                "Vanishing gradient",
                "LSTM: gates (i,f,o,g)",
                "Cell state long memory",
            ],
            "Attention in NLP": [
                "Alignment scores",
                "Context vector",
                "Replaces RNN bottleneck",
                "Transformer attention",
            ],
        },
    },
]


def make_module_mindmap(mod: dict):
    color = mod["color"]

    fig, ax = plt.subplots(figsize=(16, 13))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#F8F9FA")
    ax.set_facecolor("#F8F9FA")

    fig.suptitle(
        f"MLForBeginners — {mod['title']}",
        fontsize=19,
        fontweight="bold",
        color="#2C3E50",
        y=0.97,
    )

    cx, cy = 0.0, 0.0

    draw_fancy_node(
        ax, cx, cy, mod["center"], color,
        fontsize=15, width=2.1, height=0.78,
        bold=True, text_color="white",
        style="round,pad=0.25", zorder=6,
    )

    branches = mod["branches"]
    n_branches = len(branches)
    branch_r = 3.6
    leaf_r   = 2.2

    b_positions, b_angles = radial_positions(n_branches, branch_r, angle_offset=np.pi / 2)

    # Alternate colors among branches using a palette derived from main color
    # We tint/shade based on index
    branch_accent_colors = [
        "#2C3E50", "#34495E", "#555555", "#3D3D3D", "#4A4A4A"
    ]

    for b_idx, (branch_name, leaves) in enumerate(branches.items()):
        bx, by = b_positions[b_idx]
        bang   = b_angles[b_idx]
        bc     = branch_accent_colors[b_idx % len(branch_accent_colors)]

        draw_bezier(ax, cx, cy, bx, by, color, lw=2.4, alpha=0.55)
        draw_fancy_node(
            ax, bx, by, branch_name, bc,
            fontsize=12, width=2.0, height=0.55,
            bold=True, text_color="white", zorder=5,
        )

        n_leaves = len(leaves)
        spread = min(np.pi * 0.65, np.pi * 0.14 * n_leaves)
        leaf_pos = sub_positions(bx, by, bang, n_leaves,
                                 child_radius=leaf_r, spread=spread)

        for leaf_text, (lx, ly, _) in zip(leaves, leaf_pos):
            draw_bezier(ax, bx, by, lx, ly, color, lw=1.3, alpha=0.45)
            draw_fancy_node(
                ax, lx, ly, leaf_text, color,
                fontsize=9, width=1.65, height=0.40,
                text_color="white", alpha=0.85, zorder=4,
            )

    ax.set_xlim(-9.5, 9.5)
    ax.set_ylim(-8.5, 8.5)

    out = os.path.join(BASE_DIR, mod["filename"])
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    GENERATED.append(out)
    print(f"  [OK] {out}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n=== MLForBeginners Mind Map Generator ===\n")

    # 1. Course overview
    print("Generating course overview...")
    make_course_overview()

    # 2. Per-part maps
    print("\nGenerating per-part mind maps...")
    for part in PART_DATA:
        make_part_mindmap(part)

    # 3. Per-module maps
    print("\nGenerating per-module mind maps...")
    for mod in MODULE_MAPS:
        make_module_mindmap(mod)

    # Summary
    print(f"\n{'='*60}")
    print(f"Done! Generated {len(GENERATED)} mind map(s):")
    for f in GENERATED:
        rel = os.path.relpath(f, BASE_DIR)
        size_kb = os.path.getsize(f) // 1024
        print(f"  {rel:55s}  {size_kb:>5} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
